"""Regression tests for loader retry idempotence.

These guard the fix for
``findings/evaluator/01-loader-retry-leaves-orphaned-gt-feature-rows.md``
in the 2026-05-28 e2e run: re-running ``load-cifar10 --phase images``
on a catalog where a prior images-phase has already written
``Image_Classification`` ground-truth rows must end up with one row per
image, not the sum of every retry attempt.

The live test needs a real Deriva server (the standard
``dev-localhost`` container the README §1 setup uses) and is gated by
the ``DERIVA_ML_LIVE_LOCALHOST=1`` environment variable so a default
``uv run python -m pytest tests/`` on a fresh checkout — where no
localhost catalog is available — still passes.

Pure (no-catalog) tests in this file run unconditionally.
"""

from __future__ import annotations

import argparse
import os
import uuid

import pytest

LIVE_GATE = pytest.mark.skipif(
    os.environ.get("DERIVA_ML_LIVE_LOCALHOST") != "1",
    reason=(
        "live-localhost test gated by DERIVA_ML_LIVE_LOCALHOST=1; "
        "requires a running dev-localhost Deriva server"
    ),
)


def test_truncate_helper_is_exposed_for_callers():
    """The retry-idempotence helper must be importable by tests + tooling."""
    from scripts._cifar10_assets import _truncate_loader_classification_rows

    assert callable(_truncate_loader_classification_rows)


def _count_loader_rows(ml) -> int:
    """Count Image_Classification rows with Confidence IS NULL.

    These are the loader-written ground-truth rows (training
    executions populate Confidence — see tk-001 in the 2026-05-28
    e2e run's tacit-knowledge.md).
    """
    feat = ml.lookup_feature("Image", "Image_Classification")
    pb = ml.pathBuilder()
    feature_path = pb.schemas[feat.feature_table.schema.name].tables[
        feat.feature_table.name
    ]
    rows = list(
        feature_path.filter(feature_path.Confidence == None)  # noqa: E711
        .entities()
        .fetch()
    )
    return len(rows)


def _count_unique_loader_images(ml) -> int:
    """Count unique Image RIDs touched by loader rows."""
    feat = ml.lookup_feature("Image", "Image_Classification")
    pb = ml.pathBuilder()
    feature_path = pb.schemas[feat.feature_table.schema.name].tables[
        feat.feature_table.name
    ]
    rows = list(
        feature_path.filter(feature_path.Confidence == None)  # noqa: E711
        .entities()
        .fetch()
    )
    return len({r["Image"] for r in rows})


@LIVE_GATE
def test_phase_images_retry_does_not_accumulate_rows():
    """Reproduces evaluator/01 and guards against regression.

    1. Create a throwaway catalog.
    2. Run --phase schema, then --phase images at --num-images 200.
       Expect 200 loader rows, 200 unique images.
    3. Run --phase images again at --num-images 400 (the retry).
       Expect 400 loader rows (NOT 600), 400 unique images.
    4. Delete the throwaway catalog.

    Before the fix, step 3 left the original 200 rows behind and
    added a fresh 400, yielding 600 rows over only 400 unique
    images — the 1600/1100 ratio the e2e curator/01 finding
    recorded, scaled down.

    Gated by ``DERIVA_ML_LIVE_LOCALHOST=1`` because it needs a
    running localhost Deriva server. See module docstring.
    """
    from scripts._cifar10_assets import run_assets_phase
    from scripts._cifar10_schema import create_or_connect_catalog, run_schema_phase

    suffix = uuid.uuid4().hex[:8]
    alias = f"loader-retry-test-{suffix}"

    args = argparse.Namespace(
        hostname="localhost",
        catalog_id=None,
        create_catalog=alias,
        domain_schema=None,
    )

    ml, catalog_id, _ = create_or_connect_catalog(args)
    try:
        run_schema_phase(ml, alias)

        # First pass: 200 images.
        first = run_assets_phase(ml, max_images=200)
        assert first["features_added"] == 200
        assert _count_loader_rows(ml) == 200
        assert _count_unique_loader_images(ml) == 200

        # Second pass: 400 images (the retry).
        second = run_assets_phase(ml, max_images=400)
        assert second["features_added"] == 400
        # The fix's signature: prior loader rows were truncated.
        assert second["prior_rows_truncated"] == 200

        # The regression check: no accumulation across retries.
        assert _count_loader_rows(ml) == 400, (
            "loader-retry accumulation bug: prior --phase images rows "
            "were not truncated before the retry. See "
            "findings/evaluator/01-loader-retry-leaves-orphaned-gt-feature-rows.md."
        )
        assert _count_unique_loader_images(ml) == 400
    finally:
        # Always tear down — leaving throwaway catalogs around clutters
        # the localhost server's catalog index.
        ml.catalog.delete_ermrest_catalog(really=True)
