"""Regression test for train/test leakage in labeled split datasets.

This test guards the fix for findings/curator/02 and
findings/evaluator/02 from the 2026-05-28 e2e run. The bug was that
``_cifar10_datasets.py`` passed
``row_per="Execution_Image_Image_Classification"`` to
``split_dataset``, which partitioned feature *rows* instead of image
RIDs. On any catalog where the loader produced more than one feature
row per image, an image's two rows could land on opposite sides of the
split, putting the same image in both train and test partitions.

The fix (per the 2026-05-28 denormalizer audit, §1) replaces the
feature-association table with the vocab table in ``include_tables``
and adds ``partition_by="element"`` (deriva-ml PR #254) so the
selector layer dedupes per element-RID and enforces a disjoint-image
invariant after the split. This test exercises a fresh catalog
end-to-end and asserts:

1. ``cifar10_labeled_split`` Training ∩ Testing on image RIDs is
   empty.
2. ``cifar10_small_labeled_split`` Training ∩ Testing on image RIDs
   is empty.
3. Each partition's actual unique-image count matches the advertised
   size — 440/110 for ``cifar10_labeled_split`` and 400/100 for
   ``cifar10_small_labeled_split``.

The test is gated on ``DERIVA_ML_LIVE_LOCALHOST=1`` so unit-test
suites in CI without a live ``dev-localhost`` container skip it. To
run it locally::

    export DERIVA_ML_LIVE_LOCALHOST=1
    uv run python -m pytest tests/test_load_cifar10_split_no_leakage.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest

LIVE = os.environ.get("DERIVA_ML_LIVE_LOCALHOST") == "1"
HOSTNAME = "localhost"

# Advertised partition sizes per src/configs/datasets.py and CLAUDE.md.
LABELED_TRAIN_SIZE = 440
LABELED_TEST_SIZE = 110
SMALL_LABELED_TRAIN_SIZE = 400
SMALL_LABELED_TEST_SIZE = 100

# Per the C.A.3 bootstrap floor: 1100 images is the smallest count
# that lets the dataset hierarchy build both the small variant
# (>= 500 train images) and the labeled holdouts.
NUM_IMAGES = 1100


pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="DERIVA_ML_LIVE_LOCALHOST=1 required; needs the dev-localhost container.",
)


def _run_cli(*args: str) -> None:
    """Invoke the ``load-cifar10`` CLI under ``uv run`` and fail loudly.

    Args:
        *args: Arguments to forward to ``load-cifar10`` (everything
            after the command name).

    Raises:
        AssertionError: If the CLI exits with a non-zero status. The
            assertion message includes the full ``stdout``/``stderr``
            so failures in CI logs are diagnosable without re-running.
    """
    cmd = ["uv", "run", "load-cifar10", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"load-cifar10 failed (exit {result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _fetch_image_rids(ml, dataset_rid: str) -> set[str]:
    """Return the set of Image RIDs that are members of ``dataset_rid``.

    Uses ``Dataset.list_dataset_members`` so the test does not depend on
    the domain schema's association table layout.

    Args:
        ml: Connected :class:`DerivaML` instance.
        dataset_rid: Dataset RID to query.

    Returns:
        Set of image RIDs that are members of the dataset.
    """
    ds = ml.lookup_dataset(dataset_rid)
    members = ds.list_dataset_members()
    return {r["RID"] for r in members.get("Image", [])}


def test_split_dataset_partitions_by_image_not_feature_row(tmp_path):
    """End-to-end: a fresh load-cifar10 catalog must not leak split rows.

    Creates a throwaway catalog, runs the three load-cifar10 phases,
    queries the resulting labeled-split datasets via PathBuilder, and
    asserts the train/test partitions are disjoint on image RIDs and
    that the actual sizes match the advertised sizes.

    Cleans up the catalog in a try/finally so a failure mid-run still
    deletes the test catalog.
    """
    from deriva.core import DerivaServer, get_credential
    from deriva_ml import DerivaML

    project_name = f"no-leakage-final-{uuid.uuid4().hex[:8]}"

    # --- Phase 1: schema (creates the catalog and prints its id). ---
    _run_cli(
        "--hostname",
        HOSTNAME,
        "--create-catalog",
        project_name,
        "--phase",
        "schema",
    )

    # The alias resolves to the freshly-created catalog id.
    server = DerivaServer("https", HOSTNAME, credentials=get_credential(HOSTNAME))
    alias = server.connect_ermrest_alias(project_name)
    catalog_id = str(alias.retrieve()["alias_target"])
    catalog = server.connect_ermrest(catalog_id)

    try:
        # --- Phase 2: images. ---
        _run_cli(
            "--hostname",
            HOSTNAME,
            "--catalog-id",
            catalog_id,
            "--phase",
            "images",
            "--num-images",
            str(NUM_IMAGES),
        )

        # --- Phase 3: datasets (builds labeled splits). ---
        _run_cli(
            "--hostname",
            HOSTNAME,
            "--catalog-id",
            catalog_id,
            "--phase",
            "datasets",
            "--num-images",
            str(NUM_IMAGES),
        )

        # --- Query the catalog and find the labeled-split children. ---
        ml = DerivaML(hostname=HOSTNAME, catalog_id=catalog_id)
        pb = ml.catalog.getPathBuilder()
        dataset_tbl = pb.schemas["deriva-ml"].Dataset

        # Find the labeled and small_labeled split parent datasets.
        # Their Description column starts with the strings emitted by
        # ``_labeled_split_description`` / ``_small_labeled_split_description``.
        all_datasets = dataset_tbl.entities().fetch()
        labeled_parent = None
        small_labeled_parent = None
        for r in all_datasets:
            desc = r.get("Description") or ""
            if desc.startswith("Small CIFAR-10 labeled split:"):
                small_labeled_parent = r["RID"]
            elif desc.startswith("CIFAR-10 labeled split:"):
                labeled_parent = r["RID"]
        assert labeled_parent is not None, (
            "Could not find labeled_split parent in catalog."
        )
        assert small_labeled_parent is not None, (
            "Could not find small_labeled_split parent in catalog."
        )

        # Use deriva-ml's Dataset_Dataset_Type association directly to
        # discover Training/Testing children. ``split_dataset`` tags
        # each child dataset with the "Training" or "Testing" dataset
        # type; we read those tags via the association table.
        ds_dt = pb.schemas["deriva-ml"].Dataset_Dataset_Type

        def _train_test_children(parent_rid: str) -> tuple[str, str]:
            """Return (train_rid, test_rid) for a labeled-split parent.

            Identifies children via ``Dataset.list_dataset_children`` and
            disambiguates train vs test by the "Training"/"Testing"
            dataset-type tags applied by ``split_dataset``.

            Args:
                parent_rid: RID of the parent split dataset.

            Returns:
                Tuple of (train_dataset_rid, test_dataset_rid).
            """
            parent = ml.lookup_dataset(parent_rid)
            child_rids = [c.dataset_rid for c in parent.list_dataset_children()]
            train_rid = test_rid = None
            for crid in child_rids:
                types = {
                    r["Dataset_Type"]
                    for r in ds_dt.filter(ds_dt.Dataset == crid).entities().fetch()
                }
                # Type tag names: "Training" and "Testing" are applied
                # by split_dataset; "Labeled" is the extra training/
                # testing_types we pass.
                if "Training" in types:
                    train_rid = crid
                elif "Testing" in types:
                    test_rid = crid
            assert train_rid is not None and test_rid is not None, (
                f"Could not find Training/Testing children for {parent_rid}; "
                f"children were {child_rids}."
            )
            return train_rid, test_rid

        labeled_train, labeled_test = _train_test_children(labeled_parent)
        small_train, small_test = _train_test_children(small_labeled_parent)

        labeled_train_rids = _fetch_image_rids(ml, labeled_train)
        labeled_test_rids = _fetch_image_rids(ml, labeled_test)
        small_train_rids = _fetch_image_rids(ml, small_train)
        small_test_rids = _fetch_image_rids(ml, small_test)

        # Print sizes so a failure tells us by how much we drifted.
        print(
            f"\nlabeled_split:        train={len(labeled_train_rids)} "
            f"test={len(labeled_test_rids)} "
            f"overlap={len(labeled_train_rids & labeled_test_rids)}",
            file=sys.stderr,
        )
        print(
            f"small_labeled_split:  train={len(small_train_rids)} "
            f"test={len(small_test_rids)} "
            f"overlap={len(small_train_rids & small_test_rids)}",
            file=sys.stderr,
        )

        # --- Assertions: disjointness. ---
        assert not (labeled_train_rids & labeled_test_rids), (
            f"labeled_split leaks: {len(labeled_train_rids & labeled_test_rids)} "
            f"image RIDs appear in both train and test."
        )
        assert not (small_train_rids & small_test_rids), (
            f"small_labeled_split leaks: "
            f"{len(small_train_rids & small_test_rids)} image RIDs appear in "
            f"both train and test."
        )

        # --- Assertions: actual sizes match advertised sizes. ---
        assert len(labeled_train_rids) == LABELED_TRAIN_SIZE, (
            f"labeled_split train size {len(labeled_train_rids)} != "
            f"advertised {LABELED_TRAIN_SIZE}."
        )
        assert len(labeled_test_rids) == LABELED_TEST_SIZE, (
            f"labeled_split test size {len(labeled_test_rids)} != "
            f"advertised {LABELED_TEST_SIZE}."
        )
        assert len(small_train_rids) == SMALL_LABELED_TRAIN_SIZE, (
            f"small_labeled_split train size {len(small_train_rids)} != "
            f"advertised {SMALL_LABELED_TRAIN_SIZE}."
        )
        assert len(small_test_rids) == SMALL_LABELED_TEST_SIZE, (
            f"small_labeled_split test size {len(small_test_rids)} != "
            f"advertised {SMALL_LABELED_TEST_SIZE}."
        )

    finally:
        # Tear down the throwaway catalog so repeated test runs don't
        # accrete catalogs on dev-localhost.
        try:
            catalog.delete_ermrest_catalog(really=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: failed to delete throwaway catalog {catalog_id}: {e}",
                file=sys.stderr,
            )
