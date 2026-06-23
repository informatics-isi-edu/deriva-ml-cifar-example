"""Unit tests for build_loaders error paths in src/models/cifar10_cnn.py.

These tests use minimal mock objects rather than a real Execution because
the failure paths we care about are entirely upstream of bag download —
the function reads execution.datasets and decides what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from models.cifar10_cnn import (
    _LANE_CONFIGS,
    _ROLE_TESTING,
    _ROLE_TRAINING,
    _ROLE_VALIDATION,
    build_loaders,
)


@dataclass
class _MockExecution:
    """Minimal stand-in for deriva_ml.execution.Execution.

    build_loaders only reads execution.datasets, so that's all we model.
    """

    datasets: list[Any] = field(default_factory=list)


def test_empty_datasets_raises_clear_error() -> None:
    """When execution.datasets is empty, build_loaders raises with a
    clear "no input datasets" message — distinct from the deeper
    "no Training bag after flattening" failure, so the user can
    diagnose the misconfig (typically: a Hydra `datasets=foo` group
    that resolved to an empty list, e.g. a placeholder registry in
    src/configs/datasets.py that wasn't filled in for the catalog).
    """
    execution = _MockExecution(datasets=[])

    with pytest.raises(RuntimeError) as exc_info:
        build_loaders(execution, batch_size=32, require_training=True)

    message = str(exc_info.value)
    assert "no input datasets" in message.lower() or "empty" in message.lower(), (
        f"Expected the error to mention empty/no datasets; got: {message!r}"
    )


# -----------------------------------------------------------------------------
# Tests for the role-dispatch table introduced in Task 5.
# These reference _LANE_CONFIGS, which is added in Task 5;
# the tests live here so they run as part of the same test file.
# -----------------------------------------------------------------------------


def test_lane_configs_cover_all_leaf_roles() -> None:
    """Every leaf role has an entry in the dispatch table.

    If a new role is added in _LEAF_ROLES without a matching
    _LANE_CONFIGS entry, _make_loader will KeyError at runtime.
    Catch that early.
    """
    assert _ROLE_TRAINING in _LANE_CONFIGS
    assert _ROLE_TESTING in _LANE_CONFIGS
    assert _ROLE_VALIDATION in _LANE_CONFIGS


def test_training_lane_shuffles_with_seeded_generator() -> None:
    """Training lane: shuffle=True, seeded generator when seed is set."""
    cfg = _LANE_CONFIGS[_ROLE_TRAINING]
    assert cfg.shuffle is True
    assert cfg.use_seeded_generator is True
    assert cfg.missing == "skip"


def test_testing_lane_keeps_unlabeled_rows_no_shuffle() -> None:
    """Testing lane: missing='unknown' so we keep unlabeled rows for
    later prediction recording; shuffle=False; no seeded generator."""
    cfg = _LANE_CONFIGS[_ROLE_TESTING]
    assert cfg.shuffle is False
    assert cfg.use_seeded_generator is False
    assert cfg.missing == "unknown"


def test_validation_lane_skips_unlabeled_no_shuffle() -> None:
    """Validation lane: shuffle=False, skip unlabeled (loss/accuracy
    are undefined for unlabeled rows). No seeded generator."""
    cfg = _LANE_CONFIGS[_ROLE_VALIDATION]
    assert cfg.shuffle is False
    assert cfg.use_seeded_generator is False
    assert cfg.missing == "skip"
