"""Tests for the cifar10_cnn runner's Dataset_Type → DataLoader dispatch.

Exercises :func:`build_loaders` end-to-end with fake bags. The dispatch
contract:

- Bags with ``Dataset_Type`` containing ``Training``/``Testing``/
  ``Validation`` route to the matching DataLoader.
- ``Split`` parents are flattened to their children before dispatch.
- Qualifier terms like ``Labeled`` are ignored — they don't pick a lane.
- Bags with no recognized role term warn and are skipped.
- ``require_training=True`` raises ``RuntimeError`` if no Training bag
  is found (closes the catalog-18 F40 silent-failure mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from models.cifar10_cnn import _bag_role, build_loaders


class _TinyImageDataset(Dataset):
    """In-memory torch Dataset mimicking ``DatasetBag.as_torch_dataset`` output."""

    def __init__(self, n: int = 4, rid_prefix: str = "X") -> None:
        self._items: list[tuple[torch.Tensor, int, str]] = [
            (torch.zeros(3, 32, 32), i % 10, f"{rid_prefix}-{i}") for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        return self._items[idx]


@dataclass
class _FakeBag:
    """Stand-in for a real ``DatasetBag`` — exposes only what the runner reads."""

    dataset_rid: str
    dataset_types: list[str]
    children: list["_FakeBag"] = field(default_factory=list)
    n_samples: int = 4

    def list_dataset_children(self) -> list["_FakeBag"]:
        return self.children

    def as_torch_dataset(self, **_kwargs: Any) -> _TinyImageDataset:
        return _TinyImageDataset(n=self.n_samples, rid_prefix=self.dataset_rid)


@dataclass
class _FakeExecution:
    """Stand-in for a real ``Execution`` — exposes ``.datasets`` only."""

    datasets: list[_FakeBag]


# ---------------------------------------------------------------------------
# _bag_role: role term picked out of Dataset_Type, qualifiers ignored
# ---------------------------------------------------------------------------


def test_bag_role_picks_training_from_training_labeled():
    bag = _FakeBag(dataset_rid="T", dataset_types=["Training", "Labeled"])
    assert _bag_role(bag) == "training"


def test_bag_role_returns_none_for_unknown_role():
    bag = _FakeBag(dataset_rid="CAL", dataset_types=["Calibration"])
    assert _bag_role(bag) is None


def test_bag_role_is_case_insensitive():
    bag = _FakeBag(dataset_rid="V", dataset_types=["VALIDATION"])
    assert _bag_role(bag) == "validation"


# ---------------------------------------------------------------------------
# build_loaders: training + validation lanes engage together
# ---------------------------------------------------------------------------


def test_training_and_validation_bags_both_engage():
    """Training + Validation bag yields train_loader + val_loader, no test_loader."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training", "Labeled"], n_samples=8)
    val = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"], n_samples=4)
    execution = _FakeExecution(datasets=[train, val])

    train_loader, test_loader, val_loader, class_names = build_loaders(
        execution, batch_size=2, require_training=True
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is None
    assert len(class_names) == 10  # canonical CIFAR-10

    train_rids = [r for batch in train_loader for r in batch[2]]
    val_rids = [r for batch in val_loader for r in batch[2]]
    assert all(rid.startswith("T-") for rid in train_rids)
    assert all(rid.startswith("DAP-") for rid in val_rids)


def test_training_plus_labeled_qualifier_dispatches_to_training():
    """``["Training", "Labeled"]`` routes to the training lane (qualifier ignored)."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training", "Labeled"])
    execution = _FakeExecution(datasets=[train])

    train_loader, test_loader, val_loader, _names = build_loaders(
        execution, batch_size=2, require_training=True
    )
    assert train_loader is not None
    assert test_loader is None
    assert val_loader is None


def test_training_inside_split_parent_dispatches():
    """A Split parent containing Training + Validation children dispatches each."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training"])
    val = _FakeBag(dataset_rid="V", dataset_types=["Validation", "Labeled"])
    parent = _FakeBag(dataset_rid="P", dataset_types=["Split"], children=[train, val])
    execution = _FakeExecution(datasets=[parent])

    train_loader, _test, val_loader, _names = build_loaders(
        execution, batch_size=2, require_training=True
    )
    assert train_loader is not None
    assert val_loader is not None


# ---------------------------------------------------------------------------
# Safety rail — closes the catalog-18 F40 silent-failure mode
# ---------------------------------------------------------------------------


def test_validation_only_input_raises_when_training_required():
    """Validation-only input with ``require_training=True`` raises a clear error.

    The catalog-18 F40 regression: a Validation-typed dataset used to
    silently produce a degenerate "training run" that landed as
    ``Status=Uploaded``. The fix is to fail loudly so the execution
    ends as ``Status=Failed``.
    """
    val = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"])
    execution = _FakeExecution(datasets=[val])

    with pytest.raises(RuntimeError) as excinfo:
        build_loaders(execution, batch_size=2, require_training=True)

    msg = str(excinfo.value)
    assert "Dataset_Type=Training" in msg
    assert "DAP" in msg
    assert "Validation" in msg


def test_validation_only_input_is_fine_when_training_not_required():
    """``require_training=False`` (test_only mode) doesn't trip the safety rail."""
    val = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"])
    execution = _FakeExecution(datasets=[val])

    train_loader, test_loader, val_loader, _names = build_loaders(
        execution, batch_size=2, require_training=False
    )
    assert train_loader is None
    assert test_loader is None
    assert val_loader is not None


def test_unrecognized_role_emits_warning_and_is_skipped():
    """A bag with no recognized role term warns rather than silently dropping."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training"])
    weird = _FakeBag(dataset_rid="CAL", dataset_types=["Calibration"])
    execution = _FakeExecution(datasets=[train, weird])

    with pytest.warns(RuntimeWarning, match="Calibration"):
        train_loader, test_loader, val_loader, _names = build_loaders(
            execution, batch_size=2, require_training=True
        )

    assert train_loader is not None
    assert test_loader is None
    assert val_loader is None
