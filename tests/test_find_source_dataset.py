"""Offline unit tests for _find_latest_source_dataset_rid.

These tests use stub objects and require no live catalog connection.
"""

import types

import pytest

from scripts.load_cifar10 import _find_latest_source_dataset_rid


def _make_dataset(
    rid: str,
    dataset_types: list[str],
    source_directory: str | None,
    is_source_root: bool = False,
):
    """Return a lightweight stub Dataset object with the required properties.

    Args:
        rid: The dataset RID string.
        dataset_types: List of Dataset_Type term names for this dataset.
        source_directory: The directory path stored on the dataset (its
            basename for the root of an ``add_files`` tree, ``"train"``/
            ``"test"`` for the partition children).
        is_source_root: Whether this dataset is the root of an ``add_files``
            tree. ``_find_latest_source_dataset_rid`` selects the root via this
            structural flag, not by matching ``source_directory`` to a string.

    Returns:
        A ``types.SimpleNamespace`` instance with ``dataset_rid``,
        ``dataset_types``, ``source_directory``, and ``is_source_root``
        attributes.

    Example:
        >>> d = _make_dataset("2-XXXX", ["CIFAR_Source"], "cifar10_source", is_source_root=True)
        >>> d.is_source_root
        True
    """
    return types.SimpleNamespace(
        dataset_rid=rid,
        dataset_types=dataset_types,
        source_directory=source_directory,
        is_source_root=is_source_root,
    )


def _make_ml(datasets: list):
    """Return a stub DerivaML instance whose ``find_datasets`` yields the given list.

    The stub honours the ``sort`` keyword argument by returning the list
    as-is (callers are expected to pass datasets in newest-first order when
    ``sort=True`` is assumed).

    Args:
        datasets: List of stub Dataset objects to return from ``find_datasets``.

    Returns:
        A ``types.SimpleNamespace`` instance with a ``find_datasets`` method.

    Example:
        >>> ml = _make_ml([])
        >>> list(ml.find_datasets(sort=True))
        []
    """

    def find_datasets(deleted: bool = False, sort=None):
        return iter(datasets)

    return types.SimpleNamespace(find_datasets=find_datasets)


class TestFindLatestSourceDatasetRid:
    """Offline tests for _find_latest_source_dataset_rid."""

    def test_returns_root_cifar_source_rid(self):
        """Returns the RID of the root CIFAR_Source dataset, ignoring children and unrelated datasets.

        A root CIFAR_Source dataset has ``is_source_root == True`` and
        ``"CIFAR_Source"`` in ``dataset_types``.  Non-root children
        (``source_directory == "train"``/``"test"``, ``is_source_root`` False)
        and unrelated datasets must be ignored.

        Example:
            This test is deterministic and requires no catalog connection.
        """
        datasets = [
            _make_dataset("2-ROOT", ["CIFAR_Source"], "cifar10_source", is_source_root=True),
            _make_dataset("2-TRAIN", ["CIFAR_Source"], "train", is_source_root=False),
            _make_dataset("2-TEST", ["CIFAR_Source"], "test", is_source_root=False),
            _make_dataset("2-OTHER", ["SomeOtherType"], "cifar10_source", is_source_root=True),
        ]
        ml = _make_ml(datasets)
        rid = _find_latest_source_dataset_rid(ml)
        assert rid == "2-ROOT"

    def test_raises_runtime_error_when_no_matching_dataset(self):
        """Raises RuntimeError mentioning 'CIFAR_Source' when no matching dataset exists.

        Example:
            This test is deterministic and requires no catalog connection.
        """
        datasets = [
            _make_dataset("2-TRAIN", ["CIFAR_Source"], "train", is_source_root=False),
            _make_dataset("2-OTHER", ["SomeOtherType"], "cifar10_source", is_source_root=True),
        ]
        ml = _make_ml(datasets)
        with pytest.raises(RuntimeError, match="CIFAR_Source"):
            _find_latest_source_dataset_rid(ml)

    def test_returns_newest_when_multiple_roots_exist(self):
        """Returns the first (newest) CIFAR_Source root when several exist.

        The stub returns datasets in newest-first order (simulating
        ``find_datasets(sort=True)``).  The helper must take ``candidates[0]``,
        which is the newest root.

        Example:
            This test is deterministic and requires no catalog connection.
        """
        # Newest first, as find_datasets(sort=True) would return them.
        datasets = [
            _make_dataset("2-NEWER", ["CIFAR_Source"], "cifar10_source", is_source_root=True),
            _make_dataset("2-OLDER", ["CIFAR_Source"], "cifar10_source", is_source_root=True),
        ]
        ml = _make_ml(datasets)
        rid = _find_latest_source_dataset_rid(ml)
        assert rid == "2-NEWER"
