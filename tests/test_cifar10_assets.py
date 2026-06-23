"""Smoke tests for src/scripts/_cifar10_assets.py.

Stage 2 needs a live Deriva catalog for its actual work, so
the module-structure tests are sparse — end-to-end behavior is
exercised in the load-cifar10 smoke test in Task A13. The
stratified-sampling helper is pure and is tested directly here.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _make_class_balanced_corpus(per_class: int) -> tuple[list[Path], dict[str, str]]:
    """Build (paths, labels) with per_class items in each of 10 classes."""
    paths: list[Path] = []
    labels: dict[str, str] = {}
    for cls in CIFAR10_CLASSES:
        for i in range(per_class):
            p = Path(f"{cls}_s_{i:06d}.png")
            paths.append(p)
            labels[p.stem] = cls
    return paths, labels


def test_module_exposes_expected_api():
    from scripts._cifar10_assets import (
        upload_images,
        add_classification_features,
        run_assets_phase,
        class_from_filename,
        stratified_sample_by_class,
    )

    for fn in (
        upload_images,
        add_classification_features,
        run_assets_phase,
        class_from_filename,
        stratified_sample_by_class,
    ):
        assert callable(fn)


def test_class_from_filename_decodes_train():
    from scripts._cifar10_assets import class_from_filename

    assert class_from_filename("train_frog_42.png") == "frog"


def test_class_from_filename_decodes_test():
    from scripts._cifar10_assets import class_from_filename

    assert class_from_filename("test_cat_19.png") == "cat"


def test_class_from_filename_returns_none_for_unknown():
    from scripts._cifar10_assets import class_from_filename

    assert class_from_filename("random_image.png") is None
    assert class_from_filename("train.png") is None


def test_stratified_sample_balances_when_divisible_by_classes():
    """At --num-images 100 (10 per class), every class gets exactly 10."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=500)
    sample = stratified_sample_by_class(paths, labels, sample_size=100, seed=42)

    assert len(sample) == 100
    counts = Counter(labels[p.stem] for p in sample)
    assert set(counts.keys()) == set(CIFAR10_CLASSES)
    assert all(n == 10 for n in counts.values()), counts


def test_stratified_sample_is_deterministic_under_seed():
    """Same seed -> same sample (set equality, since final shuffle differs)."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=200)
    s1 = stratified_sample_by_class(paths, labels, sample_size=100, seed=42)
    s2 = stratified_sample_by_class(paths, labels, sample_size=100, seed=42)
    assert s1 == s2

    # Different seed should not (with high probability) give the same set
    s3 = stratified_sample_by_class(paths, labels, sample_size=100, seed=99)
    assert set(s3) != set(s1)


def test_stratified_sample_distributes_remainder():
    """sample_size=15 across 10 classes: 5 classes get 2, others get 1."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=100)
    sample = stratified_sample_by_class(paths, labels, sample_size=15, seed=42)

    assert len(sample) == 15
    counts = Counter(labels[p.stem] for p in sample)
    assert sum(counts.values()) == 15
    # No class with zero items (15 >= 10 classes).
    assert all(c in counts for c in CIFAR10_CLASSES)
    # Each class gets either 1 or 2 (base quota 1 + 0 or 1 from remainder).
    assert all(n in (1, 2) for n in counts.values()), counts


def test_stratified_sample_warns_when_below_class_count(caplog):
    """sample_size=5 < 10 classes: warn and degrade to biased sample."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=10)
    with caplog.at_level(logging.WARNING, logger="scripts._cifar10_assets"):
        sample = stratified_sample_by_class(paths, labels, sample_size=5, seed=42)

    assert len(sample) == 5
    # A warning was emitted explaining the bias.
    assert any("class-biased" in r.message for r in caplog.records)


def test_stratified_sample_returns_all_when_size_exceeds_corpus():
    """sample_size > len(items) returns the full known-class set."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=3)
    sample = stratified_sample_by_class(paths, labels, sample_size=100, seed=42)

    assert len(sample) == 30  # 10 classes * 3
    assert Counter(labels[p.stem] for p in sample) == Counter(
        {c: 3 for c in CIFAR10_CLASSES}
    )


def test_stratified_sample_none_returns_full_corpus():
    """sample_size=None returns all known-class items (shuffled deterministically)."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=4)
    sample = stratified_sample_by_class(paths, labels, sample_size=None, seed=42)

    assert len(sample) == 40
    assert set(sample) == set(paths)


def test_stratified_sample_skips_unlabeled_items():
    """Items whose stem is missing from ``labels`` are silently dropped."""
    from scripts._cifar10_assets import stratified_sample_by_class

    paths, labels = _make_class_balanced_corpus(per_class=10)
    extras = [Path("unknown_x.png"), Path("unknown_y.png")]
    sample = stratified_sample_by_class(paths + extras, labels, sample_size=50, seed=42)

    assert len(sample) == 50
    assert all(labels.get(p.stem) is not None for p in sample)
