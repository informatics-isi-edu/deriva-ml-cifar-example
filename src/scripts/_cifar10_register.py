"""CIFAR-10 Execution 1: source-image registration scaffold.

This module demonstrates the **source-registration pattern**: how to
stage a class-balanced sample of raw source images into a stable cache
directory and register them as a named, nested File dataset in the
catalog — *before* uploading the pixel data as Image assets.

The result of ``run_register_phase`` is a root File dataset RID whose
children are ``train/`` and ``test/`` sub-datasets, mirroring the
Toronto train/test split at the provenance layer.  Execution 2
(Task 6) reads those File rows and uploads the actual image bytes.

Public API:
    - ``stage_source(max_images, cache_root)`` — download + extract +
      stratified-sample → populate ``cache_root/train`` and
      ``cache_root/test`` with symlinks, write ``labels.csv``,
      return ``cache_root``.
    - ``run_register_phase(ml, max_images, cache_root)`` — create
      Execution 1, call ``exe.add_files(...)`` on the staged tree,
      return root File dataset RID.
    - ``class_from_filename(filename)`` — pure helper: decode CIFAR-10
      class from a filename of the form ``train_<class>_<id>.png``.
    - ``stratified_sample_by_class(items, labels, sample_size, seed)``
      — class-balanced sampling helper.
    - ``DEFAULT_SAMPLE_SEED`` — default RNG seed (42).

.. note::
    Lines marked ``# DOMAIN: replace for your data`` are the seams to
    edit when adapting this scaffold to a different dataset.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from deriva_ml import DerivaML
from deriva_ml.core.filespec import FileSpec
from deriva_ml.execution import ExecutionConfiguration

from scripts._cifar10_source import (
    CIFAR10_SOURCE_CACHE,
    download_cifar10_archive,
    extract_cifar10_to_png,
    write_labels_manifest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default seed for reproducible stratified sampling.
#: CIFAR-10 has 10 classes; when the requested sample size is below 10 we
#: cannot give every class at least one representative and fall back to a
#: deterministic first-N sample with a warning.
DEFAULT_SAMPLE_SEED = 42  # DOMAIN: replace for your data

#: Known CIFAR-10 class names (for ``class_from_filename`` validation).
CIFAR10_CLASSES_FROZEN = frozenset(  # DOMAIN: replace for your data
    {
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
    }
)


# ---------------------------------------------------------------------------
# Pure helpers (moved from _cifar10_assets.py)
# ---------------------------------------------------------------------------


def class_from_filename(filename: str) -> str | None:
    """Decode the CIFAR-10 class from an image filename.

    Image filenames produced by the upload stage have the shape
    ``train_<class>_<id>.png`` or ``test_<class>_<id>.png``,
    where ``<class>`` is one of the ten CIFAR-10 class names.
    This helper extracts the class name; returns ``None`` if
    the filename doesn't follow the expected pattern or the
    decoded class isn't a known CIFAR-10 class.

    Args:
        filename: Image filename (with or without leading path).

    Returns:
        The class name if the filename decodes cleanly,
        otherwise ``None``.

    Example:
        >>> class_from_filename("train_frog_42.png")
        'frog'
        >>> class_from_filename("test_cat_19.png")
        'cat'
        >>> class_from_filename("random.png") is None
        True
    """
    stem = Path(filename).name
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    if parts[0] not in ("train", "test"):
        return None
    candidate = parts[1]
    if candidate not in CIFAR10_CLASSES_FROZEN:  # DOMAIN: replace for your data
        return None
    return candidate


def stratified_sample_by_class(
    items: list[Path],
    labels: dict[str, str],
    sample_size: int | None,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> list[Path]:
    """Pick a class-balanced sample of image paths.

    Groups ``items`` by their CIFAR-10 class (looked up in ``labels``
    via each path's stem) and returns ``sample_size`` paths split as
    evenly as possible across the classes present. The result preserves
    determinism for a given ``seed``: each class's items are shuffled
    with the seed, the per-class quota is taken from the front, and
    the concatenated result is shuffled once more.

    Args:
        items: Candidate image paths. Items whose stem is missing from
            ``labels`` are skipped.
        labels: Mapping of ``image_stem -> class_name`` (the same
            mapping returned by :func:`extract_cifar10_to_png`).
        sample_size: How many paths to return. If ``None`` or larger
            than ``len(items)``, returns all known-class items shuffled
            deterministically.
        seed: Seed for the per-class and final shuffles. Default 42.

    Returns:
        A list of up to ``sample_size`` image paths with roughly
        balanced class representation.

    Notes:
        When ``sample_size`` is smaller than the number of available
        classes, every class cannot be represented; the function still
        spreads quota one-per-class until the budget is exhausted and
        emits a warning so callers know the resulting sample is biased.

    Example:
        >>> # 5 classes, 10 paths, balanced sample of 5
        >>> paths = [Path(f"x_{i}.png") for i in range(10)]
        >>> labs = {p.stem: ["a", "b", "c", "d", "e"][i % 5]
        ...         for i, p in enumerate(paths)}
        >>> sample = stratified_sample_by_class(paths, labs, 5, seed=1)
        >>> sorted(labs[p.stem] for p in sample)
        ['a', 'b', 'c', 'd', 'e']
    """
    # Group by class. Skip items whose label can't be resolved.
    by_class: dict[str, list[Path]] = {}
    for path in items:
        cls = labels.get(path.stem)
        if cls is None:
            continue
        by_class.setdefault(cls, []).append(path)

    total_known = sum(len(v) for v in by_class.values())
    if sample_size is None or sample_size >= total_known:
        # Take everything we know, but still shuffle deterministically
        # so downstream slicing (e.g. train/test halves) is class-mixed.
        rng = random.Random(seed)
        flat = [p for v in by_class.values() for p in v]
        rng.shuffle(flat)
        return flat

    num_classes = len(by_class)
    if num_classes == 0:
        return []

    if sample_size < num_classes:
        logger.warning(
            "Stratified sample requested for %d items but %d classes "
            "are available; result will be class-biased (every class "
            "cannot be represented at this size).",
            sample_size,
            num_classes,
        )

    # Deterministic per-class shuffle, then assign the quota.
    class_rng = random.Random(seed)
    sorted_classes = sorted(by_class.keys())  # stable ordering across runs
    shuffled_by_class: dict[str, list[Path]] = {}
    for cls in sorted_classes:
        bucket = list(by_class[cls])
        class_rng.shuffle(bucket)
        shuffled_by_class[cls] = bucket

    base_quota = sample_size // num_classes
    remainder = sample_size % num_classes

    picked: list[Path] = []
    # Each class gets base_quota, plus the first ``remainder`` classes
    # (after a deterministic shuffle of the class order) get one extra
    # so the remainder is spread, not biased to alphabetical leaders.
    order_rng = random.Random(seed + 1)
    class_order = list(sorted_classes)
    order_rng.shuffle(class_order)
    extras = set(class_order[:remainder])

    for cls in sorted_classes:
        quota = base_quota + (1 if cls in extras else 0)
        bucket = shuffled_by_class[cls]
        picked.extend(bucket[:quota])

    # If a class had fewer items than its quota, top up from other
    # classes' remainders so we still return ``sample_size`` items.
    if len(picked) < sample_size:
        already = set(picked)
        leftover: list[Path] = []
        for cls in sorted_classes:
            bucket = shuffled_by_class[cls]
            quota = base_quota + (1 if cls in extras else 0)
            leftover.extend(bucket[quota:])
        # Deterministic shuffle of leftovers for fairness.
        leftover_rng = random.Random(seed + 2)
        leftover_rng.shuffle(leftover)
        for path in leftover:
            if len(picked) >= sample_size:
                break
            if path in already:
                continue
            picked.append(path)

    # Final shuffle so subsequent slicing isn't class-clustered.
    final_rng = random.Random(seed + 3)
    final_rng.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# Stage function
# ---------------------------------------------------------------------------


def stage_source(
    max_images: int | None,
    cache_root: Path = CIFAR10_SOURCE_CACHE,
) -> Path:
    """Download, extract, and stage a sampled CIFAR-10 source tree.

    Populates ``cache_root/train/`` and ``cache_root/test/`` with
    symlinks to the sampled PNGs (no byte copy) and writes a
    ``labels.csv`` manifest at ``cache_root/``.  Any prior contents of
    ``cache_root`` are removed before staging so the directory always
    reflects exactly the current sample.

    The labels written to ``labels.csv`` cover exactly the sampled
    files (filename stem → class).  The upload execution (Execution 2)
    reads this manifest rather than carrying an in-memory labels dict
    across the execution boundary.

    Args:
        max_images: Total number of images to stage (split evenly
            between train and test).  ``None`` stages the full corpus
            (~60 K images).
        cache_root: Root directory for the staged source tree.
            Defaults to :data:`CIFAR10_SOURCE_CACHE`.

    Returns:
        ``cache_root`` (the path passed in).

    Example:
        >>> root = stage_source(max_images=100)  # doctest: +SKIP
        >>> sorted(root.iterdir())  # doctest: +SKIP
        [PosixPath('...labels.csv'), PosixPath('...test'), PosixPath('...train')]
    """
    # Clear any prior staging so stale files don't accumulate.  # DOMAIN: replace for your data
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)

    archive_path = download_cifar10_archive()  # DOMAIN: replace for your data

    # Extract into a subdirectory of cache_root so symlink targets stay
    # alive for the lifetime of the cache.  The ``_extract/`` dir and the
    # ``train/``/``test/`` symlink dirs are both under cache_root; the
    # entire tree is cleared atomically by ``shutil.rmtree(cache_root)``
    # at the top of the next call — no stale targets accumulate.
    extract_root = cache_root / "_extract"
    train_dir, test_dir, labels = extract_cifar10_to_png(archive_path, extract_root)

    # Class-balanced sampling — split max_images evenly between splits.  # DOMAIN: replace for your data
    if max_images is not None:
        train_limit = max_images // 2
        test_limit = max_images - train_limit
    else:
        train_limit = None
        test_limit = None

    all_train = sorted(train_dir.glob("*.png"))
    all_test = sorted(test_dir.glob("*.png"))
    train_paths = stratified_sample_by_class(
        all_train, labels, train_limit, seed=DEFAULT_SAMPLE_SEED
    )
    test_paths = stratified_sample_by_class(
        all_test, labels, test_limit, seed=DEFAULT_SAMPLE_SEED + 1
    )

    # Symlink the sampled subset into cache_root/train + cache_root/test.
    # Targets live in cache_root/_extract/{train,test}/ — same tree, so
    # the symlinks remain valid as long as cache_root is intact.
    for split, paths in (("train", train_paths), ("test", test_paths)):
        sub = cache_root / split
        sub.mkdir(parents=True, exist_ok=True)
        for img_path in paths:
            (sub / img_path.name).symlink_to(img_path.resolve())

    # Write labels only for the sampled files (stem -> class).
    sampled_labels = {
        p.stem: labels[p.stem] for p in train_paths + test_paths if p.stem in labels
    }
    write_labels_manifest(cache_root, sampled_labels)  # DOMAIN: replace for your data

    logger.info(
        "Staged %d train + %d test source images to %s",
        len(train_paths),
        len(test_paths),
        cache_root,
    )
    return cache_root


# ---------------------------------------------------------------------------
# Register phase orchestrator
# ---------------------------------------------------------------------------


def run_register_phase(
    ml: DerivaML,
    max_images: int | None,
    cache_root: Path = CIFAR10_SOURCE_CACHE,
) -> str:
    """Execution 1 — stage source images and register them as a File dataset.

    Creates a ``CIFAR_Source_Registration`` workflow execution, calls
    ``exe.add_files`` on the staged source tree, and returns the root
    File dataset RID.  The nested structure (``train/`` + ``test/``
    sub-datasets under the root) mirrors the Toronto train/test split
    at the provenance layer.

    The workflow type ``CIFAR_Source_Registration`` must be present in
    the ``Workflow_Type`` vocabulary before calling this function — it
    is seeded by ``_cifar10_schema.setup_workflow_types``.

    Args:
        ml: Connected DerivaML instance with the schema set up.
        max_images: Total number of images to stage and register.
            ``None`` processes the full corpus.
        cache_root: Staging directory.  Defaults to
            :data:`CIFAR10_SOURCE_CACHE`.

    Returns:
        The RID of the root File dataset (``cifar10_source``).

    Example:
        >>> ml = DerivaML(hostname="localhost", catalog_id="42")  # doctest: +SKIP
        >>> rid = run_register_phase(ml, max_images=100)  # doctest: +SKIP
        >>> rid.startswith("2-")  # doctest: +SKIP
        True
    """
    # Stage the source images to the stable cache directory.
    stage_source(max_images, cache_root=cache_root)

    # Create a workflow for this registration execution.  # DOMAIN: replace for your data
    workflow = ml.create_workflow(
        name="CIFAR-10 Source Registration",
        workflow_type="CIFAR_Source_Registration",  # DOMAIN: replace for your data
        description=(
            "Register sampled CIFAR-10 source images as by-reference "
            "Input provenance (Execution 1 of the two-execution ingest)"
        ),
    )
    config = ExecutionConfiguration(workflow=workflow)

    with ml.create_execution(config) as exe:
        logger.info("Source-registration execution RID: %s", exe.execution_rid)

        specs = list(
            FileSpec.create_filespecs(
                cache_root,
                description="CIFAR-10 source image (pre-upload reference)",  # DOMAIN: replace for your data
                file_types=["Image"],
            )
        )
        logger.info("Registering %d source file specs via add_files...", len(specs))

        source_dataset = exe.add_files(
            specs,
            dataset_types=["CIFAR_Source"],  # DOMAIN: replace for your data
            root_name="cifar10_source",  # DOMAIN: replace for your data
            description="CIFAR-10 source images registered as upload-execution inputs",
        )
        logger.info("Registered root File dataset %s", source_dataset.dataset_rid)

    return source_dataset.dataset_rid
