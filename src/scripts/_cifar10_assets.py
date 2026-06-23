"""CIFAR-10 Stage 2: upload images, add classification features.

This module demonstrates the **asset-upload + feature-labeling
pattern**: how to upload binary files as ``Image`` assets inside
one Execution, then re-query the catalog and add feature values
inside a separate Execution for clean provenance.  The two-step
structure (2a upload, 2b label) is the key pattern to copy when
you need separate provenance records for data ingestion vs.
annotation.

This module is the assets layer. It downloads + extracts the
CIFAR-10 archive (via ``_cifar10_source``), uploads each image
as an ``Image`` asset inside one Execution, then adds an
``Image_Classification`` feature row for each image inside a
separate Execution.

Stage 2 is fully self-contained: it does not depend on
in-memory state from any earlier step. The feature-labeling
sub-stage recovers each image's class from its filename
(format: ``train_<class>_<id>.png`` or ``test_<class>_<id>.png``)
by reading the catalog, so it can be re-run against any
catalog where stage 1 is complete and some images exist.

Public API:
    - ``upload_images(ml, archive_path=None, max_images=None,
        batch_size=500)`` — Stage 2a.
    - ``add_classification_features(ml)`` — Stage 2b. Reads
      back uploaded Image rows from the catalog.
    - ``class_from_filename(filename)`` — pure helper that
      decodes the class from a CIFAR-10 image filename.
    - ``run_assets_phase(ml, max_images=None, batch_size=500)``
      — orchestrator that runs 2a then 2b.
"""

from __future__ import annotations

import logging
import random
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deriva_ml import DerivaML
from deriva_ml.core.ermrest import UploadProgress
from deriva_ml.execution import ExecutionConfiguration

from scripts._cifar10_source import download_cifar10_archive, extract_cifar10_to_png

# Default seed for reproducible stratified sampling. CIFAR-10 has 10
# classes; when the requested sample size is below 10 we cannot give
# every class at least one representative and we fall back to a
# deterministic first-N sample with a warning.
DEFAULT_SAMPLE_SEED = 42

logger = logging.getLogger(__name__)

CIFAR10_CLASSES_FROZEN = frozenset(
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


def class_from_filename(filename: str) -> str | None:
    """Decode the CIFAR-10 class from an image filename.

    Image filenames produced by Stage 2a have the shape
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
    if candidate not in CIFAR10_CLASSES_FROZEN:
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


def _create_upload_progress_callback(
    total_files: int,
) -> tuple[Callable[[UploadProgress], None], dict[str, Any]]:
    """Create a progress callback for upload monitoring.

    Lifted from the previous load_cifar10.py with no behavior change.

    Args:
        total_files: Total number of files to be uploaded.

    Returns:
        A tuple of (callback_function, state_dict). The callback
        logs upload progress at configurable percentage intervals.

    Example:
        >>> callback, state = _create_upload_progress_callback(1000)
        >>> callable(callback)
        True
    """
    state = {"last_reported_percent": -1, "started": False, "callback_count": 0}

    if total_files < 20:
        report_every_percent = max(1, 100 // total_files) if total_files > 0 else 10
    elif total_files <= 100:
        report_every_percent = 10
    else:
        report_every_percent = 5

    def progress_callback(progress: UploadProgress) -> None:
        state["callback_count"] += 1
        if not state["started"]:
            state["started"] = True
            logger.info(
                f"  [Upload] Starting upload (reporting every ~{report_every_percent}%)..."
            )
        match = re.search(r"Uploading file (\d+) of (\d+)", progress.message)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            percent = progress.percent_complete
            report_percent = int(percent // report_every_percent) * report_every_percent
            if report_percent > state["last_reported_percent"]:
                state["last_reported_percent"] = report_percent
                logger.info(f"  [Upload] {percent:.0f}% ({current}/{total} files)")

    return progress_callback, state


def upload_images(
    ml: DerivaML,
    archive_path: Path | None = None,
    max_images: int | None = None,
    batch_size: int = 500,  # currently unused but reserved for future batching
) -> dict[str, Any]:
    """Stage 2a — upload CIFAR-10 images as Image assets.

    Downloads the CIFAR-10 archive (cached after first call),
    extracts to a temporary directory, and registers + uploads
    every PNG as an ``Image`` asset inside one Execution. Train
    images get filenames ``train_<class>_<id>.png``; test images
    get ``test_<class>_<id>.png``. The class is encoded in the
    filename so Stage 2b can recover it.

    Args:
        ml: Connected DerivaML instance with the schema set up.
        archive_path: Optional path to a pre-downloaded archive.
            If ``None``, ``download_cifar10_archive()`` is called.
        max_images: Optional total cap (split evenly between train
            and test). ``None`` means upload everything (~60K).
        batch_size: Reserved for future use; currently unused.

    Returns:
        Stats dict with keys ``total_images``, ``training_images``,
        ``testing_images``, ``execution_rid``.

    Example:
        >>> ml = DerivaML(hostname="localhost", catalog_id="42")
        >>> stats = upload_images(ml, max_images=100)
        >>> stats["total_images"]
        100
    """
    if archive_path is None:
        archive_path = download_cifar10_archive()

    workflow = ml.create_workflow(
        name="CIFAR-10 Asset Upload",
        workflow_type="CIFAR_Data_Load",
        description="Upload CIFAR-10 images to the Image asset table",
    )
    config = ExecutionConfiguration(workflow=workflow)

    if max_images:
        train_limit = max_images // 2
        test_limit = max_images - train_limit
        logger.info(f"Loading {train_limit} train + {test_limit} test images")
    else:
        train_limit = None
        test_limit = None

    train_count = 0
    test_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        train_dir, test_dir, labels = extract_cifar10_to_png(archive_path, temp_path)
        logger.info(f"Extracted CIFAR-10 to: {temp_path}")

        # Class-balanced sampling of the source images. When the caller
        # asks for fewer images than the full corpus, we want every
        # class represented evenly — picking by sorted filename order
        # (the previous behavior) produced bird/ship-only partitions
        # because CIFAR-10 filenames cluster by class (#13).
        all_train_paths = sorted(train_dir.glob("*.png"))
        all_test_paths = sorted(test_dir.glob("*.png"))
        train_paths = stratified_sample_by_class(
            all_train_paths, labels, train_limit, seed=DEFAULT_SAMPLE_SEED
        )
        test_paths = stratified_sample_by_class(
            all_test_paths, labels, test_limit, seed=DEFAULT_SAMPLE_SEED + 1
        )
        if train_limit is not None:
            train_class_counts: dict[str, int] = {}
            for p in train_paths:
                cls = labels.get(p.stem)
                if cls is not None:
                    train_class_counts[cls] = train_class_counts.get(cls, 0) + 1
            logger.info(
                "  Train sample class distribution: "
                + ", ".join(f"{c}={n}" for c, n in sorted(train_class_counts.items()))
            )
        if test_limit is not None:
            test_class_counts: dict[str, int] = {}
            for p in test_paths:
                cls = labels.get(p.stem)
                if cls is not None:
                    test_class_counts[cls] = test_class_counts.get(cls, 0) + 1
            logger.info(
                "  Test sample class distribution: "
                + ", ".join(f"{c}={n}" for c, n in sorted(test_class_counts.items()))
            )

        with ml.create_execution(config) as exe:
            logger.info(f"  Upload execution RID: {exe.execution_rid}")
            execution_rid = exe.execution_rid

            # Clear working dir
            working_dir = exe.working_dir
            if working_dir.exists():
                for item in working_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            # Register train images
            for img_path in train_paths:
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name is None:
                    logger.warning(f"No label for {image_id}, skipping")
                    continue
                new_filename = f"train_{class_name}_{image_id}.png"
                exe.asset_file_path(
                    asset_name="Image",
                    file_name=str(img_path),
                    asset_types=["Image"],
                    copy_file=True,
                    rename_file=new_filename,
                )
                train_count += 1
                if train_count % 1000 == 0:
                    logger.info(f"  Registered {train_count} train images...")

            # Register test images
            for img_path in test_paths:
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name is None:
                    logger.warning(f"No label for {image_id}, skipping")
                    continue
                new_filename = f"test_{class_name}_{image_id}.png"
                exe.asset_file_path(
                    asset_name="Image",
                    file_name=str(img_path),
                    asset_types=["Image"],
                    copy_file=True,
                    rename_file=new_filename,
                )
                test_count += 1
                if test_count % 1000 == 0:
                    logger.info(f"  Registered {test_count} test images...")

            logger.info(
                f"  Total: {train_count} train + {test_count} test = {train_count + test_count}"
            )

        # Commit after context exits (matches previous behavior)
        total_count = train_count + test_count
        progress_callback, _ = _create_upload_progress_callback(total_count)
        upload_result = exe.commit_output_assets(
            clean_folder=True, progress_callback=progress_callback
        )

        logger.info("  [Upload] 100% complete")
        logger.info(f"  Upload complete: {upload_result.total_uploaded} files uploaded")

    return {
        "total_images": train_count + test_count,
        "training_images": train_count,
        "testing_images": test_count,
        "execution_rid": execution_rid,
    }


def _truncate_loader_classification_rows(ml: DerivaML) -> int:
    """Delete any prior loader-written ``Image_Classification`` rows.

    Makes ``--phase images`` idempotent on retry. When a previous
    loader attempt's ``images`` sub-stage succeeded but the run
    failed later (e.g. in the ``datasets`` phase) and the user reran
    the loader, the prior pass's ground-truth feature rows would
    otherwise survive — see
    ``findings/evaluator/01-loader-retry-leaves-orphaned-gt-feature-rows.md``
    in the 2026-05-28 e2e run for the audit. The downstream symptoms
    were doubly-tagged images (1600 rows over 1100 unique Image RIDs)
    and a train/test partition leak through
    ``split_dataset(row_per=feature_table)``.

    The truncate is filtered to ``Confidence IS NULL`` so it touches
    only loader-written ground-truth rows. Training executions
    populate ``Confidence`` when they record predictions (see
    ``record_test_predictions`` in ``src/models/cifar10_cnn.py``) and
    those prediction rows are preserved — this matches the dual-write
    convention recorded in the e2e run's ``tacit-knowledge.md`` entry
    ``tk-001``.

    Args:
        ml: Connected DerivaML instance.

    Returns:
        The number of prior loader rows that were deleted (zero on a
        fresh catalog).

    Example:
        >>> deleted = _truncate_loader_classification_rows(ml)  # doctest: +SKIP
        >>> print(f"removed {deleted} stale GT rows")
    """
    feat = ml.lookup_feature("Image", "Image_Classification")
    pb = ml.pathBuilder()
    feature_path = pb.schemas[feat.feature_table.schema.name].tables[
        feat.feature_table.name
    ]
    # Confidence IS NULL selects loader rows (ground truth) and
    # excludes training rows (predictions). The `== None` form is the
    # ermrest path-builder spelling for IS NULL — see
    # ``core/mixins/dataset.py`` for the same idiom on ``Deleted``.
    prior = list(
        feature_path.filter(feature_path.Confidence == None)  # noqa: E711
        .entities()
        .fetch()
    )
    if not prior:
        return 0
    logger.info(
        f"  Truncating {len(prior)} prior loader Image_Classification rows "
        "(retry idempotence; preserves training-prediction rows)"
    )
    feature_path.filter(feature_path.Confidence == None).delete()  # noqa: E711
    return len(prior)


def add_classification_features(ml: DerivaML) -> dict[str, Any]:
    """Stage 2b — add Image_Classification feature for every uploaded image.

    Queries the catalog for all ``Image`` asset rows, decodes the
    class from each filename via :func:`class_from_filename`, and
    adds an ``Image_Classification`` feature row inside one
    Execution. Images whose filenames don't decode are logged and
    skipped.

    This sub-stage is fully self-contained — it reads back from
    the catalog rather than depending on any in-memory state from
    Stage 2a. It can be re-run safely against a catalog where the
    schema is set up and some images have already been uploaded:
    any prior loader-written ``Image_Classification`` rows are
    truncated first via :func:`_truncate_loader_classification_rows`,
    so retries don't accumulate orphaned ground-truth rows. Training
    executions' prediction rows (``Confidence`` populated) are
    preserved.

    Args:
        ml: Connected DerivaML instance.

    Returns:
        Stats dict with keys ``features_added``, ``images_skipped``,
        ``execution_rid``, ``prior_rows_truncated``.

    Example:
        >>> stats = add_classification_features(ml)
        >>> stats["features_added"]
        100
    """
    assets = ml.list_assets("Image")
    logger.info(f"Found {len(assets)} Image assets in catalog")

    prior_truncated = _truncate_loader_classification_rows(ml)

    workflow = ml.create_workflow(
        name="CIFAR-10 Classification Labeling",
        workflow_type="CIFAR_Data_Load",
        description="Add Image_Classification feature for each Image asset",
    )
    config = ExecutionConfiguration(workflow=workflow)

    ImageClassification = ml.feature_record_class("Image", "Image_Classification")

    feature_records = []
    skipped = 0
    for asset in assets:
        class_name = class_from_filename(asset.filename)
        if class_name is None:
            logger.warning(f"Skipping {asset.filename}: cannot decode class")
            skipped += 1
            continue
        feature_records.append(
            ImageClassification(
                Image=asset.asset_rid,
                Image_Class=class_name,
            )
        )

    with ml.create_execution(config) as exe:
        logger.info(f"  Labeling execution RID: {exe.execution_rid}")
        execution_rid = exe.execution_rid
        logger.info(f"  Adding {len(feature_records)} classification labels...")
        exe.add_features(feature_records)

    exe.commit_output_assets(clean_folder=True)
    logger.info(f"  Added {len(feature_records)} Image_Classification features")

    return {
        "features_added": len(feature_records),
        "images_skipped": skipped,
        "execution_rid": execution_rid,
        "prior_rows_truncated": prior_truncated,
    }


def run_assets_phase(
    ml: DerivaML,
    archive_path: Path | None = None,
    max_images: int | None = None,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Stage 2 orchestrator — upload images then add features.

    Args:
        ml: Connected DerivaML instance.
        archive_path: Optional pre-downloaded archive.
        max_images: Optional total image cap.
        batch_size: Reserved.

    Returns:
        Merged stats dict from upload_images + add_classification_features.

    Example:
        >>> stats = run_assets_phase(ml, max_images=100)
        >>> stats["features_added"] == stats["total_images"]
        True
    """
    upload_stats = upload_images(
        ml, archive_path=archive_path, max_images=max_images, batch_size=batch_size
    )
    feature_stats = add_classification_features(ml)
    return {**upload_stats, **feature_stats}
