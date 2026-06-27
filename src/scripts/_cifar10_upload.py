"""CIFAR-10 Execution 2: upload Image assets from the registered File dataset.

This module demonstrates the **file-dataset consumption + upload pattern**:
how to consume a previously registered File dataset as an execution *Input*,
resolve each registered file's tag-URL to a local filesystem path, upload
the image bytes as ``Image`` assets, and add ``Image_Classification``
feature rows — all with clean lineage back to the source File dataset
produced by Execution 1 (``_cifar10_register.run_register_phase``).

The two-execution design is the key pattern to copy:

- **Execution 1** (``_cifar10_register``) registers source bytes as a
  durable File dataset (provenance only — no upload).
- **Execution 2** (this module) *consumes* that File dataset as an
  Input, reads the manifest for labels, and actually uploads the bytes
  as ``Image`` assets.

This split gives separate provenance records for "which files were
ingested" vs. "what pixel data ended up in the catalog", and the
catalog's lineage graph links source→images automatically.

Public API:
    - ``tag_url_to_path(url: str) -> Path`` — pure resolver that turns a
      deriva-ml tag URL (``tag://host,date:file:///abs/path``) into a
      ``Path``.
    - ``read_labels_manifest(path: Path) -> dict[str, str]`` — pure
      reader that turns a ``labels.csv`` (written by
      :func:`scripts._cifar10_source.write_labels_manifest`) into a
      ``{filename: class}`` dict keyed by ``<stem>.png``.
    - ``add_classification_features(ml) -> dict`` — Execution 2b: add
      ``Image_Classification`` feature rows for every uploaded image.
    - ``_truncate_loader_classification_rows(ml) -> int`` — helper used
      by ``add_classification_features`` to make retries idempotent.
    - ``run_upload_phase(ml, source_dataset_rid) -> dict`` — Execution 2
      orchestrator: consume the File dataset, upload Image assets, add
      ``Image_Classification`` features, return stats.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

from deriva_ml import DerivaML
from deriva_ml.dataset.aux_classes import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration

from scripts._cifar10_register import class_from_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (no catalog/network)
# ---------------------------------------------------------------------------


def tag_url_to_path(url: str) -> Path:
    """Resolve a deriva-ml tag URL to its local filesystem path.

    Parses ``tag://<host>,<date>:file:///abs/path`` and returns the
    ``Path`` object for the file portion.  The ``file://`` segment
    follows the first ``:file://`` occurrence in the URL, so the tag
    authority (host and date) is stripped cleanly.

    Args:
        url: A tag URL of the form
            ``tag://HostA,2026-06-25:file:///var/cache/img.png``.

    Returns:
        ``Path`` object for the absolute local filesystem path.

    Example:
        >>> tag_url_to_path(
        ...     "tag://HostA,2026-06-25:file:///var/cache/img.png"
        ... ).as_posix()
        '/var/cache/img.png'
    """
    file_part = url.split(":file://", 1)[1]
    return Path(urlsplit("file://" + file_part).path)


def read_labels_manifest(path: Path) -> dict[str, str]:
    """Read a ``labels.csv`` manifest into a filename→class dict.

    Inverse of :func:`scripts._cifar10_source.write_labels_manifest`.
    The CSV has a ``filename,class`` header; each row is
    ``<stem>.png,<class>``.

    Args:
        path: Path to the ``labels.csv`` file.

    Returns:
        Dict mapping ``"<stem>.png"`` to class name, e.g.
        ``{"cat_2.png": "cat", "dog_3.png": "dog"}``.

    Example:
        >>> from pathlib import Path
        >>> m = read_labels_manifest(Path("/cache/cifar10_source/labels.csv"))
        ... # doctest: +SKIP
        >>> m["cat_2.png"]  # doctest: +SKIP
        'cat'
    """
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["filename"]: row["class"] for row in reader}


# ---------------------------------------------------------------------------
# Feature-labeling helpers (Execution 2b)
# ---------------------------------------------------------------------------


def _truncate_loader_classification_rows(ml: DerivaML) -> int:
    """Delete any prior loader-written ``Image_Classification`` rows.

    Makes the labeling sub-stage idempotent on retry.  When a previous
    loader attempt's labeling sub-stage succeeded but the run failed
    later and the user re-ran the loader, prior ground-truth feature rows
    would otherwise accumulate.  The truncate is filtered to
    ``Confidence IS NULL`` so it touches only loader-written ground-truth
    rows; training executions' prediction rows (``Confidence`` populated)
    are preserved.

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
    # ermrest path-builder spelling for IS NULL.
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
    """Execution 2b — add Image_Classification feature for every uploaded image.

    Queries the catalog for all ``Image`` asset rows, decodes the class
    from each filename via :func:`scripts._cifar10_register.class_from_filename`,
    and adds an ``Image_Classification`` feature row inside one Execution.
    Images whose filenames don't decode are logged and skipped.

    This sub-stage is fully self-contained — it reads back from the catalog
    rather than depending on any in-memory state from Execution 2a.  Any
    prior loader-written ``Image_Classification`` rows are truncated first
    via :func:`_truncate_loader_classification_rows`, so retries don't
    accumulate orphaned ground-truth rows.  Training executions' prediction
    rows (``Confidence`` populated) are preserved.

    Args:
        ml: Connected DerivaML instance.

    Returns:
        Stats dict with keys ``features_added``, ``images_skipped``,
        ``execution_rid``, ``prior_rows_truncated``.

    Example:
        >>> stats = add_classification_features(ml)  # doctest: +SKIP
        >>> stats["features_added"]  # doctest: +SKIP
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


# ---------------------------------------------------------------------------
# Execution 2 orchestrator
# ---------------------------------------------------------------------------


def run_upload_phase(
    ml: DerivaML,
    source_dataset_rid: str,
) -> dict[str, Any]:
    """Execution 2 — consume the File dataset, upload Images, add features.

    Creates a ``CIFAR_Image_Upload`` workflow execution that consumes the
    root File dataset from Execution 1 as an *Input*.  For each
    partition child (``train`` / ``test``) it walks the registered File
    members, resolves each tag-URL to a local cache path, and stages the
    image bytes as an ``Image`` asset via
    :meth:`Execution.asset_file_path`.  After the execution context
    exits, assets are committed to Hatrac via
    :meth:`Execution.commit_output_assets`.  Finally, a second small
    execution adds ``Image_Classification`` feature rows for every
    uploaded image.

    Lines marked ``# DOMAIN: replace for your data`` are the seams to
    edit when adapting this scaffold to a different dataset.

    Args:
        ml: Connected DerivaML instance with the schema already set up.
        source_dataset_rid: RID of the root File dataset produced by
            :func:`scripts._cifar10_register.run_register_phase`.

    Returns:
        Stats dict with keys ``total_images``, ``training_images``,
        ``testing_images``, ``upload_execution_rid``,
        ``feature_execution_rid``, ``features_added``,
        ``images_skipped``.

    Example:
        >>> ml = DerivaML(hostname="localhost", catalog_id="42")
        ... # doctest: +SKIP
        >>> stats = run_upload_phase(ml, source_dataset_rid="2-WXYZ")
        ... # doctest: +SKIP
        >>> stats["total_images"]  # doctest: +SKIP
        100
    """
    # ------------------------------------------------------------------
    # Look up the source File dataset so we can get its current version
    # for the DatasetSpec (required by ExecutionConfiguration).
    # ------------------------------------------------------------------
    source_ds = ml.lookup_dataset(source_dataset_rid)
    source_version = str(source_ds.current_version)
    logger.info(
        "Consuming source File dataset %s @ %s", source_dataset_rid, source_version
    )

    # ------------------------------------------------------------------
    # Execution 2a: upload Image assets
    # ------------------------------------------------------------------
    workflow = ml.create_workflow(
        name="CIFAR-10 Image Upload",  # DOMAIN: replace for your data
        workflow_type="CIFAR_Image_Upload",  # DOMAIN: replace for your data
        description=(
            "Execution 2: consume the registered File dataset as Input and "
            "upload each source image as an Image asset."
        ),
    )
    config = ExecutionConfiguration(
        workflow=workflow,
        # materialize=False: consume the File dataset as an Input for lineage
        # WITHOUT fetching member bytes into a bag.  The files are by-reference
        # (local tag:// URLs), which bag materialization cannot fetch — it would
        # raise BagValidationError.  We only need the File rows' URLs (table
        # metadata) to resolve the local paths ourselves.  See tk-015.
        datasets=[
            DatasetSpec(
                rid=source_dataset_rid,
                version=source_version,
                materialize=False,
            )
        ],
    )

    train_count = 0
    test_count = 0

    # Read the labels manifest from the root File dataset's labels.csv member.
    # The root dataset (the add_files tree root, identified via is_source_root)
    # holds labels.csv; partition children (source_directory "train" / "test")
    # hold the images.
    root_members = source_ds.list_dataset_members()
    file_records = root_members.get("File", [])  # DOMAIN: replace for your data
    # NOTE: add_files leaves File.Filename NULL — the name lives in the URL tag
    # path. Match on the URL basename, not Filename (see tacit-knowledge tk-014).
    labels_record = next(
        (r for r in file_records if tag_url_to_path(r["URL"]).name == "labels.csv"),
        None,
    )
    if labels_record is None:
        raise RuntimeError(
            f"labels.csv not found in root File dataset {source_dataset_rid}. "
            "Run run_register_phase first."
        )
    labels_path = tag_url_to_path(labels_record["URL"])  # DOMAIN: replace for your data
    logger.info("Reading labels manifest from %s", labels_path)
    labels = read_labels_manifest(labels_path)  # DOMAIN: replace for your data
    logger.info("Loaded %d label entries from manifest", len(labels))

    # Walk the partition children (train / test).
    partition_children = [
        child
        for child in source_ds.list_dataset_children()
        if child.is_directory
        and child.source_directory in {"train", "test"}  # DOMAIN: replace for your data
    ]

    with ml.create_execution(config) as exe:
        logger.info("Upload execution RID: %s", exe.execution_rid)
        upload_execution_rid = exe.execution_rid

        for child in partition_children:
            partition = child.source_directory  # "train" or "test"  # DOMAIN
            child_members = child.list_dataset_members()
            image_records = child_members.get("File", [])  # DOMAIN
            logger.info(
                "  Partition %r: %d File records", partition, len(image_records)
            )

            for file_rec in image_records:
                url: str = file_rec["URL"]  # DOMAIN: tag URL field name

                # add_files leaves File.Filename NULL — derive the name from the
                # URL tag path, not the Filename column (see tk-014).
                local_path = tag_url_to_path(url)  # DOMAIN
                filename = local_path.name  # e.g. "cat_2.png"
                stem = local_path.stem  # e.g. "cat_2"

                # Skip the labels manifest if it somehow appears in a child.
                if filename == "labels.csv":
                    continue

                cls = labels.get(filename)  # DOMAIN: manifest is keyed by filename
                if cls is None:
                    logger.warning(
                        "No label for %r (stem=%r), skipping", filename, stem
                    )
                    continue

                # Rename: <partition>_<class>_<original_stem>.png
                # DOMAIN: adjust rename convention for your dataset.
                rename = f"{partition}_{cls}_{stem}.png"
                exe.asset_file_path(
                    asset_name="Image",  # DOMAIN: asset table name
                    file_name=str(local_path),
                    asset_types=["Image"],  # DOMAIN
                    copy_file=True,
                    rename_file=rename,
                )

                if partition == "train":
                    train_count += 1
                else:
                    test_count += 1

                if (train_count + test_count) % 1000 == 0:
                    logger.info(
                        "  Registered %d images so far...", train_count + test_count
                    )

        logger.info(
            "  Total staged: %d train + %d test = %d",
            train_count,
            test_count,
            train_count + test_count,
        )

    # Commit image assets to Hatrac after the execution context exits.
    logger.info("Committing Image assets to Hatrac...")
    exe.commit_output_assets(clean_folder=True)
    logger.info("  Upload complete.")

    # ------------------------------------------------------------------
    # Execution 2b: add Image_Classification feature rows
    # ------------------------------------------------------------------
    # ``add_classification_features`` reads back all Image rows from the
    # catalog and derives the class from the uploaded filename
    # (``train_<class>_<stem>.png`` convention) — no in-memory state
    # crosses the execution boundary.  DOMAIN: replace with your own
    # feature-labeling logic when adapting this scaffold.
    feature_stats = add_classification_features(ml)
    logger.info(
        "  Added %d Image_Classification features", feature_stats["features_added"]
    )

    return {
        "total_images": train_count + test_count,
        "training_images": train_count,
        "testing_images": test_count,
        "upload_execution_rid": upload_execution_rid,
        "feature_execution_rid": feature_stats["execution_rid"],
        "features_added": feature_stats["features_added"],
        "images_skipped": feature_stats["images_skipped"],
    }
