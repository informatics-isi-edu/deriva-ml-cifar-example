"""CIFAR-10 Stage 3: create the dataset hierarchy.

This module demonstrates the **dataset-hierarchy pattern**: how
to query the catalog for existing assets, partition them via the
library's split/subsample primitives, and assemble a nested
dataset structure with derived holdout splits — all inside one
Execution. Copy this module when you need to turn a set of
uploaded assets into a structured dataset hierarchy with
train/test splits.

This module is the datasets layer. Given a catalog with the
schema set up and some Image asset rows uploaded (Stages 1 and
2 complete), it creates:

    - ``Complete`` (Labeled) — all images. Hand-built; it is the
      input to the canonical Split, not a Split itself.
    - ``Split`` (+ ``Training`` / ``Testing`` partition children) —
      the canonical Toronto train/test partition. Produced by
      ``split_dataset(selection_fn=cifar_canonical_partition)`` so
      the curators' fixed ``train_`` / ``test_`` filename prefix
      drives the partition. The two children carry the
      ``Split_Partition`` discriminator (deriva-ml v1.42, spec
      2026-06-01).
    - ``Small_Training`` (Labeled, ``Subsample``) — a stratified
      ``SMALL_TRAIN_SIZE`` sample of ``Training``. Produced by
      ``subsample()``; no parent Split — siblings are
      discoverable via the producing execution's outputs.
    - ``Small_Testing`` (Labeled, ``Subsample``) — same shape,
      drawn from ``Testing``.
    - ``Labeled_Split`` (and Training/Testing children) — 80/20
      split of training images via ``split_dataset()``.
    - ``Small_Labeled_Split`` (and Training/Testing children) —
      fixed 400/100 split when the training pool is >=500, else
      an 80/20 fallback for small-scale work.

The roles-don't-propagate rule: ``split_dataset`` assigns
``Training`` / ``Testing`` based on each partition's role in the
split, not by inheriting from the source. See CONTEXT.md
(``Datasets — types and partitions``) for the canonical framing.

Stage 3 reads back ``Image`` rows from the catalog and uses
each filename's ``train_`` or ``test_`` prefix to decide which
dataset each image belongs to. No in-memory state from Stage 2
is needed.

Public API:
    - ``create_dataset_hierarchy(ml, batch_size=500)`` — does
      all the work in one Execution.
    - ``run_datasets_phase(ml, batch_size=500)`` — orchestrator
      alias for symmetry with run_schema_phase / run_assets_phase.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from deriva_ml import DerivaML
from deriva_ml.dataset.split import split_dataset, subsample
from deriva_ml.execution import ExecutionConfiguration

logger = logging.getLogger(__name__)

SMALL_TRAIN_SIZE = 500
SMALL_TEST_SIZE = 500

# Column to stratify by for class-balanced splits — column on the
# Image_Class vocab table (reached transparently through the
# Image_Classification feature populated in stage 2b). The vocab
# table is what we pass in include_tables; see the 2026-05-28
# denormalizer audit (findings/investigation/03-recommendations.md
# §1) for why the feature-association table is the wrong handle.
STRATIFY_COLUMN = "Image_Class.Name"


class SmallVariantDegenerateError(RuntimeError):
    """Raised when the 'small' Toronto split family would equal the full split.

    The ``Small_Training`` / ``Small_Testing`` datasets are intended to be
    a strictly smaller random sample of the corresponding full Toronto
    train/test pools. When the source pools are smaller than (or equal to)
    ``SMALL_TRAIN_SIZE`` / ``SMALL_TEST_SIZE``, the sample step falls
    through to "use everything," producing dataset rows whose members are
    byte-identical to the full split. Any ``small`` vs ``full`` comparison
    on that catalog is then meaningless, with no warning surfaced at run
    time. We refuse to create that ambiguity and raise this error instead.

    See:
        ``deriva-ml-model-template-e2e/findings/curator/01-small-variant-equals-full-at-500-images.md``
    """


def _require_small_variant_distinct(train_pool: int, test_pool: int) -> None:
    """Refuse to build a degenerate small Toronto split family.

    A small variant must draw strictly fewer than the source pool size in
    each partition; otherwise the "small" datasets are byte-identical to
    the full datasets. Equality (``pool == SMALL_*_SIZE``) is also
    degenerate because every source RID would be picked.

    Args:
        train_pool: Number of available training images in the catalog.
        test_pool: Number of available testing images in the catalog.

    Raises:
        SmallVariantDegenerateError: If either pool is too small to yield
            a strictly smaller sample. The message tells the operator
            what ``--num-images`` minimum would clear the threshold and
            suggests the labeled-split family as the right tool for tiny
            catalogs.

    Example:
        >>> _require_small_variant_distinct(train_pool=600, test_pool=600)
        >>> _require_small_variant_distinct(train_pool=250, test_pool=250)
        Traceback (most recent call last):
            ...
        scripts._cifar10_datasets.SmallVariantDegenerateError: ...
    """
    if train_pool > SMALL_TRAIN_SIZE and test_pool > SMALL_TEST_SIZE:
        return

    # --num-images is split evenly between train and test partitions in
    # _cifar10_assets.upload_images(), so the threshold to recover a
    # non-degenerate small family is roughly 2*(SMALL_*_SIZE + 1).
    min_num_images = 2 * (max(SMALL_TRAIN_SIZE, SMALL_TEST_SIZE) + 1)
    raise SmallVariantDegenerateError(
        f"At this catalog size (train_pool={train_pool}, test_pool={test_pool}) "
        f"the 'small' Toronto split family would be byte-identical to the full "
        f"Toronto split. SMALL_TRAIN_SIZE={SMALL_TRAIN_SIZE} and "
        f"SMALL_TEST_SIZE={SMALL_TEST_SIZE} require strictly larger source "
        f"pools to yield a distinct sample. "
        f"Re-run load-cifar10 with --num-images >= {min_num_images} so each "
        f"partition exceeds the small-variant sample size, or skip the small "
        f"Toronto split and use the labeled-split family instead — "
        f"split_dataset() partitions the training images directly and stays "
        f"distinct at any catalog size."
    )


def _build_dataset_descriptions(
    train_count: int,
    test_count: int,
    small_train_count: int,
    small_test_count: int,
) -> dict[str, str]:
    """Build dataset descriptions parameterized by actual member counts.

    The CIFAR-10 loader's ``--num-images`` flag controls how many
    Toronto source images get pulled into the catalog, so the resulting
    dataset member counts depend on the run, not on the Toronto
    defaults (50,000 / 10,000). Hard-coding the Toronto numbers in the
    descriptions produced misleading metadata at any
    ``--num-images < default`` (see e2e finding curator/03 from
    2026-05-26).

    This helper centralises the description text so it can be
    unit-tested without a live catalog round-trip.

    Args:
        train_count: Actual count of images in the ``Training`` dataset.
        test_count: Actual count of images in the ``Testing`` dataset.
        small_train_count: Actual count of images in ``Small_Training``
            (``min(SMALL_TRAIN_SIZE, train_count)``).
        small_test_count: Actual count of images in ``Small_Testing``
            (``min(SMALL_TEST_SIZE, test_count)``).

    Returns:
        Mapping with one entry per Toronto-family dataset key. Keys are
        ``complete``, ``split``, ``training``, ``testing``,
        ``small_training``, and ``small_testing``. ``small_split`` is
        not produced — ``Small_Training`` and ``Small_Testing`` are
        sibling outputs of the same execution (no parent Split), per
        the v1.42 ``subsample()`` migration.

    Example:
        >>> d = _build_dataset_descriptions(
        ...     train_count=250, test_count=250,
        ...     small_train_count=250, small_test_count=250,
        ... )
        >>> "250 labeled images" in d["training"]
        True
        >>> "small_split" in d
        False
    """
    total = train_count + test_count
    return {
        "complete": (
            f"Complete CIFAR-10 dataset: {total:,} labeled images "
            f"({train_count:,} train + {test_count:,} test)."
        ),
        "split": (
            "CIFAR-10 dataset split into training and testing subsets "
            f"({train_count:,} / {test_count:,} images)."
        ),
        "training": (f"CIFAR-10 training partition: {train_count:,} labeled images."),
        "testing": (f"CIFAR-10 testing partition: {test_count:,} labeled images."),
        "small_training": (
            f"Small CIFAR-10 training set: {small_train_count:,} "
            "stratified images for quick testing."
        ),
        "small_testing": (
            f"Small CIFAR-10 testing set: {small_test_count:,} "
            "stratified images for quick testing."
        ),
    }


def _labeled_split_description(train_count: int) -> str:
    """Describe the 80/20 stratified labeled split of the training set.

    Args:
        train_count: Total number of images in the ``Training`` dataset.

    Returns:
        A description that reports the resulting child partition sizes
        (80/20 of ``train_count``, rounded down/up to keep the total).

    Example:
        >>> _labeled_split_description(250)
        'CIFAR-10 labeled split: stratified 80/20 from training images (200 / 50, seed=42).'
    """
    test_n = train_count // 5
    train_n = train_count - test_n
    return (
        "CIFAR-10 labeled split: stratified 80/20 from training images "
        f"({train_n:,} / {test_n:,}, seed=42)."
    )


def _small_labeled_split_description(train_count: int) -> str:
    """Describe the Small_Labeled_Split of the training set.

    At ``train_count >= 500`` the loader uses a fixed 400/100 stratified
    split; otherwise it falls back to an 80/20 fraction.

    Args:
        train_count: Total number of images in the ``Training`` dataset.

    Returns:
        A description that reports the resulting partition sizes.

    Example:
        >>> _small_labeled_split_description(600)
        'Small CIFAR-10 labeled split: stratified 400/100 from training (seed=42).'
        >>> _small_labeled_split_description(250)
        'Small CIFAR-10 labeled split: stratified 80/20 from training (200 / 50, seed=123).'
    """
    if train_count >= 500:
        return (
            "Small CIFAR-10 labeled split: stratified 400/100 from training (seed=42)."
        )
    test_n = train_count // 5
    train_n = train_count - test_n
    return (
        "Small CIFAR-10 labeled split: stratified 80/20 from training "
        f"({train_n:,} / {test_n:,}, seed=123)."
    )


def cifar_canonical_partition(
    df: pd.DataFrame,
    partition_sizes: dict[str, int],
    seed: int,
) -> dict[str, np.ndarray]:
    """Partition CIFAR-10 images by the curators' fixed filename prefix.

    ``split_dataset()`` accepts a ``selection_fn`` callable that maps
    the denormalized dataframe to a per-partition index mapping. For
    CIFAR-10 the canonical Toronto train/test partition is fully
    predicate-determined by filename prefix — ``train_*`` images are
    Training, ``test_*`` images are Testing — so ``partition_sizes``
    and ``seed`` are ignored (the selector is deterministic).

    The caller must pass ``include_tables=["Image"]`` so the
    denormalized dataframe carries the ``Image.Filename`` column this
    predicate inspects. The column case mirrors the catalog column
    (``Filename``), not the lowercase asset-object attribute.

    Args:
        df: Denormalized member dataframe assembled by
            ``split_dataset`` from the source dataset.
        partition_sizes: Requested per-partition sizes. Ignored —
            sizes are determined entirely by the predicate.
        seed: Random seed. Ignored — the selector is deterministic.

    Returns:
        Mapping ``{"Training": indices, "Testing": indices}`` of
        integer index arrays selecting rows of ``df``.

    Example:
        >>> df = pd.DataFrame(
        ...     {"Image.Filename": ["train_a.png", "test_b.png", "train_c.png"]}
        ... )
        >>> parts = cifar_canonical_partition(df, {"Training": 0, "Testing": 0}, seed=0)
        >>> sorted(parts.keys())
        ['Testing', 'Training']
        >>> parts["Training"].tolist()
        [0, 2]
        >>> parts["Testing"].tolist()
        [1]
    """
    is_train = df["Image.Filename"].str.startswith("train_")
    return {
        "Training": np.flatnonzero(is_train.values),
        "Testing": np.flatnonzero(~is_train.values),
    }


def create_dataset_hierarchy(ml: DerivaML, batch_size: int = 500) -> dict[str, str]:
    """Create the full CIFAR-10 dataset hierarchy.

    Queries the catalog for all ``Image`` asset rows, splits them
    by filename prefix (``train_`` vs ``test_``), creates the
    parent and child dataset rows, assigns members in batches,
    and finally creates the labeled-split families via
    ``split_dataset()``.

    All work happens inside one Execution for clean provenance.

    Args:
        ml: Connected DerivaML instance.
        batch_size: Batch size for ``add_dataset_members`` calls.

    Returns:
        Mapping of dataset name to its RID. Keys include
        ``complete``, ``split``, ``training``, ``testing``,
        ``small_training``, ``small_testing``,
        ``labeled_split``, ``labeled_training``,
        ``labeled_testing``, ``small_labeled_split``,
        ``small_labeled_training``, ``small_labeled_testing``.

        ``small_split`` is **not** present — the parent ``Small_Split``
        dataset is no longer created. ``Small_Training`` and
        ``Small_Testing`` are sibling outputs of the same execution
        (per the v1.42 ``subsample()`` migration; see CONTEXT.md
        ``Datasets — types and partitions``). Code that previously
        read ``datasets["small_split"]`` must migrate.

    Raises:
        SmallVariantDegenerateError: If the catalog holds too few
            training or testing images for the ``Small_Training`` /
            ``Small_Testing`` datasets to be a strict subset of the
            full split. The error fires before any catalog writes
            so the caller can re-run ``load-cifar10`` with a larger
            ``--num-images``.

    Example:
        >>> datasets = create_dataset_hierarchy(ml)
        >>> datasets["training"]
        'X-12345-NXYZ'
    """
    assets = ml.list_assets("Image")
    logger.info(f"Found {len(assets)} Image assets to organize")

    train_rids = [a.asset_rid for a in assets if a.filename.startswith("train_")]
    test_rids = [a.asset_rid for a in assets if a.filename.startswith("test_")]
    all_rids = train_rids + test_rids
    logger.info(f"  Train: {len(train_rids)}, Test: {len(test_rids)}")

    # Refuse to build a degenerate small Toronto split family — see
    # curator/01 finding from the 2026-05-26 e2e run. Surfacing this
    # before any catalog writes keeps the catalog clean: if the small
    # variant can't be distinct, we don't want the full-size datasets
    # half-created either, since the operator will need to re-run with
    # more images anyway. ``subsample()`` will also raise on
    # ``size > total``, but this guard fails faster (no catalog writes
    # at all) and produces operator-friendly remediation text.
    _require_small_variant_distinct(
        train_pool=len(train_rids), test_pool=len(test_rids)
    )

    workflow = ml.create_workflow(
        name="CIFAR-10 Dataset Hierarchy",
        workflow_type="CIFAR_Data_Load",
        description="Create CIFAR-10 dataset hierarchy from uploaded images",
    )
    config = ExecutionConfiguration(workflow=workflow)

    # Descriptions reflect the actual catalog state for this run (see
    # finding curator/03, 2026-05-26): hard-coding the Toronto defaults
    # (50,000 / 10,000) lied at any --num-images < default.
    small_train_count = min(SMALL_TRAIN_SIZE, len(train_rids))
    small_test_count = min(SMALL_TEST_SIZE, len(test_rids))
    descriptions = _build_dataset_descriptions(
        train_count=len(train_rids),
        test_count=len(test_rids),
        small_train_count=small_train_count,
        small_test_count=small_test_count,
    )

    datasets: dict[str, str] = {}

    def _batched_add(ds_rid: str, rids: list[str], label: str) -> None:
        ds = ml.lookup_dataset(ds_rid)
        added = 0
        for i in range(0, len(rids), batch_size):
            batch = rids[i : i + batch_size]
            ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
        logger.info(f"  {label}: added {added}/{len(rids)} images")

    with ml.create_execution(config) as exe:
        logger.info(f"  Datasets execution RID: {exe.execution_rid}")

        # Complete is hand-built — it is the *input* to the canonical
        # Split, not itself a Split. Members are assigned in batches
        # via the catalog directly; the execution scopes provenance.
        complete = exe.create_dataset(
            description=descriptions["complete"],
            dataset_types=["Complete", "Labeled"],
        )
        datasets["complete"] = complete.dataset_rid
        if all_rids:
            logger.info("Assigning Image RIDs to Complete...")
            _batched_add(datasets["complete"], all_rids, "Complete")

        # Canonical Toronto train/test partition — produced by
        # split_dataset() with a predicate selector. The selector
        # ignores partition_sizes and seed (the partition is fully
        # determined by the train_/test_ filename prefix), but
        # split_dataset's signature still requires test_size — any
        # nonzero value satisfies validation. include_tables=["Image"]
        # guarantees the denormalized dataframe carries
        # Image.filename for the predicate to inspect.
        #
        # Both children automatically receive the Split_Partition tag
        # (deriva-ml v1.42, spec 2026-06-01). Role types
        # (Training/Testing) are assigned by split_dataset based on
        # the partition's role in the split, NOT inherited from the
        # source — that is the canonical role-axis vs origin-axis
        # decomposition; see CONTEXT.md.
        if all_rids:
            logger.info("Creating canonical Split via split_dataset(selection_fn)...")
            canonical = split_dataset(
                ml,
                datasets["complete"],
                exe,
                test_size=max(len(test_rids), 1),
                selection_fn=cifar_canonical_partition,
                element_table="Image",
                include_tables=["Image"],
                training_types=["Labeled"],
                testing_types=["Labeled"],
                partition_by="element",
                split_description=descriptions["split"],
            )
            datasets["split"] = canonical.split.rid
            datasets["training"] = canonical.training.rid
            datasets["testing"] = canonical.testing.rid

            # Small_Training and Small_Testing — stratified subsamples
            # of the canonical Training / Testing partitions. The
            # source relationship is recorded in execution provenance
            # (the source becomes an input of this execution, the
            # subsample is an output); no parent Split, no
            # Dataset_Dataset edge. ``Subsample`` is appended to
            # dataset_types automatically.
            logger.info("Subsampling Small_Training from Training...")
            small_training_result = subsample(
                ml,
                datasets["training"],
                exe,
                size=SMALL_TRAIN_SIZE,
                seed=42,
                stratify_by_column=STRATIFY_COLUMN,
                element_table="Image",
                include_tables=["Image", "Image_Class"],
                dataset_types=["Training", "Labeled"],
                description=descriptions["small_training"],
                partition_by="element",
            )
            datasets["small_training"] = small_training_result.subsample.rid

            logger.info("Subsampling Small_Testing from Testing...")
            small_testing_result = subsample(
                ml,
                datasets["testing"],
                exe,
                size=SMALL_TEST_SIZE,
                seed=43,
                stratify_by_column=STRATIFY_COLUMN,
                element_table="Image",
                include_tables=["Image", "Image_Class"],
                dataset_types=["Testing", "Labeled"],
                description=descriptions["small_testing"],
                partition_by="element",
            )
            datasets["small_testing"] = small_testing_result.subsample.rid

    exe.commit_output_assets(clean_folder=True)

    # Labeled splits derived from Training. Stratify by the
    # Image_Classification feature so each child partition keeps a
    # balanced class distribution — without this, the 400/100 small
    # split inherited the source class skew and ended up bird/ship-only
    # at --num-images 500 (#13).
    #
    # ``split_dataset`` runs inside an Execution the caller has already
    # opened: deriva-ml never invents workflow provenance, so this
    # script (which is the caller) is responsible for registering the
    # workflow and opening an execution that says "the bytes in this
    # script decided to do these splits." We reuse a single workflow
    # and a single execution across both labeled splits so the lineage
    # is coherent (one Execution row, two Split datasets nested under
    # the source Training dataset).
    if train_rids:
        split_workflow = ml.create_workflow(
            name="CIFAR-10 Labeled Split",
            workflow_type="Dataset_Split",
            description=(
                "Stratified labeled splits from the CIFAR-10 training set; "
                "produces Labeled_Split and Small_Labeled_Split."
            ),
        )
        with ml.create_execution(
            ExecutionConfiguration(
                workflow=split_workflow,
                description="Create Labeled_Split and Small_Labeled_Split from the training set",
            )
        ) as split_exe:
            logger.info(
                "Creating Labeled_Split (80/20 of training) in execution %s...",
                split_exe.execution_rid,
            )
            # Stratification shape per the 2026-05-28 denormalizer audit
            # (deriva-ml-model-template-e2e/findings/investigation/
            # 03-recommendations.md §1):
            #
            #   * include_tables passes the vocab table (Image_Class)
            #     directly, NOT the feature-association table
            #     (Execution_Image_Image_Classification). The
            #     feature-association is then a transparent bridge:
            #     the planner gets row_per=Image auto-default, and the
            #     vocab value projects as one column per image.
            #     Passing the feature-association table here triggers
            #     Rule 5 (DerivaMLDenormalizeDownstreamLeaf) because
            #     it is strict-downstream of Image and the planner
            #     does not aggregate.
            #   * stratify_by_column points at the vocab column
            #     (Image_Class.Name), not at the feature-row dotted
            #     path that the prior shape used.
            #   * partition_by="element" (PR informatics-isi-edu/
            #     deriva-ml#254) forces dedup at the selector layer
            #     and asserts disjoint element-RID partitions after
            #     the split — protection against the original
            #     train/test leakage even if multiple feature rows
            #     exist per image.
            labeled = split_dataset(
                ml,
                datasets["training"],
                split_exe,
                test_size=0.2,
                seed=42,
                stratify_by_column=STRATIFY_COLUMN,
                training_types=["Labeled"],
                testing_types=["Labeled"],
                element_table="Image",
                include_tables=["Image", "Image_Class"],
                partition_by="element",
                split_description=_labeled_split_description(len(train_rids)),
            )
            datasets["labeled_split"] = labeled.split.rid
            datasets["labeled_training"] = labeled.training.rid
            datasets["labeled_testing"] = labeled.testing.rid

            logger.info("Creating Small_Labeled_Split...")
            # _require_small_variant_distinct guarantees train_rids > 500,
            # so the fixed 400/100 split always fits with images to spare.
            # Same shape as the labeled_split call above — see that
            # block for the audit reference.
            small_labeled = split_dataset(
                ml,
                datasets["training"],
                split_exe,
                test_size=100,
                train_size=400,
                seed=42,
                stratify_by_column=STRATIFY_COLUMN,
                training_types=["Labeled"],
                testing_types=["Labeled"],
                element_table="Image",
                include_tables=["Image", "Image_Class"],
                partition_by="element",
                split_description=_small_labeled_split_description(len(train_rids)),
            )
            datasets["small_labeled_split"] = small_labeled.split.rid
            datasets["small_labeled_training"] = small_labeled.training.rid
            datasets["small_labeled_testing"] = small_labeled.testing.rid

        split_exe.commit_output_assets(clean_folder=True)

    return datasets


def run_datasets_phase(ml: DerivaML, batch_size: int = 500) -> dict[str, str]:
    """Stage 3 orchestrator alias.

    Args:
        ml: Connected DerivaML instance.
        batch_size: Batch size for dataset-member additions.

    Returns:
        Mapping of dataset name to RID (see create_dataset_hierarchy).

    Example:
        >>> rids = run_datasets_phase(ml)
        >>> rids["training"]
        'X-12345-NXYZ'
    """
    return create_dataset_hierarchy(ml, batch_size=batch_size)
