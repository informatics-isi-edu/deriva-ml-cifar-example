# Dataset Design: CIFAR-10 input datasets (the hierarchy `load-cifar10` builds)

**Slug:** cifar10-input-datasets
**Covers slugs:** `cifar10-complete`, `cifar10-toronto-split`,
`cifar10-small-subsample`, `cifar10-labeled-split`,
`cifar10-small-labeled-split`, `cifar10-small-labeled-testing`,
`cifar10-small-training` (the experiment designs reference several of these
per-dataset slugs; all resolve to a section of this doc).
**Status:** Built   <!-- Draft | Approved | Built | Validated | Released -->
**Date:** 2026-06-23

> **Reverse-engineered.** Reconstructed *after* the datasets existed, from
> `src/scripts/_cifar10_datasets.py` (`create_dataset_hierarchy`) and the load
> into localhost catalog 100. RIDs below are the **catalog-100** instances; the
> shipped `src/configs/datasets.py` entries are empty placeholders (no RID) by
> design — fill them per README §7 before running against a catalog.

## Purpose

The full set of input datasets `load-cifar10` creates when a CIFAR-10 catalog
is provisioned. Together they give experiments the data to train and evaluate
on: a labeled root, the canonical Toronto train/test split, small stratified
subsamples for fast runs, and training-derived **labeled holdout splits** (the
family the example model defaults to). One doc covers the whole hierarchy
because the datasets are produced by a single execution and only differ by the
split/subsample operation that derived them.

## Requirements (shared)

- **Source data:** all `Image` rows uploaded in the loader's images phase
  (Toronto CIFAR-10 train + test batches), each carrying an
  `image-classification` ground-truth label.
- **Element types:** `Image` (the only member type across every dataset here).
- **Determinism:** splits and subsamples use a fixed **seed 42** (the same
  canonical value as the model default), so "default everything" runs are
  reproducible.
- **Catalog-100 scale:** 2000 images (1000 train-origin + 1000 test-origin);
  the README's first-time default is 10,000.

## The datasets (what the loader builds)

### Complete — the labeled root
- **Role/Content/Origin:** `Complete` / `Labeled` (+`CIFAR_10`) / —.
- Every loaded `Image`; the superset all other datasets derive from. Not
  consumed directly by an experiment.
- **Catalog 100:** `11PM`.

### Canonical Toronto Split → Training / Testing
- Produced by `split_dataset(selection_fn=cifar_canonical_partition)` —
  partitioned by **source batch** (Toronto train images → `Training`,
  `test_batch` → `Testing`), not by a ratio.
- **Tags:** parent `Split`/`Labeled`/`Split`; children `Training`/`Testing`,
  `Labeled`, Origin **`Split_Partition`** (auto-applied; roles don't propagate
  from the parent). The model's `_flatten_to_leaves` expands the `Split` parent
  to these children.
- **Catalog 100:** Split `15M2`, Training `15M8`, Testing `15MJ`.

### Small subsamples — Small_Training / Small_Testing
- Stratified `subsample()` of `Training` / `Testing`, **`SMALL_TRAIN_SIZE=500`**
  / **`SMALL_TEST_SIZE=500`**. Siblings — no parent Split (the v1.42 migration
  dropped the old `Small_Split` parent).
- **Tags:** `Training`/`Testing`, `Labeled`, Origin **`Subsample`**.
- **Hard floor:** each source pool must be **strictly larger** than its sample
  size or the subsample would be byte-identical to its source —
  `_require_small_variant_distinct` raises `SmallVariantDegenerateError`. This
  is why catalog 100 needed `--num-images ≥ 1002` (see `tk-002`).
- **Consumed by:** `cifar10_default` (uses `cifar10_small_training` — note:
  training-only, no bundled labeled test partition).
- **Catalog 100:** Small_Training `19J8`, Small_Testing `1AHY`.

### Labeled holdout splits — Labeled_Split & Small_Labeled_Split (→ Training / Testing)
- **The main input family for evaluation.** Produced by `split_dataset()` over
  the Toronto **training** images only (the official `test_batch` stays unseen),
  so **both** partitions carry ground-truth labels — the right choice for ROC
  analysis, accuracy metrics, and any evaluation work. `Small_Labeled_Split` is
  the reduced-size sibling (400/100, or sampled at larger scale).
- **Tags:** parent `Split`/`Labeled`/`Split`; children `Training`/`Testing`,
  `Labeled`, Origin `Split_Partition`.
- **Consumed by experiments:**
  - `cifar10_small_labeled_split` — the example model's **default**; used by
    `cifar10_quick`, `cifar10_extended`, `cifar10_small_default`,
    `cifar10_small_large` (the capacity sweep).
  - `cifar10_labeled_split` — full-data experiments `cifar10_quick_full`,
    `cifar10_extended_full`.
  - `cifar10_small_labeled_testing` — `cifar10_test_only` evaluates a
    checkpoint on this labeled test child.
- **Catalog 100:** Labeled_Split `1BJM` (Training `1BJT`, Testing `1BK4`);
  Small_Labeled_Split `1DJA` (Training `1DJG`, Testing `1DJT`).

## Validation (shared)

- **Counts:** each derived dataset's member count matches its configured ratio
  / sample size; a split parent == sum of its children.
- **No leakage:** within any split, `Training` and `Testing` member RIDs are
  disjoint (the live-localhost test
  `test_split_dataset_partitions_by_image_not_feature_row` guards
  image-vs-feature-row partitioning; skipped offline).
- **Distinctness:** subsamples are strictly smaller than their source (guard
  enforces it).
- **Both partitions labeled** in the labeled-split family — the property that
  makes it evaluation-ready vs the training-only subsample.
- **Bag parity:** each downloaded bag's RIDs == its catalog members.

## Consumption

- Pinned (per catalog) in `src/configs/datasets.py` — ship as empty
  placeholders, filled with a released version per README §7, never a
  dev/"current" label.
- Downstream experiment designs: all 8 in `docs/design/experiment/`.

## Upstream designs

None — a dataset doesn't depend on a feature. Every split/subsample reads the
`image-classification` class label its `Image` members carry (an
element-property precondition, not a build dependency).

## Status & links

- **RIDs + versions (catalog 100):** listed per dataset above
  (`https://localhost/id/100/<rid>`); versions via `ml.find_datasets()` — not
  pinned (throwaway test catalog).
- **configs/datasets.py:** the `cifar10_*` dataset entries (placeholders, no
  RID shipped).
- **tacit-knowledge.md:** small-variant floor that forced `--num-images 2000`
  is `tk-002`; catalog 100 provisioning is `tk-003`.
