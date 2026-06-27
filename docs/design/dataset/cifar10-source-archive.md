# Dataset Design: CIFAR-10 source images (File-backed Input provenance)

**Slug:** cifar10-source-archive
**Status:** Built   <!-- Draft | Approved | Built | Validated | Released -->
**Date:** 2026-06-24

## Purpose

Capture *where the raw CIFAR-10 data came from* as a first-class catalog
record, before any bytes are reshaped into `Image` assets. Previously the
loader downloaded the Toronto archive, extracted it, and uploaded per-image
`Image` assets — but the catalog never recorded the **origin** of those images.
A future reader could see 2000 `Image` rows but not answer "which source files
produced them?" This closes that gap with a **two-execution ingest**: a
dedicated registration execution (Exec 1, `CIFAR_Source_Registration`) registers
each sampled source image as a by-reference `File` record (URL + MD5 + length,
no bytes uploaded) in a named nested File-dataset tree; then the upload
execution (Exec 2, `CIFAR_Image_Upload`) consumes that source File dataset as an
**Input** and produces the `Image` assets as Outputs. Because both roles (Input
source dataset, Output Image assets) attach to the same upload execution, the
chain **source File dataset → upload exec → Image assets** is a recorded,
traversable edge.

The `Image`-asset + `Image_Classification`-feature pattern that every experiment
depends on is unchanged.

## Requirements

- **Source data:** the sampled CIFAR-10 source PNGs, produced by
  `extract_cifar10_sample_to_png()` in `_cifar10_source.py` — which decodes the
  Toronto pickle batches in memory, class-balanced-samples the requested counts,
  and writes only the sampled PNGs into the stable cache directory
  `CIFAR10_SOURCE_CACHE` (`~/.cache/deriva-ml-model-template/cifar10_source`).
  No full 60K extraction to disk; no symlinks (see `tk-013`). A `labels.csv`
  manifest (filename, class) is written into the cache root by
  `write_labels_manifest()`; the upload phase reads it to recover the
  filename→class mapping without crossing execution boundaries.
- **Target size & composition:** one `File` record **per sampled source image**
  (the same train/test files the upload stages), built via
  `FileSpec.create_filespecs(cache_root)` which computes MD5 + length per file.
  The `url` is a `tag://` URI derived from the local cache path (an
  origin-identity token, not a live fetch URL).
- **Scale dependency:** this path registers hundreds–thousands of File rows in
  one `add_files` call, which goes through `add_dataset_members` →
  `resolve_rids`. That requires **deriva-ml ≥ 1.51.17** (1.51.12 added the
  `resolve_rids` chunking fix — `tk-006`/`tk-007`/`tk-008`; 1.51.14 added
  directory-tree nesting — `tk-009`/`tk-010`; 1.51.17 added the `root_name`
  parameter used to name the root dataset `cifar10_source`). Pin accordingly.
- **Element types:** `File` (the deriva-ml file table), via `exe.add_files()`.
  No new domain table.
- **Vocabularies:** `CIFAR_Source` is a `Dataset_Type` vocabulary term seeded by
  the schema phase. The registration call passes
  `dataset_types=["CIFAR_Source"]` and `root_name="cifar10_source"` to
  `add_files`, so the root dataset is named `cifar10_source` and tagged
  `CIFAR_Source` (in addition to the built-in `File` + `Directory` tags applied
  to all datasets `add_files` creates). The `CIFAR_Source` tag makes the root
  dataset discoverable via `ml.find_datasets(dataset_types=["CIFAR_Source"])`
  when the upload phase runs in isolation.
- **Compute budget:** modest — one MD5 per sampled source file; no bytes
  uploaded to hatrac (registration is by reference).

## Structure plan

- **Pattern:** File-backed datasets (the `Dataset` `add_files` returns), built
  as a **nested tree mirroring the source directory layout**. `cache_root`
  contains `train/` and `test/` subdirectories (each holding the sampled PNGs
  for that partition) plus `labels.csv` at the root. `add_files` groups files
  by parent directory and nests, producing **a named root dataset
  `cifar10_source` (`source_directory == "."`, tagged `CIFAR_Source`) that
  contains a `train` File dataset + a `test` File dataset** — the Toronto
  train/test split mirrored on the source side. Not a split or subsample (no
  `split_dataset`/`subsample`).
- **Why decode-time sampling (not full extraction + symlinks):**
  `create_filespecs` walks a directory recursively, so whatever directory is
  passed must contain *only* the files intended for registration — no scratch or
  bulk extraction subdirs (see `tk-013`). An earlier design extracted the full
  60K corpus into `cache_root/_extract/` and symlinked the sampled subset into
  `cache_root/train` + `cache_root/test`, but `create_filespecs` then walked
  the extraction subdirectory too, registering all 60K files. The fix is
  decode-time sampling: `extract_cifar10_sample_to_png()` writes only the
  sampled PNGs directly into `cache_root/train` + `cache_root/test`, so
  `create_filespecs(cache_root)` registers exactly the intended set.
- **Dataset_Type tags:** `CIFAR_Source` (passed explicitly) + `File` +
  `Directory` (the auto-applied built-ins) on every dataset in the tree.
  `add_files` creates no automatic `Dataset_Dataset` edge to the `Complete`
  image dataset — that link is execution-mediated through the upload execution.

## Validation

- **Registration:** one `File` row per sampled source image (each with a
  `tag://` URL, MD5, byte length); a nested File-dataset tree — a named root
  `cifar10_source` (`source_directory == "."`) whose members are a `train` File
  dataset and a `test` File dataset. A `labels.csv` File row lives in the root
  dataset alongside the child datasets.
- **Provenance:** the upload execution (Exec 2, `CIFAR_Image_Upload`) consumes
  the source root File dataset as an Input (`Dataset_Execution`, Input role) and
  produces the `Image` assets as Outputs (every `Image.execution_rid` points to
  that same upload execution). The chain **source File dataset → upload exec →
  Image assets** is a recorded, traversable edge (verified live, `tk-017`).
  Regression coverage: `tests/test_lineage_connected.py`.
- **`lookup_lineage` on an image dataset** reaches the source files in one call
  as of **deriva-ml ≥ 1.52.0** (the pinned version): the walk descends into the
  dataset's `Image` members → their producing (upload) execution → that exec's
  Input source File dataset. Before 1.52.0 this needed a manual hop (image
  dataset → `Image` members → `Image.execution_rid` → the exec's Input source
  dataset). See `tk-018` (member-asset traversal) and `tk-020`/`tk-021`/`tk-022`
  (consumed-version faithfulness, the FK-RID fix, and the perf/guard follow-up).
- **Integrity meaning:** the MD5/length describe the exact source files that
  were sampled and uploaded this run.

## Consumption

- Not consumed by any experiment or DataLoader — it is a provenance/lineage
  record, read by a human (or a lineage walk) asking "where did these images
  come from." It does not participate in training.
- The upload execution reads `labels.csv` from the root dataset's File members
  (matched by URL basename, not `Filename` — `add_files` leaves `Filename`
  NULL, see `tk-014`) to resolve filename→class, and resolves each image File's
  `tag://` URL to the local cache path via `tag_url_to_path()`.
- Not pinned in `src/configs/datasets.py` (configs pin the *training* dataset
  families, not the source datasets). Discoverable via
  `ml.find_datasets(dataset_types=["CIFAR_Source"])`.

## Upstream designs

None — a dataset doesn't depend on a feature, and these sit at the very top of
the data lineage (they *are* the origin). The `cifar10-input-datasets` families
are conceptually downstream (their `Image` members are derived from these source
files), but `add_files` creates no automatic `Dataset_Dataset` edge to them.

## Status & links

- **Implementation:**
  - Exec 1 (source registration): `run_register_phase()` in
    `src/scripts/_cifar10_register.py`. Creates a `CIFAR_Source_Registration`
    workflow execution, stages source PNGs via `stage_source()`, calls
    `exe.add_files(specs, dataset_types=["CIFAR_Source"], root_name="cifar10_source",
    ...)`, returns the root File dataset RID.
  - Exec 2 (upload, consumes source as Input): `run_upload_phase()` in
    `src/scripts/_cifar10_upload.py`. Creates a `CIFAR_Image_Upload` workflow
    execution that consumes the source root File dataset via
    `DatasetSpec(rid=source_dataset_rid, version=..., materialize=False)` —
    `materialize=False` is required because `tag://` local URLs can't be
    byte-fetched by bag materialization (see `tk-015`).
  - The retired `_cifar10_assets.py` (`upload_images()`) no longer exists.
- **Requires:** deriva-ml ≥ **1.51.17** — 1.51.12 added the `resolve_rids`
  chunking fix needed at this scale (`tk-006`/`tk-007`/`tk-008`); 1.51.14
  added the directory-tree nesting for equal-depth siblings (`tk-009`/`tk-010`);
  1.51.17 added `root_name` to `add_files` (needed to name the root dataset
  `cifar10_source`).
- **RID + version:** catalog-specific; the source root dataset carries
  `CIFAR_Source` + `File` + `Directory` tags. Query its origin folder via
  `Dataset.source_directory` (root `'.'`, children `'train'`/`'test'`) and gate
  on `Dataset.is_directory`. Resolve RIDs via
  `ml.find_datasets(dataset_types=["CIFAR_Source"])`. Not pinned in configs.
- **tacit-knowledge.md:** `tk-005` (add_files Input role + dataset-type tagging),
  `tk-006`/`tk-007` (scale bug), `tk-008` (1.51.12 chunking fix), `tk-009`
  (nesting gap), `tk-010` (1.51.14 nesting fix + `source_directory` API),
  `tk-013` (staging-leak bug → decode-time sampling), `tk-014` (`Filename` NULL
  in by-reference File rows), `tk-015` (`materialize=False` requirement),
  `tk-016` (register-phase timing), `tk-017` (two-execution lineage verified),
  `tk-018` (`lookup_lineage` gap + upstream improvement flagged).
