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
produced them?" This closes that gap by registering each sampled source image
as a by-reference `File` record (URL + MD5 + length, no bytes copied) linked as
an **Input** of the same execution that performs the upload.

It is a *provenance layer added alongside* the existing per-image upload, not a
replacement for it. The `Image`-asset + `Image_Classification`-feature pattern
that every experiment depends on is unchanged.

## Requirements

- **Source data:** the sampled CIFAR-10 source PNGs, extracted from the Toronto
  archive (`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, fetched +
  cached by `download_cifar10_archive()` in `_cifar10_source.py`) into a local
  temp directory by `extract_cifar10_to_png()`.
- **Target size & composition:** one `File` record **per sampled source image**
  (the same train/test files the upload stages), built via
  `FileSpec.create_filespecs(...)` which computes MD5 + length per file. The
  `url` is a `tag://` URI derived from the local extracted path (an
  origin-identity token, not a live fetch URL — the temp dir is ephemeral).
- **Scale dependency:** this path registers hundreds–thousands of File rows in
  one `add_files` call, which goes through `add_dataset_members` →
  `resolve_rids`. That requires **deriva-ml ≥ 1.51.12** (earlier versions fail
  with an HTTP-414 on the unbatched resolution query — see `tk-006`/`tk-007`/
  `tk-008`). Pin accordingly.
- **Element types:** `File` (the deriva-ml file table), via `exe.add_files()`.
  No new domain table.
- **Vocabularies:** none added. The implementation does not pass a custom
  `dataset_types` term; `add_files` auto-tags the datasets it creates with the
  built-in `File` + `Directory` `Dataset_Type` terms (both seeded by the schema
  phase), and each `FileSpec` carries the built-in `Image` + `File` asset types.
  (A custom `CIFAR_Source` term was considered and dropped — it would require
  seeding a vocabulary term in the schema phase for marginal benefit.)
- **Compute budget:** modest — one MD5 per sampled source file; no bytes
  uploaded to hatrac (registration is by reference).

## Structure plan

- **Pattern:** File-backed dataset(s) (the `Dataset` `add_files` returns).
  `add_files` builds **one dataset per source directory**, nested to mirror the
  directory tree — so the loader's `train/` and `test/` source subdirs yield
  two File datasets (not one). Not a split or subsample.
- **Dataset_Type tags (three axes):** Content `File` + `Directory` (the
  auto-applied built-ins). Role and Origin are not split/subsample-derived
  (registered directly, not produced by `split_dataset`/`subsample`).
  `add_files` creates no automatic `Dataset_Dataset` edge to the `Complete`
  image dataset — the link is execution-mediated, not dataset-to-dataset.

## Validation

- **Registration:** one `File` row per sampled source image, each carrying a
  `tag://` URL, a non-empty MD5, and the file's byte length; one File dataset
  per source directory.
- **Provenance:** each `File` row links (via `File_Execution`, `Asset_Role =
  Input`) to the same execution that uploads the `Image` assets — so lineage
  from an `Image` back to its source files is reachable through that shared
  execution (the source files are Inputs, the `Image` assets Outputs of one
  execution).
- **Integrity meaning:** the MD5/length describe the exact source files that
  were uploaded this run.

## Consumption

- Not consumed by any experiment or DataLoader — it is a provenance/lineage
  record, read by a human (or a lineage walk) asking "where did these images
  come from." It does not participate in training.
- Not pinned in `src/configs/datasets.py` (configs pin the *training* dataset
  families, not the source datasets).

## Upstream designs

None — a dataset doesn't depend on a feature, and these sit at the very top of
the data lineage (they *are* the origin). The `cifar10-input-datasets` families
are conceptually downstream (their `Image` members are derived from these source
files), but `add_files` creates no automatic `Dataset_Dataset` edge to them.

## Status & links

- **Implementation:** `upload_images()` in `src/scripts/_cifar10_assets.py` —
  `exe.add_files(FileSpec.create_filespecs(...))` runs at the start of the
  upload execution, before the per-image `asset_file_path` loop. No schema
  change required.
- **Requires:** deriva-ml ≥ 1.51.12 (the `resolve_rids` chunking fix; earlier
  versions fail at this scale — see `tk-006`/`tk-007`/`tk-008`).
- **RID + version:** catalog-specific; the source datasets carry `File` +
  `Directory` tags (resolve via `ml.find_datasets()`). Not pinned.
- **tacit-knowledge.md:** `tk-005` (add_files behavior), `tk-006`/`tk-007`
  (the scale bug), `tk-008` (the 1.51.12 fix that made this work).
