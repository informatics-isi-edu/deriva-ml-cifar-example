# Spec: Two-Execution CIFAR Ingest with Connected Source Lineage

**Date:** 2026-06-25
**Status:** Approved
**Repos touched:** `deriva-ml` (library change) + `deriva-ml-cifar-example` (this repo)

## Problem

The current single-execution ingest (`_cifar10_assets.py`) registers the source
images via `add_files` *and* uploads them as `Image` assets in **one execution**.
As a result (recorded in `tacit-knowledge.md` `tk-011`), the source File datasets
do **not** appear as consumed inputs when walking a downstream image dataset's
lineage: `ml.lookup_lineage(<image dataset>)` shows only the image datasets, never
the source File tree. Source-provenance and dataset-lineage are two parallel
branches off the same execution, bridged only by sharing that execution — so a
lineage walk on a training dataset cannot reach the raw source files.

Two secondary defects:
- The **root source File dataset has no meaningful name** — only a generic
  `Description` (the same string `add_files` writes to every node) and
  `Directory_Dataset.Path = "."`.
- The source files live in an **ephemeral `TemporaryDirectory`**, so their
  registered `tag://` URLs point at paths that are deleted right after extraction
  — they can't be resolved by an independent later execution.

## Goal

Split ingest into **two genuinely decoupled executions** so the source File
dataset is a real *consumed Input* of the upload, making lineage connect
`source files → Image assets → image datasets`. Make `register` and `upload`
**copy/edit scaffold files** (the reusable pattern is captured by a skill in
`../deriva-ml-skills`; CIFAR is the worked example). Name the root File dataset
properly via a clean `add_files` change in deriva-ml.

## Architecture

New `--phase` structure: `schema | register | upload | datasets | [cleanup]`.

| Phase | Execution / Workflow | Responsibility |
|---|---|---|
| `schema` | (none) | install domain model; seed `CIFAR_Source` `Dataset_Type` term |
| **`register`** | Exec 1 — `CIFAR Source Registration` | download → extract+sample into a **stable cache dir**; write `labels.csv` at root; `add_files` → named source File dataset tree |
| **`upload`** | Exec 2 — `CIFAR Image Upload` | **consume the source File dataset as Input**; walk dir children; read root `labels.csv`; resolve tag-URLs → cache paths; upload `Image` assets + features |
| `datasets` | own execution (as today) | build Complete → Split → subsamples from `Image` RIDs (unchanged) |
| `cleanup` | (none) | delete the stable cache dir (also a flag) |

The two executions are decoupled: Exec 2 locates everything from the
catalog-recorded File dataset (its tree structure, the manifest File, the
tag-URLs) — not from in-memory state of Exec 1. They communicate only through the
catalog. The stable cache dir is what keeps the tag-URLs resolvable across them;
explicit cleanup prevents storage leakage.

## Components (this repo)

Both are standalone copy/edit scaffolds in `src/scripts/`, CIFAR-filled-in, with
clearly-marked "replace for your domain" seams. They replace `_cifar10_assets.py`.

### `_cifar10_register.py` — Exec 1
- `download_cifar10_archive()` → extract + **stratified-sample** into a **stable
  cache root** `~/.cache/deriva-ml-model-template/cifar10_source/`:
  `labels.csv` (filename → class) at root, `train/<imgs>`, `test/<imgs>`.
- `exe.add_files(FileSpec.create_filespecs(cache_root), dataset_types=["CIFAR_Source"])`
  → nested source File dataset tree; the **root auto-named** `cifar10_source`
  (from its directory basename — see the deriva-ml change) and tagged
  `CIFAR_Source`.
- Returns the root File dataset RID.
- **Domain seam:** the extract/sample/manifest logic is CIFAR-specific; the
  "stage a dir tree → `add_files` → named, tagged root" shape is what the skill
  generalizes.

### `_cifar10_upload.py` — Exec 2
- `ExecutionConfiguration(datasets=[<root File dataset RID>])` — **consume it as
  Input**.
- Walk the File dataset's directory children: `child.source_directory ∈
  {"train","test"}` → partition.
- Read the **root `labels.csv` File** (the *registered* manifest, not an
  in-memory dict) → filename → class map.
- For each image File: resolve its `tag://...file:///...` URL → cache path;
  `exe.asset_file_path(asset_name="Image", file_name=<cache path>,
  asset_types=["Image"], rename_file="<partition>_<class>_<id>.png")`;
  then `exe.commit_output_assets()`.
- `exe.add_features(Image_Classification ...)` from the manifest.
- **Domain seam:** the `Image` table, train/test meaning, and filename→class
  mapping are the "replace for your domain" parts.

### `_cifar10_datasets.py` — unchanged
Builds Complete → Split → subsamples from `Image` RIDs; runs after Exec 2.

### `load_cifar10.py` — orchestrator
Updated `--phase` choices (`schema|register|upload|datasets|cleanup|all`); wires
the new two-execution sequence; passes the root File dataset RID from `register`
to `upload`.

## deriva-ml change (coordinated, sibling repo `../deriva-ml`)

`add_files` should **name the ingest-root dataset**, since it owns node creation
and knows which node is the root. **No backward-compatibility shim required.**

- Add a `root_name: str | None = None` parameter.
- The `ingest_root` node's `Description` defaults to **the root directory's
  basename** (`ingest_root.name`) instead of the generic caller `description`;
  `root_name` overrides when given. Non-root nodes keep the per-node behavior.
- Bump deriva-ml version; this repo's `uv.lock` then consumes it.

Mechanism note: a post-hoc `dataset.description` setter exists and works (verified
on 1.51.14), but the clean change is to have `add_files` set the root description
at creation. We are taking the clean path.

## Data flow (end to end)

```
register (Exec 1):  download → stage sampled files into
                    ~/.cache/.../cifar10_source/{labels.csv, train/, test/}
                    → add_files(create_filespecs(cifar10_source),
                                dataset_types=["CIFAR_Source"])
                    → root File dataset auto-named "cifar10_source" (+ CIFAR_Source)
upload   (Exec 2):  consume root File dataset as INPUT
                    → walk children (source_directory → train/test)
                    → read root labels.csv File
                    → resolve each File tag-URL → cache path
                    → asset_file_path(Image) → commit_output_assets
                    → add Image_Classification features
datasets (own exec): Complete → Split → subsamples from Image RIDs  (unchanged)
cleanup:            delete ~/.cache/.../cifar10_source/   (flag or final phase)
```

Resulting lineage: image dataset → Exec 2 → **consumed Input = root File
dataset** → source files. The `tk-011` gap is closed.

## Error handling

- `upload` fails loudly with actionable messages if: the File dataset RID is
  missing or is not a File/Directory dataset; `labels.csv` is absent at the root;
  a tag-URL path no longer resolves on disk (cache deleted prematurely — message
  points at the cleanup ordering).
- `register` is idempotent on re-run against the same cache (re-stage is safe).
- Small-variant floor (`tk-002`) still applies at the `datasets` phase.

## Testing

- **Unit (no catalog):** directory→partition resolution; manifest (`labels.csv`)
  → class mapping; tag-URL → local-path resolution. Extend `tests/`.
- **Live-localhost (gated):** full two-execution run; assert the image dataset's
  `lookup_lineage` now lists the source File dataset as a consumed input (the
  regression test for `tk-011`).

## Sequencing (implementation order)

1. deriva-ml `add_files` root-naming change + version bump (sibling repo).
2. This repo: bump `uv.lock` to the new deriva-ml.
3. Schema phase: seed `CIFAR_Source` `Dataset_Type` term.
4. Write `_cifar10_register.py` + `_cifar10_upload.py`; retire `_cifar10_assets.py`.
5. Update `load_cifar10.py` phases + orchestration; add cleanup.
6. Tests; live-localhost verification of connected lineage.

## Status & links

- Supersedes the single-execution source registration (`tk-005`–`tk-010`).
- Fixes the lineage gap documented in `tk-011`.
- Reusable pattern to be captured by a skill in `../deriva-ml-skills`.
