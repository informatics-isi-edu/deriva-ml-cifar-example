# Tacit Knowledge

This file records **tacit knowledge** — the *why*, the *intent*, and the
*background* behind decisions made about this project's models and data.

The **catalog** is the source of record for everything else: data contents,
RIDs, dataset versions, workflow URLs and checksums, executions, lineage.
Don't replicate catalog-stored facts here. Don't ask this file what's in
the catalog — query the catalog directly (resources first, tools next).
When this file *needs* to reference a catalog entity, link to it
(`deriva://catalog/{host}/{cat}/ml/...`) instead of inlining its contents.

Each entry captures a decision: what was chosen, what alternatives were
considered, what was rejected and why, and any background context a future
reader would need to evaluate whether the decision still holds.

---

<a id="tk-001"></a>
### tk-001 — Convention — the offline smoke test stops *before* `dry_run=true`; that command needs a provisioned catalog
**When:** 2026-06-23T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

Lesson from a mislabeled smoke test: `deriva-ml-run dry_run=true` does **not**
belong in an *offline* (no-catalog) smoke test. It was included as if it were
catalog-free, failed against the fresh checkout, and the failure was wrongly
reported as a template defect. It is not — it is expected behavior when no
DerivaML catalog exists yet. The offline smoke test that genuinely never
touches a catalog is: `uv sync`, `uv run python -m pytest tests/`,
`uv run ruff check src tests`, `uv run ruff format --check src tests`,
`uv run deriva-ml-run --list-configs`, and `uv run deriva-ml-run --cfg job`
(the last resolves and prints the composed Hydra config without constructing
an `Execution`). `dry_run=true` is a *catalog* smoke step — run it only once a
catalog is provisioned.

The underlying mechanism, for reference:

During that smoke test, the dry run aborted with
`DerivaMLTableNotFound: Workflow_Type` against the shipped placeholder
connection `default_deriva` (`hostname=localhost`, `catalog_id=0`).

What was actually missing: **the whole DerivaML schema, because catalog 0 is
an empty placeholder that was never provisioned with DerivaML.** The error is
*not* about a missing workflow-type vocabulary *term*. `Execution.__init__`
validates the workflow type via `lookup_term(MLVocab.workflow_type, wt)`,
which calls `name_to_table("Workflow_Type")` *first* — and that searches the
catalog's schemas for a table named `Workflow_Type`. Catalog 0 has no
`deriva-ml` schema at all, so the table lookup fails before any term value is
ever compared. The validation fired correctly and loudly; it was reporting
"this catalog hasn't been set up for DerivaML," which is true. Against a real,
provisioned DerivaML catalog this same dry run succeeds.

Why this matters for the offline smoke test: `dry_run=true` suppresses the
catalog *writes* but still *reads* the catalog to validate the workflow type,
so it requires a reachable, DerivaML-provisioned catalog — it is **not** a
no-catalog command. The `lookup_term` read happens before the `dry_run` guard
in `deriva_ml/execution/execution.py`; that ordering is upstream `deriva-ml`
behavior, not a template concern. (`catalog_id=0` being a placeholder is
documented in the `src/configs/deriva.py` docstring.)

Implications for collaborators: README §8 presents `deriva-ml-run
dry_run=true` as the canonical "no catalog writes" smoke step, which misleads
on a fresh checkout — there's no provisioned catalog to read yet, so it
aborts. To exercise it, first `load-cifar10` into a real catalog (or point at
an existing one) and run `deriva-ml-run dry_run=true --host <h> --catalog
<id>`. For a genuinely offline structural smoke test, stop at `uv sync` +
`pytest` + `ruff` + `deriva-ml-run --list-configs` + `deriva-ml-run --cfg job`
(the last resolves and prints the composed Hydra config without constructing
an `Execution`, so it never touches a catalog).

**Weighed alternatives:** *(none captured — this is an observed behavior of the
shipped code, not a choice made in this session.)*

<a id="tk-002"></a>
### tk-002 — Convention — `load-cifar10 --num-images` must clear the small-variant floor (`>= 1002`, practically `2000`)
**When:** 2026-06-23T14:40:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (the offline smoke test couldn't reach a catalog; this is the first real load)

A `load-cifar10 --num-images 1000` run failed in the *datasets* phase (images
uploaded fine first) with `SmallVariantDegenerateError`:

> At this catalog size (train_pool=500, test_pool=500) the 'small' Toronto
> split family would be byte-identical to the full Toronto split.
> SMALL_TRAIN_SIZE=500 and SMALL_TEST_SIZE=500 require strictly larger source
> pools to yield a distinct sample. Re-run with --num-images >= 1002 ... or
> skip the small Toronto split and use the labeled-split family instead.

Why this is correct behavior, not a bug: `--num-images N` splits ~50/50 into
train/test pools (N=1000 → 500/500). The `Small_Training` / `Small_Testing`
variants are a stratified `subsample()` of 500 each. Sampling 500 from a pool
of exactly 500 yields a byte-identical copy — a degenerate "subset" that would
silently mislead anyone who later compared `Small_Training` against `Training`
expecting them to differ. `_require_small_variant_distinct` (the same guard
exercised by `test_require_small_variant_distinct_rejects_*` in the suite)
refuses rather than create the degenerate dataset. The choice of `N=1000` in
this session landed *exactly* on the degenerate boundary — a poor parameter
pick, not a template fault.

Implications for collaborators: when creating a CIFAR-10 catalog that needs
the small Toronto split family (`cifar10_small_training` / `_small_testing`,
which the example model uses by default), pass `--num-images >= 1002` so each
partition strictly exceeds the 500-sample size. In practice pick a round
number with headroom (`--num-images 2000` → 1000/1000 pools). The error is
loud and prescriptive, so the failure mode is self-correcting — but the
README's "10000 for first-time setup" and the absence of any stated *minimum*
make 1000 an easy, wrong guess.

**Weighed alternatives:** the error itself names a second path — skip the small
Toronto family and use the labeled-split family (`split_dataset()` partitions
training images directly and stays distinct at any catalog size). Chose to
re-load at a larger `--num-images` instead, because the example model's default
dataset (`cifar10_small_labeled_split`) wants the small family present.

<a id="tk-003"></a>
### tk-003 — Convention — `load-cifar10` builds a fixed set of named datasets, and `--dry-run` provisions the schema
**When:** 2026-06-23T14:42:30-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (the small-variant floor that constrains how many images the load needs)

Two durable facts learned while standing up a CIFAR-10 catalog to exercise the
template end-to-end after the offline smoke test (the catalog itself was a
throwaway, so its RIDs/host/catalog-id are deliberately *not* recorded here —
fetch live catalog state from `ml.find_datasets()` instead).

**`--dry-run` is not a no-op — it provisions the schema.** The loader's
`--dry-run` creates the catalog and installs the full schema (`cifar10_*`
domain model + `deriva-ml`), and only skips the *data* writes (image upload,
features, dataset hierarchy). So a "dry run" leaves a real, schema-provisioned
but empty catalog behind. Useful to know both for verifying auth/connectivity
cheaply and for not accidentally accumulating orphaned catalogs.

**The config-name → dataset-role mapping is a project decision** (the catalog
stores each dataset's RID, role, and origin tags, but not which
`src/configs/datasets.py` name is *intended* to pin it — that association lives
in the template). `load-cifar10` always builds the same named family; the
durable mapping is:

| `datasets.py` config name | Dataset role / origin |
|---|---|
| `cifar10_complete` | Complete (Labeled) — the root superset |
| `cifar10_split` / `_training` / `_testing` | canonical Toronto Split + its Training/Testing children (`Split_Partition`) |
| `cifar10_small_training` / `_small_testing` | stratified `Subsample` of Training/Testing (sibling pair, no parent Split) |
| `cifar10_labeled_split` / `_training` / `_testing` | training-derived labeled holdout Split + children |
| `cifar10_small_labeled_split` / `_training` / `_testing` | small labeled holdout Split + children (the example model's default) |

(The full structural detail lives in the forward design doc
`docs/design/dataset/cifar10-input-datasets.md`.)

Implications for collaborators: to run the example model against a freshly
loaded catalog, either pass `--host <h> --catalog <id>` on the CLI, or fill the
RIDs into `src/configs/datasets.py` (per README §7) — the shipped defaults
carry stale RIDs from a prior demo catalog and won't resolve. Resolve the
actual RIDs + versions for your catalog via `ml.find_datasets()` before pinning
them. If you load with `DERIVA_ML_ALLOW_DIRTY=true` (uncommitted tree), the
recorded git provenance hash won't reflect the working tree — fine for a
throwaway catalog, but don't cite that provenance as reproducible.

<a id="tk-004"></a>
### tk-004 — Convention — `add_files()` is the only by-reference asset API in deriva-ml 1.45; it tags files as Output, and `LocalFile` (the Input-side primitive) isn't shipped yet
**When:** 2026-06-24T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

Surveyed how to register an external/remote file *by reference* (record its
URL + MD5 + length without copying bytes into hatrac) — the use case being to
register the upstream CIFAR-10 source archive for provenance rather than
re-uploading it. Findings about the deriva-ml 1.45 API surface, recorded
because they shape any future by-reference work in this project:

- **`add_files(files, execution_rid, dataset_types, description)`** is the only
  by-reference API actually available. It inserts `File` rows (a `FileSpec`
  carries `url` + `md5` + `length`; remote URLs pass through, local paths are
  rewritten to `tag://` URIs), links them to the execution, and returns a
  `Dataset`. It tags the files **`Asset_Role="Output"`** of that execution.
- **There is no by-reference path into a named domain asset table** (e.g.
  `Image`). `create_asset` always wires a hatrac upload template — the
  `use_hatrac=False` flag that builds an external/URL-only table is internal
  and used solely to construct the generic `File` table. So: external file →
  generic `File` table; named asset table (`Image`, model weights, …) → byte
  upload via `asset_file_path()` + `commit_output_assets()`. No overlap.
- **`LocalFile`** (declared as `ExecutionConfiguration(assets=[LocalFile(
  path=...)])`, plus a `LocalFileConfig` for hydra) would register the file as
  an execution **Input** by reference — the *semantically correct* primitive
  for a *source the run consumes*, since `add_files`' `Output` role is wrong
  for a source archive. **But it is not present in deriva-ml 1.45.0** — not
  importable from `deriva_ml.execution`, no `LocalFileConfig` export, zero
  matches in the package. It is documented in the `work-with-assets` skill
  (plugin v1.11.1), so it is either a newer or not-yet-released API.

Implications for collaborators: to register a by-reference source today, use
`add_files()` and accept that the source archive is recorded as an `Output`
(lineage still walks from `Image` assets back to the archive via the shared
upload execution). If/when `LocalFile` ships in the pinned deriva-ml, prefer it
for *inputs* — it carries the correct Input role and keeps source bytes local.
Re-check `from deriva_ml.execution import LocalFile` against the installed
version before designing around it.

**Weighed alternatives:** considered a purpose-built named external asset table
via `create_asset(use_hatrac=False, ...)` — rejected: `use_hatrac` isn't a
public `create_asset` parameter, so it would mean hand-building schema, and the
generic `File` table already serves the by-reference role.

<a id="tk-005"></a>
### tk-005 — Convention — `add_files()` in deriva-ml 1.51.9 tags references as **Input** and creates one File-typed dataset per source directory
**When:** 2026-06-24T01:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-004](#tk-004) (the 1.45 behavior this supersedes after the version bump)

Bumped the pinned deriva-ml from 1.45.0 to **1.51.9** (git dep, via
`uv lock --upgrade-package deriva-ml`) specifically to get the corrected
by-reference semantics. Two claims in [tk-004](#tk-004) are **version-bound to
1.45 and no longer hold on 1.51.9** — recording the corrected behavior here:

- **Role flipped Output → Input.** `add_files` now links each `File` row as
  `File_Execution.Asset_Role="Input"`, by design and intrinsically (not a
  parameter): *"a `File` reference names a file the run consumed, so it is
  always an Input."* This makes `add_files` the semantically-correct primitive
  for recording a *source* a run consumes — so the source-archive use case no
  longer needs `LocalFile` to get the Input role. (`LocalFile` does now exist —
  `deriva_ml.asset.aux_classes.LocalFile(*, path: str, cache: bool=False)` —
  but it's the config-declared input path, a different ergonomic, not required
  for in-loop registration.)
- **`Execution.add_files` injects `execution_rid`.** The `Execution`-bound
  method signature is `add_files(files, dataset_types=None, description="")` —
  no `execution_rid` arg; call it as `exe.add_files(specs)`. (The `DerivaML`
  mixin method still takes `execution_rid` explicitly.)

Answers to "what datasets/types does `add_files` create" (from the 1.51.9
source, durable behavior):

- **Dataset_Type tags:** every dataset `add_files` creates is tagged **`File`**
  (always prepended), **plus** any terms passed in `dataset_types` (which must
  pre-exist in the `Dataset_Type` vocabulary — it `lookup_term`s each and
  raises on an undefined term). The same tag list is applied to every dataset
  the call creates, nested ones included. Note the two distinct axes: a
  `FileSpec`'s `file_types` are validated against the **`Asset_Type`** vocab and
  tag the `File` *rows*; `dataset_types` are validated against the
  **`Dataset_Type`** vocab and tag the *datasets*.
- **Datasets created:** **one dataset per distinct parent directory** of the
  registered files, built deepest-first and **nested to mirror the directory
  tree** (child-dir datasets become members of their parent-dir dataset); the
  call returns the top-most dataset. So a *flat* source dir → exactly **one**
  File-typed dataset containing all the file references; a dir with subfolders
  → a nested dataset hierarchy.

Implications for collaborators: registering a flat directory of CIFAR source
images via `FileSpec.create_filespecs(source_dir)` + `exe.add_files(specs)`
yields a single `File`-typed Input dataset of references, with no bytes
uploaded — pair it with the existing `asset_file_path(copy_file=False)` +
`commit_output_assets()` Output-upload to get both the source-provenance layer
and the hatrac-backed `Image` assets in one execution. If you want the source
dataset to carry a domain tag (e.g. `CIFAR_Source`) alongside `File`, add that
term to the `Dataset_Type` vocabulary in the schema phase first.

<a id="tk-006"></a>
### tk-006 — Dead end — a single `add_files()` of ~600 source images fails inside `add_dataset_members`/`resolve_rids`
**When:** 2026-06-24T02:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-005](#tk-005) (the add_files source-registration approach this implements)

Implemented the source-provenance idea: in `_cifar10_assets.py:upload_images`,
register the sampled source PNGs via `exe.add_files(FileSpec.create_filespecs(
...))` as by-reference Input before the per-image Hatrac upload. A first run
against a fresh localhost catalog with `--num-images 2000` (so ~600 sampled
source files passed in one `add_files` call) **failed in the datasets step of
`add_files` itself**, not in the template logic:

- `add_files` inserted the ~600 `File` rows, then its internal
  `Dataset.add_dataset_members(members=<600 RIDs>)` → `resolve_rids` raised
  `DerivaMLRidsNotFound` for *all* of those just-inserted RIDs.
- A secondary `400 Bad Request` followed — *"index row size 3352 exceeds btree
  maximum 2704 for `Execution_Status_Detail_idx`"* — which is a **red herring**:
  it's the execution state machine trying to write the giant 600-RID
  `RidsNotFound` error message into the indexed `Execution.Status_Detail`
  column, which can't hold a value that large. The btree error is a *symptom of
  the error message's size*, not the cause.

Root cause (as far as reading the 1.51.10 source took it): the failure is in
deriva-ml's `add_files` → `add_dataset_members` → `resolve_rids` path when many
members are passed at once. `resolve_rids` resolves the freshly-inserted File
RIDs by querying the `File` table with `RID == AnyQuantifier(*~600 rids)` (a
single large IN-style ERMrest query); for ~600 RIDs that lookup returns nothing
(candidate `File` rows not found), so every RID is reported missing. Most
likely a request/URL-size or read-after-write visibility limit on the batched
resolution — an **upstream scale limit in `add_files`**, not a template bug.
The catalog (108) was left schema-provisioned but the upload execution failed,
so no images/datasets landed.

Implications for collaborators: `add_files` is not safe to call with hundreds
of files in one shot against this stack today. Do not pass the whole sampled
corpus in a single `add_files`. Open options (none yet validated): (a) batch
`add_files` into small chunks (e.g. ≤100 files/call); (b) register a single
*source-archive* File reference (one `.tar.gz`, the original design) instead of
per-image references, which sidesteps the many-member path entirely; (c) report
upstream and pin a fix. Option (b) is both smaller and closer to the original
"record where the data came from" intent.

**Weighed alternatives:** registering per-image source references (chosen for
this attempt because it mirrors the exact files uploaded) vs. registering the
one source archive. The per-image path hit this scale wall; the archive path
avoids it and is the likely pivot.

<a id="tk-007"></a>
### tk-007 — The deriva-ml 1.51.11 `add_files` "fix" does NOT resolve the many-member failure; the membership path is still unbatched
**When:** 2026-06-24T03:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-006](#tk-006) (the failure this version was meant to fix)

Bumped deriva-ml 1.51.10 → 1.51.11 (commit `8fd41b31`) to pick up an `add_files`
fix, then re-ran the per-image source-registration load (`--num-images 2000`,
~1000 source files per train/test directory). **It failed with the exact same
error** as [tk-006](#tk-006) — byte-identical traceback: `add_files`
(`file.py:209`) → `Dataset.add_dataset_members` (`dataset.py:2134`) →
`resolve_rids` (`rid_resolution.py:216`) → `DerivaMLRidsNotFound`, plus the same
`Execution_Status_Detail_idx` btree-2704 red herring.

What 1.51.11 actually changed (and why it missed): `add_files` now **streams the
File-row inserts in batches of `chunk_size=500`** (`for batch in batched(files,
chunk_size)`). That bounds the *insert/tag/link* side. But the **dataset-build
tail is unchanged** — after all batches, it still calls
`dataset.add_dataset_members(members=<all RIDs for that directory>)` once per
source directory. For a 2000-image load that is ~1000 RIDs in a single
`add_dataset_members` → a single `resolve_rids` →
`RID == AnyQuantifier(*~1000 rids)` query, which is the call that fails. So the
fix addressed insert batching, not the membership-resolution path that `tk-006`
identified as the actual failure point.

Implications for collaborators: do not assume the by-reference `add_files` path
works for many files on 1.51.11 — it does not. The unblock still requires either
(a) an upstream fix to **batch `add_dataset_members`/`resolve_rids`** (not just
the inserts), (b) register the single source *archive* (one File row, no
many-member membership call — sidesteps it entirely), or (c) chunk our own
`add_files` calls so each directory's membership stays small. Re-verify against
whatever deriva-ml version claims the fix by reading the `add_files` *tail*
(`add_dataset_members`), not just the insert loop — the insert batching is a
decoy.

**Weighed alternatives:** *(none new — same options as [tk-006](#tk-006); this
entry records that the version-bump path (c→upstream) did not pan out on
1.51.11.)*

<a id="tk-008"></a>
### tk-008 — Fixed in deriva-ml 1.51.12 — `resolve_rids` now chunks at 500 RIDs/query; the per-image `add_files` source registration works end-to-end
**When:** 2026-06-24T04:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-006](#tk-006) (original root cause), [tk-007](#tk-007) (the 1.51.11 fix that missed)

Bumped deriva-ml to 1.51.12 (commit `a03b1db6`) and the `add_files` many-member
failure from [tk-006](#tk-006)/[tk-007](#tk-007) is **resolved**. The fix landed
in the right place this time: `resolve_rids` (`core/mixins/rid_resolution.py`)
now defines `_MAX_RIDS_PER_QUERY = 500` and **chunks** the lookup —
`for rid_chunk in batched(remaining_rids, 500): filter(RID == AnyQuantifier(
*rid_chunk))`. The in-code comment confirms the exact root cause we traced: *"a
10k-RID URL is ~70 KB → HTTP 414. Without chunking, resolving [fails]... 20
chunks of 500 fetched cleanly, no 414."* So the original failure was an HTTP
request-size (URL-length / 414) limit on the single giant `RID == Any(...)`
query — not read-after-write. (`add_files`' own tail still calls
`add_dataset_members(members=<~1000 rids>)` unbatched, but it no longer needs to
batch, because the resolution underneath it now chunks.)

Verified end-to-end: re-ran `load-cifar10 --create-catalog --num-images 2000`
with the per-image source-registration step. The upload execution completed —
2000 images, 2000 features, and the full 12-dataset hierarchy — **plus** the
`add_files` source registration produced the expected File/Directory datasets
(`['File','Directory']` tags, one per `train/`/`test/` source subdirectory, per
[tk-005](#tk-005)'s one-dataset-per-directory rule).

Implications for collaborators: the by-reference `add_files` source-provenance
approach is now viable on deriva-ml ≥ 1.51.12. Pin at or above that version when
relying on `add_files` (or any `resolve_rids` / `add_dataset_members` call) with
more than ~500 members. The lesson from the three-version chase: when an upstream
"fix" claims to address a many-member failure, verify it touched the
**resolution query** (`resolve_rids` chunking), not just the insert loop —
1.51.11 batched inserts and looked fixed but wasn't (see [tk-007](#tk-007)).

<a id="tk-009"></a>
### tk-009 — `add_files` nesting only fires for genuinely-nested dirs, NOT equal-depth siblings — so `train/`+`test/` give two flat File datasets, never a parent
**When:** 2026-06-24T05:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-005](#tk-005) (the one-dataset-per-directory behavior this refines)

Tried to get a nested **parent → train + test** File-dataset tree out of the
source `add_files` registration by staging the sampled images under one root
(`_source/train/`, `_source/test/`) and pointing `create_filespecs` at the root.
**It did not work** — the catalog still has **two flat sibling File datasets**
(1000 members each, no parent, no children), identical to passing a flat file
list.

Root cause (read from `add_files`, `core/mixins/file.py`): it buckets files into
`dir_rid_map` keyed by each file's **immediate parent directory**, then creates
one dataset per bucket and nests only when a directory is *shallower* than the
previous one (`if len(p.parts) < path_length`). Two leaf dirs at **equal depth**
(`train`, `test`) never satisfy that `<` condition, and the enclosing `_source/`
directory **contains no files directly**, so it never becomes a bucket key →
no parent dataset is ever created. The nesting algorithm only materializes a
parent for *genuinely nested* file-bearing paths (e.g. `a/` and `a/b/`), not for
sibling leaves under a common (empty) parent.

Implications for collaborators: you cannot get a "whole-collection" parent
File dataset over `train`+`test` just by arranging the directory layout — the
staging-into-one-root trick is wasted effort (it adds a staging dir + symlinks
for zero structural gain). To get a parent that contains the train + test File
datasets, **create the parent dataset explicitly** (`create_dataset` +
`add_dataset_members([train_ds, test_ds])`) rather than relying on `add_files`'
directory nesting. Absent that, the honest description of the source-provenance
layer is "two flat File datasets, one per Toronto partition."

**Weighed alternatives:** (1) staged nested layout — *tried, failed* (this
entry). (2) explicit parent dataset — viable, hand-built nesting. (3) accept two
flat datasets — simplest; the train/test source split is still captured, just
without an over-arching parent. Decision pending.

<a id="tk-010"></a>
### tk-010 — Fixed in deriva-ml 1.51.14 — `add_files` now nests equal-depth siblings under a common root; `Dataset.source_directory`/`is_directory` expose the structure
**When:** 2026-06-25T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-009](#tk-009) (the equal-depth-sibling nesting gap this fixes)

Bumped deriva-ml to 1.51.14. The [tk-009](#tk-009) nesting gap is **resolved**:
`add_files` replaced its incremental depth-comparison loop with a
`_directory_tree()` helper that finds the source directories' common ancestor
(the "ingest root") and materializes a dataset for **every** tree node — each
file-bearing directory *plus every intermediate ancestor up to the root*. So
equal-depth leaf dirs (`train`, `test`) now get a parent dataset for their
common ancestor even though that ancestor holds no files directly. `add_files`
now returns the **ingest-root** dataset.

New API for the directory structure (this is the "directory information on
datasets" the loader now uses): each directory dataset records the folder it
represents in a new `Directory_Dataset` table, surfaced as two `Dataset`
properties — `source_directory` (the path relative to the ingest root; root is
`"."`) and `is_directory` (True iff it has a `Directory_Dataset` row).

Verified end-to-end: re-ran the loader (which stages the sampled images under
`_source/train/` + `_source/test/` and calls `add_files(create_filespecs(
_source))`). The catalog now holds **three** File datasets in a tree — a root
(`source_directory='.'`, members = the two child *datasets*) with `train`
(`source_directory='train'`, 1000 File members) and `test`
(`source_directory='test'`, 1000 File members) as children — instead of the two
flat siblings from [tk-009](#tk-009). The loader logs the children via
`source_root_ds.list_dataset_children()` + `child.source_directory`.

Implications for collaborators: registering a staged directory tree via
`add_files` now yields a faithful nested dataset hierarchy on deriva-ml ≥
1.51.14 — the "build the parent dataset explicitly" workaround
([tk-009](#tk-009) option 2) is no longer needed. Query a source dataset's
origin folder with `Dataset.source_directory` and gate on `Dataset.is_directory`
to separate auto-created directory datasets from curated ones. Pin ≥ 1.51.14
when relying on this.

<a id="tk-011"></a>
### tk-011 — The `add_files` source File datasets do NOT appear in an image dataset's `lookup_lineage` — source provenance and dataset lineage are parallel, execution-bridged structures
**When:** 2026-06-25T01:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-010](#tk-010) (the nested source File datasets whose lineage visibility this checks)

Checked whether the by-reference source File datasets (the `add_files`-built
`root → train + test` tree from [tk-010](#tk-010)) surface as *consumed* inputs
when walking a downstream image dataset's lineage. They do **not**.
`ml.lookup_lineage(<a Small_Testing subsample>)` reports its producing
execution's `consumed_datasets` as the **image** datasets only (Complete + the
Split's Training/Testing) — none of the three source File datasets appear.

Why, and the durable shape: the CIFAR loader builds everything in one execution
("CIFAR-10 Asset Upload" workflow), but provenance fans into two **parallel**
branches off that shared execution, with no dataset-to-dataset edge between them:

- **Source branch:** `add_files` registers source File rows as *Input*; the
  upload produces the `Image` assets as *Output*. (File rows → Image assets.)
- **Dataset branch:** the dataset hierarchy is assembled from `Image` RIDs
  (Complete → Split → subsample), so a dataset's lineage walks
  Image-RID-membership and its producing execution — never the File datasets.

So `lookup_lineage` on an image dataset will not lead a reader to the source
images. To bridge the two you must hop manually: image dataset → its `Image`
members → the execution that produced those Images → that execution's Input
`File` rows. (Observed wrinkle: the lineage's `consumed_assets` listed a single
`File` RID that was *not* one of the 2000 source rows — even the individual
source File rows don't reliably show as consumed assets of the dataset's
producing execution; the connection is the shared *execution*, not a recorded
consumed-asset edge.)

Implications for collaborators: the `add_files` source-provenance layer is a
**separate, queryable record of origin** (browse the File datasets directly, or
filter `Dataset.is_directory`), **not** something that shows up in an image
dataset's automatic lineage walk. Don't expect `lookup_lineage` on a training
dataset to surface where the raw files came from — that linkage is
execution-mediated and must be traversed by hand. If automatic dataset→source
lineage is a requirement, it would need an explicit `Dataset_Dataset` edge from
the image Complete dataset to the source File root, which `add_files` does not
create (see [tk-005](#tk-005)).

<a id="tk-012"></a>
### tk-012 — Operational hazard — the `../deriva-ml` repo is a shared multi-agent working tree; never blind `git stash pop` there
**When:** 2026-06-26T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

While preparing to merge + release a `deriva-ml` change, a `git stash pop` in
`../deriva-ml` accidentally applied **another agent's stash** (`stash@{0}`,
labeled *"current-mods on other agent's branch — need to move to my branch"*),
spilling unrelated denormalize/local_db work into the tree as `UU`/`DU`
conflicts. The `deriva-ml` repo carries **~10 stashes from multiple
agents/branches** (audit-thread, denormalize-user-guide, describe-warnings, …)
— it is a busy shared working tree, not a private one.

Recovery was non-destructive only because **`git stash pop` that hits conflicts
does NOT drop the stash** — `stash@{0}` stayed intact, so the other agent's work
was never at risk. The conflict-spilled tree copy was redundant; restoring the
unmerged paths to `HEAD` (`git checkout -f HEAD -- <file>` for `UU`; `git rm` for
the `DU` file absent in HEAD) cleaned the tree while leaving the stash for its
owner to pop onto their own branch.

Implications for collaborators: in `../deriva-ml` (and likely the other
shared workspace repos), do **not** run bare `git stash`/`git stash pop` — a pop
grabs whatever is at `stash@{0}`, which is probably someone else's. Inspect
`git stash list` first and pop by explicit ref only for a stash you created and
recognize. When you need a scratch stash, prefer `git stash push -m "<your
unique label>" -- <specific paths>` and pop that exact entry by message/index.
Our own committed work is never affected by this (commits are branch-scoped);
the hazard is purely the shared *stash stack* and working tree.

<a id="tk-013"></a>
### tk-013 — Staging bug — `create_filespecs(cache_root)` registered all 60K extracted files because `_extract/` lived inside `cache_root`
**When:** 2026-06-26T18:30:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-011](#tk-011) (the lineage gap this two-execution redesign fixes)

First live run of the two-execution loader (`register` phase) hung for 25+
minutes at ~18% CPU with **zero `File` rows** committed. A process stack sample
showed it pinned in `_pydantic_core` validation (not network I/O). Root cause:
the register scaffold's `stage_source` extracted the full CIFAR archive into
`cache_root/_extract/` (≈60,000 PNGs) and symlinked the 2,000 sampled files into
`cache_root/train` + `cache_root/test` — keeping `_extract/` *inside* `cache_root`
so symlink targets stay alive. But `run_register_phase` then calls
`FileSpec.create_filespecs(cache_root)`, which `rglob("*")`s the **entire**
`cache_root` — so `add_files` tried to register all **60,001** files (MD5 +
pydantic-validate each), ~30× the intended 2,000, hence the multi-minute
pydantic hot loop.

The trap: `create_filespecs(dir)` walks the whole subtree recursively with no
exclusion. Putting the bulk extraction *under* the directory you hand to
`create_filespecs` silently balloons the registration to the full corpus. Unit
tests missed it because they monkeypatched `extract` to write a handful of files
— the 60K archive only appears in a real run.

Fix (applied): extract to a temp dir **outside** `cache_root`; **copy** (not
symlink) the sampled files into `cache_root/train` + `cache_root/test`; remove
the temp extraction. `cache_root` then holds *only* the sampled subset +
`labels.csv`, so `create_filespecs(cache_root)` registers exactly the sampled
files while still giving `add_files` the single nested root. The byte copy is
~2,000 tiny PNGs (~6 MB) — negligible; the "no byte copy / symlink" optimization
only mattered for the full 60K corpus, not a sample.

Implications for collaborators: whatever directory you pass to
`create_filespecs`/`add_files` must contain **only** the files you intend to
register — no scratch/extraction subdirs. Stage the exact set into a clean
directory; don't co-locate bulk working files with the registration root.

<a id="tk-014"></a>
### tk-014 — `add_files`-registered File rows have `Filename = NULL`; read the name from the `URL` tag path, not `Filename`
**When:** 2026-06-26T19:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-013](#tk-013) (same two-execution upload phase)

The upload execution (Exec 2) crashed reading back the registered source File
dataset: `AttributeError: 'NoneType' object has no attribute 'endswith'` while
matching the `labels.csv` File by `r.get("Filename", "").endswith("labels.csv")`.
Inspecting the catalog: the File row's **`Filename` column is `None`**, even
though its `URL` is `tag://host,date:file:///.../labels.csv`. So
`add_files` / `FileSpec.create_filespecs` populate the **`URL`** (the tag path)
but leave **`Filename` NULL** — the by-reference File table does not derive a
`Filename` from the path. (The `.get("Filename", "")` default is no help: the
key exists with value `None`, so the default never applies and `None.endswith`
raises.)

Two consequences the upload code got wrong:
- Matching the manifest File by `Filename` fails — must match on the **basename
  of the `URL`** instead (`Path(urlsplit(url).path).name`, or
  `tag_url_to_path(url).name`).
- Per-image stem/class lookup likewise can't use `Filename` — derive the stem
  from the `URL` path too.

Implications for collaborators: when consuming `add_files`-registered File rows,
treat **`URL` as the source of truth for identity/name**; do NOT rely on
`Filename` (it is NULL for by-reference files). Guard any `.endswith`/string op
with `(rec.get("Filename") or "")` if you must touch it, but prefer deriving the
name from `URL`.

<a id="tk-015"></a>
### tk-015 — Consuming a by-reference File dataset as an execution Input must use `DatasetSpec(materialize=False)` — bag materialization can't fetch `tag://` local URLs
**When:** 2026-06-26T19:30:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-011](#tk-011) (the lineage fix that requires consuming the File dataset as Input), [tk-014](#tk-014) (same upload phase)

The two-execution upload phase consumes the source File dataset as an Input so
that lineage connects (image dataset → upload exec → source files). But
`create_execution` **materializes every input dataset as a BDBag by default**,
and bag materialization *validates by fetching each member's bytes from its URL*.
Our File rows carry `tag://host,date:file:///…/cache/…/*.png` URLs (local,
by-reference — see [tk-005](#tk-005)). The bag fetcher can't retrieve `tag://`
local files: it looks in the bag-cache dir, finds nothing, and
`bdbag` raises `BagValidationError: Bag validation failed` on all ~2,000 members.
So the very mechanism that makes lineage connect (consume-as-Input) also triggers
a byte-fetch the by-reference design can't satisfy.

Fix: pass **`DatasetSpec(rid=…, version=…, materialize=False)`** for the consumed
File dataset. `materialize=False` downloads only the **table metadata** (the File
rows + their URLs) — no asset byte-fetch, no bag validation. That is exactly what
the upload phase needs: it reads the File records' URLs, resolves the `tag://`
paths to the local cache itself, and uploads from there. The Input edge (and thus
the lineage) is still recorded; only the (impossible) byte-fetch is skipped.

Implications for collaborators: any execution that consumes a **by-reference**
File dataset (one registered via `add_files` with local `tag://` URLs) as an
input MUST set `materialize=False` on its `DatasetSpec`. The default `True` will
fail bag validation. This is a general constraint on by-reference datasets, not
CIFAR-specific. (Hatrac-backed asset datasets materialize fine; the limitation is
specific to local/`tag://`-referenced files.)
