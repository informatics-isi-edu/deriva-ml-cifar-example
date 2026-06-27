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

## 2026-06-27 — RESOLVED: root File-dataset Folder shows its basename, not "." (deriva-ml 1.54.0)

**Outcome:** The long-standing nit — the root of an `add_files` File-dataset tree
displaying `Directory_Dataset.Path = "."` instead of its directory name — is
**fixed and verified end-to-end.** Shipped in **deriva-ml 1.54.0** (merged to
main, tag v1.54.0, pushed); cifar-example lock-bumped to it (commit b745256).

**What changed (see [[2026-06-27-directory-dataset-path-load-bearing-marker]]):**
- `add_files` now writes the root's directory **basename** to
  `Directory_Dataset.Path` (via `_root_path_name`, precedence `root_name or
  ingest_root.name or "root"`, sharing the root Description's precedence so the
  two columns agree). Children keep their relative path (`train`/`test`).
- Root identification moved off the `== "."` string onto the new STRUCTURAL
  accessor `Dataset.is_source_root` / `DatasetBag.is_source_root`
  (`is_directory and no parent is a directory dataset`) — works on both legacy
  (".") and new catalogs, so **no backfill** was needed.

**Live verification (fresh catalog 346, deriva-ml 1.54.0, `--phase all`):**
- root `AT4`: `Directory_Dataset.Path == "cifar10_source"` (basename, NOT ".")
- exactly ONE `is_source_root == True` (AT4); children `train`(ATG)/`test`(ATT)
  both False.

**For a future reader:** old catalogs (built pre-1.54.0, e.g. 328) still show the
root Path as "." — that's expected (no backfill), and `is_source_root` finds the
root on them anyway. Only NEW loads get the descriptive basename.

---

## 2026-06-27 — The deriva-ml test suite is slow + NOT safe to parallelize; scope to the diff, don't reach for pytest-xdist

**Context:** The deriva-ml offline test suite (`uv run python -m pytest tests/ -m "not integration"`)
is long-running — its live-catalog tests hit a local Deriva instance, and even
the focused `tests/core/test_file.py` + `tests/dataset/test_bag_api_coverage.py`
group (45 tests) took **8m19s**. Background runs of the full suite were killed
three times mid-run (32%/32%/46%, zero failures each) because wall-clock exceeded
the background slot.

**Why pytest-xdist is NOT the fix (the non-obvious part):** `pytest-xdist` is
**not installed**, and more importantly the suite is **not parallel-safe as
written**. `tests/conftest.py` defines `catalog_manager` at `scope="session"`
(conftest.py:57) — a SINGLE shared test catalog for the whole run — and tests
mutate it via `catalog_manager.reset()` / `CatalogState`. Under xdist each worker
gets its own session, so workers would either collide on the same Deriva catalog
(reset-mid-test races → flaky failures, corrupted state) or each need to
provision its own catalog (the fixture isn't built for that). So `pytest -n auto`
would BREAK the suite, not speed it up. Making it parallel-safe is real
infra work (per-worker catalog provisioning, or cleanly splitting catalog-free
offline tests from live ones), not a quick win.

**How to apply — for any deriva-ml test run:**
1. **Scope to the diff.** Run only the impacted dirs/files (e.g.
   `tests/core tests/dataset tests/schema` for a dataset/file change), not
   `tests/`. This is the durable speed-up and what we shipped on
   (the change surface here was narrow + self-contained).
2. For a true full-suite green light, **run it in your own terminal** where
   nothing recycles the process, OR accept targeted-suite + review as the
   regression evidence (what we did: the 45 targeted tests exercise every
   changed line; whole-branch review verified line-by-line).
3. Don't add `pytest-xdist` expecting a free win — it needs the session-scoped
   shared-catalog fixture redesigned first.

---

## 2026-06-27 — Two distinct meanings of `"."`: catalog root-marker (migrated) vs. flat-layout PARTITIONS sentinel (kept)

**Decision:** When migrating the `source_directory == "."` root-identification
readers to the new structural `is_source_root` accessor (the root-Path-basename
change), the `"."` in `deriva-ml-skills/.../upload_phase_template.py` was
**deliberately left as-is**, while every other `== "."` reader was migrated.

**Why the distinction matters.** There are two unrelated uses of `"."` in this
area, and only one is a catalog read:
1. **Catalog root-marker (MIGRATED):** `d.source_directory == "."` read the
   stored `Directory_Dataset.Path` to find the tree root. The writer change
   makes the root store its basename, so these readers would break — they now
   use `d.is_source_root` (structural, parent-graph-based). Sites:
   cifar `load_cifar10.py`, `test_lineage_connected.py`; skills
   `loader_orchestrator_template.py`.
2. **Flat-layout PARTITIONS sentinel (KEPT):** in
   `upload_phase_template.py`, `partition == "."` compares against the
   **caller-supplied `PARTITIONS` list**, documented in `setup-ml-catalog`'s
   SKILL.md as "`["."]` for a flat layout." This `"."` is a template *API
   convention* meaning "the root dataset itself, no partition children" — it is
   never read from the catalog, so the writer change does not touch it. Changing
   it to match `source_ds.source_directory` would have **broken** the documented
   flat-layout contract. Left as-is with a clarifying comment.

**How to apply:** Before mechanically migrating a `== "."` comparison, check
whether the left side is a value READ FROM THE CATALOG (`Directory_Dataset.Path`
/ `source_directory`) or a SENTINEL in a caller-supplied list/API. Only the
former is affected by the root-Path-basename change. [[2026-06-27-directory-dataset-path-load-bearing-marker]]

---

## 2026-06-27 — `Directory_Dataset.Path = "."` for the root is a load-bearing marker, not just a display value

**Context:** The root node of an `add_files` nested File-dataset tree stores
`Directory_Dataset.Path = "."` (deriva-ml `core/mixins/file.py:382`), while
children store their relative path (`train`, `test`). The root's *name* lives
separately in `Dataset.Description` (e.g. `cifar10_source`). A user browsing the
catalog sees the root "Folder" column as `.`, which reads as uninformative.

**Why `"."` is not just cosmetic — it is the root-identification key.** An audit
(2026-06-27) found five functional readers across three repos that use
`source_directory == "."` to *find the root* of the tree, not merely to display
it:
- `deriva-ml-cifar-example/src/scripts/load_cifar10.py:177` —
  `_find_latest_source_dataset_rid` filters `source_directory == "."`.
- `deriva-ml-cifar-example/src/scripts/_cifar10_upload.py:319` — root holds
  `labels.csv`; children are the `train`/`test` partitions.
- `deriva-ml-cifar-example/tests/test_lineage_connected.py:214` — finds the root
  via `== "."`.
- `deriva-ml-skills/skills/setup-ml-catalog/scripts/loader_orchestrator_template.py:75`
  and `upload_phase_template.py:102` — shipped-to-users templates with the same
  `== "."` root check.
Plus four deriva-ml tests assert the root Path is exactly `"."`
(`tests/core/test_file.py:173,278`, `tests/dataset/test_bag_api_coverage.py:519,530`)
and the convention is documented in `create_schema.py:69`, both
`source_directory` docstrings, and a design plan.

**Implication for any "make the root show its name" change.** Changing the
stored value from `"."` to the basename is a **cross-repo coordinated change**
(deriva-ml writer + accessors + 4 tests + schema comment; cifar-example 2
readers + 2 tests; deriva-skills 2 templates). Every `== "."` reader must switch
to a different root-identification method *in lockstep with the writer*, or root
detection silently breaks. The open design question is **how to identify the
root once `"."` is gone** — basename comparison is fragile (collides if a child
were ever named like the root); a dedicated null/boolean root marker, or
identifying the root as the dataset with no `Directory_Dataset` parent, is
cleaner. This warrants brainstorm→spec→plan, not an inline edit. [[2026-06-27-end-to-end-run]]

---

## 2026-06-27 — End-to-end run on deriva-ml 1.53.0 (catalog 328): the labeled-split evaluation path

**Decision:** For a fresh end-to-end demonstration (load → train → evaluate),
use the `cifar10_quick` experiment preset (model `cifar10_quick` + dataset
`cifar10_small_labeled_split`) against a catalog loaded with
`--num-images 1100`.

**Why these choices:**
- **`cifar10_small_labeled_split`, not a plain `*_split`.** Only the
  labeled-split family carries ground truth on *both* the train and test
  partitions, so the run can record real predictions and the loader can report
  emission-time accuracy. The plain `*_split` configs are training-only and
  yield no checkable evaluation. (Run recorded 100 `Image_Classification`
  feature rows for the test partition, each with `Image_Class` + `Confidence`.)
- **`--num-images 1100`.** The small labeled-split family needs
  `≥ 1002` images (`2*(SMALL_TEST_SIZE+1)`, with SMALL_TRAIN=SMALL_TEST=500 in
  `src/scripts/_cifar10_datasets.py`); below that the small variant would be
  byte-identical to the full split and the datasets phase raises
  `SmallVariantDegenerateError`. 1100 is the known-good bootstrap floor (also
  used by `tests/test_load_cifar10_split_no_leakage.py`).
- **`--phase all` on a fresh catalog.** `--phase register` alone fails on an
  empty catalog (`DerivaMLInvalidTerm: CIFAR_Source_Registration` — the
  Workflow_Type vocab term doesn't exist yet). `all` runs the schema phase
  first, which installs the vocabulary.

**The config-wiring step is mandatory and easy to miss.** `src/configs/datasets.py`
ships `cifar10_small_labeled_split` as an empty list. After the load you must
fill it with the new catalog's `Small_Labeled_Split` RID + version
(`DatasetSpecConfig(rid="ZB8", version="0.1.0.post1.dev1")` for catalog 328),
or the run fails with "dataset not found." This is the single most common
first-run snag — README §7 is the procedure. (Revert to the empty placeholder
before committing the template.)

**Lineage verified end-to-end (the point of the tk-018..tk-024 work).**
`lookup_lineage("10BT")` on the training execution shows it consuming `ZB8` (one
`Dataset_Execution` input edge, `consumed_assets: []` — no per-file bloat,
confirming the 1.53.0 add_files change). Walking `lookup_lineage("ZB8")` then
reconstructs the full source chain back through the CIFAR-10 Source Registration
executions (`Y6J → QTY → D0E → 4AP`) to the root `cifar10_source` dataset `AT4`,
with `walked_complete=True`, `cycle_detected=False`. So source images → labeled
split → training run is fully connected provenance on a freshly built catalog.
