# Two-Execution CIFAR Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split CIFAR ingest into two decoupled executions (register → upload) so the source File dataset is a real consumed Input of the upload, connecting source→image lineage; name the ingest-root dataset via a clean `add_files` change in deriva-ml.

**Architecture:** A `register` execution stages sampled source images into a stable cache dir (with a `labels.csv` manifest at the root) and registers them via `add_files`, producing a named, nested source File dataset tree. A separate `upload` execution consumes that File dataset as an Input, walks its directory children to classify train/test, reads the registered manifest for labels, resolves each File's tag-URL to the cache path, and uploads `Image` assets. The two executions communicate only through the catalog. A coordinated deriva-ml change makes `add_files` name the ingest-root dataset from its directory basename (or an explicit `root_name`).

**Tech Stack:** Python, `uv`, deriva-ml (sibling repo `../deriva-ml`), pytest, ruff.

## Global Constraints

- Repos: `../deriva-ml` (library change) and this repo (`deriva-ml-cifar-example`). Each has its own `pyproject.toml`/`uv.lock`/tests; `cd` into the right one per command.
- **No backwards-compat shims** (workspace rule) — the `add_files` change replaces behavior; do not preserve the old root-description default.
- Use `uv run` for everything; in this repo use `uv run python -m pytest` (the bare `pytest` shim has a stale shebang).
- deriva-ml is a **git dependency** in this repo; consume a new version with `uv lock --upgrade-package deriva-ml && uv sync`.
- `DERIVA_ML_ALLOW_DIRTY=true` only for dev iteration on a dirty tree; never in a final/verification run.
- macOS DataLoaders stay `num_workers=0` (unrelated to this work but a standing repo rule).
- Stable cache root: `~/.cache/deriva-ml-model-template/cifar10_source/`.
- Requires the resulting deriva-ml version pinned in this repo's `uv.lock`.

---

### Task 1: deriva-ml — `add_files` names the ingest-root dataset

**Files:**
- Modify: `../deriva-ml/src/deriva_ml/core/mixins/file.py` (signature at lines 134-141; node-creation block at lines 277-289)
- Test: `../deriva-ml/tests/` (add a focused unit test; match the repo's existing test layout/naming)

**Interfaces:**
- Produces: `add_files(files, execution_rid, dataset_types=None, description="", root_name=None, chunk_size=500) -> Dataset`. When `root_name` is None, the ingest-root dataset's `Description` is the root directory's basename (`ingest_root.name`); when given, it is `root_name`. Non-root nodes keep `description`.

- [ ] **Step 1: Write the failing test**

In `../deriva-ml`, add a unit test for the description-selection logic. The dataset creation needs a catalog, so test the pure helper instead: extract the root-description decision into a tiny module-level function and test it. Add to a new test file (e.g. `tests/test_add_files_root_name.py`):

```python
from deriva_ml.core.mixins.file import _root_description
from pathlib import Path


def test_root_description_defaults_to_basename():
    root = Path("/tmp/abc/cifar10_source")
    assert _root_description(root, root_name=None, description="generic") == "cifar10_source"


def test_root_description_uses_explicit_root_name():
    root = Path("/tmp/abc/cifar10_source")
    assert _root_description(root, root_name="CIFAR-10 source", description="generic") == "CIFAR-10 source"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ../deriva-ml && uv run python -m pytest tests/test_add_files_root_name.py -v`
Expected: FAIL with `ImportError` / `cannot import name '_root_description'`.

- [ ] **Step 3: Add the helper and use it in `add_files`**

In `../deriva-ml/src/deriva_ml/core/mixins/file.py`, add a module-level helper near `_directory_tree` (after line 102):

```python
def _root_description(ingest_root: Path, root_name: str | None, description: str) -> str:
    """Description for the ingest-root dataset.

    Defaults to the root directory's basename so the root dataset is
    self-identifying (e.g. ``cifar10_source``); an explicit ``root_name``
    overrides. Non-root nodes use ``description`` and are identified
    structurally via ``Directory_Dataset.Path``.
    """
    return root_name if root_name else ingest_root.name
```

Add `root_name: str | None = None` to the `add_files` signature (between `description` and `chunk_size`, lines 134-141), document it in the docstring, and replace the node-creation block (lines 282-290) so the root node gets its own description:

```python
        root_desc = _root_description(ingest_root, root_name, description)
        node_dataset: dict[Path, "Dataset"] = {
            directory: Dataset.create_dataset(
                self,  # type: ignore[arg-type]
                dataset_types=dataset_types,
                execution_rid=execution_rid,
                description=root_desc if directory == ingest_root else description,
            )
            for directory in nodes
        }
```

Update the stale comment above it (the "ingest root keeps the bare caller description" lines) to describe the new behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ../deriva-ml && uv run python -m pytest tests/test_add_files_root_name.py -v`
Expected: PASS (both).

- [ ] **Step 5: Lint + full deriva-ml test suite (no regressions)**

Run: `cd ../deriva-ml && uv run ruff check src tests && uv run python -m pytest -q`
Expected: lint clean; suite passes (no new failures).

- [ ] **Step 6: Commit (in ../deriva-ml)**

```bash
cd ../deriva-ml && git add src/deriva_ml/core/mixins/file.py tests/test_add_files_root_name.py
git commit -m "feat(add_files): name the ingest-root dataset (root_name, default = dir basename)"
```

- [ ] **Step 7: Release a new deriva-ml version**

Run: `cd ../deriva-ml && uv run bump-version patch`
Expected: clean tree required; tag + commit created and pushed automatically. Note the new version string for Task 2.

---

### Task 2: This repo — consume the new deriva-ml

**Files:**
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Consumes: the `add_files(..., root_name=...)` API from Task 1.
- Produces: the installed venv has the new deriva-ml; `root_name` is importable/usable.

- [ ] **Step 1: Upgrade the lock + sync**

Run: `uv lock --upgrade-package deriva-ml && uv sync`
Expected: deriva-ml updated to the Task-1 version.

- [ ] **Step 2: Verify the new param is present**

Run:
```bash
uv run python -c "import inspect; from deriva_ml import DerivaML; print('root_name' in inspect.signature(DerivaML.add_files).parameters)"
```
Expected: `True`.

- [ ] **Step 3: Confirm existing suite still passes**

Run: `uv run python -m pytest tests/ -q`
Expected: 66 passed, 2 skipped (unchanged).

- [ ] **Step 4: Commit**

```bash
git add uv.lock
git commit -m "build: bump deriva-ml for add_files root_name"
```

---

### Task 3: Schema — seed the `CIFAR_Source` Dataset_Type term

**Files:**
- Modify: `src/scripts/_cifar10_schema.py` (the function that seeds Dataset_Type terms)
- Test: `tests/test_cifar10_schema.py`

**Interfaces:**
- Produces: after the schema phase, the `Dataset_Type` vocabulary contains `CIFAR_Source`, so `add_files(..., dataset_types=["CIFAR_Source"])` validates.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cifar10_schema.py` a test that the module exposes `CIFAR_Source` among the dataset-type terms it intends to seed (mirror the existing `test_module_exposes_expected_api` style; assert the constant/term name is referenced):

```python
def test_cifar_source_dataset_type_is_declared():
    import scripts._cifar10_schema as schema
    src = open(schema.__file__).read()
    assert "CIFAR_Source" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cifar10_schema.py::test_cifar_source_dataset_type_is_declared -v`
Expected: FAIL.

- [ ] **Step 3: Seed the term**

In `src/scripts/_cifar10_schema.py`, in the dataset-type-seeding code (near the existing `DATASET_TYPES` list / `add_term` calls around line 54+), add a `CIFAR_Source` Dataset_Type term with description `"Source files a CIFAR-10 ingest registered by reference (the upload's Input provenance)."`. Use the same `add_term(table="Dataset_Type", ...)` pattern already in the file (only add if not present, matching the existing idempotent seeding).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cifar10_schema.py::test_cifar_source_dataset_type_is_declared -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scripts/_cifar10_schema.py tests/test_cifar10_schema.py
git commit -m "feat(schema): seed CIFAR_Source Dataset_Type term"
```

---

### Task 4: Source layer — stable cache extraction + labels manifest

**Files:**
- Modify: `src/scripts/_cifar10_source.py`
- Test: `tests/test_cifar10_source.py`

**Interfaces:**
- Consumes: `download_cifar10_archive() -> Path`, `extract_cifar10_to_png(archive, out_dir) -> (train_dir, test_dir, labels)` (existing).
- Produces: `CIFAR10_SOURCE_CACHE: Path` (= `~/.cache/deriva-ml-model-template/cifar10_source`); `write_labels_manifest(root: Path, labels: dict[str, str]) -> Path` writing `root/labels.csv` with header `filename,class` (one row per `<stem>.png,<class>`); returns the manifest path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cifar10_source.py`:

```python
def test_write_labels_manifest_round_trips(tmp_path):
    from scripts._cifar10_source import write_labels_manifest
    labels = {"frog_42": "frog", "cat_7": "cat"}
    path = write_labels_manifest(tmp_path, labels)
    assert path == tmp_path / "labels.csv"
    rows = path.read_text().strip().splitlines()
    assert rows[0] == "filename,class"
    assert "frog_42.png,frog" in rows
    assert "cat_7.png,cat" in rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cifar10_source.py::test_write_labels_manifest_round_trips -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

In `src/scripts/_cifar10_source.py` add:

```python
import csv

CIFAR10_SOURCE_CACHE = DEFAULT_CACHE_DIR / "cifar10_source"


def write_labels_manifest(root: Path, labels: dict[str, str]) -> Path:
    """Write a filename->class manifest to ``root/labels.csv``.

    Keys of ``labels`` are filename stems (no extension); each row is
    ``<stem>.png,<class>``. This is the durable, registered label source the
    decoupled upload execution reads (no in-memory labels dict crosses the
    execution boundary).
    """
    manifest = root / "labels.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "class"])
        for stem, cls in sorted(labels.items()):
            writer.writerow([f"{stem}.png", cls])
    return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cifar10_source.py::test_write_labels_manifest_round_trips -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scripts/_cifar10_source.py tests/test_cifar10_source.py
git commit -m "feat(source): stable cache path + labels.csv manifest writer"
```

---

### Task 5: Register scaffold — `_cifar10_register.py` (Exec 1)

**Files:**
- Create: `src/scripts/_cifar10_register.py`
- Test: `tests/test_cifar10_register.py`

**Interfaces:**
- Consumes: `download_cifar10_archive`, `extract_cifar10_to_png`, `write_labels_manifest`, `CIFAR10_SOURCE_CACHE` (Task 4); `stratified_sample_by_class` (currently in `_cifar10_assets.py` — move it to a shared spot or import it).
- Produces: `stage_source(max_images, cache_root=CIFAR10_SOURCE_CACHE) -> Path` (stages sampled `train/`+`test/` PNGs + `labels.csv` under `cache_root`, returns `cache_root`); `run_register_phase(ml, max_images, cache_root=CIFAR10_SOURCE_CACHE) -> str` (creates Exec 1 with workflow `CIFAR Source Registration`, calls `exe.add_files(FileSpec.create_filespecs(cache_root), dataset_types=["CIFAR_Source"], root_name="cifar10_source")`, returns the root File dataset RID).

- [ ] **Step 1: Write the failing test (pure staging, no catalog)**

Add `tests/test_cifar10_register.py`:

```python
def test_stage_source_lays_out_train_test_and_manifest(tmp_path, monkeypatch):
    import scripts._cifar10_register as reg
    # Fake extract: write 2 train + 2 test pngs + labels
    from PIL import Image
    def fake_extract(archive, out):
        (out / "train").mkdir(parents=True); (out / "test").mkdir(parents=True)
        for d, names in (("train", ["airplane_1", "cat_2"]), ("test", ["dog_3", "frog_4"])):
            for n in names:
                Image.new("RGB", (4, 4)).save(out / d / f"{n}.png")
        labels = {"airplane_1": "airplane", "cat_2": "cat", "dog_3": "dog", "frog_4": "frog"}
        return out / "train", out / "test", labels
    monkeypatch.setattr(reg, "extract_cifar10_to_png", fake_extract)
    monkeypatch.setattr(reg, "download_cifar10_archive", lambda: tmp_path / "fake.tar.gz")
    root = reg.stage_source(max_images=4, cache_root=tmp_path / "src")
    assert (root / "labels.csv").exists()
    assert sorted(p.name for p in (root / "train").glob("*.png"))
    assert sorted(p.name for p in (root / "test").glob("*.png"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cifar10_register.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_cifar10_register.py`**

Create `src/scripts/_cifar10_register.py` with `stage_source` (download → extract → stratified-sample → copy/symlink sampled files into `cache_root/train` + `cache_root/test`, clearing any prior cache first → `write_labels_manifest(cache_root, sampled_labels)` → return `cache_root`) and `run_register_phase` (create execution with `workflow_type="CIFAR_Source_Registration"`, then `exe.add_files(...)` as in Interfaces). Mark the CIFAR-specific seams with `# DOMAIN: replace for your data` comments. Move `stratified_sample_by_class` + `class_from_filename` here (or a shared `_cifar10_sampling.py`) and update `_cifar10_assets.py`'s imports if it still references them this task; the full retirement of `_cifar10_assets.py` is Task 7.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cifar10_register.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/scripts/_cifar10_register.py && uv run ruff format src/scripts/_cifar10_register.py
git add src/scripts/_cifar10_register.py tests/test_cifar10_register.py
git commit -m "feat(register): Exec 1 scaffold — stage to cache + add_files named root"
```

---

### Task 6: Upload scaffold — `_cifar10_upload.py` (Exec 2)

**Files:**
- Create: `src/scripts/_cifar10_upload.py`
- Test: `tests/test_cifar10_upload.py`

**Interfaces:**
- Consumes: the root File dataset RID from Task 5; `Dataset.source_directory`/`is_directory`/`list_dataset_children` (deriva-ml ≥ 1.51.14); the registered `labels.csv`.
- Produces: `read_labels_manifest(path: Path) -> dict[str, str]` (filename→class, inverse of `write_labels_manifest`); `tag_url_to_path(url: str) -> Path` (parse `tag://host,date:file:///abs/path` → `Path("/abs/path")`); `run_upload_phase(ml, source_dataset_rid) -> dict` (Exec 2: consume the File dataset as Input, walk children by `source_directory` for partition, read manifest for class, resolve each File's tag-URL, `asset_file_path(asset_name="Image", ...)` + `commit_output_assets()`, then `add_features(Image_Classification ...)`; returns stats).

- [ ] **Step 1: Write the failing tests (pure resolvers, no catalog)**

Add `tests/test_cifar10_upload.py`:

```python
def test_tag_url_to_path():
    from scripts._cifar10_upload import tag_url_to_path
    url = "tag://HostA,2026-06-25:file:///var/cache/cifar10_source/train/cat_2.png"
    assert tag_url_to_path(url).as_posix() == "/var/cache/cifar10_source/train/cat_2.png"


def test_read_labels_manifest_round_trips(tmp_path):
    from scripts._cifar10_source import write_labels_manifest
    from scripts._cifar10_upload import read_labels_manifest
    written = write_labels_manifest(tmp_path, {"cat_2": "cat", "dog_3": "dog"})
    m = read_labels_manifest(written)
    assert m["cat_2.png"] == "cat"
    assert m["dog_3.png"] == "dog"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_cifar10_upload.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_cifar10_upload.py`**

Create `src/scripts/_cifar10_upload.py` with:

```python
from pathlib import Path
from urllib.parse import urlsplit
import csv


def tag_url_to_path(url: str) -> Path:
    """Resolve a deriva-ml tag URL to its local filesystem path.

    ``tag://<host>,<date>:file:///abs/path`` -> ``Path("/abs/path")``. The
    file:// portion follows the first colon after the tag authority.
    """
    file_part = url.split(":file://", 1)[1]
    return Path(urlsplit("file://" + file_part).path)


def read_labels_manifest(path: Path) -> dict[str, str]:
    """Read ``labels.csv`` (written by ``write_labels_manifest``) into a
    filename->class dict."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["filename"]: row["class"] for row in reader}
```

Then `run_upload_phase(ml, source_dataset_rid)`: build `ExecutionConfiguration(workflow=<CIFAR Image Upload>, datasets=[DatasetSpec(rid=source_dataset_rid, ...)])`; inside the execution, get the consumed File dataset, find the root's `labels.csv` File (the child whose path ends in `labels.csv`), `read_labels_manifest`; for each directory child (`source_directory` in {"train","test"}) and each File member, `tag_url_to_path`, then `exe.asset_file_path(asset_name="Image", file_name=str(path), asset_types=["Image"], rename_file=f"{partition}_{cls}_{stem}.png")`; after the block `exe.commit_output_assets()`; then `add_features` for `Image_Classification`. Mark domain seams.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_cifar10_upload.py -v`
Expected: PASS (both).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/scripts/_cifar10_upload.py && uv run ruff format src/scripts/_cifar10_upload.py
git add src/scripts/_cifar10_upload.py tests/test_cifar10_upload.py
git commit -m "feat(upload): Exec 2 scaffold — consume File dataset, resolve, upload Image assets"
```

---

### Task 7: Orchestrator — new phases + cleanup; retire `_cifar10_assets.py`

**Files:**
- Modify: `src/scripts/load_cifar10.py`
- Delete: `src/scripts/_cifar10_assets.py` (after moving its still-used helpers in Tasks 5/6)
- Test: `tests/` (a config/CLI smoke test for the new `--phase` choices)

**Interfaces:**
- Consumes: `run_register_phase` (Task 5), `run_upload_phase` (Task 6), `run_datasets_phase` (existing).
- Produces: `--phase` choices `schema|register|upload|datasets|cleanup|all`; a `--keep-source-cache` flag (skip cleanup); `all` runs schema→register→upload→datasets→cleanup in order, threading the source dataset RID from register to upload.

- [ ] **Step 1: Write the failing test**

Add a test asserting the CLI exposes the new phases (mirror existing CLI/smoke tests):

```python
def test_phase_choices_include_register_upload_cleanup():
    import scripts.load_cifar10 as l
    src = open(l.load_cifar10.__file__ if hasattr(l, "load_cifar10") else l.__file__).read()
    for p in ("register", "upload", "cleanup"):
        assert p in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/ -k phase_choices -v`
Expected: FAIL.

- [ ] **Step 3: Implement the orchestration**

In `src/scripts/load_cifar10.py`: replace the `images` phase with `register` + `upload`; add `cleanup`; update `--phase` choices and `main()` routing so `all` runs schema → register (capture `source_rid`) → upload(source_rid) → datasets → cleanup. Add `--keep-source-cache`. `cleanup` deletes `CIFAR10_SOURCE_CACHE` (use `shutil.rmtree(..., ignore_errors=True)`). Remove `run_assets_phase` usage; delete `src/scripts/_cifar10_assets.py` once its helpers live in the register/upload/sampling modules.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/ -q`
Expected: all pass (existing + new); update any test that imported from `_cifar10_assets.py`.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src tests && uv run ruff format src tests
git add -A
git commit -m "feat(loader): two-execution phases (register/upload) + cleanup; retire _cifar10_assets"
```

---

### Task 8: Live verification — connected lineage (the tk-011 regression test)

**Files:**
- Test: `tests/test_lineage_connected.py` (live-localhost, gated by `DERIVA_ML_LIVE_LOCALHOST=1` like the existing gated tests)

**Interfaces:**
- Consumes: the full loader (`all` phase) and `ml.lookup_lineage`.

- [ ] **Step 1: Write the gated test**

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DERIVA_ML_LIVE_LOCALHOST") != "1",
    reason="DERIVA_ML_LIVE_LOCALHOST=1 required; needs the dev-localhost container.",
)


def test_image_dataset_lineage_lists_source_file_dataset():
    """After a full two-execution load, an image dataset's lineage must list
    the source File dataset as a consumed input (the tk-011 fix)."""
    # Load a small catalog via the loader (or reuse a fixture catalog), then:
    #   lin = ml.lookup_lineage(<a Small_Testing RID>)
    #   consumed = {d.rid for d in lin.lineage.consumed_datasets}
    #   assert <source root File dataset RID> in consumed
    ...
```

- [ ] **Step 2: Run the full loader against localhost**

Run (clean tree, real provenance):
```bash
uv run python src/scripts/load_cifar10.py --hostname localhost --create-catalog cifar10_2exec --num-images 2000
```
Expected: exit 0; register creates the named `cifar10_source` root File dataset; upload consumes it; 2000 images + features + 12 datasets.

- [ ] **Step 3: Manually verify connected lineage, then fill in the test**

Run a Python check: `ml.lookup_lineage(<Small_Testing RID>)` — assert the source root File dataset RID is in `consumed_datasets`. Fill the test body with the concrete assertion (parameterize via env or a known catalog), confirm it passes with `DERIVA_ML_LIVE_LOCALHOST=1`.

- [ ] **Step 4: Run the gated test**

Run: `DERIVA_ML_LIVE_LOCALHOST=1 uv run python -m pytest tests/test_lineage_connected.py -v`
Expected: PASS (lineage now connects); without the env var it SKIPS.

- [ ] **Step 5: Commit + record tacit knowledge**

```bash
git add tests/test_lineage_connected.py
git commit -m "test: connected source->image lineage (tk-011 regression, live-gated)"
```

Append a `tk-0NN` entry: the two-execution ingest now connects source→image lineage (the upload consumes the source File dataset as Input); reference the spec and tk-011.

---

### Task 9: Docs — update design docs + README phase list

**Files:**
- Modify: `docs/design/dataset/cifar10-source-archive.md`; `README.md` (the `load-cifar10` phase list / usage); `CIFAR10.md` if it documents the loader stages.

- [ ] **Step 1: Update the source-archive design doc**

Reflect the two-execution structure, the named root (`cifar10_source` + `CIFAR_Source` tag), the labels.csv manifest, and the connected lineage. Set Status appropriately.

- [ ] **Step 2: Update README / CIFAR10.md**

Document the new `--phase` values (`register|upload|datasets|cleanup`), the stable source cache + `--keep-source-cache`, and that the source dataset is now a consumed Input.

- [ ] **Step 3: Commit**

```bash
git add docs/ README.md CIFAR10.md
git commit -m "docs: two-execution ingest — phases, named source root, connected lineage"
```

---

## Self-Review

**Spec coverage:** two executions (Tasks 5,6,7) ✓; decoupled via catalog + stable cache (Tasks 4,5,6) ✓; labels.csv manifest registered + read (Tasks 4,6) ✓; consume File dataset as Input (Task 6) ✓; named root via deriva-ml change (Task 1) + CIFAR_Source tag (Task 3) ✓; lock bump (Task 2) ✓; cleanup phase/flag (Task 7) ✓; connected-lineage verification (Task 8) ✓; docs (Task 9) ✓; copy/edit scaffolds (Tasks 5,6 marked DOMAIN seams) ✓; skill in deriva-ml-skills is out of scope (forward pointer only) ✓.

**Placeholder scan:** Task 8's test body is intentionally a guided stub filled in against live data in Step 3 (it cannot be fully written without a live catalog RID); all other steps carry concrete code/commands.

**Type consistency:** `write_labels_manifest`/`read_labels_manifest` are inverses over the same `filename,class` CSV; `CIFAR10_SOURCE_CACHE` used consistently; `run_register_phase` returns the source dataset RID consumed by `run_upload_phase`; `root_name` param name matches Task 1 ↔ Task 5.
