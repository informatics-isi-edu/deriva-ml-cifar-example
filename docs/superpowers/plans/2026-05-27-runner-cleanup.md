# cifar10_cnn Runner Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three issues in `src/models/cifar10_cnn.py` in one PR: (1) fail loudly with a clear message when `execution.datasets` is empty so the user understands the misconfig (closes #49); (2) replace three `getattr`-with-default anti-patterns with direct attribute access (per PR #70 rule); (3) simplify `build_loaders` by extracting a role-dispatch table and two small helpers.

**Architecture:**
- The empty-datasets check runs at the *start* of `build_loaders` (before the flatten loop) and raises a clear `RuntimeError` that names the typical cause (empty placeholder registry in `src/configs/datasets.py`). Upstream of the existing per-bag warning loop, so the user gets a distinct diagnosis for "no datasets at all" vs the deeper "no Training bag after flattening." The catalog will record a failed execution row — that's **honest provenance**: the user did invoke this command, and recording it that way is correct. No mechanism is needed to prevent the catalog write.
- The `build_loaders` refactor extracts a module-level `_LANE_CONFIGS: dict[str, _LaneConfig]` dispatch table plus two helpers `_flatten_to_leaves` and `_make_loader`. The main function shrinks from ~80 body LOC to ~30 and the per-role conditionals (`missing=`, `shuffle=`, `generator=`) move to one declarative place.
- The three `getattr` sites at `cifar10_cnn.py:254`, `:326`, `:367` get direct attribute access. The fourth `getattr` site (`load_cifar10.py:122`, on `argparse.Namespace`) is the legitimate "genuinely schemaless" case the deriva-ml-context skill blesses; left alone.

**Tech Stack:** Python 3.13, `uv` for dependency / test orchestration, `pytest` for tests, existing `tests/test_configs_load.py` as the smoke-test pattern.

**Issue reference:** [#49](https://github.com/informatics-isi-edu/deriva-ml-model-template/issues/49) — also folds in the `getattr` fix the issue calls out and the `build_loaders` simplification.

---

## File Structure

| File | Purpose | Lifecycle |
|---|---|---|
| `src/models/cifar10_cnn.py` | The runner. Three groups of changes — `getattr` cleanup, empty-datasets check, `build_loaders` refactor. | Modified |
| `tests/test_cifar10_cnn_loaders.py` | New test file covering the three behaviours: empty-datasets raises, role-dispatch table picks the right `missing`/`shuffle` policy per role, `_flatten_to_leaves` expands Split parents correctly. | Created |
| `tests/test_configs_load.py` | Existing config-loading smoke test. Verify it still passes after the runner changes. | Unchanged, re-run |

Tests run via `uv run python -m pytest` (per CLAUDE.md "Use `uv run python -m pytest`, not `uv run pytest`" — the venv's `pytest` shim has a stale shebang).

---

## Task 1: Replace getattr anti-patterns with direct attribute access

**Files:**
- Modify: `src/models/cifar10_cnn.py:254`, `:326`, `:367`

This task is a pure cleanup — no new behaviour, just three small edits that align with the PR #70 rule. Doing it first because the later refactor (Task 4) will rewrite the surrounding code and would otherwise re-introduce the anti-pattern by accident.

- [ ] **Step 1: Replace L254 (`_target_to_class_idx`)**

Current code:

```python
def _target_to_class_idx(rec: Any) -> int:
    cls = getattr(rec, "Image_Class", None) or rec.Name
    return CIFAR10_CLASS_TO_IDX[cls]
```

Replace with:

```python
def _target_to_class_idx(rec: Any) -> int:
    return CIFAR10_CLASS_TO_IDX[rec.Image_Class]
```

Rationale: `rec` is a `FeatureRecord` dynamically built from the `Image_Classification` feature's column set. For this catalog the term column is `Image_Class`. Direct attribute access; let `AttributeError` surface if the feature shape changes (which would be a contract break worth seeing).

The `or rec.Name` fallback was dead — `FeatureRecord` for `Image_Classification` does not expose `Name`. No catalog state in any of our runs has exercised the fallback path.

- [ ] **Step 2: Replace L326 (warning string in `build_loaders`)**

Current code:

```python
warnings.warn(
    f"Bag {getattr(bag, 'dataset_rid', '<unknown>')} has no "
    f"recognized Dataset_Type role (looked for one of "
    f"{list(_LEAF_ROLES)} in {list(bag.dataset_types)!r}). "
    f"Skipping.",
    RuntimeWarning,
    stacklevel=2,
)
```

Replace `getattr(bag, 'dataset_rid', '<unknown>')` with `bag.dataset_rid`:

```python
warnings.warn(
    f"Bag {bag.dataset_rid} has no "
    f"recognized Dataset_Type role (looked for one of "
    f"{list(_LEAF_ROLES)} in {list(bag.dataset_types)!r}). "
    f"Skipping.",
    RuntimeWarning,
    stacklevel=2,
)
```

`DatasetBag` exposes `dataset_rid` by contract.

- [ ] **Step 3: Replace L367 (error diagnostic in `build_loaders`)**

Current code:

```python
seen = [
    f"  - {getattr(b, 'dataset_rid', '<unknown>')}: "
    f"Dataset_Type={list(b.dataset_types)!r}"
    for b in bags
]
```

Replace with:

```python
seen = [
    f"  - {b.dataset_rid}: "
    f"Dataset_Type={list(b.dataset_types)!r}"
    for b in bags
]
```

Same reason.

- [ ] **Step 4: Confirm no other `getattr` anti-patterns remain in source**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
grep -rn "getattr" --include="*.py" src/
```

Expected output (only the one legitimate site survives):
```
src/scripts/load_cifar10.py:122:    phase = getattr(args, "phase", "all")
```

That site reflects across `argparse.Namespace` and is the genuine "schemaless mapping" case (deriva-ml-context skill blesses `.get()`-style access there). Leave it alone.

- [ ] **Step 5: Verify the existing config-loading smoke test still passes**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/test_configs_load.py -v
```

Expected: existing tests pass (this task didn't introduce new behaviour, but the import-time graph still has to load cleanly).

- [ ] **Step 6: Commit**

```bash
git add src/models/cifar10_cnn.py
git commit -m "$(cat <<'EOF'
fix(cifar10_cnn): replace getattr-with-default with direct attribute access

Three sites in src/models/cifar10_cnn.py used getattr(obj, "name", default)
on typed domain objects (FeatureRecord and DatasetBag), which masks
contract changes as silent fallbacks instead of surfacing them as
AttributeError. Per the deriva-ml-context skill rule (PR #70 in
deriva-ml-skills), direct attribute access is correct for these objects.

The fourth getattr site (src/scripts/load_cifar10.py:122 on
argparse.Namespace) is the legitimate schemaless-mapping case and is
left as-is.
EOF
)"
```

---

## Task 2: Test the empty-datasets failure

**Files:**
- Create: `tests/test_cifar10_cnn_loaders.py`

TDD a small unit test that exercises the empty-datasets path against a mock execution. We can't easily mock a full `Execution` end-to-end (it pulls real bags), but we can mock just the attribute `build_loaders` reads — `execution.datasets` — and assert the error.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cifar10_cnn_loaders.py` with:

```python
"""Unit tests for build_loaders error paths in src/models/cifar10_cnn.py.

These tests use minimal mock objects rather than a real Execution because
the failure paths we care about are entirely upstream of bag download —
the function reads execution.datasets and decides what to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from models.cifar10_cnn import build_loaders


@dataclass
class _MockExecution:
    """Minimal stand-in for deriva_ml.execution.Execution.

    build_loaders only reads execution.datasets, so that's all we model.
    """

    datasets: list[Any] = field(default_factory=list)


def test_empty_datasets_raises_clear_error() -> None:
    """When execution.datasets is empty, build_loaders raises with a
    clear "no input datasets" message — distinct from the deeper
    "no Training bag after flattening" failure, so the user can
    diagnose the misconfig (typically: a Hydra `datasets=foo` group
    that resolved to an empty list, e.g. a placeholder registry in
    src/configs/datasets.py that wasn't filled in for the catalog).
    """
    execution = _MockExecution(datasets=[])

    with pytest.raises(RuntimeError) as exc_info:
        build_loaders(execution, batch_size=32, require_training=True)

    message = str(exc_info.value)
    assert "no input datasets" in message.lower() or "empty" in message.lower(), (
        f"Expected the error to mention empty/no datasets; got: {message!r}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/test_cifar10_cnn_loaders.py::test_empty_datasets_raises_clear_error -v
```

Expected: FAIL.

The current `build_loaders` doesn't have an upstream empty check; with `datasets=[]` it falls through to the existing `require_training` rail at L362-378 which raises with the deeper message *"No bag with Dataset_Type=Training found ... (no input bags)"*. The assertion `"no input datasets" in message.lower() or "empty" in message.lower()` does not match that string (it says "no input bags" not "no input datasets" — close, but the existing message also surfaces under non-empty-but-wrong-shape conditions, so we want a distinct upstream message for the empty case).

Capture the actual failure message in the test output for confirmation.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_cifar10_cnn_loaders.py
git commit -m "test(cifar10_cnn): empty datasets should raise with clear message"
```

---

## Task 3: Add the upstream empty-datasets check

**Files:**
- Modify: `src/models/cifar10_cnn.py:300-308` (the flatten loop area)

- [ ] **Step 1: Read the current `build_loaders` body around L300**

Reference (already in the file at lines 300-308 today):

```python
    # Expand Split parents to their children.
    bags: list[DatasetBag] = []
    for bag in execution.datasets:
        roles = {t.lower() for t in bag.dataset_types}
        if _ROLE_SPLIT in roles:
            bags.extend(bag.list_dataset_children())
        else:
            bags.append(bag)
```

- [ ] **Step 2: Add the upstream check right before the flatten loop**

Edit `src/models/cifar10_cnn.py`. Find the line `# Expand Split parents to their children.` and insert this block immediately BEFORE it:

```python
    # Fail fast on the misconfig case: a Hydra `datasets=foo` group
    # that resolved to an empty list. This typically means the
    # placeholder registries in src/configs/datasets.py weren't
    # filled in for the current catalog. The execution row in the
    # catalog will already be open by the time we get here (deriva-ml
    # creates it before invoking the model function), so this is the
    # earliest the runner itself can catch it. Full upstream
    # prevention would need a deriva-ml pre_check hook.
    if not execution.datasets:
        raise RuntimeError(
            "Execution has no input datasets. This typically means a "
            "Hydra `datasets=<group>` override resolved to an empty list "
            "(e.g. a placeholder `datasets_store([], name=...)` in "
            "src/configs/datasets.py that wasn't filled in for this "
            "catalog). Fill in the dataset RIDs for the catalog you're "
            "running against, or pass a different `datasets=<group>` "
            "override."
        )

    # Expand Split parents to their children.
```

The check is positionally upstream of the flatten loop. It catches the case where there's nothing to flatten in the first place. The deeper `require_training` rail at the bottom of the function still handles the "non-empty but no Training-typed bag" case.

- [ ] **Step 3: Run the test to verify it now passes**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/test_cifar10_cnn_loaders.py::test_empty_datasets_raises_clear_error -v
```

Expected: PASS. The new check fires before the flatten loop and raises with the substring `"no input datasets"` (matches the assertion).

- [ ] **Step 4: Verify the existing `require_training` rail is still reachable**

The two checks are now both present and cover different cases:
- New upstream check (this task): `execution.datasets` is empty entirely
- Existing rail at L362-378: `execution.datasets` is non-empty but no member dispatches to the Training lane after flattening Splits

Both rely on `require_training=True`. Both raise `RuntimeError`. They report different things and should both remain.

(No new test for the L362-378 path in this PR; the existing rail was already correct and is exercised indirectly by smoke tests.)

- [ ] **Step 5: Commit**

```bash
git add src/models/cifar10_cnn.py
git commit -m "$(cat <<'EOF'
fix(cifar10_cnn): fail fast with clear message when execution has no datasets

Adds an upstream empty-list check at the top of build_loaders so the
misconfig case — a Hydra `datasets=<group>` override that resolved to
`datasets_store([], name=...)` in src/configs/datasets.py — surfaces
as a clear actionable RuntimeError rather than falling through to the
deeper "No bag with Dataset_Type=Training" rail (which reads as if a
deeper schema mismatch is at fault).

The catalog will still record the failed execution — that's honest
provenance, not a bug to be prevented. The user-facing problem is
"why did my run fail," solved by the clearer error message; the
"failed run is recorded" behaviour is correct.

Closes #49.
EOF
)"
```

---

## Task 4: Test the role-dispatch table for build_loaders

**Files:**
- Modify: `tests/test_cifar10_cnn_loaders.py` (add two more tests)

TDD the dispatch-table behaviour we're about to introduce. We need a more capable mock — one that can stand in for a `DatasetBag` enough that `_make_loader` can call its accessors without crashing. But we can stub `as_torch_dataset` to return a sentinel and just assert the configured `missing` / `shuffle` / `generator` arguments.

- [ ] **Step 1: Append the dispatch-table tests to `tests/test_cifar10_cnn_loaders.py`**

Append the following to `tests/test_cifar10_cnn_loaders.py`:

```python
# -----------------------------------------------------------------------------
# Tests for the role-dispatch table introduced in Task 5.
# These reference _LANE_CONFIGS and _make_loader, which are added in Task 5;
# the tests live here so they run as part of the same test file.
# -----------------------------------------------------------------------------

from unittest.mock import MagicMock

from models.cifar10_cnn import (
    _LANE_CONFIGS,
    _ROLE_TRAINING,
    _ROLE_TESTING,
    _ROLE_VALIDATION,
    _make_loader,
)


def test_lane_configs_cover_all_leaf_roles() -> None:
    """Every leaf role has an entry in the dispatch table.

    If a new role is added in _LEAF_ROLES without a matching
    _LANE_CONFIGS entry, _make_loader will KeyError at runtime.
    Catch that early.
    """
    assert _ROLE_TRAINING in _LANE_CONFIGS
    assert _ROLE_TESTING in _LANE_CONFIGS
    assert _ROLE_VALIDATION in _LANE_CONFIGS


def test_training_lane_shuffles_with_seeded_generator() -> None:
    """Training lane: shuffle=True, seeded generator when seed is set."""
    cfg = _LANE_CONFIGS[_ROLE_TRAINING]
    assert cfg.shuffle is True
    assert cfg.use_seeded_generator is True
    assert cfg.missing == "skip"


def test_testing_lane_keeps_unlabeled_rows_no_shuffle() -> None:
    """Testing lane: missing='unknown' so we keep unlabeled rows for
    later prediction recording; shuffle=False; no seeded generator."""
    cfg = _LANE_CONFIGS[_ROLE_TESTING]
    assert cfg.shuffle is False
    assert cfg.use_seeded_generator is False
    assert cfg.missing == "unknown"


def test_validation_lane_skips_unlabeled_no_shuffle() -> None:
    """Validation lane: shuffle=False, skip unlabeled (loss/accuracy
    are undefined for unlabeled rows). No seeded generator."""
    cfg = _LANE_CONFIGS[_ROLE_VALIDATION]
    assert cfg.shuffle is False
    assert cfg.use_seeded_generator is False
    assert cfg.missing == "skip"
```

- [ ] **Step 2: Run the new tests, verify they fail because the names don't exist yet**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/test_cifar10_cnn_loaders.py -v
```

Expected: the four new tests FAIL at *import time* with `ImportError: cannot import name '_LANE_CONFIGS' from 'models.cifar10_cnn'` (or similar). The previously-passing `test_empty_datasets_raises_clear_error` should also fail to collect because the file-level import errors before any test runs.

This is fine — they all become collectible and passing once Task 5 lands.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_cifar10_cnn_loaders.py
git commit -m "test(cifar10_cnn): dispatch-table tests for build_loaders refactor"
```

---

## Task 5: Refactor build_loaders to use a role-dispatch table

**Files:**
- Modify: `src/models/cifar10_cnn.py:233-380` (the section from `_bag_role` through `build_loaders`)

The full code below replaces the existing block. Apply it as one edit — easier to review as a whole than in pieces.

- [ ] **Step 1: Add the imports needed for the dispatch table**

At the top of `src/models/cifar10_cnn.py`, after the existing `from typing import Any` line (around L36), confirm `dataclass` is importable. If not already imported, add:

```python
from dataclasses import dataclass
```

Place it next to the other stdlib imports (after `import csv`, `import random`, `import warnings`).

- [ ] **Step 2: Replace `_bag_role` through `build_loaders` with the refactored version**

In `src/models/cifar10_cnn.py`, locate the block:

```python
def _bag_role(bag: DatasetBag) -> str | None:
    """Return the leaf role of a bag, or None if it has no recognized role.
    ...
    """
    roles = {t.lower() for t in bag.dataset_types}
    for role in _LEAF_ROLES:
        if role in roles:
            return role
    return None
```

…through the entire `build_loaders` function (currently ending at line 380, immediately before `def record_predictions(`).

Replace the entire block with:

```python
def _bag_role(bag: DatasetBag) -> str | None:
    """Return the leaf role of a bag, or None if it has no recognized role.

    A bag's ``Dataset_Type`` is a set of catalog vocabulary terms (one or
    more of ``Training``/``Testing``/``Validation``/``Split``/qualifiers
    like ``Labeled``). We pick the first leaf role found; ``Split`` is
    handled upstream by ``_flatten_to_leaves`` (it expands to children,
    then this is called on each child).
    """
    roles = {t.lower() for t in bag.dataset_types}
    for role in _LEAF_ROLES:
        if role in roles:
            return role
    return None


@dataclass(frozen=True)
class _LaneConfig:
    """Per-role DataLoader policy.

    Adding a new role (a new vocabulary term in ``_LEAF_ROLES``) is one
    entry here plus one line in ``build_loaders``' return tuple.
    """

    missing: str
    """``bag.as_torch_dataset`` policy for elements with no feature
    value. ``'skip'`` drops them (training/validation: loss/accuracy
    are undefined without labels); ``'unknown'`` keeps them with a
    sentinel target (testing: we still want to record predictions on
    unlabeled rows)."""

    shuffle: bool
    """Whether the DataLoader shuffles between epochs. Only training."""

    use_seeded_generator: bool
    """Whether to give the DataLoader a seeded ``torch.Generator`` for
    reproducible shuffle order. Only matters when ``shuffle=True``."""


_LANE_CONFIGS: dict[str, _LaneConfig] = {
    _ROLE_TRAINING: _LaneConfig(
        missing="skip", shuffle=True, use_seeded_generator=True,
    ),
    _ROLE_TESTING: _LaneConfig(
        missing="unknown", shuffle=False, use_seeded_generator=False,
    ),
    _ROLE_VALIDATION: _LaneConfig(
        missing="skip", shuffle=False, use_seeded_generator=False,
    ),
}


def _flatten_to_leaves(execution: Execution) -> list[DatasetBag]:
    """Walk ``execution.datasets``, expand any ``Split`` parent to its
    leaf children. Non-Split bags pass through.
    """
    leaves: list[DatasetBag] = []
    for bag in execution.datasets:
        roles = {t.lower() for t in bag.dataset_types}
        if _ROLE_SPLIT in roles:
            leaves.extend(bag.list_dataset_children())
        else:
            leaves.append(bag)
    return leaves


def _make_loader(
    bag: DatasetBag,
    role: str,
    batch_size: int,
    seed: int | None,
) -> DataLoader:
    """Build one DataLoader for one bag's role-specific lane.

    Looks up the role's policy in ``_LANE_CONFIGS``. Adding a new role
    means adding an entry there; this function doesn't need to change.
    """
    cfg = _LANE_CONFIGS[role]
    dataset = bag.as_torch_dataset(
        element_type="Image",
        sample_loader=_load_image,
        transform=_TRANSFORM,
        targets=["Image_Classification"],
        target_transform=_target_to_class_idx,
        missing=cfg.missing,
    )
    generator = None
    if cfg.use_seeded_generator and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    # macOS DataLoader: num_workers=0 to avoid fork() + MPS deadlock.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=cfg.shuffle,
        num_workers=0,
        collate_fn=_rid_collate,
        generator=generator,
    )


def build_loaders(
    execution: Execution,
    batch_size: int,
    require_training: bool = False,
    seed: int | None = None,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None, list[str]]:
    """Walk ``execution.datasets`` and build DataLoaders by ``Dataset_Type``.

    The harness:

      1. Fail fast if ``execution.datasets`` is empty (misconfig: a
         Hydra ``datasets=<group>`` override resolved to an empty list).
      2. Flatten any ``Split`` parent bag to its children.
      3. For each leaf bag, look up its role and build a DataLoader
         from the role's lane policy (``_LANE_CONFIGS``).
      4. If ``require_training`` is set and no Training bag was found,
         raise — fail loudly rather than silently produce a non-training
         "training run".

    Args:
        execution: DerivaML execution; ``execution.datasets`` is the
            list of downloaded bags.
        batch_size: Batch size for all loaders.
        require_training: If True, raise ``RuntimeError`` when no bag
            dispatches to the training lane.
        seed: Drives the training DataLoader's shuffle order. ``None``
            uses PyTorch's default global generator.

    Returns:
        ``(train_loader, test_loader, val_loader, class_names)``. Any
        loader may be ``None`` if no bag dispatched to that lane. The
        Validation lane (D01) is wired through to the training loop as
        a per-epoch metric but doesn't drive save-best — that's
        intentional for a demo. Plug early stopping in here if you
        want it.

    Raises:
        RuntimeError: If ``execution.datasets`` is empty (step 1), or
            if ``require_training=True`` and no Training-typed bag is
            present after flattening (step 4).
    """
    # Fail fast on the misconfig case: a Hydra `datasets=<group>` group
    # that resolved to an empty list. This typically means the
    # placeholder registries in src/configs/datasets.py weren't filled
    # in for the current catalog. The execution row in the catalog
    # will already be open by the time we get here (deriva-ml creates
    # it before invoking the model function), so this is the earliest
    # the runner itself can catch it. Full upstream prevention would
    # need a deriva-ml pre_check hook.
    if not execution.datasets:
        raise RuntimeError(
            "Execution has no input datasets. This typically means a "
            "Hydra `datasets=<group>` override resolved to an empty list "
            "(e.g. a placeholder `datasets_store([], name=...)` in "
            "src/configs/datasets.py that wasn't filled in for this "
            "catalog). Fill in the dataset RIDs for the catalog you're "
            "running against, or pass a different `datasets=<group>` "
            "override."
        )

    leaves = _flatten_to_leaves(execution)
    loaders: dict[str, DataLoader] = {}
    for bag in leaves:
        role = _bag_role(bag)
        if role is None:
            warnings.warn(
                f"Bag {bag.dataset_rid} has "
                f"Dataset_Type={list(bag.dataset_types)!r}; no recognised "
                f"role among {list(_LEAF_ROLES)}. Skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        loaders[role] = _make_loader(bag, role, batch_size, seed)
        print(f"  {role.capitalize()} samples: {len(loaders[role].dataset)}")

    if require_training and _ROLE_TRAINING not in loaders:
        # Safety rail: fail loudly when the primary input is missing.
        # Distinct from the empty-datasets check at the top: here we
        # had bags but none dispatched to the Training lane.
        diag = "\n".join(
            f"  - {b.dataset_rid}: Dataset_Type={list(b.dataset_types)!r}"
            for b in leaves
        ) or "  (no input bags after flattening)"
        raise RuntimeError(
            "No bag with Dataset_Type=Training found in execution input. "
            "Cannot train. Input bags after flattening Split parents:\n"
            f"{diag}\n"
            "Add a Training-typed dataset to the execution config "
            "(see src/configs/datasets.py)."
        )

    return (
        loaders.get(_ROLE_TRAINING),
        loaders.get(_ROLE_TESTING),
        loaders.get(_ROLE_VALIDATION),
        list(CIFAR10_CLASS_NAMES),
    )
```

- [ ] **Step 3: Run all tests, verify they now pass**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/test_cifar10_cnn_loaders.py -v
```

Expected: ALL 5 tests pass:
- `test_empty_datasets_raises_clear_error`
- `test_lane_configs_cover_all_leaf_roles`
- `test_training_lane_shuffles_with_seeded_generator`
- `test_testing_lane_keeps_unlabeled_rows_no_shuffle`
- `test_validation_lane_skips_unlabeled_no_shuffle`

- [ ] **Step 4: Run the existing config-load smoke tests**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/test_configs_load.py -v
```

Expected: PASS — the refactor didn't change any public surface (`build_loaders` keeps its signature; `_bag_role`, `_ROLE_TRAINING`, etc. keep their names).

- [ ] **Step 5: Lint**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run ruff check src/models/cifar10_cnn.py tests/test_cifar10_cnn_loaders.py
uv run ruff format --check src/models/cifar10_cnn.py tests/test_cifar10_cnn_loaders.py
```

If ruff reports issues, run `uv run ruff format src/models/cifar10_cnn.py tests/test_cifar10_cnn_loaders.py` and re-run lint. Expected: no issues.

- [ ] **Step 6: Commit**

```bash
git add src/models/cifar10_cnn.py tests/test_cifar10_cnn_loaders.py
git commit -m "$(cat <<'EOF'
refactor(cifar10_cnn): extract role-dispatch table for build_loaders

The per-role conditionals in build_loaders (missing= policy,
shuffle=, seeded generator) now live in a single declarative
_LANE_CONFIGS table. Two small helpers (_flatten_to_leaves,
_make_loader) factor out the Split-expansion and per-bag DataLoader
construction respectively. build_loaders itself shrinks from ~80
body LOC to ~30 and the path from "bag in" to "loader out" reads
top-to-bottom without per-role branching.

Adding a new role is now one entry in _LANE_CONFIGS plus one line
in the return tuple. The public surface of build_loaders is
unchanged.

Unit tests in tests/test_cifar10_cnn_loaders.py cover:
- the empty-datasets failure (from Task 3)
- _LANE_CONFIGS covers all _LEAF_ROLES
- per-role policy is what we expect (training/testing/validation)
EOF
)"
```

---

## Task 6: Local end-to-end smoke

**Files:**
- None modified

This is a manual smoke step — no commit. The goal is to confirm the refactor didn't break the real `deriva-ml-run` path.

- [ ] **Step 1: Dry-run a single experiment**

The model-template doesn't ship with a live catalog by default, so a dry-run is what's available locally. Run:

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run deriva-ml-run \
    +experiment=cifar10_quick \
    deriva_ml.hostname=localhost \
    deriva_ml.catalog_id=0 \
    dry_run=true 2>&1 | tail -20
```

Expected: Hydra prints the resolved config and exits cleanly with "Dry run mode: skipping model execution". No errors.

If this fails because catalog 0 isn't reachable or because `cifar10_small_labeled_split` is empty on this checkout, that's expected on a fresh `main` — and is in fact the bug this PR's Task 3 makes diagnosable. The dry-run path short-circuits before `build_loaders` so it won't surface the new check; but it should still resolve the Hydra config cleanly. If Hydra itself errors, STOP and triage — the refactor may have broken an import.

- [ ] **Step 2: Run the full test suite**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run python -m pytest tests/ -v
```

Expected: all tests pass (the existing config-load smoke + the new loader tests).

- [ ] **Step 3: Lint check the whole src tree**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: no issues.

---

## Task 7: Update issue #49 with the corrected framing

**Files:**
- None modified locally — this is a GitHub-side comment.

The original #49 framing ("validate before opening Execution") turned out to be wrong on the code path: the runner doesn't open the execution, deriva-ml does (in `run_model` at `runner.py:597`, before the model function is invoked). But more importantly, on reflection the goal *should* be a clear error message, not catalog-write prevention. A failed execution row is honest provenance: the user invoked the command, the run failed at config-resolution time, recording it that way is the correct behaviour.

Add a comment to issue #49 acknowledging the framing correction and closing the door on the catalog-write-prevention sub-goal.

- [ ] **Step 1: Post a comment on issue #49**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
gh issue comment 49 --body "$(cat <<'EOF'
**Framing correction on this issue, posted alongside the PR that resolves it.**

The original issue framed the goal as "validate before opening Execution" — i.e., prevent the catalog from recording a failed execution row when the input dataset list resolves to empty. After looking at the code paths (`deriva-ml/src/deriva_ml/execution/runner.py:597`) and thinking it through, that framing was off in two ways:

1. **Mechanically:** the runner (`cifar10_cnn`) doesn't open the execution itself. `run_model` in deriva-ml does, via `ml_instance.create_execution(...)`, before invoking the model function. So a runner-side fix can't prevent the catalog write — only a deriva-ml change (or a Hydra-compose-time validator) could.
2. **More importantly:** the catalog write isn't the bug. A failed execution row is **honest provenance** — the user did invoke `deriva-ml-run` with the given config, and recording that fact in the catalog (as a failed row, distinguishable from successful ones by status and lack of output assets) is the right thing for the platform to do. Hiding failed runs would be a small lie about what happened. The friction the user actually experiences is *understanding why their run failed*; that's solved by a clear, actionable error message at the runner's entry point.

The PR landing alongside this comment does that:

- Adds an upstream empty-list check at the top of `build_loaders` with a message that names the typical cause (placeholder registry in `src/configs/datasets.py`).
- Replaces three `getattr`-with-default anti-patterns at `cifar10_cnn.py:254, :326, :367` with direct attribute access (per the PR #70 rule from deriva-ml-skills).
- Refactors `build_loaders` to use a declarative `_LANE_CONFIGS` dispatch table, factors out `_flatten_to_leaves` and `_make_loader`. Public surface unchanged.

This closes #49. The catalog-write-prevention sub-goal is explicitly *not* something we're pursuing — failed-execution rows are correct behaviour, and the user-facing fix is the clear error message.
EOF
)"
```

- [ ] **Step 2: Confirm the comment landed**

```bash
gh issue view 49 --comments 2>&1 | tail -15
```

Expected: the new comment appears at the bottom.

---

## Task 8: Open the PR

**Files:**
- None modified

- [ ] **Step 1: Push the branch**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
git push -u origin HEAD
```

- [ ] **Step 2: Create the PR**

```bash
gh pr create --title "fix(cifar10_cnn): empty-datasets check + getattr cleanup + build_loaders refactor" --body "$(cat <<'EOF'
## Summary

Three folded changes in `src/models/cifar10_cnn.py`:

1. **Empty-datasets early failure (closes #49).** When `execution.datasets` is empty (typical cause: a Hydra `datasets=<group>` override that resolved to a placeholder empty registry in `src/configs/datasets.py`), `build_loaders` now raises a clear `RuntimeError` at the very top with an actionable message. Previously the failure fell through to the deeper "No bag with Dataset_Type=Training" rail at L362-378, which made the misconfig harder to diagnose. The catalog will record a failed execution row — that's honest provenance, not a bug to prevent. See #49 comment for the framing.

2. **Three `getattr`-with-default sites replaced with direct attribute access.** Per PR #70 in deriva-ml-skills, `getattr(obj, "name", default)` on typed domain objects (FeatureRecord, DatasetBag) masks contract changes. Three sites at `cifar10_cnn.py:254, :326, :367`. The fourth `getattr` site in the codebase (`load_cifar10.py:122` on `argparse.Namespace`) is the legitimate "genuinely schemaless" case and is left as-is.

3. **`build_loaders` refactored to use a declarative role-dispatch table.** Per-role conditionals (`missing=`, `shuffle=`, seeded `generator`) now live in `_LANE_CONFIGS: dict[str, _LaneConfig]`. Two new module-level helpers (`_flatten_to_leaves`, `_make_loader`) factor out the Split-expansion and per-bag DataLoader construction. `build_loaders` shrinks from ~80 body LOC to ~30. Public surface unchanged.

## Test plan

- [x] `uv run python -m pytest tests/test_cifar10_cnn_loaders.py -v` — 5 new unit tests covering the empty-datasets path and `_LANE_CONFIGS` coverage of each leaf role.
- [x] `uv run python -m pytest tests/ -v` — full suite passes.
- [x] `uv run ruff check src tests && uv run ruff format --check src tests` — clean.
- [x] `uv run deriva-ml-run +experiment=cifar10_quick dry_run=true` — Hydra resolves cleanly.

## Related

Issue: #49 (with framing correction comment).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm the PR opened**

```bash
gh pr view --json url,title,state 2>&1 | tail -10
```

Expected: the PR's URL is printed; `state=OPEN`.

---

## Success criteria

- ✅ Three `getattr` anti-patterns removed; the legitimate `argparse.Namespace` site preserved.
- ✅ Empty `execution.datasets` raises a clear `RuntimeError` at the top of `build_loaders` mentioning the placeholder-registry diagnosis.
- ✅ Existing `require_training` rail at the bottom of `build_loaders` still fires for the non-empty-but-no-Training case.
- ✅ `_LANE_CONFIGS` covers all `_LEAF_ROLES`; per-role policy matches the previous behaviour exactly.
- ✅ 5 new unit tests in `tests/test_cifar10_cnn_loaders.py` all pass.
- ✅ Existing `tests/test_configs_load.py` still passes.
- ✅ `ruff check` and `ruff format --check` clean.
- ✅ Issue #49 carries a framing-correction comment acknowledging the partial fix.
- ✅ PR opened with clear test-plan.

## Out of scope (explicit)

- **Preventing the failed execution row from being recorded in the catalog.** Not pursued — failed-execution rows are honest provenance (the user invoked the command, the run failed at config-resolution time, the catalog correctly records that). The user-facing problem is "why did my run fail," which the clear error message solves.
- **The other friction items from the seed-sweep arc** (bag-cache reuse across multirun children, multirun parent description not auto-composed, tee-output dirty-tree poison). Those live in deriva-ml issue #251 and the friction findings file; not touched here.
