# End-to-End DerivaML Platform Test — Design

**Author:** Carl Kesselman (with Claude)
**Date:** 2026-05-13
**Status:** Superseded by [docs/test-plans/2026-05-20-e2e-multipersona.md](../../test-plans/2026-05-20-e2e-multipersona.md) (multi-persona variant). Kept here for historical reference — the May 2026 run was executed against this spec and produced the May 19 session journal. The next run uses the multi-persona plan.

## 1. Scope and goals

This is an end-to-end integration test of the **DerivaML platform stack**,
using the model template as the harness:

- `deriva-ml` (core Python library)
- `deriva-mcp-core` + `deriva-ml-mcp` (MCP servers)
- `deriva-skills` + `deriva-ml-skills` (Claude Code skill plugins)
- `deriva-ml-model-template` (this repo — both subject and harness)

### Primary goals

1. Verify the full data → training → analysis pipeline works end to end
   against a freshly-created catalog.
2. Exercise the skill plugins and MCP tools at every step where they
   apply. Skills-first: the CLI is invoked *through* skills wherever a
   skill exists for the work.
3. Surface and inline-fix bugs in any of the five repos as they appear.
4. Surface and inline-fix skill/tool gaps — missing skills, ambiguous
   routing, broken tools, bad descriptions.

### Secondary goals

5. Validate features (existing features match expectation; new
   features round-trip correctly through write → query → bag).
6. Exercise dataset operations (new splits, subsets, training on a
   created split).
7. Verify client caching kicks in on repeated local operations
   (dataset BDBag, asset download, schema/vocabulary lookups).
8. Exercise script generation + provenance for *new* Workflows (not
   just the bundled CIFAR-10 one).
9. Capture decisions and rationale via `maintain-experiment-notes`
   plus a chronological session journal.

### Non-goals

- Performance benchmarking.
- Coverage of every model config / experiment combination — we will
  pick representative ones.
- Multi-host / cluster scenarios — `localhost` only.

## 2. Workspace, isolation, and cleanup

### 2.1 Worktree setup

```bash
git worktree add ../deriva-ml-model-template-e2e e2e-test/2026-05-13
cd ../deriva-ml-model-template-e2e
```

All test operations run from the worktree. The `main` checkout of the
model template stays untouched and is available for sibling-repo
verification work.

### 2.2 Sibling repos

`deriva-ml`, `deriva-mcp-core`, `deriva-ml-mcp`, `deriva-skills`, and
`deriva-ml-skills` are edited in place at their existing locations
under `/Users/carl/GitHub/DerivaML/`. They are *not* worktreed — fixes
commit straight to their normal branch (usually `main` per workspace
convention; confirm before pushing). The model-template worktree
consumes them via the same `uv` resolution it normally does — a fix in
a sibling becomes live with `uv sync --reinstall-package <pkg>` in the
worktree.

### 2.3 Commit hygiene inside the worktree

Two kinds of commits land on `e2e-test/2026-05-13`:

| Kind | Example | Fate |
|---|---|---|
| Test-run mutation | `test: [E2E-DROP] repoint dev/*_localhost.py at catalog 1499` | Dropped at end. Never cherry-picked. |
| Genuine template change | `fix(configs): asset_localhost.py missing roc_lr_sweep entry` | Cherry-picked to `main` at end. |

Test-mutation commits get a `[E2E-DROP]` marker in the subject so
cherry-pick selection at the end is trivial:

```
test: [E2E-DROP] repoint dev/*_localhost.py at catalog 1499
```

Genuine-fix commits use the normal Conventional Commits subject style
with no marker.

### 2.4 Refresh after sibling-repo fixes

Whenever a fix lands in a sibling repo, the worktree (and any
already-running MCP/skill surface) must pick up the new code before
re-attempting the failed step.

| Repo fixed | Refresh action |
|---|---|
| `deriva-ml` | `uv sync --reinstall-package deriva-ml` in the worktree. |
| `deriva-mcp-core` | Restart the MCP server process. `uv sync --reinstall-package deriva-mcp-core` if the worktree imports it. |
| `deriva-ml-mcp` | Same as `deriva-mcp-core` — restart server; `uv sync` reinstall if imported. |
| `deriva-skills` / `deriva-ml-skills` | Reload the plugins inside Claude Code. (Agent prompts the user; cannot self-reload.) |

Procedure after every sibling-repo fix: commit fix → refresh → re-attempt
the failing step → journal the bug → fix → re-attempt sequence → only
move on after the re-attempt succeeds.

If a fix to one sibling regresses an earlier-validated step, that is
itself a finding. Fix inline, re-verify the earlier step before
continuing forward.

### 2.5 End-of-session cleanup

1. From the workspace root, review worktree commits:
   `git -C ../deriva-ml-model-template-e2e log main..HEAD --oneline`.
2. Cherry-pick non-`[E2E-DROP]` commits onto the model template's
   `main`.
3. Optionally delete the test catalog via `deriva_ml_*` (user decides
   at session end).
4. `git worktree remove ../deriva-ml-model-template-e2e --force`.
5. `git branch -D e2e-test/2026-05-13`.

Worktree removal, branch deletion, and catalog deletion all require
explicit user confirmation. No auto-cleanup.

## 3. Pre-test work (Phase 0)

Three things land on `main` of `deriva-ml-model-template` *before* the
worktree is created.

### 0a. Write the design spec

This document, committed to
`docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md`.

### 0b. Switch CIFAR-10 source from Kaggle to the Toronto open mirror

Target file: `src/scripts/load_cifar10.py` (and `pyproject.toml` if the
`kaggle` Python package is a dependency).

**Source URL:** `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
(canonical upstream, Python pickle format).

**What changes:**

- Drop `verify_kaggle_credentials()`, `subprocess` calls to `kaggle`/`7z`,
  outer-zip extraction, CSV-based label loading.
- Add `urllib.request` (or `requests`) download with a local on-disk
  cache, `tarfile` extraction, pickle-load the six batch files (5 train
  + 1 test), write out PNGs for downstream code that expects them.
- Update `iter_images()` so test images yield labels too (the Toronto
  distribution has labels for the test set; Kaggle did not).
- Update dataset descriptions and types so `Testing` becomes `Labeled`
  rather than `Unlabeled`.
- Cache the tarball at `~/.cache/deriva-ml-model-template/cifar-10-python.tar.gz`
  (gitignored, survives across runs).
- Doc updates: README §Prerequisites, CIFAR10.md, CLAUDE.md
  "Gotchas" — drop the Kaggle/`~/.kaggle/kaggle.json` requirement.
- Smoke-test: `load-cifar10 --create-catalog e2e-precheck --num-images 50 --hostname localhost`
  end-to-end.

Commit on `main` with `fix(scripts): load CIFAR-10 from Toronto open mirror instead of Kaggle`.

### 0c. Investigate downstream impact of labeled-test-set change

Because the Toronto distribution ships labeled test data, the
`cifar10_split` vs `cifar10_labeled_split` distinction may no longer be
meaningful. Look-and-decide step during execution: either consolidate
the dataset configs, or update descriptions only, with rationale
recorded in the journal.

## 4. Test phases

Each phase follows the same shape. Skill-first; CLI invocations route
through skills; direct + indirect verification at every phase boundary.

### Per-phase template

1. **Try the skill** — invoke the skill that should fire for this work.
2. **Direct catalog check** — verify expected entities/relationships
   via `deriva-py` (`DatapathBuilder`, raw `ermrest_catalog.get`) or
   `deriva-ml` (`ml.find_datasets`, `ml.list_vocabulary_terms`,
   `ml.model.schemas[...]`). Bypass MCP and skills.
3. **Indirect check via tool/skill** — query the same information
   via `deriva_ml_*` MCP tools and/or skill invocations.
4. **Diff the two.** Disagreement is a finding. If direct and indirect
   disagree and `deriva-ml` is a plausible culprit (since it sits in
   both channels), drop one level lower (raw `ermrest_catalog.get`) to
   break the tie before assigning blame.
5. **Journal** the phase outcome, tool/skill choices, and any findings.

### Skill-issue meta-loop

If the skill in step 1 fails to fire, mis-routes, or behaves
unexpectedly, diagnose as:

- **(a) Triggering issue** — description doesn't match the request.
- **(b) Routing issue** — fires but picks the wrong tool/CLI/sub-skill.
- **(c) Behavior issue** — fires and routes correctly but the result
  is wrong.
- **(d) Missing skill** — no skill exists for the work.

Then pick one or more of: **refine the existing skill**, **add an
eval** (if the failure mode is one we'd expect to regress), or **write
a new skill** with `skill-creator`. Commit in the relevant skill repo.
Reload plugins. Re-attempt. Tag the journal entry `#skill-issue` with
sub-tag `#refined`, `#new-skill`, or `#eval-added`.

### Phases

#### Phase 1 — Catalog bootstrap and config repointing

- **Skill:** `route-project-setup`.
- **Routes to:** `load-cifar10 --create-catalog e2e-test-20260513 --hostname localhost --num-images 500 --show-urls`.
- **Direct check:** open the catalog via `DerivaML(...)`. Confirm
  domain schema exists; `Image`, `Image_Class`, `Image_Classification`,
  `Dataset_Type`, `Workflow_Type` tables exist; expected vocabulary
  terms present; 13 datasets exist with expected names
  (`complete`, `split`, `training`, `testing`, `small_split`,
  `small_training`, `small_testing`, `labeled_split`,
  `labeled_training`, `labeled_testing`, `small_labeled_split`,
  `small_labeled_training`, `small_labeled_testing`); each dataset
  has expected `current_version`; expected dataset hierarchy.
- **Indirect check:** `deriva_ml_list_datasets`, `deriva_ml_get_dataset`,
  `deriva_ml_list_vocabularies`, `deriva_ml_list_vocabulary_terms`,
  `deriva_ml_list_features`.
- **Diff** counts, names, versions, hierarchy.
- **Repoint** `src/configs/dev/{deriva,datasets,assets,roc_analysis}_localhost.py`
  with the new catalog ID and dataset RIDs. Commit on the worktree
  branch as `test: [E2E-DROP] repoint dev/*_localhost.py at catalog <id>`.

#### Phase 2 — Quick training (dry-run, then real)

- **Skill:** `route-run-workflows`.
- **Routes to:** `deriva-ml-run +experiment=cifar10_quick dry_run=true`
  (dry-run first to validate config + Hydra resolution); then re-run
  the same command with `dry_run=true` removed (real training).
- **Direct check:** verify Workflow row with git commit hash matching
  `HEAD`; Execution row with status `completed`; Execution_Asset rows
  for model weights and predictions CSV.
- **Indirect check:** same via `deriva_ml_list_executions`,
  `deriva_ml_get_execution`, `deriva_ml_list_assets`.
- **Diff** and journal.

#### Phase 3 — Feature validation (existing features match expectation)

- **Skill:** `create-feature` (used in *query* mode, not *create*).
- Inspect the `Image_Classification` feature schema. Query feature
  values for a sample of ≥10 images.
- **Direct check:** raw query of the feature association table.
  Verify `Image_Class` term matches the filename-encoded class (e.g.,
  `train_frog_42.png` → `Image_Class=frog`).
- **Indirect check:** `deriva_ml_feature_values` (or whichever tool
  the skill picks).
- **Diff.** This phase also validates that `create-feature` routes
  correctly when the request is "show me existing features" rather
  than "create a feature." Routing failure here is a `#skill-issue`.

#### Phase 4 — Multirun (parent/child execution lineage)

- **Skill:** `route-run-workflows`.
- **Routes to:** `deriva-ml-run +multirun=quick_vs_extended` (or
  `lr_sweep`).
- **Direct check:** parent Execution row + N child Execution rows
  with the parent FK populated; each child has its own assets; the
  bag for the parent includes children via FK traversal. (The recent
  `09caed4 test: add bag FK traversal regression + multirun validator`
  guards this — re-verify here.)
- **Indirect check:** `deriva_ml_list_executions` with parent filter;
  walk lineage via tool.
- **Diff** and journal. Confirm `multirun_descriptions.py` content lands
  on the parent execution row.

#### Phase 5 — Client cache validation

Cross-cutting check, interleaved after phases 2 and 4.

- Re-run a step that downloads the dataset bag (e.g., `dry_run=true`
  training). Confirm via timing or log messages that the second
  download is cache-served.
- Use `deriva_ml_*` tools to fetch the same vocabulary/schema info
  twice. Confirm cache behavior (or note its absence as a finding).
- **Verification:** "cached" log messages present, second invocation
  noticeably faster than first.
- **Journal:** which cache layer kicked in (deriva-py client cache,
  BDBag cache, DerivaML's own cache), where the cache lives on disk,
  any cases where cache *should* serve but doesn't (tag `#cache-miss`).

#### Phase 6 — Feature creation (new feature, round-trip)

- **Skill:** `create-feature` (in *create* mode).
- Create a new feature on a target table — e.g.,
  `Prediction_Confidence_Bucket` on `Image` (low/med/high), or
  `Quality_Score`. Let the skill choose the idiom (vocab vs scalar vs
  composite); capture the rationale via `maintain-experiment-notes`.
- Populate it for a subset of images via an Execution (mutations
  happen inside Executions per the steering principle).
- **Direct check:** feature schema in `ml.model.schemas[...]`; feature
  values queryable by raw table read.
- **Indirect check:** values via `deriva_ml_feature_values`; appears
  in dataset bag after re-download.
- **Diff** and journal.

#### Phase 7 — Dataset operations (new split + train on it)

- **Skill:** `dataset-lifecycle`.
- Create a new split — 70/30 stratified train/test from labeled
  training (fall back from 60/20/20 if 3-way isn't supported; that
  fallback is itself a finding to consider extending `split_dataset()`).
- Create a custom subset (e.g., only `cat`+`dog`+`frog` classes) to
  test the subset path.
- Register a new experiment config in `src/configs/dev/experiments.py`
  pointing at the new split.
- Run a small training experiment against the new split via
  `route-run-workflows`.
- **Direct check:** new datasets in catalog with correct
  `current_version`; new Workflow row with script ref, URL, commit
  hash, workflow-type vocab term; training run records new dataset
  RID as input.
- **Indirect check:** same via `deriva_ml_*`.
- **Diff** and journal. (Phase 8 is folded in here.)

#### Phase 8 — Script generation + new-workflow provenance

Folded into Phase 7. Specifically validates: does the new Workflow row
correctly link script + commit hash + URL + type? Does the platform's
script-generation help produce a runnable artifact?

#### Phase 9 — ROC notebook

- **Skill:** `route-run-workflows` (notebooks live there) or
  `compare-model-runs` (since ROC is run comparison). Try
  `route-run-workflows` first; if it doesn't route to notebooks,
  that's `#skill-issue`.
- Update `dev/assets_localhost.py` + `dev/roc_analysis_localhost.py`
  with the asset RIDs from phases 2 and 4 (queried via MCP tools).
- Run: `deriva-ml-run-notebook notebooks/roc_analysis.ipynb deriva_ml=localhost_<id> assets=<roc_asset_config>`.
- **Direct check:** ROC plot asset row appears in catalog; executed
  notebook archived as metadata asset; AUC > 0.5 (sane).
- **Indirect check:** same via `deriva_ml_list_assets`.
- **Diff** and journal. (Recent commit `2760ee8` clarified notebook
  host/catalog override semantics — re-verify here.)

#### Phase 10 — Model comparison

- **Skill:** `compare-model-runs`.
- Rank the multirun children from Phase 4 by accuracy.
- **Direct check:** read prediction-CSV asset rows directly; compute
  ranking manually.
- **Indirect check:** ranking from the skill.
- **Diff.** Journal how the skill discovers prediction assets and
  whether it has to be told the metric or infers it.

#### Phase 11 — Maintain experiment notes (cross-cutting)

- **Skill:** `maintain-experiment-notes`.
- Invoked at decision points throughout the test, not as a single
  phase. Examples: sample selection in Phase 3, feature-shape choice
  in Phase 6, split strategy in Phase 7, prediction-CSV selection in
  Phase 9, any spec-deviation rationale.
- **Pre-condition:** look up where the skill writes its output before
  the first invocation. If output is RIDs in the catalog, that's a
  finding (notes die with the test catalog at cleanup); fix the
  skill or work around with an explicit export step.
- **Diff** check is conceptually different here: rationale captured
  via the skill should be readable back; if not, that's a finding.

### Phase ordering

- Phases 1 → 2 → 3 → 4 are linear (each needs the previous).
- Phase 5 (cache) interleaves after Phase 2 and again after Phase 4.
- Phase 6 (new feature) runs after Phase 3 (existing-feature validation).
- Phase 7 (new split + new workflow) runs after Phase 4 (so we have
  multirun lineage to inform Phase 10).
- Phases 9, 10 run last (aggregation).
- Phase 11 is cross-cutting.

## 5. Logging conventions and decision capture

Two distinct artifacts, two distinct purposes.

### 5.1 Session journal — chronological, lightweight, scannable

**Path:** `/Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md`

- Lives at the **workspace root** (one level above the model template).
  Survives worktree teardown. Not in any single repo's git history.
- One file per test session. Cross-day resumption uses
  `## YYYY-MM-DD continuation` headings.

**Format:** flat chronological log. Each entry is one paragraph,
~3–5 lines:

```markdown
### 2026-05-13 14:22 — Phase 1: catalog bootstrap

**Skill:** route-project-setup → ran `load-cifar10 --create-catalog
e2e-test-20260513 --num-images 500 --hostname localhost`.
**MCP tools:** deriva_ml_list_datasets (3 calls — surveying created
datasets), deriva_ml_get_dataset (verifying RIDs).
**Direct/indirect diff:** ✓ agree (13 datasets, all hierarchies match).
**Outcome:** Catalog 1499 created. Repointed dev/*_localhost.py.
**Decisions:** Used `--num-images 500` per spec. Picked
dev/datasets_localhost.py over dev/datasets.py per CLAUDE.md.
```

**Tags** — entries that involved a finding get a tag at the end:

- `#bug-fixed` — found a bug, fixed inline, re-attempted, moved on.
- `#skill-issue` — skill mis-routed / failed to fire / wrong behavior.
  Sub-tags: `#refined`, `#new-skill`, `#eval-added`.
- `#tool-issue` — MCP tool missing / broken / bad description.
- `#doc-gap` — README, CLAUDE.md, or skill doc wrong/incomplete.
- `#surprise` — worked unexpectedly; worth recording rationale.
- `#cache-miss` — cache should serve but doesn't (Phase 5).
- `#diff` — direct and indirect channels disagree.

At session end, `grep` by tag gives a finding-class summary.

### 5.2 `maintain-experiment-notes` invocations — knowledge transfer

Invoked at decision points throughout the test, per the skill's
purpose ("capture tacit knowledge — the reasoning behind decisions
that would otherwise be lost"). The skill writes the *why*; the
journal captures the *what*.

Triggering moments (non-exhaustive):

- Phase 3: which images for feature spot-check, and why.
- Phase 6: scalar vs vocab-typed for new feature, and why.
- Phase 7: split ratios and stratification, and why.
- Phase 9: which prediction CSVs feed the ROC notebook, and why.
- Any spec deviation (e.g., 3-way → 2-way split fallback).

**Output location:** TBD, established during Phase 0/early Phase 1
by reading the skill file. If the skill writes to RIDs in the catalog,
treat as `#skill-issue` and fix/work around.

### 5.3 MCP tool call logging level

Default: log at **skill level** ("invoked `dataset-lifecycle` to
create a 70/30 split, which used `deriva_ml_create_dataset` +
`deriva_ml_add_dataset_member`").

Drill into args/results **only when something surprised:**

- Unexpected error or empty result.
- Behavior didn't match the tool description.
- Tool used in a non-obvious way to get the job done.

### 5.4 Artifacts and their lifetimes

| Artifact | Path | Lifetime | Why |
|---|---|---|---|
| Design spec | `deriva-ml-model-template/docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md` | Permanent (on `main`) | Reference material for the test |
| Session journal | `/Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md` | Permanent (workspace, not in any repo) | Survives worktree teardown |
| `maintain-experiment-notes` output | TBD (per 5.2) | TBD | Domain-specific rationale |
| Test mutation commits | `e2e-test/2026-05-13` branch in worktree | Dropped at end | Dev config changes |
| Genuine fix commits | `e2e-test/2026-05-13` branch in worktree | Cherry-picked to `main` | Real template/sibling bugs |

### 5.5 Sibling-repo journal

Skill/tool fixes land in `deriva-ml-mcp`, `deriva-mcp-core`,
`deriva-skills`, `deriva-ml-skills`, or `deriva-ml`. Journal entries
reference the commit SHA. We do not keep a parallel log inside those
repos — their own commit history is enough.

## 6. Direct vs indirect verification — channels

| Channel | What it is | When to use |
|---|---|---|
| **Direct** | `deriva-py` (`DatapathBuilder`, `ermrest_catalog.get`) or `deriva-ml` (`ml.find_datasets`, `ml.list_vocabulary_terms`, `ml.model.schemas`) | Always at phase boundaries. Bypasses MCP/skills. |
| **Indirect** | `deriva_ml_*` MCP tools + skill invocations | Always at phase boundaries. Mirrors the user-facing path. |
| **Tie-breaker** | Raw `ermrest_catalog.get(...)` or `DatapathBuilder` with no `deriva-ml` helpers | When direct and indirect disagree and `deriva-ml` is a plausible culprit. |
| **Chaise** | Web UI visual check | Deferred. Not in this test plan. |

**Direct-check style per phase type:**

- **Schema/structure** (1, 6, 7) — `ml.model.schemas[...]` introspection
  or raw `/schema` REST.
- **Data** (2, 4, 8) — `ml.catalog.getPathBuilder().domain.Image.entities()`
  or raw ERMrest predicate queries to count rows.
- **Provenance** (4, 8) — follow FK chains in the catalog model
  directly, not via any helper.
- **Features** (3, 6) — query the feature association table directly.

## 7. Acceptance — when is the test "done"

The test session is complete when all of the following are true:

1. All 10 sequential phases (Phases 1–10) have been executed with
   both direct and indirect verification passing (no unresolved
   `#diff` findings).
2. Phase 11 (`maintain-experiment-notes`) — the cross-cutting
   rationale-capture phase, not part of the sequential numbering —
   has been invoked at all decision points identified during
   execution.
3. Every finding (bug, skill-issue, tool-issue, doc-gap) has been
   either fixed inline or explicitly noted in the journal as
   "not fixing in this session — followup needed" with rationale.
4. The session journal is complete and reviewable.
5. Genuine template fixes have been cherry-picked from the worktree
   branch back onto the model template's `main`.
6. Sibling-repo fixes have been committed (and, where appropriate per
   existing workflow, pushed to `main`).
7. The worktree and its branch are torn down — with explicit user
   confirmation.
8. The test catalog (`e2e-test-20260513`) is either deleted or
   explicitly preserved for follow-up — user decides at session end.

**Done does NOT mean:**

- Every finding is fixed. Some may be deferred. The bar is recorded.
- Zero diffs. Diffs are findings, not failures of the test itself.
- Every model config / experiment / dataset combination is exercised.
  We picked representative ones.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Catalog ends up in an unrecoverable state mid-test (schema migration fails, etc.) | Cheap recovery: `load-cifar10 --create-catalog e2e-test-20260513-retry`. Repoint configs. Re-attempt. ~5 min cost. |
| A skill fix in `deriva-skills` regresses something earlier | Per Section 4 meta-loop, re-verify earlier phase before continuing forward. Journal records the regression. |
| `maintain-experiment-notes` writes to catalog RIDs (lost on cleanup) | Treat as a finding-with-fix; modify skill or work around with explicit export before cleanup. |
| Toronto-host CIFAR download is slow or flaky | Cache the tarball on first download at `~/.cache/deriva-ml-model-template/`. Cache survives across runs. |
| MCP server won't restart cleanly after a fix | Document the restart procedure in the journal; if recurring, fix in MCP server lifecycle code. |
| Worktree out of sync with a sibling-repo fix | Always `uv sync --reinstall-package <pkg>` after the fix; document each occurrence in the journal. |
| Test runs longer than one session | Journal supports `## YYYY-MM-DD continuation` headings. Session journal survives worktree teardown. Catalog persists. Cross-day resumption is supported. |

## 9. Open questions to resolve during execution

These items are not blockers for starting the test but need answers
during execution:

1. **3-way splits in `dataset-lifecycle`** — does `split_dataset()`
   support train/val/test, or only train/test? If 2-way only, decide:
   extend the API, or fall back to 2-way and add a finding.
2. **Where `maintain-experiment-notes` writes** — RID-attached in
   catalog (problem for cleanup) or repo files (fine)? Check before
   first invocation.
3. **Phase 0c outcome** — do the labeled-test-set changes consolidate
   the `cifar10_split` vs `cifar10_labeled_split` distinction, or
   just update descriptions?
4. **Cache layer attribution** — which cache layer (deriva-py client,
   BDBag, DerivaML's own) actually serves repeat operations? Confirm
   during Phase 5.
