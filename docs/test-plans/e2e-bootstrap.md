# E2E Bootstrap (Phase 0)

**Companion to:** [`2026-05-20-e2e-multipersona.md`](2026-05-20-e2e-multipersona.md) (the persona scenario).

---

Run once, by the orchestrator (or the user), *before* the
multipersona scenario starts. Phase 0 is **infrastructure setup**,
not part of any persona's work. A failure here is a Phase 0
finding and may block the run entirely.

## Why Phase 0 is not the Curator

The Curator persona inherits a bootstrapped catalog rather than
creating it. This is a deliberate choice:

- **The scenario measures the persona experience, not infrastructure
  setup.** `load-cifar10` is mechanical (one CLI invocation) and
  reveals no judgment-laden friction. A Curator arc that includes
  bootstrap dilutes the persona's role away from their actual
  value-add: *exploring and characterising the data the team will
  work with*.
- **`load-cifar10` is the test harness, not the test subject.** Its
  bugs were shaken out in earlier runs. Re-running it through a
  persona adds no new signal.
- **In real organisations, role overlap varies.** Some shops have
  separate data-engineering and data-curation roles. Others combine
  them in one person. The persona is an abstraction, not a
  roleplay — treat Phase 0 as "the data-engineering hat" the same
  human (or a different one) wears before the curation hat goes on.
- **Bootstrap failure modes are still surfaced.** Phase 0 part E
  (step 9 below) is a fail-fast sanity gate on the catalog the
  bootstrap produced. If `load-cifar10` breaks the catalog or
  produces obviously wrong state, that's a Phase 0 finding *before*
  any persona starts.

## What Phase 0 produces

By the time Phase 0 is done, the following is true:

- A single shared git worktree exists at
  `../deriva-ml-model-template-e2e` on branch `e2e-test/<YYYY-MM-DD>`,
  cut from `main` of this repo. All persona work happens here.
- A fresh catalog exists at `localhost` named `e2e-test-<YYYYMMDD>`.
- The catalog has the cifar10 domain schema populated by
  `load-cifar10` (Image table, vocabularies including `Image_Class`,
  the built-in datasets, ground-truth `Image_Classification` feature
  values).
- `src/configs/deriva.py` in the e2e worktree has been edited so
  `default_deriva` points at the new catalog id (a `[E2E-DROP]`
  commit). `src/configs/datasets.py` has been edited with the
  loader-produced RIDs (also `[E2E-DROP]`). The base config files
  are edited *directly* — `configs/dev/` no longer exists in this
  template; the dev-overlay pattern was retired with the 2026-05-21
  rewrite.
- The dev-localhost MCP container is rebuilt against the current
  sibling versions and Claude Code's MCP server connection is
  authenticated.

## Phase 0 steps (in order)

**Preflight first, then authentication.** P0 begins with a sync
audit (step 0) and an MCP-auth handshake (step 1). Both are
fail-fast gates: if the workspace is drifted or auth can't be
established, no further P0 work is reachable.

### 0. Sync audit (preflight)

Verify the workspace is internally consistent before doing any
setup work. Past e2e runs have surfaced two distinct kinds of
drift the orchestrator can't recover from later: stale Claude Code
plugins (skill docs were one minor version behind the API they
document) and a stale MCP container image (a `deriva-mcp-test`
image built against an older `deriva-ml`). Both look healthy on
inspection (plugin lists, `docker ps`) yet ship the wrong code.

Run these checks in order; bail at the first failure rather than
papering over it:

**a. Repo state.** For each of `deriva-ml`, `deriva-mcp-core`,
`deriva-ml-mcp`, `deriva-skills`, `deriva-ml-skills`,
`deriva-ml-model-template`:

```
git -C <repo> fetch --prune origin
git -C <repo> status -b --short      # expect: clean, == origin/main
git -C <repo> log --oneline -1 main  # note the SHA
```

No repo should have uncommitted changes or be ahead/behind its
`origin/main`.

**b. Stale local branches.** For each repo above, list local
branches whose upstream is `gone` (merged + branch deleted
upstream). They are harmless but accumulate, and
`git fetch --prune` will mark them:

```
git -C <repo> for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads \
  | awk '$2 ~ /gone/ {print $1}'
```

Delete any whose tip is also in `main` (`git branch -d`).

**c. Lockfile freshness.** In `deriva-ml-mcp`,
`deriva-ml-model-template`, and `deriva-ml-skills`:

```
uv sync --upgrade-package deriva-ml
uv sync --upgrade-package deriva   # deriva-py
```

If either produces a diff to `uv.lock`, commit it as
`chore(deps): sync ...` and push before proceeding. The run
becomes unreconstructable if the lockfile drifts mid-test.

**d. Local venv sanity.** From the model-template:

```
uv run python -c "
import deriva_ml, inspect
from deriva_ml.dataset.split import split_dataset
print(deriva_ml.__version__)
print('execution param:', 'execution' in inspect.signature(split_dataset).parameters)
"
```

Version should match the lockfile pin; the `execution` param check
is a fast sentinel that catches "split_dataset signature drift" —
a stand-in for "is the venv on the new contract".

**e. Claude Code plugin freshness.** The skill docs that the
personas will lean on must match the API they describe.

```
claude plugin list | grep deriva
```

For each `deriva*@deriva-plugins` entry, compare its version
against the latest tag on origin:

```
git -C deriva-skills    tag --list | sort -V | tail -1
git -C deriva-ml-skills tag --list | sort -V | tail -1
```

If installed < latest tag, run
`claude plugin update <name>@deriva-plugins` and restart Claude
Code before continuing.

**f. MCP container freshness.** This is the trap. The compose
file declares two services that build distinct images (`deriva-mcp`
and `deriva-mcp-test` — the latter extends the former but yields a
*separate* tag), and rebuilding one does NOT rebuild the other.
Verify the actual running test image:

```
docker exec deriva-mcp-test python -c '
import deriva_ml, importlib.metadata as md
print("deriva-ml:    ", deriva_ml.__version__)
print("deriva-ml-mcp:", md.version("deriva-ml-mcp"))
'
```

Both versions must match the SHAs from step (a). If either lags:

```
cd deriva-docker/deriva
docker compose --env-file ~/.deriva-docker/env/localhost.env \
               build --no-cache deriva-mcp-test
docker compose --env-file ~/.deriva-docker/env/localhost.env \
               up -d --force-recreate deriva-mcp-test
```

Re-run the version check before proceeding to step 1. **Do not
rely on `--no-cache deriva-mcp` to rebuild the test image** —
they are separate tags. Always name the `-test` service
explicitly.

**g. `main` is at template state.** The persona arcs start from a
worktree cut from `main`, so `main` itself must be in its
pristine, no-prior-run state. Every previous multipersona run
produced `[E2E-DROP]` commits that mutate `src/configs/deriva.py`,
`src/configs/datasets.py`, and `tacit-knowledge.md`. Wrap-up of
the previous run drops those commits when cherry-picking back to
`main`, but the bookkeeping is easy to get wrong, and a poisoned
`main` means the *next* multipersona run inherits the prior run's
catalog id, dataset RIDs, and Bootstrap note. The persona cannot
detect the drift — they just see a stale catalog ref in the config
they're "starting fresh" with.

Check each file is at its template state:

- `src/configs/deriva.py` should have `catalog_id=0` in
  `default_deriva` (the placeholder). Anything else means a prior
  `[E2E-DROP]` leaked through.
- `src/configs/datasets.py` should have empty placeholder list
  literals for every dataset group, not RID strings. The
  docstring at the top of the file calls itself out as
  "intentionally empty by default."
- `tacit-knowledge.md` should be the template header only — a
  few short lines of intro + a horizontal-rule separator +
  nothing else. No "Bootstrap" entry, no per-persona decision
  logs, no model-tuning notes.

Fast cross-check (ignores commented-out example lines in
`datasets.py`, which legitimately contain RID strings inside
`# DatasetSpecConfig(rid="..."` examples):

```
grep -E "^[^#]*catalog_id=[1-9]" src/configs/deriva.py \
  && echo "FAIL: deriva.py has a real catalog_id"
grep -E "^[^#]*rid=\"[0-9]" src/configs/datasets.py \
  && echo "FAIL: datasets.py has RIDs filled in"
[ "$(wc -l < tacit-knowledge.md)" -gt 17 ] \
  && echo "FAIL: tacit-knowledge.md is non-template"
```

If any check fails: `git log --oneline -- <path>` to find the
offending `[E2E-DROP]` commit, then `git revert <sha>` (or
`git restore --source=<known-good-sha> <path>` if reverting is
messy because of subsequent template-evolution commits) and push
to `origin/main` *before* proceeding.

If any sub-check (a-g) fails, fix it and re-run from (a). The
cost of bailing here is minutes; the cost of running a
multipersona arc against drifted siblings or a poisoned `main`
is the entire run.

### 1. Authenticate the dev-localhost MCP server (OAuth)

The `dev-localhost` MCP server uses a browser-based OAuth flow
that must be completed once per Claude Code session before its
tools become available. P0 starts here so the run fails fast: if
auth can't be completed, the rest of P0 produces nothing usable.

Prerequisites (workspace setup, not P0 steps themselves):

- The dev-localhost MCP container is built and running. If it
  isn't, run
  `cd deriva-docker/deriva && docker compose up -d deriva-mcp`
  (or the equivalent for your local rig). For an e2e run that
  needs fresh sibling versions, rebuild via
  `docker compose build --no-cache deriva-mcp` first.
- The MCP server is registered with Claude Code (it appears in
  `claude mcp list`). If it isn't, follow the deriva-docker setup
  notes to register it.

Procedure:

**a.** Confirm the server is registered and its current state:

```
claude mcp list
```

Expected line:
`dev-localhost: https://localhost/mcp (HTTP) - ! Needs authentication`

If it says `Connected` already, skip to (d). If `Failed to
connect`, the container isn't healthy — return to the
prerequisites above and resolve before continuing.

**b.** Trigger the authorization URL:

```
mcp__dev-localhost__authenticate
```

The tool prints an `https://localhost/authn/authorize?...` URL
and a fallback path (`mcp__dev-localhost__complete_authentication`)
for the case where the redirect lands on a port nothing is
listening on.

**c.** Open the URL in a browser, sign in, and complete the
consent flow. The page redirects to
`http://localhost:8080/callback?code=...&state=...`. Two
outcomes:

- **Page loads cleanly.** The MCP server received the code,
  exchanged it for a token, and the `deriva_ml_*` and other
  tools become available automatically. The session emits a
  notification listing the newly-available deferred tools.
- **Browser shows "connection error".** Nothing listened on
  port 8080. Copy the full URL from the browser's address bar
  and call `mcp__dev-localhost__complete_authentication` with
  it to finish the handshake.

**d.** Sanity-check: a follow-up `claude mcp list` should now
show `dev-localhost: ... - ✓ Connected`. Confirm a representative
tool works:

```
mcp__dev-localhost__get_catalog_info(hostname=localhost, catalog_id=1)
```

(Any catalog id is fine — even a missing one returns a meaningful
error rather than an auth failure.)

Notes:

- The orchestrator session's OAuth token is **not inherited by
  sub-agents spawned via the `Agent` tool**. A past run observed
  that sub-agents *did* inherit auth (the dev-localhost tools
  were immediately available to personas without re-auth); verify
  this holds for your run by including a check in the persona's
  startup instructions.
- If auth expires mid-run (long sessions), tool calls start
  returning auth errors. Re-run (b) and (c).
- This step is per-Claude-Code-session, not per-catalog. If you
  run a second e2e on the same day in the same session, you
  don't need to re-auth.

### 2. Create the shared e2e worktree

Pick the run date as `<YYYY-MM-DD>` (all later artifacts key off
this) and:

```
git -C deriva-ml-model-template worktree add \
    ../deriva-ml-model-template-e2e -b e2e-test/<YYYY-MM-DD>
```

Refuse to proceed if a prior catalog at the target name exists
unless the user explicitly says delete-and-reuse. If an
`e2e-test/<YYYY-MM-DD>` branch already exists, abort or use a
suffixed date — never overwrite.

Immediately re-verify the *worktree's* template-state files match
`main` (step 0(g) checked `main` itself; this checks the worktree
the personas will actually inhabit):

```
cd ../deriva-ml-model-template-e2e
grep -E "^[^#]*catalog_id=[1-9]" src/configs/deriva.py \
  && echo "FAIL: deriva.py is non-template in the worktree"
grep -E "^[^#]*rid=\"[0-9]" src/configs/datasets.py \
  && echo "FAIL: datasets.py has RIDs in the worktree"
[ "$(wc -l < tacit-knowledge.md)" -gt 17 ] \
  && echo "FAIL: tacit-knowledge.md is non-template in the worktree"
```

None of the FAIL lines should print. If any does, something
between `main`'s tip and the new branch's tip is wrong — abort
and inspect (`git diff main..e2e-test/<YYYY-MM-DD> -- \
src/configs/ tacit-knowledge.md` will be empty for a clean cut).

### 3. Verify clean state

Model template `main` is at the latest commit; no stale
`e2e-test/*` worktrees or branches conflict; prior test catalogs
(if any) are either kept intentionally or deleted with user
confirmation.

### 4. Refresh sibling versions

`uv sync --upgrade` inside the e2e worktree to pick up the latest
`deriva-ml`, `deriva-ml-mcp`, `deriva-mcp-core`, `deriva-skills`,
`deriva-ml-skills` versions. Confirm versions match their `main`
HEADs (or the run will pin to stale versions and the run is not
reconstructable from sibling tags alone). If sibling versions
have advanced enough to need a container rebuild, rebuild the
dev-localhost MCP container against those versions and restart
Claude Code's MCP servers, then **re-do step 1** to
re-authenticate the freshly restarted server.

### 5. Phase 0 part A — create the catalog

From the e2e worktree:

```
uv run load-cifar10 --hostname localhost \
    --create-catalog e2e-test-<YYYYMMDD> --phase schema
```

This creates the catalog and the domain schema only. Capture the
numeric catalog id printed by the loader — every later step needs
it.

### 6. Phase 0 part B — update `deriva.py`

Edit `src/configs/deriva.py` in the e2e worktree so the
`default_deriva` entry has `hostname="localhost"` and
`catalog_id=<new_id>`. Commit on `e2e-test/<YYYY-MM-DD>` with an
`[E2E-DROP]` marker so the commit can be dropped from `main` at
wrap-up. After this step, `uv run deriva-ml-run` (and
`deriva-ml-run-notebook`) in the e2e worktree default to the new
catalog with no CLI overrides.

### 7. Phase 0 part C — load assets and datasets

Re-invoke the loader against the now-existing catalog:

```
uv run load-cifar10 --hostname localhost \
    --catalog-id <new_id> --num-images 1100 --phase images
uv run load-cifar10 --hostname localhost \
    --catalog-id <new_id> --num-images 1100 --phase datasets
```

The small-Toronto-split family in
`src/scripts/_cifar10_datasets.py` requires `train_pool` and
`test_pool` strictly greater than the `SMALL_TRAIN_SIZE` /
`SMALL_TEST_SIZE` constants (currently 500 each); otherwise the
`datasets` phase aborts with `SmallVariantDegenerateError`. At
`--num-images N` the loader splits `N/2` / `N/2` into train and
test pools, so the floor is `N > 2 * 500 = 1000`. 1100 clears the
floor with a comfortable margin. More images means a longer load
and a larger catalog — if a future run needs faster turnaround,
either lower `SMALL_*_SIZE` in the loader or skip the small
Toronto split family.

Run the phases separately (not `--phase all`) so a failure in
`datasets` doesn't require re-uploading the images. Each phase is
intended to be idempotent against partial state.

### 8. Phase 0 part D — update `datasets.py`

Edit `src/configs/datasets.py` in the e2e worktree, replacing the
empty placeholder lists with the dataset RIDs the loader
produced. Discover them with `ml.find_datasets()` from a quick
Python session against the new catalog. Commit on
`e2e-test/<YYYY-MM-DD>` with an `[E2E-DROP]` marker.

### 9. Phase 0 part E — confirm the catalog is usable

Quick sanity check that the bootstrap produced what was expected:

- The catalog exists at the expected name and id.
- The canonical dataset hierarchy is reachable.
- `Image_Classification` feature values are populated for the
  labeled partitions.
- The class distribution is approximately uniform across the 10
  CIFAR-10 classes (a severely skewed distribution — e.g.,
  bird+ship dominating — indicates the loader has regressed).

This is a fail-fast gate on bootstrap, not the parity check the
evaluator will eventually run on the personas' work. If something
at this step looks badly wrong (no datasets, all-null features,
severe class skew), file a Phase 0 finding against `load-cifar10`
and stop — running personas against a broken catalog learns
nothing.

## What's *not* Phase 0

- **`load-cifar10` itself.** The script lives in
  `src/scripts/load_cifar10.py` and is treated as platform code,
  not test code. If it breaks during step 5 or step 7, that's a
  finding against the script (or against `deriva-ml` if the
  failure is in a library call), not test-design feedback.
- **Schema or vocabulary creation beyond what `load-cifar10`
  does.** Any curation work belongs to the Curator persona, not
  bootstrap.
- **Feature populations beyond ground-truth.** The Curator is the
  persona who decides whether additional features are needed
  downstream.
- **Mode selection, persona launches, evaluator launch.** Those
  are the *orchestrator's* job, covered separately in
  [`e2e-orchestrator.md`](e2e-orchestrator.md).
- **Recording the bootstrap as a `tacit-knowledge.md` entry.** If
  it's worth recording (often it is — sibling versions, catalog
  id, the `load-cifar10` invocation that produced this state),
  that's `capture-tacit-knowledge`'s call. Phase 0 doesn't
  prescribe it.
