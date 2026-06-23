# E2E Orchestrator

**Companion to:** [`2026-05-20-e2e-multipersona.md`](2026-05-20-e2e-multipersona.md) (the persona scenario) and [`e2e-bootstrap.md`](e2e-bootstrap.md) (Phase 0 platform setup).

---

The orchestrator drives a multipersona e2e run from start to
finish. It's not part of the scenario — the personas don't see
the orchestrator's decisions, and the orchestrator doesn't do any
of the personas' work. Its job is **sequencing, authority, and
mechanical cleanup**: spawn the personas in the right order, hand
each one the right context, decide when to pause for the user,
launch the evaluator at the end, and tear down the workspace
afterwards.

This document describes that role. The reader is either the user
running an e2e by hand or an outer agent automating the run.

## 1. Preconditions

Before the orchestrator does anything, Phase 0 must be complete.
See [`e2e-bootstrap.md`](e2e-bootstrap.md). At that point:

- The e2e worktree exists at `../deriva-ml-model-template-e2e` on
  branch `e2e-test/<YYYY-MM-DD>`.
- A fresh catalog `e2e-test-<YYYYMMDD>` exists at `localhost`,
  populated by `load-cifar10`.
- `src/configs/deriva.py` and `src/configs/datasets.py` in the
  worktree are pointed at the new catalog.
- The dev-localhost MCP server is authenticated.

The orchestrator's first action is to confirm those preconditions
hold. If any don't, return to Phase 0 — don't try to compensate.

## 2. Mode — pick one at session start

The orchestrator runs in one of two modes, chosen by the user
once at session start. The choice does not change mid-run.

**Interactive mode.** After each persona's arc finishes, the run
pauses. The user reviews what the persona did and decides whether
to redirect, ask for elaboration, request a re-do of a specific
step, or proceed to the next persona. This mode is for first-time
runs and runs where the user wants to verify the personas are
behaving sensibly.

**Autonomous mode.** All three personas run their arcs
back-to-back without checkpoints. The evaluator runs after the
personas finish and produces its evaluation report. This mode is
for repeat runs once the user trusts the personas, for overnight
execution, or for batch comparison of multiple platform versions.

The mode is selected once, at session start, by the user. If the
user is interactively monitoring and wants to step away, the
choice is to abort and re-launch in autonomous mode — not to
switch modes inside one run.

### Inquiry is allowed in either mode

The mode flag governs *checkpoint pauses* (does the orchestrator
wait between persona arcs?). It does **not** restrict persona
agents from raising a short clarifying question to the user
*during* an arc when the answer would materially improve the
work. Inquiry is distinct from a checkpoint: it's an inline
question that doesn't pause the arc, and the user's answer feeds
the next sentence the agent writes. In autonomous mode the bar is
higher (asking interrupts the autonomy contract), so personas
should default to provenance markers and inquire only when a
load-bearing claim would otherwise be `[inferred from pattern]` —
see the `capture-tacit-knowledge` skill's "When to inquire"
section for the budget, threshold, and confirmatory-shape rules.

## 3. Decision rights — what an agent can decide alone

Persona agents need clear ground rules about when to act and when
to escalate. The rules differ by mode.

| Decision | Interactive | Autonomous |
|---|---|---|
| Which existing dataset/feature/config to use for an obvious task | Decide | Decide |
| Reasonable parameter choice (split ratio, learning rate, epoch count) within typical range | Checkpoint summary | Decide; note the choice in `tacit-knowledge.md` |
| Pick between two equally-valid skills | Checkpoint summary | Decide |
| Create a new dataset / feature / config that serves the persona's professional motivation | Checkpoint, ask first | Decide if it serves the persona's interests; note rationale |
| Destructive operations (delete catalog, drop schema, force-push, rm -rf working dir) | Always ask | Always ask — abort the persona if blocked |
| Schema migrations (new column, FK change, drop table) | Always ask | Always ask — abort the persona if blocked |
| Fix a bug encountered mid-arc | Always ask | Never. File a finding and route around if possible. |
| Stop the arc because the platform won't support what the persona wants | Checkpoint, explain | File a finding; produce whatever summary the persona can with what's been done |

The bright lines: destructive operations and schema migrations
always require explicit user authorization, regardless of mode.
Persona agents never fix bugs mid-arc — that's a separate
fix-pass after the evaluator's report lands.

## 4. Multi-agent setup

Each persona runs as its own Agent-tool invocation with a
dedicated system prompt drawn from the scenario document's §2.
**All three personas share a single git worktree** on a single
dedicated e2e branch — they run sequentially in the same working
tree, not in per-persona worktrees. The catalog is also shared.

Branch / worktree convention (set up in Phase 0):

```
git worktree add ../deriva-ml-model-template-e2e \
    -b e2e-test/<YYYY-MM-DD>
```

All persona work — config edits, `tacit-knowledge.md` appends,
findings under `findings/<persona>/`, helper scripts, commits
with `[E2E-DROP]` markers — happens here, on this branch.

### Why single-worktree, not worktree-per-persona

An earlier revision of this scheme used per-persona worktrees to
prevent file-stomping between concurrent agents. Personas in this
scheme are sequential, not concurrent, so the file-stomping risk
doesn't apply. The cost of per-persona worktrees was much higher:
each persona's `tacit-knowledge.md`, config edits, and findings
lived in a separate working tree, and the orchestrator had to
merge between branches to carry the handoff forward. That made
the cross-persona knowledge-transfer artifact (`tacit-knowledge.md`)
implicit in the orchestrator's merging discipline rather than
naturally available to the next persona. Single-worktree restores
the handoff as the straightforward chain it should be: persona N
writes, persona N+1 reads from the same files.

### Concurrent variant (future)

If a future run ever wants to exercise concurrent personas (e.g.,
Curator on labeling while the Modeler trains on an earlier
dataset version), reintroduce per-persona worktrees and treat
each merge as an explicit synchronisation point. Out of scope
here.

## 5. The run, step by step

### 5.1 Confirm Phase 0 is done

Quick verification that the preconditions in §1 hold:

```
ls ../deriva-ml-model-template-e2e
git -C ../deriva-ml-model-template-e2e branch --show-current
claude mcp list | grep dev-localhost
```

If any of these surface a problem, return to
[`e2e-bootstrap.md`](e2e-bootstrap.md).

### 5.2 Ask the user for the mode

Interactive or autonomous (see §2). Default to interactive on
first runs.

### 5.3 Launch the Curator

Spawn the Curator persona via the Agent tool with the system
prompt corresponding to §2.1 of the scenario document. The
Curator inherits the e2e worktree and the catalog as Phase 0 left
them.

**Interactive mode:** when the Curator finishes, pause. Read the
Curator's summary. Ask the user whether to proceed.

**Autonomous mode:** when the Curator finishes, proceed to the
Modeler immediately.

### 5.4 Launch the Modeler

Spawn the Modeler with the §2.2 prompt. The Modeler inherits
whatever state the Curator left.

**Interactive mode:** pause after the Modeler finishes for the
user's review. **Autonomous mode:** proceed.

### 5.5 Launch the Analyst

Spawn the Analyst with the §2.3 prompt. The Analyst inherits
whatever state the Modeler left.

**Interactive mode:** pause after the Analyst finishes for the
user's review. **Autonomous mode:** proceed.

### 5.6 Launch the evaluator

Once the three persona arcs are done, spawn the evaluator agent
(see [`e2e-evaluator.md`](e2e-evaluator.md) for its rubric). The
evaluator runs **cold** — it sees the artifacts the personas
produced but not the prompts that drove them. Its output is
`docs/reports/<YYYY-MM-DD>-evaluation.md` in the e2e worktree,
plus any new files under `findings/evaluator/` it judges worth
filing.

The evaluator is **not** optional. It's the test's actual output.

### 5.7 Show the user the evaluation

In both modes, present the evaluation report to the user. The
user decides what to do with each finding: promote to a GitHub
issue, queue for a fix-pass, defer, or discard.

## 6. Wrap-up (mechanical)

Wrap-up happens **after** the evaluator's pass. The evaluator
needs the e2e worktree and catalog in place to do its work; tear
them down only when the evaluation report has landed.

1. **Cherry-pick template fixes** from the shared
   `e2e-test/<YYYY-MM-DD>` branch back to `main` of the model
   template. `[E2E-DROP]`-tagged commits (the catalog-id and RID
   repointing commits) stay on the branch; only genuine
   improvements to the template get picked back.
2. **Push or archive the e2e branch** for the historical record.
   The convention is to push to
   `origin/archive/e2e-test-<YYYY-MM-DD>` and delete the local
   branch.
3. **Worktree teardown**, with explicit user confirmation:
   ```
   git worktree remove ../deriva-ml-model-template-e2e
   git branch -D e2e-test/<YYYY-MM-DD>
   ```
   The CLAUDE.md guidance is to leave the worktree in place
   until the user is done reading the evaluation report — only
   tear down when explicitly asked, or when setting up the next
   run.
4. **Catalog disposition**, with explicit user confirmation:
   delete the test catalog or preserve it for archeology.

## 7. What's *not* the orchestrator's job

- **Doing any persona's work.** The orchestrator sequences and
  manages; the personas do.
- **Pre-judging findings.** That's the evaluator's job. The
  orchestrator should not decide that a persona's claim looks
  wrong and re-run them, or rewrite a persona's summary. If
  something looks off, file a finding from the orchestrator
  itself or surface the question to the user.
- **Editing `tacit-knowledge.md`.** Personas write to it through
  `capture-tacit-knowledge`; the orchestrator doesn't.
- **Fixing bugs.** Same as the personas: findings flow to a
  fix-pass after the evaluator runs.
- **Phase 0 setup.** That's its own document.
