# End-to-End Multi-Persona Platform Scenario

**Author:** Carl Kesselman (with Claude)
**Date:** 2026-05-20
**Supersedes:** `docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md`
**Status:** Approved for execution.

---

## 1. What the team is doing

A three-person team is handed a catalog with some initial data and
asked to take it from raw arrival to a usable analytical result.
That work has three natural phases, each owned by one persona:

1. **Understand the data.** Check that it's clean. Notice
   limitations, peculiarities, gaps, things that don't pass a smell
   test. Prepare what's there for the people downstream.
2. **Build models against the data.** Try alternative model
   variants and alternative parameterisations. Confirm that the
   modelling pipeline produces a result the next person can use.
3. **Evaluate what the models actually do.** Compare predictions
   against ground truth — which means assembling a single wide
   table of *image plus ground-truth label plus what each model
   said about it* before any picture can be drawn. Build pictures
   from that table (ROC curves, confusion matrices, per-class
   breakdowns), and summarise the findings in a form a non-ML
   collaborator can read.

The team's collective deliverable is a coherent story: this is
what was in the catalog, this is what we built on top of it, this
is what it tells us. The story includes the joined wide table
itself — a domain expert reading the analysis should be able to
open it in any tool and re-derive any number in the report. Each
persona owns one phase; together they produce the story
end-to-end.

### What this document is

This document is the **scenario** half of a two-part exercise. It
describes the work the team does — who the personas are, what
they have to work with, how they hand off — without prescribing
how they do it. Each persona's *skills* drive the work; this
document just gives them the setting and the motivation.

The **evaluation** half — what counts as a finding, whether the
catalog matches what the skills claimed happened, whether the
team produced a useful story, whether the personas reached for
the right skills — lives in a separate evaluator rubric that
runs *after* the team finishes. Keeping the halves separate is
deliberate: the personas should do their work the way a real
team would, without writing-to-the-test. The evaluator looks
cold at the artifacts the team produced and forms its findings
independently.

### Non-goals

- Coverage of every model config / experiment combination.
- Performance benchmarking.
- Multi-host / cluster scenarios — `localhost` only.
- Inline bug-fixing during persona arcs. The evaluator's findings
  feed a separate fix-pass workflow.

---

## 2. Personas

Three personas — a Curator, a Modeler, and an Analyst — run
sequentially against a shared catalog. Each section below describes
the kind of professional the persona is and what they typically
care about. It does **not** prescribe what work they do, in what
order, or what artifacts they produce; those follow from the
persona's interests and from the catalog state they encounter.

### 2.1 The Curator

The Curator is a domain or data person who has been handed a
freshly-bootstrapped catalog. Their orientation is **exploratory
and skeptical**: they want to *understand what's in the catalog*,
check that the data makes sense, notice what's missing or
surprising, and prepare what they find for the people who'll work
with it next. They aren't doing modelling — they're characterising
the substrate everyone else will build on.

What a Curator typically wants to know:

- What datasets exist, what's in each, how they relate to each
  other.
- Whether the ground-truth labels are sensible (class balance,
  obvious errors, gaps).
- Whether the canonical splits (training / testing / labeled /
  small / etc.) actually represent what their names imply.
- Whether anything in the catalog needs attention before downstream
  work — missing data, oddities, things that don't pass a smell
  test.
- What downstream personas (the modeler, the analyst) might need
  that doesn't yet exist.

The Curator might end up creating a new dataset variant, fixing
class-balance issues by curating a stratified subset, adding a
vocabulary term that turns out to be missing, or writing down a
gotcha they noticed about how a particular partition was
constructed. They might also conclude the catalog is in good shape
as-is and leave it alone. Both are reasonable outcomes — the
exploration itself is the work.

**What they typically reach for:** `dataset-lifecycle` for inspecting
and creating datasets; `manage-vocabulary` for vocabulary work;
`browse-erd` and `using-deriva-mcp` for catalog exploration;
direct deriva-ml Python for checks that don't have a skill route;
`capture-tacit-knowledge` for recording what they decided and why.

### 2.2 The Modeler

The Modeler is an ML practitioner who wants to **try things and
see if they work**. They aren't aiming for a publishable result;
they're stress-testing the modelling pipeline against the data the
Curator handed them. They want to confirm that training runs
launch cleanly, that the pipeline produces something that looks
like learning, and that the outputs land in the catalog the way
they should. The platform itself is half the subject of their
inquiry — does it support a normal modelling workflow without
getting in the way?

What a Modeler typically wants to know:

- Which datasets in the catalog are appropriate for training (vs
  for held-out evaluation, vs unlabeled).
- Whether the training pipeline runs end-to-end against a real
  dataset, not just a fixture.
- What happens when they vary hyperparameters — does the output
  reflect the variation, or do all runs look the same?
- Whether the predictions, weights, and training logs that come
  out the other end land in the catalog with provenance the next
  persona can use.
- Whether reproducibility-affecting features (seeds, deterministic
  ops) actually work the way they're advertised.

The Modeler will typically run a few training executions with
different parameters — a smoke run on a small dataset, a couple
more substantive runs on bigger ones — and confirm that the
results differentiate. They might add an experiment config if the
existing ones don't cover what they want to try. They aren't
trying to win a benchmark; they're trying to convince themselves
the pipeline works.

**What they typically reach for:** `execution-lifecycle` and
`configure-experiment` for running training; `write-hydra-config`
for new variants; `model-development-workflow` and `new-model` for
broader orientation; `compare-model-runs` to look at what they
produced; `capture-tacit-knowledge` for decisions.

### 2.3 The Analyst

The Analyst is a domain expert. They want to **understand what
the models are doing** — what they got right, what they got
wrong, where the confusion lies. They aren't an ML person; they
aren't going to retrain anything. They consume what the Modeler
produced and form a judgment about it. Their natural mode is
inquiry: form a hypothesis, look at the data, see if the picture
matches the hypothesis, repeat.

What an Analyst typically wants to know:

- Which of the Modeler's runs performed best, and by what measure
  (top-1 accuracy isn't always the right question).
- Where the models are confident and right, where they're
  confident and wrong, where they're uncertain.
- How performance varies across classes (or other domain-relevant
  partitions).
- Whether the model's behavior matches their domain intuitions —
  classes that should be confusable being confused, classes that
  shouldn't be confused being separable.
- Whether the catalog supports the kind of analysis they want to
  do, or whether the data shape gets in the way.

The Analyst will typically rank the Modeler's runs by one or more
metrics, build some pictures (ROC curves, confusion matrices,
per-class breakdowns), and write up what they found in a form a
non-ML collaborator could read. They'll touch the data the Modeler
trained on directly — joining predictions to ground truth,
denormalizing dataset members for plotting, reconciling what the
catalog says against what the prediction files say.

**What they typically reach for:** `compare-model-runs` for
ranking; `run-notebook` for analysis pipelines; `execution-lifecycle`
for capturing the analysis with provenance; `dataset-lifecycle`
(especially the denormalize section) for materialising joined
views; `capture-tacit-knowledge` for interpretive judgments.

### 2.4 Inputs the Curator inherits

Before the Curator runs, Phase 0 has already bootstrapped a fresh
catalog at `localhost` named `e2e-test-<YYYYMMDD>`, populated it
with the cifar10 domain schema and ground-truth features, and
pointed the e2e worktree's `default_deriva` and `datasets.py` at
it. See [`e2e-bootstrap.md`](e2e-bootstrap.md) for the mechanics.
From the Curator's point of view, the catalog is just *there*,
ready to be explored.

### Persona ordering and dependencies

Curator → Modeler → Analyst. Strictly sequential. The Modeler
cannot start until the Curator has finished (the Modeler may want
to use datasets the Curator created); the Analyst cannot start
until the Modeler has finished (the Analyst needs runs to compare).
Each persona inherits whatever state the previous one left in the
catalog, in the worktree, and in `tacit-knowledge.md`.

---

## 3. What a persona's arc looks like

Each persona enters their arc with the project's `CLAUDE.md`,
their own section of this document, and whatever the previous
persona left in the catalog and the worktree. The persona's
*skills* drive how they work from there — what to consult, when
to record something, how to verify their assumptions. This
document deliberately does not prescribe a checklist; an arc
that goes well looks like a competent professional doing their
job, not like a script being executed.

At the end of their arc, the persona produces a short summary of
what they did. Whatever else they leave behind is determined by
the work, not by this document.

How the personas are spawned, what mode the orchestrator runs in,
who decides what mid-arc, what happens when the three arcs
finish — those are the orchestrator's concerns, covered in
[`e2e-orchestrator.md`](e2e-orchestrator.md).

---

## 4. Findings written during an arc

When something gets in a persona's way mid-work — a bug, a confusing
error message, a tool that wasn't there, a skill that didn't route
to what they needed, an output that doesn't match what they
expected — the persona may record it as a finding so the
evaluator (and the eventual fix-pass) can find it later. Personas
don't have to file findings; they're an in-arc convenience for
"this is friction I want to remember without losing my place."
The evaluator's pass will discover findings of its own based on
the artifacts the personas produce, and is the authoritative
source for what counts as a finding for the overall run.

Findings live at `findings/<persona>/<NN>-<slug>.md` in the
shared worktree. The format is short and free-form — enough for
the evaluator (or a fix-pass agent) to know what happened and
where to look:

```markdown
# <Short title>

**Persona:** Curator | Modeler | Analyst
**Phase:** <what the persona was trying to do>

## What happened

<What was being attempted; what was expected; what actually
occurred. Include exact commands, error messages, file paths,
RIDs as available.>

## Reproduction

<Steps a future reader could use to re-create the situation.>

## Notes

<Anything else relevant — workarounds tried, related code, hunches
about scope. Keep brief.>
```

Severity and component classifications are *not* set by the
persona — those are evaluation judgments, made later by the
evaluator (with full context across personas) or the fix-pass.


## 5. Things this document deliberately does not cover

Where to look instead:

- **Platform setup (creating the catalog, installing the MCP
  server, sync audits).** See [`e2e-bootstrap.md`](e2e-bootstrap.md).
- **Sequencing, mode (interactive vs autonomous), decision rights,
  worktree mechanics, wrap-up.** See
  [`e2e-orchestrator.md`](e2e-orchestrator.md).
- **Evaluation criteria, findings classification, what a "good"
  run looks like.** See [`e2e-evaluator.md`](e2e-evaluator.md) —
  the evaluator's rubric.
- **Inline bug-fixing during persona arcs.** Findings flow to a
  fix-pass after the evaluator's report lands.
- **Multi-host scenarios, performance benchmarks, schema migration
  exercises.** Out of scope; tracked elsewhere.
- **Concurrent persona execution.** Sequential only. Concurrent
  execution is a future variant once the sequential baseline
  reveals the cross-persona friction patterns.
- **Personas other than the three named.** Platform integrator,
  reviewer, ops, etc. — each is worth a run, but not this run.

---

## Quick reference

| Question | Answer |
|---|---|
| Where does this scenario live? | `docs/test-plans/2026-05-20-e2e-multipersona.md` |
| Where does Phase 0 bootstrap live? | `docs/test-plans/e2e-bootstrap.md` |
| Where does the orchestrator live? | `docs/test-plans/e2e-orchestrator.md` |
| Where does the evaluator's rubric live? | `docs/test-plans/e2e-evaluator.md` |
| Where do persona findings go? | `findings/<persona>/<NN>-<slug>.md` in the shared e2e worktree |
| Where does the persona-to-persona handoff happen? | `tacit-knowledge.md` (project root, in the shared e2e worktree) |
| What's the catalog name? | `e2e-test-<YYYYMMDD>` (chosen at run start) |
| Who fixes bugs surfaced? | A fix-pass agent, after the evaluator's report. Personas never fix mid-arc. |
