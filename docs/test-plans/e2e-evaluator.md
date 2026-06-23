# E2E Evaluator

**Companion to:** [`2026-05-20-e2e-multipersona.md`](2026-05-20-e2e-multipersona.md) (the persona scenario), [`e2e-bootstrap.md`](e2e-bootstrap.md) (Phase 0), and [`e2e-orchestrator.md`](e2e-orchestrator.md).

---

The evaluator is a separate agent. It runs after the three
personas finish, reads what they produced, and writes a report
saying how the run went. It is the **test's actual output**: the
scenario produces artifacts, this agent judges them.

This document is the evaluator's rubric — what it looks for, how
it forms findings, what its report looks like.

## 1. The evaluator's stance

The evaluator approaches the run as a **fair, attentive reader
who was not in the room**. It sees the artifacts the team
produced — the e2e branch, the catalog state, `tacit-knowledge.md`,
the report under `docs/reports/`, the persona findings — but
**not** the prompts that drove the personas. This matters: the
evaluator must judge the work on its own merits, the way a future
contributor or auditor would, not by checking it against the
instructions the personas received.

Three orientations the evaluator holds:

- **Cold reader.** Could someone who has never seen this project
  orient themselves from what's in the worktree and the catalog?
  If `tacit-knowledge.md` reads as a coherent record of decisions,
  good. If it reads as state-replication padding, that's a
  finding.
- **Domain professional.** Does the work serve the personas'
  professional roles? Did the Curator characterise the data
  usefully? Did the Modeler produce results the Analyst could
  actually compare? Does the Analyst's report answer the question
  it set out to ask?
- **Platform reviewer.** Did the platform support the work without
  getting in the way? Where did it? Were the skills that *should*
  have fired the ones that *did* fire?

The evaluator is **not** there to second-guess the personas'
specific choices. If the Curator decided XEM stratified at 15/class
was the right Validation slice, the evaluator doesn't relitigate
that — they confirm the work is internally coherent and serves
the next persona.

## 2. What the evaluator reads

In rough order of importance:

- **The catalog itself.** The catalog is the source of record for
  *what happened*. Every other artifact is a claim about the
  catalog's state. The evaluator must verify those claims.
- **`tacit-knowledge.md`** — the team's accumulated decision
  record. Read top to bottom as the project's history.
- **`docs/reports/<YYYY-MM-DD>-analysis.md`** (or similar) — the
  Analyst's written summary, the team's primary deliverable to
  the outside world.
- **The e2e branch's commit log** — `git log e2e-test/<YYYY-MM-DD>`.
  What did the personas actually do? What did they touch that
  they didn't mention?
- **`findings/<persona>/*.md`** — friction the personas captured
  in-arc.
- **Persona arc summaries** — short writeups each persona produces
  at the end of their turn.
- **Executed notebooks and the report's referenced output assets**
  — confirm they actually exist and contain what the report says
  they contain.

Catalog access is via the dev-localhost MCP server (already
authenticated at Phase 0) and direct `deriva-ml` Python. The
evaluator should freely use both.

## 3. What the evaluator looks for

Four threads. The evaluator weaves through all of them; they're
ordered roughly by how load-bearing each is, not by how the
evaluator narrates them in the report.

### 3.1 Catalog ↔ claim agreement

The most important thread. **Every claim the personas made about
the catalog must hold up under direct inspection.** This is the
test that historically catches the highest-severity bugs.

The evaluator picks a sample of claims from the artifacts and
verifies them against the catalog directly:

- The Curator says they created dataset XEM at version 0.1.0 with
  150 members? Look it up — it's there with that version and
  that count, or there's a finding.
- The Modeler says executions Y1M, YDT, XRJ are all `Uploaded`
  with weight + log + CSV assets each? Direct query confirms or
  refutes.
- The Analyst says the denormalize on JZ8 returned 1900 rows
  matching the (Execution, Image) tuple set bit-for-bit? Re-run
  the parity check.

Discrepancies are findings, and they're usually high-severity
because they mean the indirect channel (skills, MCP tools) is
reporting state that doesn't match the direct channel (deriva-ml
Python). Capture both readings verbatim in the finding.

**Cross-channel tie-breaker.** If direct deriva-ml Python and an
MCP tool disagree about catalog state, drop one level lower:
raw `ermrest_catalog.get(...)` or `DatapathBuilder` with no
deriva-ml helpers. That tells you whether the bug is in
deriva-ml or in the layer above.

### 3.2 Coherence of the team's deliverables

Read the artifacts as a continuous story. Does the chain hold?

- Did the **Curator** characterise the data well enough that the
  Modeler could pick datasets to train on without guessing? Did
  the Curator flag oddities a downstream consumer would want to
  know about?
- Did the **Modeler** produce runs the Analyst can rank — same
  test bag, comparable metrics, predictions actually committed?
- Does the **Analyst's report** answer the question it sets out to
  ask, with figures that support the conclusions and caveats
  where the data is too thin?

The form of the test is: a fresh reader, given only the
worktree and the catalog, could they reconstruct what the team
did and why? If yes, the team produced a coherent story. If
no — if some step requires the reader to have been in the room —
that's a coherence finding.

### 3.3 Quality of `tacit-knowledge.md`

The file is the team's accumulated rationale. The evaluator
reads it as a future contributor would: top to bottom, asking
*"is this useful to me?"*

Specific failure modes to watch for:

- **State replication.** Tables of catalog state (dataset RIDs,
  member counts, vocabulary contents, sibling versions) that
  duplicate what `deriva://catalog/.../ml/...` resources would
  return. `capture-tacit-knowledge` explicitly says don't write
  these; if they're here, that's a finding about either the
  persona's use of the skill or the skill's effectiveness at
  steering them off it.
- **PR-number / release-coordinate citations.** "PR #246 fixed
  the row-loss bug" is a citation to a transient git coordinate
  that ages badly. The behaviour ("the PagedFetcher row-completeness
  invariant") is what should be recorded; the PR is at most a
  footnote.
- **TODO-list framing.** "Analyst should run roc_analysis.ipynb
  next" or "we still need to release dataset X" are workflow
  directives, not tacit knowledge. The skill explicitly says no.
- **Handoff-as-narrative.** Sections labelled "handoff to the
  next persona" that describe *what was done* rather than
  *what's tacit* — those are arc summaries, not tacit knowledge.
  The orchestrator's summaries cover what was done. tk-entries
  should be the *why*.
- **`[inferred from pattern]` claims that were load-bearing**
  without inquiry. The skill's "When to inquire" section says to
  ask before recording load-bearing pattern-inferred claims; if
  the evaluator sees such a claim that affects downstream work,
  that's a finding about the persona's discipline.

The reverse failure modes (things to specifically value):

- **Dead-end entries.** "We tried X, it didn't work because Y,
  abandoning unless Z changes." This is the highest-leverage
  tacit knowledge.
- **Gotchas surfaced by the work.** "The Image_Classification
  feature lives in the domain schema, not deriva-ml — a naive
  direct query will KeyError." This is exactly what the file is
  for.
- **Convention entries.** "Whenever we do X in this project, we
  also do Y because Z." Especially valuable when the convention
  isn't otherwise documented.

### 3.4 Platform fitness

Did the platform support the team's work, or did it get in the
way? Synthesise across:

- **Persona findings** — friction they captured in-arc. Are they
  legitimate, or do they reflect a persona reaching for the wrong
  tool? The evaluator may upgrade some to higher severity, downgrade
  others to non-findings, or merge near-duplicates.
- **Skill use observed in the commit log and tacit entries.** Did
  the right skills fire at the right moments? Did personas reach
  for raw deriva-ml Python in places where a skill should have
  routed them? That's a skill gap.
- **Missed friction.** Things the evaluator sees in retrospect
  that the personas didn't flag — a confusing error that was
  routed-around silently, an MCP tool used in an awkward shape
  because the natural shape doesn't exist, a skill description
  that didn't match the API.

The evaluator may file its own findings under
`findings/evaluator/<NN>-<slug>.md`. These are the most
authoritative findings in the run because they reflect a
cross-persona reading the personas themselves couldn't have done.

## 4. The evaluation report

Output goes to `docs/reports/<YYYY-MM-DD>-evaluation.md` in the
e2e worktree. The report is the test's deliverable. It is
**concise and structured** — bureaucratic checklists are not the
goal; useful findings are.

Recommended shape (not a rigid template — adapt to what the run
actually surfaced):

```markdown
# Multipersona E2E Run — <YYYY-MM-DD> Evaluation

**Catalog:** id=<N>, hostname=<h>, name=<e2e-test-YYYYMMDD>
**Branch:** e2e-test/<YYYY-MM-DD>
**Sibling versions:** <deriva-ml vX.Y.Z, deriva-ml-mcp vA.B.C, ...>

## Headline

<One paragraph. Did the team produce a coherent story? Did the
platform support them? What's the most important finding the
user should act on?>

## Catalog ↔ claim agreement

<The §3.1 thread. What was verified, what held up, what didn't.
Discrepancies become findings.>

## Coherence of the team's deliverables

<The §3.2 thread. Curator → Modeler → Analyst as a chain. Did
each pass produce what the next needed?>

## tacit-knowledge.md quality

<The §3.3 thread. Read top-to-bottom; assess as a fresh reader
would. Cite specific entries.>

## Platform fitness

<The §3.4 thread. Persona findings, skill use, missed friction.
Cite finding files by path.>

## Comparison vs prior runs

<If prior evaluation reports exist under docs/reports/, compare
on the dimensions that matter: number of findings, severity
distribution, recurring vs new issues. A healthy trend is one
where findings decrease and the remaining ones are less severe.>

## Recommended actions

<Suggestions to the user, organised by likely disposition: fix
inline now, file as GitHub issue, defer, dismiss. Not
prescriptive — the user decides.>
```

## 5. Findings the evaluator files

Format matches the persona findings (`findings/evaluator/<NN>-<slug>.md`,
same template), but evaluator findings carry **classifications
the personas can't set** — severity, component, category. These
are evaluation judgments that need cross-arc context.

Severities:

- **Blocker** — the platform produced a result that's actively
  wrong (catalog state disagrees with what the tools said, data
  loss, silent corruption). The run did not actually succeed
  even if it appeared to.
- **High** — a primary deliverable was blocked or compromised.
  A persona had to route around a broken skill, a tool produced
  the wrong shape, an API silently swallowed an error.
- **Medium** — friction that slowed the work but didn't block
  it. A confusing error, an awkward tool shape, a skill that
  didn't fire when it should have.
- **Low** — polish: a misleading message, a doc gap, a small
  ergonomic improvement.

Categories:

- **Bug** — platform code does the wrong thing.
- **Skill issue** — a skill misroutes, fails to fire, or gives
  bad advice.
- **Doc gap** — behaviour is correct but not findable.
- **Missing feature** — the platform doesn't support something
  the persona needed.
- **Polish** — small ergonomic ask.

## 6. What the evaluator does NOT do

- **Re-run the personas.** If a persona's work looks wrong, file
  a finding. Don't try to redo it.
- **Edit `tacit-knowledge.md`.** It's the team's record. The
  evaluator reads it and judges it; it does not modify it.
- **Fix bugs.** Findings flow to a fix-pass agent. The evaluator
  identifies friction; the fix-pass resolves it.
- **Promote findings to GitHub issues automatically.** That's
  the user's call after reading the report.
- **Tear down the worktree or the catalog.** That's the
  orchestrator's wrap-up.

## 7. Disposition

The user reads the evaluation report and decides per finding:

- **Fix inline** via a fix-pass agent — surface bugs the next run
  shouldn't hit.
- **Promote to GitHub issue** with appropriate label and
  severity.
- **Defer** — note in the report, leave for a later pass.
- **Dismiss** — note in the report with a brief reason (false
  positive, won't fix, out of scope).

The evaluation report itself stays in `docs/reports/` as the
historical record of how this run went.
