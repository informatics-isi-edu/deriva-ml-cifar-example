# docs/findings/

This directory holds **session-bounded historical findings** from
named test sessions — typically a single markdown file per session
summarizing what came out of it. These files are dated, scoped, and
read-only after the session ends.

## Not to be confused with

A live e2e multi-persona test run produces findings under a
different path:

```
findings/                                 # at repo root, NOT under docs/
    setup/<NN>-<slug>.md                  # Phase 0 / pre-curator findings
    curator/<NN>-<slug>.md                # Curator persona findings
    developer/<NN>-<slug>.md              # Developer persona findings
    analyst/<NN>-<slug>.md                # Analyst persona findings
    REPORT-<YYYY-MM-DD>.md                # consolidated friction map
```

That directory is created and populated by the test run per
[`docs/test-plans/2026-05-20-e2e-multipersona.md`](../test-plans/2026-05-20-e2e-multipersona.md).

## Why two directories?

- **`docs/findings/`** is *project* documentation — historical
  findings that should travel with the repo and be browsable by
  anyone reading the docs. Dated; archival.
- **`findings/`** (repo root) is *run* output — the live artifacts
  produced by an in-progress or recently-completed e2e session.
  Per-persona, structured, intended for promotion to GitHub issues
  or fix-pass review.

Files in this directory predate the multi-persona test format and
remain useful as context. They are not part of the next run's
workflow.
