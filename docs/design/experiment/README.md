# Experiment Design Documents

One Markdown design document per experiment, named `<slug>.md` (e.g.
`dropout-vs-baseline.md`). Write the design **before** you configure and run
the experiment — it is the up-front contract the config in `src/configs/`
then implements.

This is the design-first phase of `/deriva-ml:experiment-lifecycle` (Phase 1).
Author docs here with `/deriva-ml:design-experiment`, which carries the
standardized template (Goal · Hypothesis · Requirements · Validation · Analysis
plan · Status & links) and a worked example.

## How this relates to the other records

- **This directory** = the **plan**: what question the experiment tests, what
  evidence answers it, the success/refute/inconclusive criteria, the cost
  budget. Written first, then implemented by a config.
- **`tacit-knowledge.md`** (project root) = the **running journal**: what you
  *learned* during and after the run. The two cross-link — a design doc's
  "Status & links" section points at the journal entries its run produced.
- **`src/configs/experiments.py`** = where the experiment is actually
  *configured* (a model config paired with a dataset config). The design doc
  explains *why* an experiment exists; the config defines *what* runs.

A design doc is cheap; finding out you ran the wrong experiment after the run
is not. Fill every section — a section you can't fill is a design question you
haven't answered yet.
