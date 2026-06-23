# Dataset Design Documents

One Markdown design document per dataset, named `<slug>.md` (e.g.
`dev-subset.md`). Write the design **before** you create, split,
subsample, or curate the dataset — it is the up-front contract the build then
implements.

This is the design-first phase of `/deriva-ml:dataset-lifecycle` (Phase 1).
Author docs here with `/deriva-ml:design-experiment`, which carries the
standardized dataset template (Purpose · Requirements · Structure plan ·
Validation · Consumption · Status & links) and a worked example. The dataset
template is parallel in shape to the experiment one.

## How this relates to the other records

- **This directory** = the **plan**: what the dataset is for, its required
  size/composition/balance, the structure (standalone / split / subsample /
  curated) and the three-axis `Dataset_Type` tags, and how correctness is
  validated (class balance, no train/test leakage, bag parity, counts).
- **`tacit-knowledge.md`** (project root) = the **running journal** of what you
  learned building it. The two cross-link.
- **`src/configs/datasets.py`** = where the produced RID + released version are
  pinned for downstream experiments.

A design doc is cheap; the validation criteria it captures (leakage, balance)
are exactly what gets skipped when a dataset is built without one.
