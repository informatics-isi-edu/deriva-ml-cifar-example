# Model Design Documents

One Markdown design document per model, named `<slug>.md` (e.g.
`2layer-cnn.md`). Write the design **before** you author the model
function and its config — it is the up-front contract the model code and
`model_config` then implement.

This is the design-first (Specify) phase of
`/deriva-ml:model-development-workflow` (Phase 1). Author docs here with
`/deriva-ml:design-experiment`, which carries the standardized model template
(Goal · Requirements · Validation · Upstream designs · Status & links) and a
worked example.

## How this relates to the other records

- **This directory** = the **plan**: the prediction task, the architecture and
  hyperparameters (the source the `model_config` group is derived from), the
  input features it trains on, any input assets (checkpoints), and the
  validation criteria (target metric + threshold). Written first, then
  implemented by the model code and config.
- **`src/configs/`** = the *model layer* config (`model_config`) and the
  *experiment layer* config (`experiments.py`) that composes this model with a
  dataset. The design's Requirements are what those configs must satisfy.
- **`tacit-knowledge.md`** (project root) = the **running journal**: what you
  *learned* building and training it. The two cross-link.

A model-design names its **input** feature-designs upstream; its **output**
(prediction) features are downstream — list them under Requirements, but each
prediction feature-design names *this* model as its producer (not the reverse).
