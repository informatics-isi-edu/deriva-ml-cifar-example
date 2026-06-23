# Experiment Design: Quick pipeline-validation run on the small labeled split

**Slug:** cifar10-quick
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

> **Reverse-engineered.** This design was reconstructed *after* the config in
> `src/configs/experiments.py` already existed, from the config parameters and
> the `Experiments.md` "Purpose" line. The Goal/Requirements are recovered
> faithfully from the config; the **Hypothesis** and **Validation** thresholds
> are *inferred* (the original author did not record explicit success criteria)
> and should be treated as a proposed contract, not a recalled one.

## Goal

Does the end-to-end CIFAR-10 training pipeline (data load → train → record
predictions) run correctly and cheaply on the small labeled split? This is a
pipeline-validation run, not a model-performance run.

## Hypothesis

*(Inferred.)* A 3-epoch, 32→64-channel run on the small labeled split completes
in well under a minute on CPU and produces a learning signal — top-1 test
accuracy meaningfully above the 10% random-guess baseline — without erroring on
asset upload, feature recording, or the prediction CSV.

## Requirements

- **Data:** `cifar10_small_labeled_split` (labeled on both partitions,
  leak-free), version pinned in `src/configs/datasets.py`.
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training` (provisioned by the loader).
- **Compute budget:** trivial — seconds to ~1 minute on CPU; at most 1 cycle.
  This run exists to validate plumbing, not to be repeated for tuning.

## Validation

- **Metric:** top-1 accuracy on the held-out labeled test partition, recorded
  as `Image_Classification` feature rows + a probability CSV asset.
- **Baseline:** 10% (random guess over 10 classes).
- **Confirms the hypothesis if:** the run completes with no errors AND test
  accuracy > 10% (any learning signal).
- **Refutes the hypothesis if:** the run errors, or accuracy ≈ 10% (no signal —
  indicates a pipeline/label-wiring bug, not just an under-trained model).
- **Inconclusive if:** accuracy is marginally above 10% on this small test
  partition — expected at 3 epochs; do not read it as a capability claim.

## Analysis plan

Single-run read of the recorded predictions via
`deriva_ml_list_feature_values` on `Image_Classification`. No multi-run
comparison — this is the low-capacity anchor of the capacity sweep (see
`cifar10-small-default`, `cifar10-small-large`), but its own purpose is
pipeline validation.

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(now authored — model_config
  `cifar10_quick`: 32→64 ch, 128 hidden, dropout 0.0, lr 1e-3, 3 epochs, batch 128)*.
- **Dataset design:** `cifar10-small-labeled-split` *(now authored — the
  small, leak-free, both-partitions-labeled split)*.

## Status & links

- **Config:** `cifar10_quick` in `src/configs/experiments.py`
  (model_config `cifar10_quick` × datasets `cifar10_small_labeled_split`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet — link the run's journal entry here once executed)
