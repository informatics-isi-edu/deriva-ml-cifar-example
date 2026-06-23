# Experiment Design: Inference-only evaluation of pretrained weights

**Slug:** cifar10-test-only
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

> **Reverse-engineered.** Reconstructed after the config existed. Goal/Requirements
> recovered from the config; **Validation** thresholds are *inferred*. This
> experiment differs structurally from the others: it does **no training**.

## Goal

Given an already-trained checkpoint, what is its top-1 accuracy on the small
labeled test partition — evaluated without any further training?

## Hypothesis

*(Inferred.)* Loading pretrained weights (`cifar10_cnn_weights.pt`) and running
forward-only inference reproduces the checkpoint's training-time test accuracy
on the same-distribution labeled test set, and records predictions
(`Image_Classification` rows + probability CSV) suitable for downstream ROC /
confusion-matrix analysis.

## Requirements

- **Data:** `cifar10_small_labeled_testing` (labeled test partition only — no
  training data needed).
- **Assets:** **required** — a weights asset containing
  `cifar10_cnn_weights.pt`, referenced by RID via the `assets=` config group.
  Without it, the run cannot proceed (this is the key precondition that sets
  this experiment apart).
- **Vocabularies:** `Workflow_Type` — an inference/evaluation type (the model
  config sets `test_only=True`; no `Training` workflow is registered).
- **Compute budget:** minimal — a single forward pass over the test partition;
  seconds.

## Validation

- **Metric:** top-1 accuracy on the labeled test partition, plus per-image
  predicted class + probabilities recorded as features/CSV.
- **Baseline:** the test accuracy reported by the run that *produced* the
  weights (should match within numerical tolerance).
- **Confirms the hypothesis if:** evaluation completes, predictions are
  recorded, and accuracy matches the source run's test accuracy.
- **Refutes the hypothesis if:** accuracy diverges materially from the source
  run — indicates a weights/architecture mismatch, wrong preprocessing, or a
  label-mapping bug in the inference path.
- **Inconclusive if:** no source-run accuracy is on record to compare against —
  then this run *establishes* the checkpoint's test accuracy rather than
  verifying it.

## Analysis plan

Single-run read of recorded predictions via `deriva_ml_list_feature_values`;
feeds the ROC analysis notebook (`notebooks/roc_analysis.ipynb`) and any
confusion-matrix work. No multi-run comparison required.

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(not yet authored — model_config
  `cifar10_test_only`: same architecture, `test_only=True`, loads
  `cifar10_cnn_weights.pt`)*.
- **Dataset design:** `cifar10-small-labeled-testing` *(not yet authored —
  labeled test partition only)*.
- **Asset precondition:** a trained-weights asset (produced by one of the
  training experiments above) — not a design doc, but a hard input dependency.

## Status & links

- **Config:** `cifar10_test_only` in `src/configs/experiments.py`
  (model_config `cifar10_test_only` × datasets `cifar10_small_labeled_testing`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
