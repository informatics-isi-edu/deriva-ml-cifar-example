# Experiment Design: Capacity-sweep high end (large capacity) on the small labeled split

**Slug:** cifar10-small-large
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

## Goal

Does increasing model capacity (64→128 ch, 256 hidden) and training longer (20
epochs) raise top-1 test accuracy on the small labeled split — the high end of
the capacity sweep held against the same data and seed?

## Hypothesis

At fixed dataset and seed, the large-capacity / 20-epoch model
achieves the highest top-1 test accuracy of the *unregularized* sweep points
(`cifar10_quick` < `cifar10_small_default` < `cifar10_small_large`) — though at
the small data scale, added capacity without regularization may begin to
overfit, narrowing or eliminating the gain over the default-capacity point.

## Requirements

- **Data:** `cifar10_small_labeled_split` — same dataset/version/seed as the
  other sweep points.
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training`.
- **Compute budget:** moderate — 20 epochs at large capacity; minutes on CPU.
- **Reproducibility:** `seed=42` shared across the sweep.

## Validation

- **Metric:** top-1 accuracy on the held-out labeled test partition.
- **Baseline:** `cifar10_quick` and `cifar10_small_default` on the same split.
- **Confirms the hypothesis if:** `small_large` test accuracy ≥ `small_default`
  ≥ `quick` (monotone with capacity).
- **Refutes the hypothesis if:** `small_large` < `small_default` — capacity
  saturation / overfitting at this data scale (note: this config is
  *unregularized*; `cifar10_extended` adds dropout + weight decay at similar
  capacity and is the regularized comparison point).
- **Inconclusive if:** within run-to-run noise of `small_default` — re-run at
  multiple seeds.

## Analysis plan

Multi-run comparison via `/deriva-ml:compare-model-runs` across the
small-labeled-split sweep; this is the **high-capacity (unregularized)** corner.
Contrast against `cifar10_extended` (similar capacity *with* regularization) to
separate the capacity effect from the regularization effect.

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(model_config
  `cifar10_large`: 64→128 ch, 256 hidden, dropout 0.0, lr 1e-3, 20 epochs, batch 64)*.
- **Dataset design:** `cifar10-small-labeled-split`.

## Status & links

- **Config:** `cifar10_small_large` in `src/configs/experiments.py`
  (model_config `cifar10_large` × datasets `cifar10_small_labeled_split`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
