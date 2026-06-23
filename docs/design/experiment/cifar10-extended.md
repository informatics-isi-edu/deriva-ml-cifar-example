# Experiment Design: Extended high-capacity regularized training on the small labeled split

**Slug:** cifar10-extended
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

## Goal

Does the high-capacity, regularized, long-training configuration achieve the
best attainable top-1 test accuracy on the small labeled split — the
"production-quality" end of the project's training options?

## Hypothesis

The extended config (64→128 ch, 256 hidden, dropout 0.25, weight
decay 1e-4, lr 1e-3, 50 epochs) outperforms the quick and default configs on
the same held-out test partition, and the regularization (dropout + weight
decay) keeps the train/test accuracy gap narrower than an unregularized
high-capacity model would.

## Requirements

- **Data:** `cifar10_small_labeled_split` (same dataset as `cifar10_quick` and
  the capacity sweep — enables direct accuracy comparison).
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training`.
- **Compute budget:** the largest of the small-split family — 50 epochs at
  high capacity; allow several minutes to ~tens of minutes on CPU.

## Validation

- **Metric:** top-1 accuracy on the held-out labeled test partition
  (`Image_Classification` feature rows + probability CSV).
- **Baseline:** the `cifar10_quick` (3-epoch) and `cifar10_small_default`
  (10-epoch default capacity) runs on the *same* split.
- **Confirms the hypothesis if:** extended test accuracy > both baselines AND
  the train/test gap is no wider than the default config's.
- **Refutes the hypothesis if:** accuracy ≤ the default config (added capacity
  + epochs not paying off — likely overfitting or under-LR at this data scale).
- **Inconclusive if:** accuracy gain is within run-to-run noise — re-run with a
  different `seed` to estimate variance, or move to `cifar10_extended_full`.

## Analysis plan

Multi-run comparison via `/deriva-ml:compare-model-runs` against the other
small-labeled-split runs (`cifar10_quick`, `cifar10_small_default`,
`cifar10_small_large`) — this experiment is the high-capacity, long-training
corner of that comparison.

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(model_config
  `cifar10_extended`: 64→128 ch, 256 hidden, dropout 0.25, wd 1e-4, lr 1e-3,
  50 epochs, batch 64)*.
- **Dataset design:** `cifar10-small-labeled-split`.

## Status & links

- **Config:** `cifar10_extended` in `src/configs/experiments.py`
  (model_config `cifar10_extended` × datasets `cifar10_small_labeled_split`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
