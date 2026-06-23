# Experiment Design: Best-accuracy extended training on the full labeled split

**Slug:** cifar10-extended-full
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

> **Reverse-engineered.** Reconstructed after the config existed. Goal/Requirements
> recovered from the config; **Hypothesis** and **Validation** thresholds are
> *inferred*.

## Goal

What is the best top-1 test accuracy the template can reach — high-capacity,
regularized, 50-epoch training on the **full** labeled split?

## Hypothesis

*(Inferred.)* The extended config on the full labeled split is the
single best-performing experiment in the suite: more data than the small-split
`cifar10_extended` plus high capacity and regularization (dropout 0.25, weight
decay 1e-4) yields the highest top-1 test accuracy, and the larger training set
lets the high-capacity model generalize without the overfitting risk it carries
on the small split.

## Requirements

- **Data:** `cifar10_labeled_split` (full labeled split).
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training`.
- **Compute budget:** the largest run in the suite — full training set × 50
  epochs × high capacity. Budget tens of minutes (CPU) / a meaningful GPU slice;
  set a hard cap before starting.

## Validation

- **Metric:** top-1 accuracy on the full held-out labeled test partition.
- **Baseline:** small-split `cifar10_extended` (same model, less data) and
  full-split `cifar10_quick_full` (same data, less capacity/epochs).
- **Confirms the hypothesis if:** test accuracy > both baselines — the best
  number in the suite.
- **Refutes the hypothesis if:** accuracy ≤ small-split `cifar10_extended`
  (more data not helping at this capacity) or ≤ `cifar10_quick_full` (the
  capacity/epochs not paying off) — investigate LR schedule / convergence.
- **Inconclusive if:** the gain over `cifar10_quick_full` is within run-to-run
  noise — re-run at multiple seeds.

## Analysis plan

Multi-run comparison via `/deriva-ml:compare-model-runs`: against
`cifar10_quick_full` (the `quick_vs_extended_full` multirun, isolating
capacity/epochs at full data scale) and against small-split `cifar10_extended`
(isolating data scale at fixed model).

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(now authored — model_config
  `cifar10_extended`: 64→128 ch, 256 hidden, dropout 0.25, wd 1e-4, lr 1e-3,
  50 epochs, batch 64)*.
- **Dataset design:** `cifar10-labeled-split` *(now authored — full split)*.

## Status & links

- **Config:** `cifar10_extended_full` in `src/configs/experiments.py`
  (model_config `cifar10_extended` × datasets `cifar10_labeled_split`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
