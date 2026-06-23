# Experiment Design: Capacity-sweep midpoint (default capacity) on the small labeled split

**Slug:** cifar10-small-default
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

> **Reverse-engineered.** Reconstructed after the config existed. Goal/Requirements
> recovered from the config and the capacity-sweep rationale documented in the
> `src/configs/experiments.py` block comment; **Validation** thresholds are
> *inferred*.

## Goal

What is the test accuracy of the *default-capacity* model (32→64 ch, 10 epochs)
on the small labeled split — the middle data point of a clean
capacity-vs-accuracy sweep that holds the dataset and seed constant?

## Hypothesis

*(Inferred.)* Holding the dataset and seed fixed and varying only model
capacity + training duration, the default-capacity / 10-epoch run lands
*between* the low end (`cifar10_quick`: 32→64 ch, 3 epochs) and the high end
(`cifar10_small_large`: 64→128 ch, 20 epochs) in top-1 test accuracy, producing
a monotone capacity-vs-accuracy curve.

## Requirements

- **Data:** `cifar10_small_labeled_split` — **the same dataset and version** as
  `cifar10_quick`, `cifar10_small_large` (and `cifar10_extended`). Holding this
  constant is the whole point of the sweep.
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training`.
- **Compute budget:** small — 10 epochs at default capacity; a few minutes CPU.
- **Reproducibility:** `seed=42` (config default) shared across the sweep so
  accuracy differences are attributable to capacity, not RNG.

## Validation

- **Metric:** top-1 accuracy on the held-out labeled test partition.
- **Baseline:** the other two sweep points on the identical split
  (`cifar10_quick`, `cifar10_small_large`).
- **Confirms the hypothesis if:** `quick` ≤ `small_default` ≤ `small_large` in
  test accuracy (monotone with capacity/epochs).
- **Refutes the hypothesis if:** the ordering inverts (e.g. `small_default` ≥
  `small_large`) — signals saturation or overfitting at the small data scale.
- **Inconclusive if:** the three points fall within run-to-run noise of each
  other — re-run each at 2–3 seeds to separate signal from variance.

## Analysis plan

Multi-run comparison via `/deriva-ml:compare-model-runs` across the three (or
four, incl. `cifar10_extended`) small-labeled-split runs, plotting capacity
against test accuracy. This experiment is the **middle** point.

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(not yet authored — model_config
  `default_model`: 32→64 ch, 128 hidden, dropout 0.0, lr 1e-3, 10 epochs, batch 64)*.
- **Dataset design:** `cifar10-small-labeled-split` *(not yet authored)*.

## Status & links

- **Config:** `cifar10_small_default` in `src/configs/experiments.py`
  (model_config `default_model` × datasets `cifar10_small_labeled_split`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
