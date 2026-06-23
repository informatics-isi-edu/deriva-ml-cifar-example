# Experiment Design: Quick baseline validation on the full labeled split

**Slug:** cifar10-quick-full
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

> **Reverse-engineered.** Reconstructed after the config existed. Goal/Requirements
> recovered from the config; **Hypothesis** and **Validation** thresholds are
> *inferred*.

## Goal

Does the quick configuration (3 epochs, 32→64 ch) produce a sensible baseline
when run on the **full** labeled split rather than the small one — and how much
does the larger training set change the result at fixed (low) capacity/epochs?

## Hypothesis

*(Inferred.)* Same model as `cifar10_quick`, but on the full labeled split:
more training data at the same 3-epoch budget yields a baseline top-1 test
accuracy at least as high as the small-split `cifar10_quick`, demonstrating the
data-scale axis (full vs small) independent of the capacity axis.

## Requirements

- **Data:** `cifar10_labeled_split` (full labeled split, leak-free, both
  partitions labeled).
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training`.
- **Compute budget:** larger than the small-split quick run (full training set,
  still only 3 epochs) — minutes on CPU.

## Validation

- **Metric:** top-1 accuracy on the full held-out labeled test partition.
- **Baseline:** the small-split `cifar10_quick` run (same model, smaller data)
  and 10% random guess.
- **Confirms the hypothesis if:** test accuracy > 10% AND ≥ the small-split
  `cifar10_quick` accuracy (more data helps, or at least doesn't hurt, at fixed
  capacity).
- **Refutes the hypothesis if:** accuracy < small-split quick — would suggest
  3 epochs is too few to exploit the larger set, or an LR/batch mismatch.
- **Inconclusive if:** within run-to-run noise of the small-split quick run.

## Analysis plan

Two-run comparison (`/deriva-ml:compare-model-runs`) against the small-split
`cifar10_quick`, isolating the data-scale effect. Paired with
`cifar10_extended_full` as the `quick_vs_extended_full` multirun.

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(not yet authored — model_config
  `cifar10_quick`: 32→64 ch, 128 hidden, dropout 0.0, lr 1e-3, 3 epochs, batch 128)*.
- **Dataset design:** `cifar10-labeled-split` *(not yet authored — full,
  leak-free, both-partitions-labeled split)*.

## Status & links

- **Config:** `cifar10_quick_full` in `src/configs/experiments.py`
  (model_config `cifar10_quick` × datasets `cifar10_labeled_split`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
