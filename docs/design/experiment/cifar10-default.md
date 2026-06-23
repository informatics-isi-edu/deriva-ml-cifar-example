# Experiment Design: Standard balanced training on the small training set

**Slug:** cifar10-default
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

> **Reverse-engineered.** Reconstructed after the config existed, from config
> parameters and `Experiments.md`. Goal/Requirements recovered from the config;
> **Hypothesis** and **Validation** thresholds are *inferred*.

## Goal

What test accuracy does the balanced "default everything" configuration (10
epochs, standard hyperparameters) reach on the small training set? This is the
project's reference standard run.

## Hypothesis

*(Inferred.)* The default config (32→64 ch, 128 hidden, lr 1e-3, 10 epochs,
batch 64, seed 42) trains stably and reaches a modest-but-clear top-1 test
accuracy (well above random; roughly in the 30–50% band at this scale), serving
as the baseline other configs are judged against.

## Requirements

- **Data:** `cifar10_small_training` (training-only subsample; **note:** no
  labeled held-out test partition is bundled in this dataset — see Validation).
- **Assets:** none (train from scratch).
- **Vocabularies:** `Workflow_Type` term `Training`.
- **Compute budget:** small — a few minutes on CPU; at most 1–2 cycles.

## Validation

- **Metric:** top-1 accuracy. **Caveat:** `cifar10_small_training` is a
  training-only dataset; rigorous held-out evaluation requires a labeled test
  partition (the `*_labeled_split` family). For a directly-comparable evaluated
  baseline, prefer `cifar10_small_default` (same model, on the labeled split).
- **Baseline:** 10% random guess.
- **Confirms the hypothesis if:** the run completes and training loss decreases
  monotonically to convergence, with train accuracy clearly above random.
- **Refutes the hypothesis if:** training diverges or loss plateaus at chance.
- **Inconclusive if:** no labeled test partition is available to compute a
  generalization number — switch to `cifar10_small_default` for that.

## Analysis plan

Single-run read of the training log / recorded feature values. For an
apples-to-apples *test* accuracy comparable to the capacity sweep, use
`cifar10_small_default` instead (identical model on the labeled split).

## Upstream designs

- **Model design:** `cifar10-2layer-cnn` *(not yet authored — model_config
  `default_model`: 32→64 ch, 128 hidden, dropout 0.0, lr 1e-3, 10 epochs, batch 64)*.
- **Dataset design:** `cifar10-small-training` *(not yet authored — stratified
  training-only subsample, no bundled labeled test partition)*.

## Status & links

- **Config:** `cifar10_default` in `src/configs/experiments.py`
  (model_config `default_model` × datasets `cifar10_small_training`).
- **Executions:** (none recorded yet)
- **tacit-knowledge.md:** (none yet)
