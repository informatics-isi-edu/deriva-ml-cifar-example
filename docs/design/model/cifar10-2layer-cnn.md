# Model Design: 2-layer CNN for CIFAR-10 classification

**Slug:** cifar10-2layer-cnn
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

## Goal

A small 2-layer convolutional network that classifies 32×32 RGB CIFAR-10
images into the 10 canonical classes — the project's single, canonical model,
used as the end-to-end example of integrating PyTorch with DerivaML. All eight
experiments run *this* model; they differ only in its `model_config`
hyperparameters and the dataset they consume.

## Requirements

The source the `model_config` group is derived from:

- **Architecture (`SimpleCNN`):**
  `Conv2d(3→conv1, 3×3, pad 1) → ReLU → MaxPool(2×2)` →
  `Conv2d(conv1→conv2, 3×3, pad 1) → ReLU → MaxPool(2×2)` →
  `Flatten → Linear(conv2·8·8 → hidden) → ReLU → Dropout → Linear(hidden → 10)`.
  Two 2×2 pools take 32×32 → 8×8.
- **Hyperparameters (the `model_config` knobs, with the `default_model` defaults):**
  `conv1_channels=32`, `conv2_channels=64`, `hidden_size=128`,
  `dropout_rate=0.0`, `learning_rate=1e-3`, `epochs=10`, `batch_size=64`,
  `weight_decay=0.0`, `seed=42`. Optimizer = Adam; loss = cross-entropy.
  The eight registered presets (`default_model`, `cifar10_quick`,
  `cifar10_large`, `cifar10_regularized`, `cifar10_fast_lr`, `cifar10_slow_lr`,
  `cifar10_extended`, `cifar10_test_only`) vary these knobs.
- **`seed=42`** matches the split seed in `_cifar10_datasets.py` so
  "default everything" runs are byte-reproducible (seeds weight init, shuffle
  order, and numpy/random).
- **Input features (consumed):** the `Image_Classification` feature on `Image`
  — specifically its `Image_Class` term, used as the *ground-truth training
  target* (mapped to a class index via `CIFAR10_CLASS_TO_IDX`). See the
  `image-classification` feature design.
- **Output features (produced):** the *same* `Image_Classification` feature —
  the model writes predicted `Image_Class` + `Confidence` rows on the test
  partition (via `record_predictions`). This is a dual-purpose feature (ground
  truth in, predictions out); the feature design records the convention. Output
  features are listed here but NOT under Upstream designs (that would cycle).
- **Output assets:** `cifar10_cnn_weights.pt` (weights + optimizer state +
  config + per-epoch log), `training_log.txt`, `prediction_probabilities.csv`,
  and (test-only) `evaluation_results.txt` — all written as `Execution_Asset`.
- **Input assets (test-only mode):** a weights asset containing
  `cifar10_cnn_weights.pt`, supplied via the `assets=` config group.

## Validation

- **Metric:** top-1 accuracy on the held-out labeled test partition. The
  module docstring states an expected **~60–70%** with default parameters on
  full CIFAR-10; at the small-split / few-epoch scale used in most experiments,
  expect well below that but clearly above the 10% random baseline.
- **Validated on:** the test partition of the labeled-split family
  (`cifar10_small_labeled_split` / `cifar10_labeled_split`).
- **Sanity checks (enforced or observable in code):** training loss decreases
  toward convergence; no NaN; softmax probabilities sum to 1 (used directly as
  `Confidence`); unlabeled rows (label `-1`) are skipped in loss/accuracy via
  `ignore_index=-1`; an *emission-time* accuracy is recomputed and printed
  alongside recorded predictions to catch CSV-vs-log desync.

## Upstream designs

- **Input feature design:** `image-classification` (the `Image_Class` ground
  truth this model trains on). *Inputs only here* — the model's own prediction
  output is recorded under Requirements → Output features, not as a dependency.
- Extends no prior model (single canonical architecture).

## Status & links

- **Model file + config groups:** `src/models/cifar10_cnn.py` (`cifar10_cnn`,
  `SimpleCNN`); `model_config` group in `src/configs/cifar10_cnn.py` (8 presets).
- **Workflow:** `cifar10_cnn` workflow in `src/configs/workflow.py`.
- **Consumed by experiment designs:** all 8 in `docs/design/experiment/`.
- **tacit-knowledge.md:** (none yet — link the first training run's entry here)
