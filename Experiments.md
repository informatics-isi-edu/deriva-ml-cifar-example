# Experiments

Canonical registry of all defined experiments and multiruns. Keep this file
in sync with `src/configs/experiments.py` and `src/configs/multiruns.py`.

## Quick Reference

| Experiment | Model Config | Dataset | Description |
|------------|-------------|---------|-------------|
| `cifar10_quick` | `cifar10_quick` | `cifar10_small_labeled_split` | 3 epochs, fast validation |
| `cifar10_default` | `default_model` | `cifar10_small_training` | 10 epochs, standard |
| `cifar10_extended` | `cifar10_extended` | `cifar10_small_labeled_split` | 50 epochs, best accuracy |
| `cifar10_small_default` | `default_model` | `cifar10_small_labeled_split` | 10 epochs, capacity-sweep middle |
| `cifar10_small_large` | `cifar10_large` | `cifar10_small_labeled_split` | 20 epochs, capacity-sweep high end |
| `cifar10_quick_full` | `cifar10_quick` | `cifar10_labeled_split` | 3 epochs, full dataset |
| `cifar10_extended_full` | `cifar10_extended` | `cifar10_labeled_split` | 50 epochs, full dataset |
| `cifar10_test_only` | `cifar10_test_only` | `cifar10_small_labeled_testing` | Inference only |

## Multiruns

| Multirun | Overrides | Description |
|----------|----------|-------------|
| `quick_vs_extended` | `+experiment=cifar10_quick,cifar10_extended` | Compare 3-epoch vs 50-epoch |
| `quick_vs_extended_full` | `+experiment=cifar10_quick_full,cifar10_extended_full` | Full dataset comparison |
| `lr_sweep` | `+experiment=cifar10_quick`, lr=0.0001..0.1 | Learning rate grid search |
| `epoch_sweep` | `+experiment=cifar10_extended`, epochs=5..50 | Training duration sweep |
| `lr_batch_grid` | `+experiment=cifar10_quick`, lr x batch_size | 2x2 grid search |

---

## Experiment Details

### `cifar10_quick`

- **Config group overrides**: `model_config=cifar10_quick`, `datasets=cifar10_small_labeled_split`
- **Parameters**: 3 epochs, 32->64 channels, batch 128, lr=1e-3
- **Purpose**: Fast validation of training pipeline

### `cifar10_default`

- **Config group overrides**: `model_config=default_model`, `datasets=cifar10_small_training`
- **Parameters**: 10 epochs, 32->64 channels, batch 64, lr=1e-3
- **Purpose**: Standard balanced training

### `cifar10_extended`

- **Config group overrides**: `model_config=cifar10_extended`, `datasets=cifar10_small_labeled_split`
- **Parameters**: 50 epochs, 64->128 channels, dropout 0.25, weight decay 1e-4
- **Purpose**: Best accuracy, production-quality training

### `cifar10_small_default`

- **Config group overrides**: `model_config=default_model`, `datasets=cifar10_small_labeled_split`
- **Parameters**: 10 epochs, 32->64 channels, batch 64, lr=1e-3
- **Purpose**: Middle point of the capacity sweep on the small labeled split (same dataset/seed as `cifar10_quick` and `cifar10_small_large`, varying only model capacity)

### `cifar10_small_large`

- **Config group overrides**: `model_config=cifar10_large`, `datasets=cifar10_small_labeled_split`
- **Parameters**: 20 epochs, 64->128 channels, 256 hidden units
- **Purpose**: High end of the capacity sweep on the small labeled split — directly comparable to `cifar10_quick` and `cifar10_small_default` (same dataset, same held-out test partition)

### `cifar10_quick_full`

- **Config group overrides**: `model_config=cifar10_quick`, `datasets=cifar10_labeled_split`
- **Parameters**: Same as `cifar10_quick` but on full 10,000-image dataset
- **Purpose**: Baseline validation on full data

### `cifar10_extended_full`

- **Config group overrides**: `model_config=cifar10_extended`, `datasets=cifar10_labeled_split`
- **Parameters**: Same as `cifar10_extended` but on full 10,000-image dataset
- **Purpose**: Best accuracy on full data

### `cifar10_test_only`

- **Config group overrides**: `model_config=cifar10_test_only`, `datasets=cifar10_small_labeled_testing`
- **Parameters**: No training; loads pre-trained weights and evaluates
- **Purpose**: Inference-only evaluation of existing checkpoints

---

## Adding New Experiments

1. Define the experiment in `src/configs/experiments.py`
2. Document it in this file
3. Test with `uv run deriva-ml-run +experiment=<name> dry_run=true`
4. Commit both the code and this file together
