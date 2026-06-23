"""Multirun descriptions for multi-experiment runs.

Rich markdown descriptions for the parent execution when running multiple
experiments together. Each description documents why experiments are being
compared and what outcomes are expected.

Usage:
    # Run a pre-defined multirun
    uv run deriva-ml-run +multirun=quick_vs_extended
"""

# =============================================================================
# Sweep Descriptions
# =============================================================================
# These are standalone markdown descriptions that can be used with any multirun.
# They are defined here as Python strings for easy reference and documentation.

QUICK_VS_EXTENDED_DESCRIPTION = """## CIFAR-10 CNN Multi-Experiment Comparison

**Objective:** Compare model performance across two training configurations to evaluate
the trade-off between training speed and model accuracy.

### Experiments

| Experiment | Epochs | Architecture | Regularization | Dataset |
|------------|--------|--------------|----------------|---------|
| `cifar10_quick` | 3 | 32→64 channels, 128 hidden | None | Small Split (1,000 images) |
| `cifar10_extended` | 50 | 64→128 channels, 256 hidden | Dropout 0.25, Weight Decay 1e-4 | Small Split (1,000 images) |

### Configuration Details

**cifar10_quick** - Fast validation baseline
- Conv1: 32 channels → Conv2: 64 channels
- Hidden layer: 128 units
- Batch size: 128
- Learning rate: 1e-3
- No regularization

**cifar10_extended** - Production-quality training
- Conv1: 64 channels → Conv2: 128 channels
- Hidden layer: 256 units
- Batch size: 64
- Learning rate: 1e-3
- Dropout: 0.25
- Weight decay: 1e-4

### Expected Outcomes

- The quick model should train in under 1 minute but have low accuracy
- The extended model should achieve higher accuracy but may overfit on the small dataset
- This comparison helps validate the training pipeline before running on full data
"""

FULL_DATASET_DESCRIPTION = """## CIFAR-10 Full Dataset Comparison

**Objective:** Evaluate model architectures on the complete CIFAR-10 dataset
(10,000 images) to get realistic performance estimates.

### Experiments

| Experiment | Epochs | Architecture | Dataset |
|------------|--------|--------------|---------|
| `cifar10_quick_full` | 3 | 32→64 channels | Full Split (10,000 images) |
| `cifar10_extended_full` | 50 | 64→128 channels | Full Split (10,000 images) |

### Notes

- Full dataset runs take significantly longer
- Extended model should show less overfitting with more training data
- Use this sweep for final model selection before production
"""

# =============================================================================
# Learning Rate Sweep
# =============================================================================
# Explores the effect of learning rate on model convergence and final accuracy

LEARNING_RATE_SWEEP_DESCRIPTION = """## Learning Rate Hyperparameter Sweep

**Objective:** Identify the optimal learning rate for the CIFAR-10 CNN architecture
by comparing training dynamics across a range of values.

### Experiments

| Learning Rate | Expected Behavior |
|--------------|-------------------|
| 0.0001 | Slow convergence, may underfit in limited epochs |
| 0.001 | Standard rate, good balance of speed and stability |
| 0.01 | Fast convergence, risk of overshooting minima |
| 0.1 | Aggressive, likely unstable training |

### Methodology

All experiments use the same:
- Architecture: 32→64 channels, 128 hidden units
- Dataset: Small Split (1,000 images)
- Epochs: 10 (enough to observe convergence behavior)
- Batch size: 128

### Metrics to Compare

- Training loss curve shape
- Final training/test accuracy
- Presence of training instability (loss spikes)
- Generalization gap (train vs test accuracy)

### Command

```bash
uv run deriva-ml-run --multirun \\
    +experiment=cifar10_lr_sweep \\
    model_config.learning_rate=0.0001,0.001,0.01,0.1
```
"""

# =============================================================================
# Epoch Sweep
# =============================================================================
# Explores the effect of training duration on model performance

EPOCH_SWEEP_DESCRIPTION = """## Training Duration (Epochs) Sweep

**Objective:** Analyze how training duration affects model performance and identify
the point of diminishing returns or overfitting onset.

### Experiments

| Epochs | Expected Behavior |
|--------|-------------------|
| 5 | Underfitting, model still learning basic features |
| 10 | Reasonable performance, may still be improving |
| 25 | Good performance, watch for overfitting signs |
| 50 | Extended training, likely overfitting on small dataset |

### Methodology

All experiments use the same:
- Architecture: 64→128 channels, 256 hidden units (extended config)
- Dataset: Small Split (1,000 images)
- Learning rate: 0.001
- Batch size: 64
- Regularization: Dropout 0.25, weight decay 1e-4

### Metrics to Compare

- Training vs test accuracy divergence (overfitting indicator)
- Final test accuracy plateau
- Training time per epoch
- Optimal early stopping point

### Command

```bash
uv run deriva-ml-run --multirun \\
    +experiment=cifar10_epoch_sweep \\
    model_config.epochs=5,10,25,50
```
"""
