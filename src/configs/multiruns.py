"""Multirun configurations for experiment sweeps.


This module defines named multirun configurations that bundle together Hydra
overrides and rich markdown descriptions. Use these to run reproducible
experiment sweeps with full documentation.

Usage:
    # Run a defined multirun (no --multirun flag needed)
    uv run deriva-ml-run +multirun=quick_vs_extended

    # Override parameters from the multirun config
    uv run deriva-ml-run +multirun=lr_sweep model_config.epochs=5

    # Show available multiruns
    uv run deriva-ml-run --info

Benefits:
    - Explicit declaration of sweep experiments
    - Rich markdown descriptions for parent executions
    - Reproducible sweeps documented in code
    - Same Hydra override syntax as command line
    - No need to remember --multirun flag
"""

from deriva_ml.execution import multirun_config
from configs.multirun_descriptions import (
    QUICK_VS_EXTENDED_DESCRIPTION,
    FULL_DATASET_DESCRIPTION,
    LEARNING_RATE_SWEEP_DESCRIPTION,
    EPOCH_SWEEP_DESCRIPTION,
)

# =============================================================================
# Experiment Comparisons
# =============================================================================
# Compare different experiment configurations

multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=cifar10_quick,cifar10_extended",
    ],
    description=QUICK_VS_EXTENDED_DESCRIPTION,
)

multirun_config(
    "quick_vs_extended_full",
    overrides=[
        "+experiment=cifar10_quick_full,cifar10_extended_full",
    ],
    description=FULL_DATASET_DESCRIPTION,
)

# =============================================================================
# Hyperparameter Sweeps
# =============================================================================
# Sweep over parameter ranges to find optimal values.
# Built on existing experiments with parameter overrides.

multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=cifar10_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description=LEARNING_RATE_SWEEP_DESCRIPTION,
)

multirun_config(
    "epoch_sweep",
    overrides=[
        "+experiment=cifar10_extended",
        "model_config.epochs=5,10,25,50",
    ],
    description=EPOCH_SWEEP_DESCRIPTION,
)

# =============================================================================
# Grid Searches
# =============================================================================
# Sweep multiple parameters simultaneously (creates N*M runs)

multirun_config(
    "lr_batch_grid",
    overrides=[
        "+experiment=cifar10_quick",
        "model_config.epochs=10",
        "model_config.learning_rate=0.001,0.01",
        "model_config.batch_size=64,128",
    ],
    description="""## Learning Rate and Batch Size Grid Search

**Objective:** Find optimal combination of learning rate and batch size.

### Parameter Grid

| Parameter | Values |
|-----------|--------|
| Learning Rate | 0.001, 0.01 |
| Batch Size | 64, 128 |

**Total runs:** 4 (2 x 2 grid)

### Expected Outcomes

- Smaller batch sizes may need lower learning rates
- Larger batch sizes can often tolerate higher learning rates
- Look for the combination with best test accuracy and stable training
""",
)
