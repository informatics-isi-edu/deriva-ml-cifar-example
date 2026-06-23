# Creating a New Model

This guide walks you through adding a new model to your DerivaML project step by step.

## Overview

Adding a new model involves three steps:

1. **Create the model function** - Your ML code with DerivaML integration
2. **Create the configuration** - Hydra-zen config defining parameters and variants
3. **Update the entry point** - Connect your model to the Hydra CLI

## Step 1: Create the Model Function

Create a new file in `src/models/` for your model. The function must follow this signature pattern:

```python
# src/models/my_model.py
"""My custom model implementation."""

from deriva_ml import DerivaML
from deriva_ml.execution import Execution


def my_model(
    # Your model parameters
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 128,
    # Framework-injected parameters (always include these)
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    """Train my custom model.

    Args:
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        hidden_size: Size of hidden layers.
        ml_instance: DerivaML instance (injected by framework).
        execution: Execution context (injected by framework).
    """
    # Access execution configuration
    if execution:
        # Get input datasets
        datasets = execution.datasets
        for ds in datasets:
            print(f"Input dataset: {ds.rid} version {ds.version}")

        # Download and access dataset files
        for dataset_spec in datasets:
            execution.download_execution_dataset(dataset_spec.rid)

        # Get working directory for outputs
        working_dir = execution.execution_working_dir
        print(f"Working directory: {working_dir}")

    # Your training code here
    print(f"Training with lr={learning_rate}, epochs={epochs}")

    # Register output files for upload
    if execution:
        # Register a model checkpoint
        model_path = execution.asset_file_path("Model", "model.pt")
        # Save your model to model_path...

        # Register metrics
        metrics_path = execution.asset_file_path("Execution_Metadata", "metrics.json")
        # Write metrics to metrics_path...

    print("Training complete!")
```

### Key Points

- **Parameter order matters**: Put your model parameters first, then `ml_instance` and `execution`
- **Use `execution.download_execution_dataset()`** to download input datasets
- **Use `execution.asset_file_path()`** to register output files for upload
- **Check `if execution:`** for code that depends on the execution context (allows dry runs)

## Step 2: Create the Configuration

Create a configuration file in `src/configs/` that defines your model's parameters and variants:

```python
# src/configs/my_model.py
"""Configuration for my custom model.

Configuration Group: model_config
---------------------------------
Defines hyperparameters and variants for my_model.

REQUIRED: A configuration named "default_model" or a new default must be set.
"""
from hydra_zen import builds, store

from models.my_model import my_model

# Build the base configuration
# - populate_full_signature=True: Include all function parameters
# - zen_partial=True: Create a partial function (deferred execution)
MyModelConfig = builds(
    my_model,
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    hidden_size=128,
    populate_full_signature=True,
    zen_partial=True,
)

# Create a store for this configuration group
model_store = store(group="model_config")

# Register the default configuration
model_store(MyModelConfig, name="my_model_default")

# Register variants (override specific parameters)
model_store(MyModelConfig, epochs=3, name="my_model_quick")
model_store(MyModelConfig, epochs=50, hidden_size=256, name="my_model_extended")
model_store(MyModelConfig, learning_rate=1e-2, name="my_model_fast_lr")
model_store(MyModelConfig, learning_rate=1e-4, epochs=30, name="my_model_slow_lr")
```

### Configuration Tips

- **Use `builds()`** to create a structured config from your function
- **Use `zen_partial=True`** for deferred execution (the function is called later by the runner)
- **Register variants** by overriding specific parameters in `store()` calls
- **Name conventions**: Use descriptive names like `quick`, `extended`, `fast_lr`

## Step 3: Update Defaults (Optional)

If you want your new model to be the default, update `src/configs/base.py`:

```python
# In src/configs/base.py, update the hydra_defaults:
DerivaModelConfig = create_model_config(
    DerivaML,
    description="Simple model run",
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "my_model_default"},  # Change this
    ],
)
```

Or create an experiment preset in `src/configs/experiments.py`:

```python
experiment_store(
    {
        "model_config": "my_model_extended",
        "datasets": "my_training_data",
        "assets": "pretrained_weights",
    },
    name="my_experiment",
)
```

## Step 4: Run Your Model

```bash
# Run with your model configuration
uv run deriva-ml-run model_config=my_model_default

# Run a quick test
uv run deriva-ml-run model_config=my_model_quick

# Override parameters inline
uv run deriva-ml-run model_config=my_model_default model_config.epochs=25

# Dry run (no catalog writes)
uv run deriva-ml-run model_config=my_model_default dry_run=true

# Run experiment preset
uv run deriva-ml-run +experiment=my_experiment
```

## Complete Example

This template ships only a no-op placeholder model (`example_model` in
`src/configs/model.py`). For a complete, runnable model — one that downloads and
restructures image datasets, trains a CNN with configurable architecture, logs
metrics, saves checkpoints, and tracks all provenance in the catalog — see the
separate worked-example repo
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).

## Checklist

Before running your model in production:

- [ ] Model function follows the signature pattern
- [ ] Configuration file created with default and variants
- [ ] Commit all changes to Git (DerivaML tracks code provenance)
- [ ] Test with `dry_run=true` first
- [ ] Create a version tag with `uv run bump-version patch`
