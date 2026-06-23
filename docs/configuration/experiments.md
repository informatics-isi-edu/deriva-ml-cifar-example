# Experiment Presets

Experiments allow you to define reusable combinations of configuration choices.

## Creating Experiments

Define experiments in `src/configs/experiments.py`:

```python
from hydra_zen import make_config, store
from configs.base import DerivaModelConfig

experiment_store = store(group="experiment", package="_global_")

# A quick test experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "quick"},
            {"override /datasets": "small_labeled_split"},
        ],
        description="Quick training: 3 epochs, batch size 128",
        bases=(DerivaModelConfig,),
    ),
    name="quick",
)

# Full training run
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "extended"},
            {"override /datasets": "labeled_split"},
            {"override /assets": "pretrained_weights"},
        ],
        description="Full training with extended epochs",
        bases=(DerivaModelConfig,),
    ),
    name="extended",
)
```

## Using Experiments

Run a single experiment:

```bash
uv run deriva-ml-run +experiment=quick
```

Run multiple experiments in sequence (ad-hoc multirun):

```bash
uv run deriva-ml-run --multirun +experiment=quick,extended
```

Override within an experiment:

```bash
uv run deriva-ml-run +experiment=extended model_config.epochs=100
```

## Named Multiruns

For predefined multirun configurations, define them in `src/configs/multiruns.py`:

```python
from deriva_ml.execution import multirun_config

multirun_config(
    "lr_sweep",
    overrides=[
        "+experiment=quick",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1",
    ],
    description="Learning rate sweep",
)

multirun_config(
    "quick_vs_extended",
    overrides=[
        "+experiment=quick,extended",
    ],
    description="Compare quick and extended training",
)
```

Run a named multirun:

```bash
uv run deriva-ml-run +multirun=lr_sweep
```

## Experiment Design Patterns

### Hyperparameter Sweep

```python
# Define model variants first
model_store(MyModelConfig, learning_rate=1e-2, name="lr_high")
model_store(MyModelConfig, learning_rate=1e-3, name="lr_medium")
model_store(MyModelConfig, learning_rate=1e-4, name="lr_low")

# Create experiments for each
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "lr_high"},
        ],
        description="High learning rate experiment",
        bases=(DerivaModelConfig,),
    ),
    name="sweep_lr_high",
)
```

Or use a named multirun for parameter sweeps:
```bash
uv run deriva-ml-run +multirun=lr_sweep
```

### Dataset Comparison

```python
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "dataset_v1"},
            {"override /model_config": "default"},
        ],
        description="Evaluation on dataset v1",
        bases=(DerivaModelConfig,),
    ),
    name="compare_v1",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "dataset_v2"},
            {"override /model_config": "default"},
        ],
        description="Evaluation on dataset v2",
        bases=(DerivaModelConfig,),
    ),
    name="compare_v2",
)
```

### Ablation Study

```python
# Full model
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "full_model"},
            {"override /assets": "all_features"},
        ],
        description="Full model with all features",
        bases=(DerivaModelConfig,),
    ),
    name="ablation_full",
)

# Without feature A
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "no_feature_a"},
            {"override /assets": "no_feature_a"},
        ],
        description="Ablation: without feature A",
        bases=(DerivaModelConfig,),
    ),
    name="ablation_no_a",
)
```

## Best Practices

1. **Name experiments descriptively**: Use names that describe what's different
2. **Document your experiments**: Add a `description` field explaining the purpose
3. **Version your data**: Use specific dataset versions in experiments
4. **Track results**: Each run creates an Execution record in the catalog
5. **Prefer named multiruns** over ad-hoc `--multirun` sweeps for reproducibility

## Multirun Output

When using multiruns, Hydra creates separate output directories:

```
outputs/
├─ 2024-01-15/
│  ├─ 10-30-00/          # First experiment
│  │  ├─ .hydra/
│  │  └─ output.log
│  └─ 10-35-00/          # Second experiment
│     ├─ .hydra/
│     └─ output.log
```

Each run gets its own Execution record in the DerivaML catalog with full provenance tracking.
