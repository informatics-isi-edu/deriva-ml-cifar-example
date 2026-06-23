# Configuration Groups

This guide details each configuration group and how to customize them.

## DerivaML Connection (`deriva_ml`)

**File:** `src/configs/deriva.py`

**Purpose:** Define catalog connection settings.

```python
from hydra_zen import store
from deriva_ml.core.config import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# Local development server
deriva_store(
    DerivaMLConfig(
        hostname="localhost",
        catalog_id="45",
    ),
    name="local",
)

# Production server
deriva_store(
    DerivaMLConfig(
        hostname="<hostname>",
        catalog_id="2",
    ),
    name="eye_ai",
)

# REQUIRED: default configuration
deriva_store(
    DerivaMLConfig(
        hostname="localhost",
        catalog_id="45",
    ),
    name="default_deriva",
)
```

**Usage:**
```bash
uv run deriva-ml-run deriva_ml=eye_ai
```

## Datasets (`datasets`)

**File:** `src/configs/datasets.py`

**Purpose:** Define input dataset specifications.

```python
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

datasets_store = store(group="datasets")

# Training dataset
training = [
    DatasetSpecConfig(
        rid="ABC1",
        version="1.0.0",
        materialize=True,  # Download files
    ),
]

# Testing dataset
testing = [
    DatasetSpecConfig(rid="ABC2", version="2.0.0"),
]

# Multiple datasets
combined = [
    DatasetSpecConfig(rid="ABC1", version="1.0.0"),
    DatasetSpecConfig(rid="ABC2", version="2.0.0"),
]

# Register configurations
datasets_store(training, name="training")
datasets_store(testing, name="testing")
datasets_store(combined, name="combined")
datasets_store(training, name="default_dataset")  # REQUIRED
```

**DatasetSpecConfig Options:**

| Field | Type | Description |
|-------|------|-------------|
| `rid` | str | Dataset RID (required) |
| `version` | str | Version string (e.g., "1.0.0") |
| `materialize` | bool | Download asset files (default: True) |
| `description` | str | Human-readable description |

**Usage:**
```bash
uv run deriva-ml-run datasets=testing
```

## Assets (`assets`)

**File:** `src/configs/assets.py`

**Purpose:** Define input assets like model weights or configuration files.

```python
from hydra_zen import store
from deriva_ml.execution import with_description

assets_store = store(group="assets")

# Plain RID strings
assets_store(
    with_description(
        ["XYZ1", "XYZ2"],
        "Model weights and config file.",
    ),
    name="pretrained",
)

# For large files with caching
from deriva_ml.asset.aux_classes import AssetSpecConfig
assets_store(
    with_description(
        [AssetSpecConfig(rid="XYZ1", cache=True)],
        "Large model weights, cached locally.",
    ),
    name="cached_weights",
)

# REQUIRED: default_asset (plain list, no with_description)
assets_store([], name="default_asset")
```

**Usage:**
```bash
uv run deriva-ml-run assets=pretrained
```

## Model Configuration (`model_config`)

**File:** `src/configs/<model_name>.py`

**Purpose:** Define model hyperparameters and variants.

```python
from hydra_zen import builds, store
from models.my_model import my_model

# Build base configuration
MyModelConfig = builds(
    my_model,
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    populate_full_signature=True,
    zen_partial=True,
)

model_store = store(group="model_config")

# Register variants
model_store(MyModelConfig, name="default_model")  # REQUIRED
model_store(MyModelConfig, epochs=3, name="quick")
model_store(MyModelConfig, epochs=50, name="extended")
model_store(MyModelConfig, learning_rate=1e-2, name="fast_lr")
```

**Usage:**
```bash
# Use a variant
uv run deriva-ml-run model_config=quick

# Override inline
uv run deriva-ml-run model_config.epochs=25
```

## Workflow (`workflow`)

**File:** `src/configs/workflow.py`

**Purpose:** Define workflow metadata for provenance tracking.

```python
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

MyWorkflow = builds(
    Workflow,
    name="Model Run",
    workflow_type=["Training", "Image Classification"],
    description="Train a model on my dataset.",
    populate_full_signature=True,
)

workflow_store = store(group="workflow")
workflow_store(MyWorkflow, name="default_workflow")
```

## Required Defaults

Each configuration group **must** have a default configuration. The naming convention is:

| Group | Default Name |
|-------|--------------|
| `deriva_ml` | `default_deriva` |
| `datasets` | `default_dataset` |
| `assets` | `default_asset` |
| `model_config` | `default_model` |
| `workflow` | `default_workflow` |

If a default is missing, Hydra will fail with a composition error.
