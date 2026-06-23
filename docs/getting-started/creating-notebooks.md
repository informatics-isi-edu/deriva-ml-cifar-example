# Creating a New Notebook

This guide walks you through adding a new analysis notebook to your DerivaML project.

## Overview

Adding a new notebook involves three steps:

1. **Create the configuration** - Define parameters for your notebook
2. **Create the notebook** - Use `run_notebook()` for initialization
3. **Test and run** - Verify it works end-to-end

## Step 1: Create the Configuration

Create a configuration file in `src/configs/` for your notebook.

### Simple Notebook (Standard Parameters Only)

If your notebook only needs the standard fields (`assets`, `datasets`, `deriva_ml`):

```python
# src/configs/my_analysis.py
"""Configuration for my analysis notebook.

This module registers the configuration for the my_analysis notebook.
"""
from deriva_ml.execution import notebook_config

notebook_config(
    "my_analysis",
    defaults={
        "assets": "my_assets",      # Which asset group to use
        "datasets": "my_datasets",   # Which dataset group to use
    },
    description="My analysis notebook",
)
```

### Notebook with Custom Parameters

If your notebook needs additional configuration options:

```python
# src/configs/my_analysis.py
"""Configuration for my analysis notebook with custom parameters."""
from dataclasses import dataclass

from deriva_ml.execution import BaseConfig, notebook_config


@dataclass
class MyAnalysisConfig(BaseConfig):
    """Configuration for my analysis notebook.

    Attributes:
        threshold: Confidence threshold for predictions.
        show_plots: Whether to display plots inline.
        output_format: Format for output files ('csv' or 'json').
    """
    threshold: float = 0.5
    show_plots: bool = True
    output_format: str = "csv"


notebook_config(
    "my_analysis",
    config_class=MyAnalysisConfig,
    defaults={"assets": "my_assets"},
    description="My analysis with custom parameters",
)
```

### Configuration Tips

- **Use descriptive field names** that make the configuration self-documenting
- **Provide sensible defaults** so notebooks run without extra configuration
- **Add docstrings** to explain what each parameter controls
- **Match defaults to asset/dataset groups** you've defined in `assets.py` and `datasets.py`

## Step 2: Create the Notebook

Create your notebook in the `notebooks/` directory. Name the file to match the
config you registered in step 1 — DerivaML's convention is that notebook
`X.ipynb` uses `notebook_config("X", ...)`. For this example, save the notebook
as `notebooks/my_analysis.ipynb`.

Use the `run_notebook()` API for initialization:

```python
# Cell 1: Initialization
from deriva_ml.execution import run_notebook

# Initialize with one call - this handles:
# - Loading configuration
# - Connecting to DerivaML
# - Creating workflow and execution
# - Downloading input datasets
#
# run_notebook() derives the config name from the notebook's filename
# (my_analysis.ipynb -> "my_analysis"). Pass an explicit name (e.g.
# run_notebook("my_analysis")) only when the notebook filename and
# config name diverge.
ml, execution, config = run_notebook()

# Access configuration values
print(f"Threshold: {config.threshold}")
print(f"Show plots: {config.show_plots}")
print(f"Output format: {config.output_format}")

# Access standard fields
print(f"Assets: {config.assets}")
print(f"Datasets: {config.datasets}")
```

```python
# Cell 2: Your analysis code
import pandas as pd

# Access downloaded datasets
for dataset in execution.datasets:
    print(f"Processing dataset: {dataset.rid}")
    # Dataset files are already downloaded to execution.execution_working_dir

# Use configuration values
if config.threshold > 0:
    # Apply threshold filtering...
    pass

if config.show_plots:
    # Display plots...
    pass
```

```python
# Cell 3: Save outputs
# Register output files for upload
output_path = execution.asset_file_path("Execution_Metadata", f"results.{config.output_format}")

# Write your results
# results_df.to_csv(output_path)

print(f"Results saved to: {output_path}")
```

```python
# Cell 4: Commit (final cell)
# Commit all registered output assets to the catalog
execution.commit_output_assets()
print("Outputs committed successfully!")
```

## Step 3: Test Your Notebook

### Interactive Testing

1. Open the notebook in JupyterLab:
   ```bash
   uv run jupyter lab
   ```

2. Select your repository's kernel (e.g., `your-repo-name`)

3. Run cells interactively to debug

### Command Line Testing

```bash
# Show available configs
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --list-configs

# Dry run (if supported by your notebook)
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
  dry_run=true

# Full run with default configuration
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb

# Run with overrides
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
  threshold=0.8 show_plots=false
```

## Common Patterns

### Accessing Dataset Files

```python
# Datasets are downloaded during run_notebook()
# Access them via execution
for dataset_spec in execution.datasets:
    dataset_dir = execution.execution_working_dir / "datasets" / dataset_spec.rid
    # Process files in dataset_dir
```

### Accessing Input Assets

```python
# Input assets are specified in config.assets
for asset_rid in config.assets:
    # Download or reference the asset
    pass
```

### Registering Multiple Output Types

```python
# Register different output types
plots_path = execution.asset_file_path("Image", "analysis_plot.png")
data_path = execution.asset_file_path("Execution_Metadata", "results.csv")
model_path = execution.asset_file_path("Model", "trained_model.pt")

# Save your files to these paths...
```

### Conditional Execution

```python
# Use config values to control behavior
if config.show_plots:
    import matplotlib.pyplot as plt
    # Generate plots...
    plt.savefig(plots_path)

if config.output_format == "json":
    results.to_json(data_path)
else:
    results.to_csv(data_path)
```

## Complete Example

This template ships a generic notebook config scaffold (`src/configs/analysis.py`)
but no executed notebook. For a complete, runnable analysis notebook — one that
uses custom configuration parameters, processes probability files from model
outputs, generates ROC curves, and uploads analysis results to the catalog — see
the separate worked-example repo
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).

## Checklist

Before running your notebook in production:

- [ ] Configuration file created in `src/configs/`
- [ ] Notebook uses `run_notebook()` for initialization
- [ ] All outputs registered with `execution.asset_file_path()`
- [ ] Final cell calls `execution.commit_output_assets()`
- [ ] Notebook runs start-to-finish without intervention
- [ ] Commit all changes to Git
- [ ] Verify configuration with `--list-configs` (the config menu) and `--cfg job` (the resolved config)
- [ ] Strip output cells before committing (`nbstripout` should handle this)
