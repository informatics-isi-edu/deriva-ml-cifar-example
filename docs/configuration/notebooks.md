# Notebook Configuration

Notebooks use a simplified configuration API that handles the common boilerplate automatically.

## The Simplified API

### `notebook_config()`

Registers a notebook configuration with minimal code:

```python
from deriva_ml.execution import notebook_config

notebook_config(
    "my_notebook",                          # Configuration name
    defaults={"assets": "my_assets"},       # Override default groups
    description="My analysis notebook",     # Optional description
)
```

### `run_notebook()`

Initializes a notebook with a single call:

```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook()
```

This single call:
1. Loads all configuration modules
2. Resolves the configuration with Hydra
3. Connects to the DerivaML catalog
4. Creates a workflow and execution
5. Downloads input datasets
6. Returns ready-to-use objects

#### Config name resolution

By DerivaML convention, notebook `X.ipynb` uses the config registered as
`notebook_config("X", ...)`. `run_notebook()` derives the config name from
the calling notebook's filename, so you don't pass it explicitly. The
explicit form is still supported and is the right fallback when the notebook
filename and config name diverge:

```python
# Equivalent to run_notebook() when called from my_notebook.ipynb.
ml, execution, config = run_notebook("my_notebook")
```

Pass an explicit name only when the notebook filename and config name
disagree (rare).

## Configuration Patterns

### Simple Notebook

For notebooks that only use standard fields:

```python
# src/configs/my_analysis.py
from deriva_ml.execution import notebook_config

notebook_config(
    "my_analysis",
    defaults={
        "assets": "probability_files",
        "datasets": "test_results",
    },
)
```

### Notebook with Custom Parameters

For notebooks that need additional configuration:

```python
# src/configs/my_analysis.py
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class MyAnalysisConfig(BaseConfig):
    """Configuration for my analysis.

    Inherits from BaseConfig which provides:
    - deriva_ml: Connection settings
    - datasets: List of DatasetSpecConfig
    - assets: List of asset RIDs
    - dry_run: Whether to skip catalog writes
    """
    # Add your custom fields
    threshold: float = 0.5
    max_iterations: int = 100
    output_format: str = "csv"

notebook_config(
    "my_analysis",
    config_class=MyAnalysisConfig,
    defaults={"assets": "my_assets"},
)
```

### Using the Configuration

```python
# In your notebook (my_analysis.ipynb)
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook()

# Access standard fields (from BaseConfig)
print(f"Connected to: {config.deriva_ml.hostname}")
print(f"Datasets: {config.datasets}")
print(f"Assets: {config.assets}")

# Access custom fields
print(f"Threshold: {config.threshold}")
print(f"Max iterations: {config.max_iterations}")
```

## Command-Line Overrides

Override configuration from the command line:

```bash
# Show available configs
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --list-configs

# Override standard fields
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
  assets=different_assets

# Override custom fields
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
  threshold=0.8 max_iterations=50

# Override connection
uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
  deriva_ml=eye_ai
```

## BaseConfig Fields

The `BaseConfig` class provides these standard fields:

| Field | Type | Description |
|-------|------|-------------|
| `deriva_ml` | DerivaMLConfig | Connection settings |
| `datasets` | list[DatasetSpecConfig] | Input datasets |
| `assets` | list[str] | Input asset RIDs |
| `dry_run` | bool | Skip catalog writes |
| `description` | str | Execution description |

## Complete Example

The template ships a generic notebook config scaffold in
`src/configs/analysis.py`. Rename it (and add more) to match the notebooks you
build under `notebooks/`:

```python
# src/configs/analysis.py
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class AnalysisConfig(BaseConfig):
    """Configuration for an analysis notebook.

    Attributes:
        show_detail: Example boolean parameter. Replace with parameters your
            notebook actually needs.
    """
    show_detail: bool = True

notebook_config(
    "analysis",
    config_class=AnalysisConfig,
    defaults={"assets": "no_assets", "datasets": "no_datasets"},
    description="Analysis of the selected asset group.",
)
```

```python
# notebooks/analysis.ipynb - Cell 1
from deriva_ml.execution import run_notebook

# Config name is auto-derived from the notebook filename
# (analysis.ipynb -> "analysis").
ml, execution, config = run_notebook()

print(f"Show detail: {config.show_detail}")
print(f"Assets: {config.assets}")
```
