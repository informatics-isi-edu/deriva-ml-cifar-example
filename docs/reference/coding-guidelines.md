# Coding Guidelines

This document captures operational practices and coding standards for DerivaML projects. These guidelines ensure reproducibility and maintainability.

## Configuration

- Each model should live in its own repository following this template
- Use `uv` to manage all dependencies
- The generated `uv.lock` **must** be committed to the repository
- It should be possible to rebuild the environment from scratch using `uv.lock`

## Git Workflow

- **Always** work in a Git branch and create pull requests, even for solo projects
- Rebase your branch regularly to keep it up to date with main
- **You MUST commit changes before running models** - this maximizes DerivaML's provenance tracking
- No change is too small to properly track
- During debugging, use `dry_run=true` to skip catalog writes

## Coding Standards

### Documentation

- Use Google docstring format for all functions and classes
- See: [Google Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.
    """
```

### Type Hints

Use type hints wherever possible:

```python
from typing import List, Optional
from deriva_ml import DerivaML
from deriva_ml.execution import Execution

def process_data(
    items: List[str],
    threshold: float = 0.5,
    ml_instance: Optional[DerivaML] = None,
) -> dict[str, float]:
    ...
```

### Linting and Formatting

Run before committing:

```bash
# Check for issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/
```

### Signature Verification

Always check function/class signatures before modifying calls:

```python
import inspect
from deriva_ml import DerivaML

# Check before using
print(inspect.signature(DerivaML.__init__))
```

Or check the source code directly.

## Versioning and Releases

- Use `bump-version` to create version tags before significant runs
- DerivaML uses semantic versioning:
  - **Major**: Breaking changes
  - **Minor**: New features
  - **Patch**: Bug fixes

```bash
uv run bump-version major|minor|patch
```

Check current version:

```bash
uv run python -m setuptools_scm
```

## Notebooks

- **Never commit notebooks with output cells**
- Install and enable `nbstripout`:
  ```bash
  uv run nbstripout --install
  ```
- Notebooks should focus on a single task (analysis/visualization)
- Prefer scripts for training models
- Notebooks **must** run start-to-finish without intervention
- Use `deriva-ml-run-notebook` for reproducible execution

## Executions and Experiments

- Always run code from hydra-zen configuration files
- Commit code before running
- Use `dry_run=true` during debugging
- Remove `dry_run`, tag a version, and run to completion when ready

## Data Management

- **Do not commit data files to Git**
- Store all data in DerivaML catalogs
- Use datasets and assets for input data
- Use execution outputs for results

## Extensibility

DerivaML is designed to be extended via inheritance:

```python
from deriva_ml import DerivaML

class MyDomainML(DerivaML):
    """Domain-specific extensions for my project."""

    def my_custom_method(self):
        """Add domain-specific functionality."""
        ...
```

Instantiate your custom class in scripts and notebooks for domain-specific functionality.

## File Organization

```
src/
├─ configs/           # All configuration files
│  ├─ __init__.py
│  ├─ base.py
│  ├─ deriva.py
│  ├─ datasets.py
│  ├─ assets.py
│  ├─ workflow.py
│  └─ model.py
├─ models/            # Model implementations
│  └─ model_protocol.py
└─ scripts/           # Data loading / generation scripts
```

## Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Configuration files | `snake_case.py` | `model.py` |
| Configuration names | `snake_case` | `extended_training` |
| Model functions | `snake_case` | `train_model` |
| Classes | `PascalCase` | `MyModelConfig` |
| Constants | `UPPER_CASE` | `DEFAULT_EPOCHS` |
