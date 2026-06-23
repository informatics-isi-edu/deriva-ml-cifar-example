# DerivaML Model Template

This repository provides a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## What's Included

This template ships as a **runnable skeleton** — every config group resolves
and dry-runs end to end with a no-op placeholder model, so you can verify the
plumbing before writing any code. You fill it in:

- A Python-first hydra-zen configuration scaffold (no YAML): one self-documenting
  module per config group in `src/configs/`, each with a live default plus a
  single commented example you uncomment and fill in
- CLI entry points via `deriva-ml-run` and `deriva-ml-run-notebook`
- A model interface to implement (`src/models/model_protocol.py`) and a
  placeholder `default_model` to replace (`src/configs/model.py`)
- Experiment presets and named multirun (sweep) configurations
- GitHub Actions for automated versioning

For a complete worked example, see
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).

## Quick Links

- [Quick Start Guide](getting-started/quick-start.md) - Get up and running in minutes
- [Customizing this Template](customization.md) - Turn the skeleton into your project
- [Creating a New Model](getting-started/creating-models.md) - Step-by-step guide for adding models
- [Creating a New Notebook](getting-started/creating-notebooks.md) - Step-by-step guide for adding notebooks
- [Configuration Guide](configuration/overview.md) - Understanding hydra-zen configuration

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git and GitHub account
- Access to a DerivaML catalog — use an existing server or run one locally with [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker)

## Project Layout

```
.
├── pyproject.toml                  # Project metadata and dependencies
├── CLAUDE.md                       # Claude Code project instructions
├── src/
│   ├── configs/                    # Hydra-zen configurations (Python, no YAML)
│   │   ├── base.py                 # Base DerivaModelConfig
│   │   ├── deriva.py               # Catalog connection settings
│   │   ├── datasets.py             # Dataset specifications
│   │   ├── assets.py               # Asset RID configurations
│   │   ├── workflow.py             # Workflow definitions
│   │   ├── model.py                # Model function + hyperparameter configs
│   │   ├── experiments.py          # Experiment presets
│   │   ├── multiruns.py            # Named multirun (sweep) configurations
│   │   ├── analysis.py             # Analysis notebook config
│   │   └── dev/                    # Alternate per-environment catalog configs
│   ├── models/                     # Model implementations
│   │   └── model_protocol.py       # The interface a model must implement
│   └── scripts/                    # Data loading / generation scripts
├── notebooks/                      # Analysis notebooks (add your own)
└── docs/                           # Documentation (auto-published)
```

## Related Resources

- [DerivaML Documentation](https://informatics-isi-edu.github.io/deriva-ml/)
- [DerivaML MCP Server](https://github.com/informatics-isi-edu/deriva-mcp) - AI assistant integration
- [Hydra-zen Documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/)
