# Running Experiments

This guide covers the complete workflow for running ML experiments with DerivaML.

## Pre-Run Checklist

Before running an experiment:

- [ ] Code changes committed to Git
- [ ] Dependencies up to date (`uv sync`)
- [ ] Authenticated with Deriva (`uv run deriva-globus-auth-utils login --host ...`)
- [ ] Configuration verified (use `--list-configs` for the menu or `--cfg job` for the resolved config)
- [ ] Version tag created for significant runs

## Single Experiment

### Basic Run

```bash
# Run with default configuration
uv run deriva-ml-run

# Run with specific model config
uv run deriva-ml-run model_config=extended

# Run with multiple overrides
uv run deriva-ml-run model_config=extended datasets=full_training
```

### Dry Run (Development)

During development, use dry runs to test without creating catalog records:

```bash
uv run deriva-ml-run dry_run=true
```

This:
- Downloads input datasets
- Runs your model code
- Skips creating execution records
- Skips uploading outputs

### Inline Overrides

Override specific parameters without creating new configurations:

```bash
# Override model parameters
uv run deriva-ml-run model_config.epochs=100 model_config.learning_rate=0.01

# Override connection from command line
uv run deriva-ml-run --host localhost --catalog 45
```

## Multiple Experiments (Multirun)

### Using Named Multiruns

```bash
# Run a predefined named multirun
uv run deriva-ml-run +multirun=quick_vs_extended
```

### Sweeping Parameters

```bash
# Sweep learning rates
uv run deriva-ml-run --multirun model_config.learning_rate=0.1,0.01,0.001

# Sweep multiple parameters (creates all combinations)
uv run deriva-ml-run --multirun \
  model_config.learning_rate=0.1,0.01 \
  model_config.epochs=10,50
```

### Ad-hoc Multirun with Experiment Presets

```bash
# Run multiple experiments as ad-hoc multirun
uv run deriva-ml-run --multirun +experiment=baseline,extended,regularized

# Use preset but sweep one parameter
uv run deriva-ml-run --multirun +experiment=baseline model_config.epochs=10,25,50
```

## Notebook Experiments

### Interactive Development

```bash
# Start JupyterLab
uv run jupyter lab

# Work interactively in the notebook
# Use your repository's kernel
```

### Reproducible Execution

```bash
# Run notebook and upload to catalog
uv run deriva-ml-run-notebook notebooks/analysis.ipynb

# With configuration overrides
uv run deriva-ml-run-notebook notebooks/analysis.ipynb assets=different_assets
```

### View Configuration Options

```bash
uv run deriva-ml-run-notebook notebooks/analysis.ipynb --list-configs
```

## Monitoring Progress

### View Outputs

Hydra creates output directories for each run:

```
outputs/
└─ 2024-01-15/
   └─ 10-30-00/
      ├─ .hydra/
      │  ├─ config.yaml      # Full resolved config
      │  ├─ hydra.yaml       # Hydra settings
      │  └─ overrides.yaml   # Command-line overrides
      └─ output.log          # Captured stdout/stderr
```

### Check Catalog Records

After a run completes, find the execution in the catalog:

1. Get the Chaise URL from the output
2. Or use MCP tools: `list_executions`
3. Or browse the Execution table in Chaise

## Post-Run Tasks

### Commit Output Assets (if not automatic)

For scripts, output assets are committed as part of the run. For notebooks:

```python
# At the end of your notebook
execution.commit_output_assets()
```

### Document Results

Consider adding notes to the execution record:
- What you learned
- Whether results were expected
- Next steps

### Clean Up

```bash
# Remove old Hydra outputs (optional)
rm -rf outputs/2024-01-*/
```

## Troubleshooting

### "No credentials found"

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

### "Configuration not found"

Check that your config file is imported in `src/configs/__init__.py`.

### "Dataset not found"

Verify the dataset RID exists in the catalog:
```bash
# Use MCP tools or check Chaise
```

### Multirun fails partway

- Check which runs succeeded in the catalog
- Resume with remaining experiments only
- Or re-run all (DerivaML will create new execution records)

## Best Practices

1. **Start with dry runs** during development
2. **Version before significant runs** using `bump-version`
3. **Use experiment presets** for reproducibility
4. **Document your experiments** in the catalog
5. **Clean up regularly** to avoid disk space issues
6. **Check outputs before uploading** to catch errors early
