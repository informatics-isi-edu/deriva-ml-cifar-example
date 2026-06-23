# DerivaML CIFAR-10 Example

This repository is the **CIFAR-10 reference example for DerivaML** — a complete, runnable ML project built on the [`deriva-ml-model-template`](https://github.com/informatics-isi-edu/deriva-ml-model-template) skeleton. It shows the template's structure filled in with a real worked example: a 2-layer CNN classifier for CIFAR-10, the `cifar10` domain schema loader, hydra-zen experiment and multirun configurations, and a ROC analysis notebook.

DerivaML is a Python library for reproducible ML workflows backed by a Deriva catalog; it captures code provenance, configuration, and outputs for reproducibility. To start a new project from a clean skeleton, use the [`deriva-ml-model-template`](https://github.com/informatics-isi-edu/deriva-ml-model-template) instead — this repo is the example you study, not the one you fork.

## Documentation

**[View Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)**

Quick links:
- [Quick Start Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/quick-start/) - Get up and running in minutes
- [Environment Setup](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/environment-setup/) - Detailed setup instructions
- [Creating a New Model](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-models/) - Step-by-step guide for adding models
- [Creating a New Notebook](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-notebooks/) - Step-by-step guide for adding notebooks
- [Configuration Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/configuration/overview/) - Understanding hydra-zen configuration
- [Coding Guidelines](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/coding-guidelines/) - Best practices and standards

## What's Included

- Python-first configuration using hydra-zen (no YAML)
- CLI entry points via `deriva-ml-run` and `deriva-ml-run-notebook`
- An example model (CIFAR-10 CNN) with 7 configuration variants
- Experiment presets and named multirun configurations
- A ROC analysis notebook with hydra-zen configuration
- GitHub Actions for automated versioning and documentation

## Quick Start

> **Note:** Running models requires access to a DerivaML catalog. If you don't have access to an existing Deriva server, you can run one locally using [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker).

### 1. Clone This Example

Clone this repository to study and run the worked CIFAR-10 example:

```bash
git clone https://github.com/informatics-isi-edu/deriva-ml-cifar-example.git
cd deriva-ml-cifar-example
```

To start a **new** project from a clean skeleton instead, create a repository from the [`deriva-ml-model-template`](https://github.com/informatics-isi-edu/deriva-ml-model-template): [Creating a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).

### 2. Enable GitHub Pages

After creating your repository from this template, enable GitHub Pages for automatic documentation deployment:

1. Go to your repository **Settings > Pages**
2. Under "Build and deployment", set **Source** to **"GitHub Actions"**
3. Save

The documentation workflow will automatically deploy on each push to main.

### 3. Initialize Environment

```bash
# Create environment and install dependencies
uv sync

# For notebook support
uv sync --group=jupyter
uv run nbstripout --install
uv run deriva-ml-install-kernel

# For PyTorch (required by the CIFAR-10 example)
uv sync --group=pytorch
```

### 4. Set Up Claude Code (Optional)

If using [Claude Code](https://claude.ai/code), connect to a DerivaML MCP server and install the two skills plugins (`deriva` for generic catalog operations, `deriva-ml` for ML workflows).

**Connect the MCP server.** The MCP server stack is split into two pieces: [`deriva-mcp-core`](https://github.com/informatics-isi-edu/deriva-mcp-core) (catalog/schema/vocabulary tools) plus the [`deriva-ml-mcp`](https://github.com/informatics-isi-edu/deriva-ml-mcp) plugin (DerivaML domain tools — datasets, executions, features, assets). When you stand up a [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker) localhost stack, both ship together as the `deriva-mcp-test` service at `https://localhost/mcp` over HTTP with OAuth.

Register the connection with Claude Code:

```bash
claude mcp add -t http dev-localhost https://localhost/mcp \
    --client-id deriva-mcp --callback-port 8080
```

Verify with `claude mcp list` — the entry should show `dev-localhost: https://localhost/mcp (HTTP) - ✓ Connected`. The `deriva-mcp` client-id is pre-registered with the Credenza auth service in the deriva-docker deployment; `--callback-port 8080` is where Claude listens for the OAuth callback.

**Trust the dev-localhost CA.** Claude Code's MCP HTTP transport runs in Node.js, which has its own CA bundle and won't trust the deriva-docker self-signed cert by default. Without this step, the connection fails with a TLS error:

```bash
# Extract the CA from the running container
mkdir -p ~/.config/deriva
docker cp deriva-mcp-test:/usr/local/share/ca-certificates/deriva-dev-ca.crt \
    ~/.config/deriva/deriva-dev-ca.crt
```

Then add to your workspace's `.claude/settings.local.json`:

```json
{
  "env": {
    "NODE_EXTRA_CA_CERTS": "/Users/<you>/.config/deriva/deriva-dev-ca.crt"
  }
}
```

The first MCP call after this opens an OAuth consent page in your browser; approve once and the bearer token is cached.

For non-dockerized setups (native install, production HTTP, or stdio with a local credential), see the [`deriva-mcp-core` deployment guide](https://github.com/informatics-isi-edu/deriva-mcp-core/blob/main/docs/deployment-guide.md).

**Install the skills plugins.** Both plugins share one marketplace:

```
/plugin marketplace add informatics-isi-edu/deriva-plugins
/plugin install deriva
/plugin install deriva-ml
```

`deriva` covers generic Deriva catalog operations (schema, vocabulary, query patterns, Chaise display); `deriva-ml` adds the DerivaML domain layer (dataset lifecycle, executions, features, experiments, Hydra-zen configs, model development). The `deriva-ml` plugin assumes `deriva` is loaded for cross-references — install both.

To pick up new plugin versions automatically, enable `"autoUpdate": true` for the `deriva-plugins` marketplace entry in `~/.claude/settings.json` and restart Claude Code. Otherwise rerun `/plugin install deriva` and `/plugin install deriva-ml` when a release ships.

For checking versions of the underlying components (deriva-py, deriva-mcp-core, deriva-ml, deriva-ml-mcp), the troubleshooting skills cover it:

- `/deriva:troubleshoot-deriva-errors` — versioning for the foundation (deriva-py, deriva-mcp-core, `deriva` plugin)
- `/deriva-ml:troubleshoot-execution` — versioning for the DerivaML layer (deriva-ml, deriva-ml-mcp, `deriva-ml` plugin)

### 5. Authenticate

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

### 6. Load CIFAR-10 into a catalog

The example model needs CIFAR-10 data and dataset definitions in your catalog.

**Prerequisites:** none beyond `uv` and a Deriva localhost instance.
The CIFAR-10 archive (~170 MB) is downloaded automatically from the
Toronto open mirror on first run and cached at
`~/.cache/deriva-ml-model-template/`.

```bash
# Create a fresh catalog and load 10K images (good for first-time setup)
uv run python src/scripts/load_cifar10.py \
    --hostname <hostname> --create-catalog cifar10_test --num-images 10000

# Or load into an existing catalog
uv run python src/scripts/load_cifar10.py \
    --hostname <hostname> --catalog-id <id> --num-images 10000
```

The loader prints the catalog ID and the RID of every dataset it creates
(`Complete`, `Training`, `Small_Labeled_Split`, etc.). **Save these RIDs** —
you need them for the next step.

### 7. Update configs for your catalog

`src/configs/datasets.py` ships with RIDs from a previous demo catalog.
After running `load-cifar10`, replace each `DatasetSpecConfig(rid=...)` with
the RID the loader reported, and update each `version=` to the version the
loader assigned (visible via `ml.find_datasets()` after loading).

| Config name | Loader output |
|---|---|
| `cifar10_complete` | `Complete` |
| `cifar10_split` | `Split` (canonical Toronto train/test partition, produced by `split_dataset(selection_fn=...)`) |
| `cifar10_training` | child `Training` of `Split` |
| `cifar10_testing` | child `Testing` of `Split` |
| `cifar10_small_training` | `Small_Training` (stratified `subsample()` of `Training`; no parent Split) |
| `cifar10_small_testing` | `Small_Testing` (stratified `subsample()` of `Testing`; no parent Split) |
| `cifar10_labeled_split` | `Labeled_Split` |
| `cifar10_labeled_training` | child `Training` of `Labeled_Split` |
| `cifar10_labeled_testing` | child `Testing` of `Labeled_Split` |
| `cifar10_small_labeled_split` | `Small_Labeled_Split` |
| `cifar10_small_labeled_training` | child `Training` of `Small_Labeled_Split` |
| `cifar10_small_labeled_testing` | child `Testing` of `Small_Labeled_Split` |

Note: `cifar10_small_split` no longer exists. The parent `Small_Split`
dataset was dropped in the v1.42 migration — `Small_Training` and
`Small_Testing` are sibling subsample outputs of the same execution.

For multi-environment setups, register parallel `*_<env>` configs in
`src/configs/dev/` rather than editing the defaults — the example model in
this template uses `cifar10_small_labeled_split`, so the corresponding
`*_<env>` variant is the minimum you need to override.

Also point `src/configs/deriva.py` (or a new entry under `src/configs/dev/`)
at your hostname and catalog ID, **or** override at the CLI:
`--host <hostname> --catalog <id>`.

### 8. Run

> **Commit before running.** DerivaML records the git commit hash for
> provenance. Uncommitted changes raise a warning and pollute the audit
> trail of any run that uses them.

```bash
# Run the example model with defaults
uv run deriva-ml-run

# Dry run (no catalog writes)
uv run deriva-ml-run dry_run=true

# Use an experiment preset
uv run deriva-ml-run +experiment=cifar10_quick

# Named multirun
uv run deriva-ml-run +multirun=quick_vs_extended

# Show available configs
uv run deriva-ml-run --list-configs

# Run a notebook
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb

# Override host/catalog from command line
uv run deriva-ml-run --host localhost --catalog 45 +experiment=cifar10_quick
```

## Project Layout

```
.
├── pyproject.toml                  # Project metadata and dependencies
├── Experiments.md                  # Registry of defined experiments
├── CLAUDE.md                       # Claude Code project instructions
├── src/
│   ├── configs/                    # Hydra-zen configurations (Python, no YAML)
│   │   ├── base.py                 # Base DerivaModelConfig
│   │   ├── deriva.py               # Catalog connection settings
│   │   ├── datasets.py             # Dataset specifications
│   │   ├── assets.py               # Asset RID configurations
│   │   ├── workflow.py             # Workflow definitions
│   │   ├── cifar10_cnn.py          # Model variant configs
│   │   ├── experiments.py          # Experiment presets
│   │   ├── multiruns.py            # Named multirun configurations
│   │   ├── roc_analysis.py         # ROC analysis notebook config
│   │   └── dev/                    # Alternate catalog configs
│   ├── models/                     # Model implementations
│   │   └── cifar10_cnn.py          # CIFAR-10 CNN model
│   └── scripts/                    # Data loading scripts
│       └── load_cifar10.py         # CIFAR-10 dataset loader
├── notebooks/
│   └── roc_analysis.ipynb          # ROC curve analysis notebook
└── docs/                           # Documentation (auto-published)
    └── design/                     # Design docs (plan before you build)
        ├── experiment/             #   per-experiment designs (<slug>.md)
        ├── dataset/                #   per-dataset designs
        ├── feature/                #   per-feature designs
        └── model/                  #   per-model designs
```

## Versioning

Create version tags before significant runs (DerivaML records the git commit for provenance):

```bash
uv run bump-version patch   # Bug fixes
uv run bump-version minor   # New features
uv run bump-version major   # Breaking changes
```

## CIFAR-10 Example

The template includes a complete CIFAR-10 CNN example. See [CIFAR10.md](CIFAR10.md) for usage and dataset details, or the [full CIFAR-10 documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/cifar10-example/) for architecture and model variants.

## Using Claude Code with DerivaML

With the MCP server connected and the `deriva` + `deriva-ml` skills plugins installed (see step 4), you can interact with catalogs through natural language and get guided workflows for common tasks. Skills auto-trigger based on context, or you can invoke them directly with `/deriva:<skill-name>` for generic catalog operations (e.g. `/deriva:getting-started`, `/deriva:manage-vocabulary`) and `/deriva-ml:<skill-name>` for ML workflows (e.g. `/deriva-ml:dataset-lifecycle`, `/deriva-ml:experiment-lifecycle`, `/deriva-ml:new-model`).

To see what's available, ask Claude *"help with deriva"* or run `/deriva:help` / `/deriva-ml:help` — these list the skills in each plugin organized by task: environment setup, catalog structure, data management, running experiments, and troubleshooting.

## Further Reading

- [Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)
- [DerivaML Library](https://informatics-isi-edu.github.io/deriva-ml/) - Core library documentation
- [DerivaML User Guide](https://deriva-ml.readthedocs.io/) - Tutorials, concepts, and API reference
- [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) - Configuration framework
