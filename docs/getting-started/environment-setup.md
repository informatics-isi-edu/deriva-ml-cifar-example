# Environment Setup

This guide covers setting up and managing your development environment.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Python package manager
- Git
- Access to a Deriva catalog — use an existing server or run one locally with [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker)

## Installing uv

If you haven't installed `uv` yet:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See the [official uv documentation](https://docs.astral.sh/uv/) for more options.

## Initializing Your Environment

From the repository root:

```bash
# Create environment and install dependencies
uv sync
```

This creates:
- A `.venv/` directory with an isolated Python environment
- A `uv.lock` file pinning exact dependency versions

**Important:** Commit `uv.lock` to your repository to ensure reproducible environments.

## Optional Dependency Groups

Install extra groups on demand:

```bash
# Jupyter notebook support
uv sync --group=jupyter

# PyTorch
uv sync --group=pytorch

# TensorFlow
uv sync --group=tensorflow

# Documentation building
uv sync --group=docs
```

To always install certain groups, add them to `default-groups` in `pyproject.toml`:

```toml
[tool.uv]
default-groups = ["dev", "jupyter"]
```

## Notebook Setup

For notebook development:

```bash
# Install Jupyter support
uv sync --group=jupyter

# Install nbstripout to auto-strip output cells on commit
uv run nbstripout --install

# Register a Jupyter kernel for this environment
uv run deriva-ml-install-kernel

# Verify available kernels
uv run jupyter kernelspec list
```

## Activating the Environment

You can run commands directly with `uv run`:

```bash
uv run python script.py
uv run pytest
```

Or activate the environment for a shell session:

```bash
# Bash/Zsh
source .venv/bin/activate

# Fish
source .venv/bin/activate.fish

# Csh/Tcsh
source .venv/bin/activate.csh

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

When finished, run `deactivate` to leave the environment.

## Updating Dependencies

### Update a Specific Package

```bash
# Update DerivaML to latest version
uv sync --upgrade-package deriva-ml

# Update multiple packages
uv sync --upgrade-package deriva-ml --upgrade-package pandas
```

### Update All Packages

```bash
# Regenerate lock file with latest versions
uv lock --upgrade

# Install updated packages
uv sync
```

**Caution:** Upgrading PyTorch or TensorFlow may require compatible GPU drivers. Consider pinning these versions in `pyproject.toml`.

After upgrading, commit your updated `uv.lock` file.

## Authentication

Before accessing catalog data, authenticate with Globus:

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

This opens a browser for Globus authentication. Credentials are cached locally.

For multiple servers:

```bash
uv run deriva-globus-auth-utils login --host <hostname>
uv run deriva-globus-auth-utils login --host <dev-hostname>
```

## GitHub Actions

The template includes GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `release.yml` | Version tag (`v*`) | Creates GitHub releases with auto-generated notes |
| `publish-docs.yml` | Push to main | Builds and deploys documentation to GitHub Pages |

These run automatically - no setup required.

## Troubleshooting

### "No credentials found"

Re-authenticate:
```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

### "Token expired"

Force re-authentication:
```bash
uv run deriva-globus-auth-utils login --host <hostname> --force
```

### Kernel not found in Jupyter

Re-register the kernel:
```bash
uv run deriva-ml-install-kernel
```

### Dependency conflicts

Try regenerating the lock file:
```bash
rm uv.lock
uv lock
uv sync
```

### Permission denied on .venv

Remove and recreate:
```bash
rm -rf .venv
uv sync
```
