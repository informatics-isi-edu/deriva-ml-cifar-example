# Versioning

DerivaML uses semantic versioning to track model releases. This enables reproducibility and clear communication about changes.

## Semantic Versioning

Version numbers follow the format `MAJOR.MINOR.PATCH`:

| Component | When to Increment | Example |
|-----------|-------------------|---------|
| **MAJOR** | Breaking changes to model interface or outputs | 1.0.0 → 2.0.0 |
| **MINOR** | New features, backward compatible | 1.0.0 → 1.1.0 |
| **PATCH** | Bug fixes, small improvements | 1.0.0 → 1.0.1 |

## Creating Versions

Use the `bump-version` script:

```bash
# Bug fix or small tweak
uv run bump-version patch

# New feature or significant improvement
uv run bump-version minor

# Breaking change or major milestone
uv run bump-version major
```

This script:
1. Calculates the next version number
2. Creates a Git tag
3. Pushes the tag to GitHub
4. Triggers a GitHub release with auto-generated notes

## Checking Current Version

```bash
# Get current version from Git tags
uv run python -m setuptools_scm
```

Example outputs:
- `1.0.0` - Clean release
- `1.0.1.dev3+g1234567` - 3 commits after 1.0.1, at commit 1234567

## When to Version

### Create a PATCH version for:
- Bug fixes
- Documentation updates
- Minor configuration tweaks
- Code cleanup/refactoring

### Create a MINOR version for:
- New model variants
- New configuration options
- Performance improvements
- New notebook analyses

### Create a MAJOR version for:
- Changes to model architecture
- Changes to output format
- Changes to execution interface
- Breaking configuration changes

## Version Tags in DerivaML

When you run a model, DerivaML records:

1. **Git commit hash**: Exact code state
2. **Version tag** (if on a tag): Semantic version
3. **Repository URL**: Where the code lives
4. **Branch name**: Which branch was used

## Best Practices

### Before Important Runs

```bash
# Ensure clean state
git status

# Commit any changes
git add .
git commit -m "Prepare for production run"

# Create version tag
uv run bump-version minor

# Run the model
uv run deriva-ml-run
```

### For Experiment Sweeps

```bash
# Version before starting sweep
uv run bump-version minor

# Run multiple experiments using a named multirun
uv run deriva-ml-run +multirun=quick_vs_extended

# Or run ad-hoc multirun with experiment presets
uv run deriva-ml-run --multirun +experiment=run1,run2,run3
```

All experiments share the same version, making them easy to compare.

### For Development

During active development, you don't need to version every run:

```bash
# Development runs (no versioning needed)
uv run deriva-ml-run dry_run=true

# Ready for real run? Create a version
git add . && git commit -m "Ready for testing"
uv run bump-version patch
uv run deriva-ml-run
```

## GitHub Integration

The template includes a GitHub Action (`.github/workflows/ci.yml`) that:

1. Runs on version tag push (`v*`)
2. Creates a GitHub Release
3. Auto-generates release notes from PR titles

To see releases: `https://github.com/your-org/your-repo/releases`

## Comparing Versions

In the DerivaML catalog, you can:

1. Filter executions by code version
2. Compare results across versions
3. Reproduce any previous version by checking out its tag

```bash
# Check out a specific version
git checkout v1.2.0

# Recreate the environment
uv sync

# Run with same configuration
uv run deriva-ml-run +experiment=original_experiment
```
