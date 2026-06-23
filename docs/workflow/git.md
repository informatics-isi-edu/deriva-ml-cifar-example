# Git Workflow

DerivaML tracks code provenance by linking execution records to Git commits. Following a proper Git workflow ensures accurate tracking and reproducibility.

## Core Principle

**Always commit before running.** DerivaML captures your Git commit hash when you run a model or notebook. If you have uncommitted changes, the provenance record won't accurately reflect the code that produced your results.

## Recommended Workflow

### 1. Work in Branches

Even for solo projects, use branches:

```bash
# Create a feature branch
git checkout -b feature/add-new-model

# Make changes...
# Commit changes...

# Push and create PR
git push -u origin feature/add-new-model
```

### 2. Commit Before Running

```bash
# Check status
git status

# Stage and commit
git add .
git commit -m "Add new model with extended training"

# Now run
uv run deriva-ml-run model_config=extended
```

### 3. Use Meaningful Commits

```bash
# Good: Descriptive commit messages
git commit -m "Add dropout regularization to CNN model"
git commit -m "Increase training epochs from 10 to 50"
git commit -m "Fix data loading for multi-class labels"

# Bad: Vague messages
git commit -m "updates"
git commit -m "fix"
```

### 4. Tag Significant Runs

Before important runs, create a version tag:

```bash
# Create a patch version
uv run bump-version patch

# Or for significant changes
uv run bump-version minor
```

## Debugging Workflow

During development and debugging, use dry runs to avoid creating execution records:

```bash
# Dry run - downloads data but doesn't create records
uv run deriva-ml-run dry_run=true

# Make changes based on results
# ...

# Once satisfied, commit and do a real run
git add .
git commit -m "Fix model architecture"
uv run deriva-ml-run
```

## Branch Strategy

```
main
 │
 ├── feature/new-model
 │    ├── commit: "Add model skeleton"
 │    ├── commit: "Implement training loop"
 │    └── commit: "Add validation metrics"
 │
 └── experiment/hyperparameter-sweep
      ├── commit: "Set up sweep configs"
      └── commit: "Run sweep experiments"
```

## Pull Request Guidelines

1. **One feature per PR**: Keep changes focused
2. **Run tests before merging**: Ensure code works
3. **Squash if needed**: Clean up messy history
4. **Delete branches after merge**: Keep repo clean

## Gitignore Best Practices

The template `.gitignore` excludes:

```
# Environment
.venv/
__pycache__/

# Outputs
outputs/
*.pyc

# Data (stored in DerivaML, not Git)
data/
*.csv
*.pkl

# Secrets
.env
credentials.json
```

## Emergency: Uncommitted Changes

If you accidentally ran with uncommitted changes:

1. The execution record still exists but provenance is imperfect
2. Commit your changes immediately
3. Note the execution RID and the commit hash
4. Add a comment to the execution record if needed

## Working with Large Files

Don't commit large files to Git:

- **Model weights**: Upload to DerivaML as assets
- **Datasets**: Store in DerivaML catalogs
- **Large outputs**: Upload via execution outputs

Use DerivaML to track these instead:

```python
# Register large output for upload
model_path = execution.asset_file_path("Model", "weights.pt")
torch.save(model.state_dict(), model_path)
execution.commit_output_assets()
```
