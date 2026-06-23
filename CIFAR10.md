# CIFAR-10 Example

The template includes a complete CIFAR-10 CNN example demonstrating the full DerivaML workflow: data loading, model training, hyperparameter sweeps, and ROC analysis.

## Quick Start

```bash
# Load data into catalog
uv run load-cifar10 --host <hostname> --catalog_id <id> --num_images 500

# Train the model
uv run deriva-ml-run +experiment=cifar10_quick

# Run a learning rate sweep
uv run deriva-ml-run +multirun=lr_sweep

# Analyze results with ROC curves
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb
```

## Dataset Types

The CIFAR-10 example includes multiple dataset configurations:

| Type | Source of test partition | Use Case |
|------|--------------------------|----------|
| `cifar10_small_labeled_split` | 20% holdout of training images | Quick experiments, ROC analysis (recommended) |
| `cifar10_labeled_split` | 20% holdout of training images | Full experiments, ROC analysis |
| `cifar10_split` | Toronto test_batch (official) | Full training on official 50K/10K split (use `cifar10_small_training` / `cifar10_small_testing` for the small subsample variant) |

Both families carry ground-truth labels in the Toronto distribution. The distinction is *which images* form the test partition:

- **`*_labeled_split`** — test partition is a held-out 20% of the 50K training images (created by `split_dataset()`). Use for ROC analysis, cross-validation, or when you want the official test_batch reserved for final evaluation.
- **`*_split`** — test partition is the official Toronto test_batch (10K images from a separate source). Use when you want results comparable to the standard CIFAR-10 benchmark.

## Loader Walkthrough

The CIFAR-10 loader (`src/scripts/load_cifar10.py`) is structured as
a thin orchestrator that composes three single-purpose stage modules.
Each module demonstrates one DerivaML pattern you would reuse when
building a loader for your own data.

| Stage | Module | Pattern demonstrated |
|---|---|---|
| 1 | [`_cifar10_schema.py`](src/scripts/_cifar10_schema.py) | Catalog + schema setup: create or connect, register the domain model (asset table, vocabulary, feature), declare workflow and dataset types |
| 2 | [`_cifar10_assets.py`](src/scripts/_cifar10_assets.py) | Asset upload + feature labeling: upload binary files in one Execution, re-query the catalog and add feature values in a separate Execution |
| 3 | [`_cifar10_datasets.py`](src/scripts/_cifar10_datasets.py) | Dataset hierarchy: query existing assets, partition by attribute, assemble nested datasets with derived holdout splits |

### Key design choices

**Stages are independent.** Each stage reads back from the catalog
rather than relying on in-memory state from earlier stages. Stage 2
queries `Image` rows to label them; Stage 3 queries `Image` rows + their
features to assign dataset membership. This means each stage works
standalone — you can run them via `load-cifar10 --phase <stage>`
against any catalog where the prior stages have completed, even from
a different process invocation.

**One Execution per logical step.** Stage 2 uses two Executions
(one for upload, one for labeling) because they are logically
distinct steps with different provenance. Stage 3 uses one
Execution because creating the entire dataset hierarchy is one
logical step.

**Class encoded in filename.** Stage 2a names uploaded files
`train_<class>_<id>.png` or `test_<class>_<id>.png`. This is what
lets Stage 2b recover the class without needing in-memory state
from Stage 2a, and what lets Stage 3 partition by train/test
prefix without needing in-memory state from either earlier
stage.

### Reusing for your own data

When adapting this template for your own data:

1. Replace `_cifar10_source.py` (the data-source layer — downloads
   the archive, extracts to a PNG layout) with your own source code.
2. Tweak `_cifar10_schema.py` to declare your domain model
   (asset table for your data type, vocabulary for your categories,
   feature for whatever labels apply).
3. Tweak `_cifar10_assets.py` to upload your data with whatever
   naming convention lets your features and datasets be derived
   from the catalog state.
4. Tweak `_cifar10_datasets.py` to assemble the dataset
   hierarchy that matches your experimental design.

The orchestrator (`load_cifar10.py`) stays as a thin CLI shim —
you only edit it to update imports and the summary banner labels.

## Model

The CIFAR-10 example uses a 2-layer CNN (`src/models/cifar10_cnn.py`) for image classification.

**Architecture:** Conv2d(3, C1) → ReLU → MaxPool → Conv2d(C1, C2) → ReLU → MaxPool → Linear(C2×8×8, hidden) → ReLU → Linear(hidden, 10)

**Data flow:** Downloads dataset as BDBag → `restructure_assets()` creates ImageFolder layout → torchvision DataLoader → training/evaluation → saves weights + prediction CSV as execution assets.

### Model Configs

Defined in `src/configs/cifar10_cnn.py`:

| Config | Channels | Hidden | Epochs | LR | Notes |
|--------|----------|--------|--------|------|-------|
| `default_model` | 32→64 | 128 | 10 | 1e-3 | Standard training |
| `cifar10_quick` | 32→64 | 128 | 3 | 1e-3 | Fast validation |
| `cifar10_large` | 64→128 | 256 | 20 | 1e-3 | More capacity |
| `cifar10_regularized` | 32→64 | 128 | 20 | 1e-3 | Dropout 0.25, weight decay 1e-4 |
| `cifar10_fast_lr` | 32→64 | 128 | 15 | 1e-2 | Fast convergence |
| `cifar10_slow_lr` | 32→64 | 128 | 30 | 1e-4 | Stable convergence |
| `cifar10_extended` | 64→128 | 256 | 50 | 1e-3 | Best accuracy, full regularization |
| `cifar10_test_only` | 32→64 | 128 | — | — | Load weights, evaluate only |

### Experiment Presets

Defined in `src/configs/experiments.py`:

| Experiment | Model Config | Dataset | Purpose |
|-----------|-------------|---------|---------|
| `cifar10_quick` | quick | small labeled split | Fast pipeline validation |
| `cifar10_default` | default | small training | Standard training |
| `cifar10_extended` | extended | small labeled split | Best accuracy on small set |
| `cifar10_quick_full` | quick | full labeled split | Baseline on full data |
| `cifar10_extended_full` | extended | full labeled split | Production run |
| `cifar10_test_only` | test_only | small labeled testing | Evaluate pretrained weights |

### ROC Analysis Notebook

The `notebooks/roc_analysis.ipynb` notebook compares model predictions across experiments by generating ROC curves. Configured via `src/configs/roc_analysis.py`. Takes asset RIDs (prediction CSVs) as input.

See the [full CIFAR-10 documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/cifar10-example/) for additional details.
