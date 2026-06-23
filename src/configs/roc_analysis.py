"""Configuration for ROC analysis notebook.

This module defines the hydra-zen configuration for the ROC curve analysis notebook.

By DerivaML convention, notebook ``X.ipynb`` uses the config registered as
``notebook_config("X", ...)``. ``run_notebook()`` derives the config name from
the calling notebook's filename, so the explicit string is only needed when
the notebook filename and config name diverge.

Usage:
    In notebook:

        from deriva_ml.execution import run_notebook

        ml, execution, config = run_notebook()
        # Config name auto-derived from notebook filename (roc_analysis.ipynb).
        # Ready to use!
        # - config.assets: probability file RIDs
        # - config.show_per_class: whether to show individual class curves

    From command line:

        # Run with specific configuration
        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            --host localhost --catalog 65 \\
            --config roc_quick_vs_extended

        # Or override assets directly
        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            assets=roc_lr_sweep

Available Configurations:
    - roc_analysis: Default (quick vs extended on small dataset)
    - roc_quick_vs_extended: Compare quick vs extended training (small dataset)
    - roc_full_quick_vs_extended: Compare quick vs extended training (full dataset)
    - roc_lr_sweep: Compare learning rate sweep (0.0001, 0.001, 0.01, 0.1)
    - roc_epoch_sweep: Compare epoch sweep (5, 10, 25, 50 epochs)
    - roc_lr_batch_grid: Compare LR x batch size grid (2x2)

Configuration Groups:
    - deriva_ml: DerivaML connection settings (default_deriva, eye_ai, etc.)
    - assets: Asset RID lists for probability files
"""

from dataclasses import dataclass

from deriva_ml.execution import BaseConfig, notebook_config


@dataclass
class ROCAnalysisConfig(BaseConfig):
    """Configuration for ROC analysis notebook.

    Attributes:
        show_per_class: If True, plot individual ROC curves for each class.
            If False, only show micro/macro averaged curves.
        confidence_threshold: Minimum confidence threshold for predictions
            to be included in the analysis (0.0 to 1.0).
    """

    show_per_class: bool = True
    confidence_threshold: float = 0.0


# =============================================================================
# ROC Analysis Notebook Configurations
# =============================================================================

# Default config. The description is intentionally GENERIC: this config is the
# CLI entry point for ROC analysis of whatever prediction-probability asset
# group the user selects (``deriva-ml-run-notebook ... assets=<group>``). A
# description that named the default asset group ("quick vs extended") would go
# stale the moment a different ``assets=`` group is passed, leaving the recorded
# execution's Description claiming a comparison that did not run. Keep it
# group-agnostic so the prose is true for every override; the resolved
# ``assets=`` override is appended to the execution Description automatically and
# carries the specific group that actually ran.
# Note: Use "no_datasets" to avoid type conflicts with with_description wrapped datasets
notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_quick_vs_extended", "datasets": "no_datasets"},
    description="ROC curve analysis of the selected prediction-probability asset group",
)

# -----------------------------------------------------------------------------
# Model Comparison Configurations
# -----------------------------------------------------------------------------
# Each configuration references prediction probability files from the
# corresponding multirun experiment. All use "no_datasets" since ROC analysis
# only needs assets (prediction probability files).

notebook_config(
    "roc_quick_vs_extended",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_quick_vs_extended", "datasets": "no_datasets"},
    description="ROC analysis: quick (3 epochs) vs extended (50 epochs) training",
)

notebook_config(
    "roc_lr_sweep",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_lr_sweep", "datasets": "no_datasets"},
    description="ROC analysis: learning rate sweep (0.0001, 0.001, 0.01, 0.1)",
)

notebook_config(
    "roc_epoch_sweep",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_epoch_sweep", "datasets": "no_datasets"},
    description="ROC analysis: epoch sweep (5, 10, 25, 50 epochs)",
)

notebook_config(
    "roc_lr_batch_grid",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_lr_batch_grid", "datasets": "no_datasets"},
    description="ROC analysis: LR x batch size grid (2x2)",
)
