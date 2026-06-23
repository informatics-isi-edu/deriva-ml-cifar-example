"""
Configuration Package for Deriva ML Model Template
===================================================

This package contains hydra-zen configuration modules that define the parameters
for model execution. Each module registers its configurations with the hydra-zen
store when imported.

Configuration Groups
--------------------
- deriva_ml: Connection settings for the Deriva catalog
- datasets: Dataset specifications (which data to use)
- assets: Asset RIDs (model weights, checkpoints, etc.)
- workflow: Workflow definitions
- model_config: Model hyperparameters
- experiments: Preset experiment configurations
- multiruns: Named multirun configurations with rich markdown descriptions

Adding New Configurations
-------------------------
For notebooks, use the simplified API::

    # configs/my_analysis.py
    from deriva_ml.execution import notebook_config

    notebook_config(
        "my_analysis",
        defaults={"assets": "my_assets"},
    )

For notebooks with custom parameters::

    # configs/my_analysis.py
    from dataclasses import dataclass
    from deriva_ml.execution import BaseConfig, notebook_config

    @dataclass
    class MyAnalysisConfig(BaseConfig):
        threshold: float = 0.5

    notebook_config(
        "my_analysis",
        config_class=MyAnalysisConfig,
        defaults={"assets": "my_assets"},
    )

For model configurations, use hydra-zen directly::

    from hydra_zen import store, builds

    my_store = store(group="model_config")
    my_store(builds(my_model, param=value, zen_partial=True), name="my_model")
"""

# Re-export load_configs from deriva-ml for convenience.
from deriva_ml.execution import load_configs

__all__ = ["load_configs", "load_all_configs"]


def load_all_configs():
    """Discover and register every hydra-zen config in the ``configs`` package.

    Thin wrapper around :func:`deriva_ml.execution.load_configs` bound to this
    package's import root. Importing the config modules has the side effect of
    populating the hydra-zen store, so calling this once makes every model,
    dataset, experiment, workflow, and asset group available for resolution.

    Returns:
        The value returned by ``load_configs("configs")`` (the populated
        hydra-zen store).

    Example:
        >>> from configs import load_all_configs
        >>> load_all_configs()  # registers all groups into the hydra-zen store
    """
    return load_configs("configs")
