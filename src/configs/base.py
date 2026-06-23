"""Base configuration for the model runner.

This module creates and exports the main model runner configuration that
can be used as a base for experiments.

Usage:
    from configs.base import BaseConfig, DerivaModelConfig
"""

from __future__ import annotations

from functools import partial
from typing import Any

from hydra_zen import builds, store

from deriva_ml import DerivaML
from deriva_ml.execution import (
    BaseConfig,
    DerivaBaseConfig,
    base_defaults,
    run_model,
)

# Hydra defaults list applied to the top-level model runner config.
_HYDRA_DEFAULTS: list[Any] = [
    "_self_",
    {"deriva_ml": "default_deriva"},
    {"datasets": "default_dataset"},
    {"assets": "default_asset"},
    {"workflow": "default_workflow"},
    {"model_config": "default_model"},
    {"optional script_config": "none"},
]

# Default Execution.description when the user does not supply one. The
# canonical ``run_model`` in deriva-ml composes the final description by
# combining this base string with the resolved Hydra overrides via
# ``_format_description_with_overrides`` -- mirroring how ``run_notebook``
# handles its own description. Keeping a short, recognisable base here
# makes ad-hoc runs scannable in ``ml.list_executions()`` output.
_DEFAULT_DESCRIPTION = "Simple model run"

# Create the main configuration schema for the model runner. The
# canonical ``run_model`` itself composes the final description from the
# Hydra overrides at dispatch time, so the config builds directly against
# it with no wrapper indirection.
DerivaModelConfig = builds(
    partial(run_model, ml_class=DerivaML),
    description=_DEFAULT_DESCRIPTION,
    populate_full_signature=True,
    hydra_defaults=_HYDRA_DEFAULTS,
)

# Register with the hydra-zen store
store(DerivaModelConfig, name="deriva_model")

__all__ = ["BaseConfig", "DerivaBaseConfig", "DerivaModelConfig", "base_defaults"]
