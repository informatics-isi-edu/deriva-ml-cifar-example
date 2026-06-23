"""
Dataset generation script configurations.

Each config wraps a generation function from src/scripts/ with its parameters.
These are registered under the "script_config" hydra group and invoked via
`uv run deriva-ml-run +experiment=<name>` where the experiment overrides
script_config instead of model_config.
"""

from hydra_zen import store

script_store = store(group="script_config")

# "none" placeholder — used as default when no script is needed
script_store(None, name="none")
