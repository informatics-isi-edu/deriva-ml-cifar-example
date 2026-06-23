"""Smoke test that all hydra-zen configs register without error.

Catches the common breakage class where deriva-ml API changes (renamed
fields, removed kwargs, moved imports) silently desync from the template's
config modules. A failure here means a downstream `deriva-ml-run` invocation
would fail at startup.
"""

from hydra_zen import store


EXPECTED_MODEL_CONFIGS = {
    "default_model",
    "cifar10_quick",
    "cifar10_large",
    "cifar10_regularized",
    "cifar10_fast_lr",
    "cifar10_slow_lr",
    "cifar10_extended",
    "cifar10_test_only",
}

EXPECTED_EXPERIMENTS = {
    "cifar10_quick",
    "cifar10_default",
    "cifar10_extended",
    "cifar10_quick_full",
    "cifar10_extended_full",
    "cifar10_test_only",
}


def _registered_names(group: str) -> set[str]:
    """Return the set of config names registered under a hydra-zen group."""
    return {entry["name"] for entry in store if entry["group"] == group}


def test_load_all_configs_registers_expected_groups():
    """Loading the configs package registers every documented group."""
    from configs import load_all_configs

    load_all_configs()

    assert EXPECTED_MODEL_CONFIGS <= _registered_names("model_config")
    assert EXPECTED_EXPERIMENTS <= _registered_names("experiment")
    assert _registered_names("datasets")
    assert _registered_names("assets")
    assert _registered_names("workflow")
    assert _registered_names("deriva_ml")
