"""Define experiments.

Experiments are pre-configured combinations of model, dataset, and asset settings.
They use Hydra's defaults list to override specific config groups and inherit from
the main DerivaModelConfig.

Usage:
    # Run a single experiment
    uv run deriva-ml-run +experiment=cifar10_quick

    # Run multiple experiments using a multirun config
    uv run deriva-ml-run +multirun=quick_vs_extended

    # Override experiment settings
    uv run deriva-ml-run +experiment=cifar10_quick datasets=cifar10_small_training

For hyperparameter sweeps and grid searches, use multirun configs defined in
configs/multiruns.py - they are self-contained and don't require separate
experiment definitions.

IMPORTANT: When overriding optional fields (like script_config), set them to
MISSING in make_config() so Hydra fills them from the defaults list instead of
using the base's None default, which would shadow the resolved value.

Reference:
    https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig

# Use _global_ package to allow overrides at the root level
experiment_store = store(group="experiment", package="_global_")

# =============================================================================
# CIFAR-10 CNN Experiments
# =============================================================================
# These experiments use the CIFAR-10 CNN model with different configurations.
# Each experiment inherits from DerivaModelConfig (a builds() of run_model)
# and overrides specific config groups.

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Quick CIFAR-10 training: 3 epochs, 32->64 channels, batch size 128 for fast validation",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "cifar10_small_training"},
        ],
        description="Default CIFAR-10 training: 10 epochs, 32->64 channels, standard hyperparameters",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_default",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Extended CIFAR-10 training: 50 epochs, 64->128 channels, dropout 0.25, weight decay 1e-4",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended",
)

# =============================================================================
# Capacity-comparison family on the small labeled split
# =============================================================================
# These three experiments hold the dataset constant at
# ``cifar10_small_labeled_split`` (labeled on both partitions, leak-free)
# and vary only model capacity + training duration. Because every run
# evaluates on the *same* held-out test partition, the resulting
# accuracies are directly comparable — capacity-vs-accuracy isolated from
# any data confound. ``cifar10_quick`` (above) is the low end of the same
# family (3 epochs, 32->64 ch); these add the middle and high ends so a
# downstream analyst can plot a clean capacity sweep. All three write
# predictions (Image_Classification feature rows + a probability CSV) on
# the held-out test partition.

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Default CIFAR-10 on small labeled split: 10 epochs, 32->64 channels, lr=1e-3, batch 64",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_small_default",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_large"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Large CIFAR-10 on small labeled split: 20 epochs, 64->128 channels, 256 hidden units",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_small_large",
)

# =============================================================================
# Full Dataset Experiments
# =============================================================================

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_labeled_split"},
        ],
        description="Quick CIFAR-10 on full dataset: 3 epochs, 32->64 channels for baseline validation",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick_full",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_labeled_split"},
        ],
        description="Extended CIFAR-10 on full dataset: 50 epochs, 64->128 channels, full regularization",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended_full",
)

# =============================================================================
# Test-Only Experiment
# =============================================================================
# Evaluate pre-trained model on test data without training

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_test_only"},
            {"override /datasets": "cifar10_small_labeled_testing"},
        ],
        description="CIFAR-10 evaluation only: load pre-trained weights and evaluate on labeled test set",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_test_only",
)
