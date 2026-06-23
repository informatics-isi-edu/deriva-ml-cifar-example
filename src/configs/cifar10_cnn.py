"""Model Configuration (CIFAR-10 CNN).

This module defines model configurations for the CIFAR-10 2-layer CNN.

Configuration Group: model_config
---------------------------------
Model configurations define the hyperparameters and settings for your ML model.
Each configuration is a set of parameters that can be selected at runtime.

REQUIRED: A configuration named "default_model" must be defined.
This is used as the default model configuration when no override is specified.

All model parameters are configurable via Hydra:
- Architecture: conv1_channels, conv2_channels, hidden_size, dropout_rate
- Training: learning_rate, epochs, batch_size, weight_decay

Example usage:
    # Run with default config
    uv run src/deriva_run.py

    # Run with a specific model config
    uv run src/deriva_run.py model_config=cifar10_extended

    # Override specific parameters
    uv run src/deriva_run.py model_config.epochs=50 model_config.learning_rate=0.01
"""

from __future__ import annotations

from hydra_zen import builds, store

from models.cifar10_cnn import cifar10_cnn

# Build the base CIFAR-10 CNN configuration.
# All parameters have sensible defaults for a simple training run.
# ``seed`` defaults to 42 to match ``cifar10_labeled_split`` in
# ``src/scripts/_cifar10_datasets.py`` — picking the same canonical value
# keeps "default everything" runs byte-reproducible without forcing users
# to think about RNG state. Override with e.g. ``model_config.seed=123``
# for run-to-run variance estimates.
Cifar10CNNConfig = builds(
    cifar10_cnn,
    # Architecture parameters
    conv1_channels=32,
    conv2_channels=64,
    hidden_size=128,
    dropout_rate=0.0,
    # Training parameters
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    weight_decay=0.0,
    seed=42,
    # Hydra-zen settings
    populate_full_signature=True,
    zen_partial=True,  # Execution context added later
)

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "model_config" must match the parameter name in BaseConfig.

model_store = store(group="model_config")

# REQUIRED: default_model - used when no model config is specified
model_store(
    Cifar10CNNConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default CIFAR-10 CNN: 32→64 channels, 128 hidden units, 10 epochs, "
            "batch size 64, lr=1e-3. Balanced config for standard training runs."
        )
    },
)

# Quick training - fewer epochs for testing
model_store(
    Cifar10CNNConfig,
    name="cifar10_quick",
    epochs=3,
    batch_size=128,
    zen_meta={
        "description": (
            "Quick training config: 3 epochs, batch 128. Use for rapid iteration, "
            "debugging, and verifying the training pipeline works correctly."
        )
    },
)

# Larger model - more capacity
model_store(
    Cifar10CNNConfig,
    name="cifar10_large",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    epochs=20,
    zen_meta={
        "description": (
            "Large capacity model: 64→128 channels, 256 hidden units, 20 epochs. "
            "More parameters for potentially better accuracy on complex patterns."
        )
    },
)

# With dropout for regularization
model_store(
    Cifar10CNNConfig,
    name="cifar10_regularized",
    dropout_rate=0.25,
    weight_decay=1e-4,
    epochs=20,
    zen_meta={
        "description": (
            "Regularized config: 25% dropout, weight decay 1e-4, 20 epochs. "
            "Use to reduce overfitting when training on smaller datasets."
        )
    },
)

# Fast learning rate - may converge faster but less stable
model_store(
    Cifar10CNNConfig,
    name="cifar10_fast_lr",
    learning_rate=1e-2,
    epochs=15,
    zen_meta={
        "description": (
            "Fast learning rate (1e-2): May converge faster but can be unstable. "
            "15 epochs. Try this when default lr converges too slowly."
        )
    },
)

# Slow learning rate - more stable, may need more epochs
model_store(
    Cifar10CNNConfig,
    name="cifar10_slow_lr",
    learning_rate=1e-4,
    epochs=30,
    zen_meta={
        "description": (
            "Slow learning rate (1e-4): More stable convergence, 30 epochs. "
            "Use when training is unstable or for fine-tuning pretrained weights."
        )
    },
)

# Extended training - for best accuracy
model_store(
    Cifar10CNNConfig,
    name="cifar10_extended",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    dropout_rate=0.25,
    weight_decay=1e-4,
    learning_rate=1e-3,
    epochs=50,
    zen_meta={
        "description": (
            "Extended training for best accuracy: Large model (64→128 ch, 256 hidden), "
            "regularization (dropout 0.25, weight decay 1e-4), 50 epochs. "
            "Use for final production training when accuracy is the priority."
        )
    },
)

# Test-only mode - load weights and run evaluation without training
# Use with assets=cifar10_small_experiment_weights and a testing dataset
model_store(
    Cifar10CNNConfig,
    name="cifar10_test_only",
    test_only=True,
    weights_filename="cifar10_cnn_weights.pt",
    zen_meta={
        "description": (
            "Test-only mode: Skips training, loads pretrained weights from assets, "
            "runs inference on dataset. Requires assets with cifar10_cnn_weights.pt. "
            "Use for evaluation and generating predictions on new data."
        )
    },
)
