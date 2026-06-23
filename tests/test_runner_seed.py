"""Tests for the ``cifar10_cnn`` byte-reproducible-training seed knob.

Covers the D02 fix: ``cifar10_cnn(..., seed=42)`` and the
``seed_everything`` helper that seeds Python ``random``, NumPy, and
PyTorch (CPU + CUDA if present).

The end-to-end ``cifar10_cnn`` entry point requires a live DerivaML
execution, so these tests exercise the reproducibility contract at the
SimpleCNN-construction level via ``seed_everything`` directly. That's
the same code path ``cifar10_cnn`` walks at the start of every run, so
byte-identical weight init is the load-bearing guarantee.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from models.cifar10_cnn import SimpleCNN, seed_everything


def _state_dict_hash(model: torch.nn.Module) -> str:
    """Return a SHA-256 hash of every parameter tensor in ``model``.

    Two models built under the same seeded RNG should produce the same
    hash. Two models built under different seeds — or no seeding at
    all — should not.
    """
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _build_model_after_seed(seed: int) -> torch.nn.Module:
    """Seed RNGs with ``seed`` and build a fresh SimpleCNN.

    Mirrors the exact order ``cifar10_cnn`` performs:
    ``seed_everything(seed)`` first, then ``SimpleCNN(...)``.
    """
    seed_everything(seed)
    return SimpleCNN()


def test_same_seed_produces_identical_weights():
    """Two SimpleCNN builds under the same seed must hash identically.

    This is the core reproducibility contract: a Hydra override of
    ``model_config.seed=N`` should produce byte-identical model
    initialization across runs.
    """
    h1 = _state_dict_hash(_build_model_after_seed(42))
    h2 = _state_dict_hash(_build_model_after_seed(42))
    assert h1 == h2, (
        "SimpleCNN weight init is not reproducible from the seed — "
        "seed_everything() failed to cover every RNG that nn.Module "
        "default-init pulls from."
    )


def test_different_seeds_produce_different_weights():
    """Different seeds must produce different SimpleCNN weights.

    Catches the regression where ``seed`` is plumbed in but never
    actually reaches ``torch.manual_seed`` — the function would still
    "succeed" but every run would share the global RNG state.
    """
    h_42 = _state_dict_hash(_build_model_after_seed(42))
    h_123 = _state_dict_hash(_build_model_after_seed(123))
    assert h_42 != h_123, (
        "SimpleCNN weight init is identical across distinct seeds — "
        "torch.manual_seed is not being honored."
    )


def testseed_everything_covers_python_random():
    """``seed_everything`` must seed Python's ``random`` module too.

    DerivaML dataset bag adapters reach for ``random`` for stratified
    sampling and shuffles. Without this, a CIFAR-10 run would be
    reproducible-in-PyTorch but vary in batch composition.
    """
    seed_everything(42)
    a = (random.random(), random.randint(0, 1000))
    seed_everything(42)
    b = (random.random(), random.randint(0, 1000))
    assert a == b


def testseed_everything_covers_numpy():
    """``seed_everything`` must seed NumPy's global RNG too.

    ``record_test_predictions`` builds a per-image probability array
    via ``.cpu().numpy()`` and any downstream sklearn/numpy step (ROC
    analysis, stratified sampling in ``_cifar10_datasets.py``) will
    pull from this RNG.
    """
    seed_everything(42)
    a = np.random.rand(4).tolist()
    seed_everything(42)
    b = np.random.rand(4).tolist()
    assert a == b


def test_cifar10_cnn_signature_has_seed_with_default():
    """``cifar10_cnn`` must accept ``seed`` with a default — omitting it
    is the historical call shape and must keep working.

    Pinning the default to ``42`` matches ``cifar10_labeled_split`` in
    ``_cifar10_datasets.py``; if someone bumps it, this test surfaces
    the change for a deliberate review (default seeds are load-bearing
    provenance).
    """
    import inspect

    from models.cifar10_cnn import cifar10_cnn

    sig = inspect.signature(cifar10_cnn)
    assert "seed" in sig.parameters
    seed_param = sig.parameters["seed"]
    assert seed_param.default == 42
