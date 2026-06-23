"""Dataset Configurations.

This module declares the *names* of dataset groups the experiments and
notebooks reference. Each entry is intentionally **empty by default** — the
checked-in template ships without RIDs because RIDs are catalog-specific and
would be stale on any fresh clone.

After running ``load-cifar10`` against your own catalog, fill in the RIDs and
versions printed by the loader (or read them with
``ml.find_datasets()``). Two recommended patterns:

1. **Edit this file in place.** Replace each empty list with a
   ``DatasetSpecConfig(rid=..., version=...)``. Wrap with
   ``with_description(..., "...")`` if you want a description that appears in
   ``deriva-ml-run --info``.

2. **Add a per-environment override.** Create
   ``src/configs/dev/datasets_<env>.py`` registering ``<name>_<env>`` configs
   in the same ``datasets`` group, then select on the CLI:
   ``deriva-ml-run datasets=cifar10_small_labeled_split_<env>`` (see
   ``dev/datasets_localhost.py`` for a worked example).

The empty defaults pass config validation but will fail at execution time
("Dataset '' not found") — which is the desired behavior: a fresh clone must
not silently run against someone else's RIDs.

Configuration Group: ``datasets``
"""

from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig  # noqa: F401  (re-exported for users editing this file)
from deriva_ml.execution import with_description  # noqa: F401

datasets_store = store(group="datasets")

# -----------------------------------------------------------------------------
# Empty placeholders — fill in for your catalog before running.
# -----------------------------------------------------------------------------
# Example (after running load-cifar10):
#
#   datasets_store(
#       with_description(
#           [DatasetSpecConfig(rid="28FA", version="0.21.0")],
#           "Complete CIFAR-10 dataset (10,000 images).",
#       ),
#       name="cifar10_complete",
#   )

datasets_store([], name="cifar10_complete")

# -----------------------------------------------------------------------------
# Original Toronto split — 50K training / 10K test_batch
#
# Training images and test images come from *different* Toronto source batches.
# Use this family when you want to train on the full 50K training set and
# evaluate against the official 10K test_batch (the "standard" CIFAR-10 split).
# Both halves carry ground-truth labels in the Toronto distribution.
# -----------------------------------------------------------------------------
datasets_store([], name="cifar10_split")
datasets_store([], name="cifar10_training")
datasets_store([], name="cifar10_testing")

# Note: ``cifar10_small_split`` is intentionally absent. The parent
# ``Small_Split`` dataset was dropped in the v1.42 migration —
# ``Small_Training`` and ``Small_Testing`` are now sibling subsample()
# outputs of the same execution, discoverable via that execution's
# outputs rather than via a parent dataset. See
# ``deriva-ml/docs/superpowers/specs/2026-06-01-split-partition-tag-and-subsample-design.md``.
datasets_store([], name="cifar10_small_training")
datasets_store([], name="cifar10_small_testing")

# -----------------------------------------------------------------------------
# Training-derived holdout split — 80/20 (or 400/100) of training images only
#
# Both the training and testing partitions are drawn from the 50K Toronto
# training images; the test_batch images are *not* used here. Created by
# split_dataset() with a fixed seed (42). Use this family for cross-validation
# workflows, ROC analysis, or experiments where the test_batch must stay
# unseen for final evaluation.
# -----------------------------------------------------------------------------
datasets_store([], name="cifar10_labeled_split")
datasets_store([], name="cifar10_labeled_training")
datasets_store([], name="cifar10_labeled_testing")

datasets_store([], name="cifar10_small_labeled_split")
datasets_store([], name="cifar10_small_labeled_training")
datasets_store([], name="cifar10_small_labeled_testing")

# -----------------------------------------------------------------------------
# Special-case configs (always empty by design)
# -----------------------------------------------------------------------------

# Notebooks (e.g., ROC analysis) that consume asset RIDs, not datasets.
datasets_store([], name="no_datasets")

# Script-only experiments that manage their own data.
datasets_store([], name="none")

# REQUIRED: ``default_dataset`` is used when no dataset override is specified.
# Set this to your most-frequently-used dataset after editing the configs above.
datasets_store([], name="default_dataset")
