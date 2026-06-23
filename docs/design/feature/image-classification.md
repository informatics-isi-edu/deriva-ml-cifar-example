# Feature Design: Image_Classification (class + confidence on Image)

**Slug:** image-classification
**Status:** Built   <!-- Draft | Approved | Built | Validated -->
**Date:** 2026-06-23

## Purpose

A per-image classification label drawn from the 10 CIFAR-10 classes, with an
optional confidence score. It serves **two roles on the same feature** (see the
dual-purpose note below): the *ground-truth* class the model trains and is
evaluated against, and the *predicted* class (with `Confidence`) the model
emits on the test partition for downstream ROC / confusion-matrix analysis.

## Requirements

- **Target table / element:** `Image` (the asset table installed by the schema
  loader).
- **Feature type:** controlled-vocabulary term (`Image_Class`) + an optional
  scalar (`Confidence`, float 0–1). Created via `ml.create_feature("Image",
  "Image_Classification", terms=["Image_Class"], optional=["Confidence"])`.
- **Vocabulary:** `Image_Class` controlled vocabulary, **10 terms** =
  the canonical CIFAR-10 classes (`airplane, automobile, bird, cat, deer, dog,
  frog, horse, ship, truck`), defined in `src/models/cifar10_classes.py`
  (`CIFAR10_CLASSES`) and created by the schema loader. Term order defines the
  class index used by the model (`CIFAR10_CLASS_TO_IDX`).
- **Who/what writes the values:**
  - **Ground truth** — the `load-cifar10` loader execution, from the Toronto
    labels. `Confidence` is left NULL for GT rows.
  - **Predictions** — each training/test-only execution writes predicted
    `Image_Class` + populated `Confidence` (softmax max) on the test partition
    via `record_predictions`. Provenance is the producing `Execution`.

## Dual-purpose convention (important)

`Image_Classification` is written by two kinds of execution and the rows are
**not distinguishable by table membership alone**: the loader writes ground
truth (`Confidence IS NULL`); training/eval executions write predictions
(`Confidence` populated). After any model run, the same image carries multiple
rows. When reading this feature as ground truth, **filter by the loader
execution RID or by `Confidence IS NULL`** — an unfiltered read returns GT +
every recorded prediction interleaved. The `newest` selector is not a safe
substitute for "ground truth." (The catalog feature row deliberately does *not*
carry a `Source_Label`; that provenance lives in the
`prediction_probabilities.csv` asset to avoid a schema migration.)

## Validation

- **Coverage:** every loaded `Image` has exactly one ground-truth
  `Image_Classification` row (the loader writes one per image, so a freshly
  loaded catalog reports one ground-truth feature per image).
- **Sanity:** all `Image_Class` values are among the 10 vocabulary terms;
  `Confidence` (when present) is in [0, 1].
- **Provenance:** each value links to a producing `Execution` (loader for GT,
  model run for predictions).
- **Consumer can read it:** the training loop's `as_torch_dataset(...,
  targets=["Image_Classification"])` finds the GT label where it expects it;
  ROC analysis joins predictions against GT.

## Upstream designs

None — this is an **input / ground-truth feature** that sits near the bottom of
the dependency tree. The `cifar10-2layer-cnn` model design names *this* feature
as an upstream input; this feature does not point back (keeps the graph
acyclic). Its *prediction* role is downstream output of that model, but it
reuses the same feature rather than defining a separate prediction feature.

## Status & links

- **Feature name + target table:** `Image_Classification` on `Image`.
- **Vocabulary:** `Image_Class` (10 terms), in the `cifar10_test` domain schema.
- **Consumed by:** the `cifar10-2layer-cnn` model design; produced-into by all
  training / test-only experiments.
- **tacit-knowledge.md:** (none yet — link the dual-purpose convention entry
  here if/when one is written)
