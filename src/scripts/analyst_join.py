"""Analyst join + metrics helpers for comparing model runs.

This module holds the *pure* data-shaping and metric functions an analyst
needs to compare several training runs against shared ground truth: building
the joined wide table (one row per evaluation image carrying the ground-truth
label plus each run's prediction), computing per-run top-1 accuracy, per-class
accuracy, confusion matrices, and one-vs-rest ROC/AUC from per-class softmax
probabilities.

No catalog calls and no RIDs live here — every function takes plain pandas
DataFrames. The catalog-facing driver (``scripts/analyst_analysis.py``) pulls
the feature rows and prediction CSVs, then hands the DataFrames to these
functions. Keeping the logic RID-free makes it reusable template config (not
catalog-specific) and unit-testable without a live catalog.

Ground-truth convention (see tacit-knowledge tk-003): the dual-purpose
``Image_Classification`` feature stores ground-truth rows with
``Confidence IS NULL`` and prediction rows with ``Confidence`` populated.
:func:`split_ground_truth_and_predictions` enforces that split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def _one_vs_rest_binary(true_classes: pd.Series, classes: list[str]) -> np.ndarray:
    """One-hot indicator matrix, one column per class, in ``classes`` order.

    ``sklearn.preprocessing.label_binarize`` collapses the two-class case to a
    single column, which breaks the one-vs-rest ravel used for micro-averaging.
    This wrapper always returns an ``(n_samples, n_classes)`` 0/1 matrix.

    Args:
        true_classes: Ground-truth class name per sample.
        classes: Class ordering for the columns.

    Returns:
        A ``(len(true_classes), len(classes))`` int array of 0/1 indicators.
    """
    binarized = label_binarize(true_classes, classes=classes)
    if binarized.shape[1] == 1:  # sklearn collapsed the 2-class case
        return np.hstack([1 - binarized, binarized])
    return binarized


def split_ground_truth_and_predictions(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dual-purpose feature table into ground truth vs predictions.

    The ``Image_Classification`` feature is written by two kinds of execution:
    the loader writes ground-truth rows (no confidence) and training
    executions write prediction rows (confidence populated). The two are
    distinguished only by whether ``Confidence`` is null (tk-003).

    Args:
        feature_df: Rows from the ``Image_Classification`` feature, with at
            least the columns ``Image``, ``Image_Class``, ``Confidence``,
            ``Execution``.

    Returns:
        ``(ground_truth_df, predictions_df)``. ``ground_truth_df`` has the
        rows where ``Confidence`` is null; ``predictions_df`` the rest.

    Raises:
        KeyError: If the required columns are absent.

    Example:
        >>> df = pd.DataFrame(
        ...     {
        ...         "Image": ["A", "A"],
        ...         "Image_Class": ["bird", "horse"],
        ...         "Confidence": [None, 0.42],
        ...         "Execution": ["LOAD", "RUN1"],
        ...     }
        ... )
        >>> gt, pred = split_ground_truth_and_predictions(df)
        >>> list(gt["Image_Class"]), list(pred["Image_Class"])
        (['bird'], ['horse'])
    """
    for col in ("Image", "Image_Class", "Confidence", "Execution"):
        if col not in feature_df.columns:
            raise KeyError(f"feature_df is missing required column {col!r}")
    is_gt = feature_df["Confidence"].isna()
    return feature_df[is_gt].copy(), feature_df[~is_gt].copy()


def build_joined_table(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    run_labels: dict[str, str],
) -> pd.DataFrame:
    """Build the wide joined table: one row per evaluation image.

    Each row carries the image RID, its ground-truth class, and — per run —
    that run's predicted class, confidence, and a correctness flag. The image
    set is the ground-truth images that *every* run predicted on (inner join),
    so each run's column is fully populated.

    Args:
        ground_truth_df: Ground-truth rows (``Image``, ``Image_Class``).
        predictions_df: Prediction rows (``Image``, ``Image_Class``,
            ``Confidence``, ``Execution``).
        run_labels: Mapping of execution RID -> human-readable run label
            (e.g. ``{"SR8": "quick"}``). Column names use the label.

    Returns:
        A DataFrame indexed 0..N-1 with columns ``Image``, ``true_class``,
        and for each run ``<label>_pred``, ``<label>_conf``,
        ``<label>_correct``.

    Raises:
        ValueError: If a run in ``run_labels`` has no prediction rows, or if
            the runs disagree on which images they scored.

    Example:
        >>> gt = pd.DataFrame({"Image": ["A", "B"], "Image_Class": ["bird", "cat"]})
        >>> pred = pd.DataFrame(
        ...     {
        ...         "Image": ["A", "B"],
        ...         "Image_Class": ["bird", "dog"],
        ...         "Confidence": [0.9, 0.5],
        ...         "Execution": ["R1", "R1"],
        ...     }
        ... )
        >>> t = build_joined_table(gt, pred, {"R1": "run1"})
        >>> list(t["run1_correct"])
        [True, False]
    """
    gt = ground_truth_df[["Image", "Image_Class"]].rename(
        columns={"Image_Class": "true_class"}
    )
    table = gt.copy()
    image_sets: list[set[str]] = []
    for rid, label in run_labels.items():
        run_rows = predictions_df[predictions_df["Execution"] == rid]
        if run_rows.empty:
            raise ValueError(f"run {rid!r} ({label}) has no prediction rows")
        run_rows = run_rows[["Image", "Image_Class", "Confidence"]].rename(
            columns={
                "Image_Class": f"{label}_pred",
                "Confidence": f"{label}_conf",
            }
        )
        image_sets.append(set(run_rows["Image"]))
        table = table.merge(run_rows, on="Image", how="inner")

    common = set.intersection(*image_sets) if image_sets else set()
    for s in image_sets:
        if s != common:
            raise ValueError(
                "runs scored different image sets; "
                f"intersection has {len(common)} images but a run scored {len(s)}"
            )

    for _rid, label in run_labels.items():
        table[f"{label}_correct"] = table[f"{label}_pred"] == table["true_class"]
    return table.reset_index(drop=True)


def per_run_accuracy(joined: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    """Compute top-1 accuracy per run from the joined table.

    Args:
        joined: Output of :func:`build_joined_table`.
        labels: Run labels to score (each must have a ``<label>_correct`` col).

    Returns:
        Mapping ``label -> accuracy`` in [0, 1].

    Example:
        >>> t = pd.DataFrame({"run1_correct": [True, False, True, True]})
        >>> per_run_accuracy(t, ["run1"])
        {'run1': 0.75}
    """
    return {label: float(joined[f"{label}_correct"].mean()) for label in labels}


def per_class_accuracy(
    joined: pd.DataFrame, label: str, classes: list[str]
) -> pd.Series:
    """Per-class top-1 accuracy for one run.

    Args:
        joined: Output of :func:`build_joined_table`.
        label: Run label.
        classes: Class names (rows of the returned Series, in order).

    Returns:
        Series indexed by class name, value = accuracy on images whose
        ground-truth class is that class. NaN if a class has no images.

    Example:
        >>> t = pd.DataFrame(
        ...     {
        ...         "true_class": ["cat", "cat", "dog"],
        ...         "r_correct": [True, False, True],
        ...     }
        ... )
        >>> per_class_accuracy(t, "r", ["cat", "dog"]).to_dict()
        {'cat': 0.5, 'dog': 1.0}
    """
    out = {}
    for cls in classes:
        mask = joined["true_class"] == cls
        out[cls] = (
            float(joined.loc[mask, f"{label}_correct"].mean()) if mask.any() else np.nan
        )
    return pd.Series(out, index=classes)


def confusion(joined: pd.DataFrame, label: str, classes: list[str]) -> pd.DataFrame:
    """Confusion matrix (rows=true, cols=predicted) for one run.

    Args:
        joined: Output of :func:`build_joined_table`.
        label: Run label.
        classes: Class ordering for both axes.

    Returns:
        DataFrame of integer counts, index = true class, columns = predicted.

    Example:
        >>> t = pd.DataFrame(
        ...     {"true_class": ["cat", "cat"], "r_pred": ["cat", "dog"]}
        ... )
        >>> confusion(t, "r", ["cat", "dog"]).loc["cat", "dog"]
        1
    """
    cm = confusion_matrix(joined["true_class"], joined[f"{label}_pred"], labels=classes)
    return pd.DataFrame(cm, index=classes, columns=classes)


def macro_micro_auc(
    true_classes: pd.Series,
    prob_df: pd.DataFrame,
    classes: list[str],
) -> dict[str, float]:
    """One-vs-rest macro and micro ROC AUC from per-class softmax probs.

    Args:
        true_classes: Ground-truth class name per image (aligned with
            ``prob_df`` row order).
        prob_df: One column per class (named exactly as ``classes``), value =
            softmax probability for that class.
        classes: Class ordering. ``prob_df`` must have these as columns.

    Returns:
        Mapping with keys ``macro_auc`` and ``micro_auc`` (each in [0, 1]).
        A class absent from the ground truth is skipped in the macro average.

    Example:
        >>> y = pd.Series(["cat", "dog"])
        >>> probs = pd.DataFrame({"cat": [0.8, 0.3], "dog": [0.2, 0.7]})
        >>> round(macro_micro_auc(y, probs, ["cat", "dog"])["macro_auc"], 3)
        1.0
    """
    y_bin = _one_vs_rest_binary(true_classes, classes)
    scores = prob_df[classes].to_numpy()

    per_class_auc: list[float] = []
    for i, _cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:  # class not present in ground truth
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        per_class_auc.append(auc(fpr, tpr))
    macro = float(np.mean(per_class_auc)) if per_class_auc else float("nan")

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), scores.ravel())
    micro = float(auc(fpr_micro, tpr_micro))
    return {"macro_auc": macro, "micro_auc": micro}


def micro_roc_curve(
    true_classes: pd.Series,
    prob_df: pd.DataFrame,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Micro-averaged ROC curve points + AUC for plotting.

    Args:
        true_classes: Ground-truth class name per image (row-aligned with
            ``prob_df``).
        prob_df: Per-class softmax probability columns named as ``classes``.
        classes: Class ordering.

    Returns:
        ``(fpr, tpr, auc_value)`` — arrays for the micro-averaged ROC curve
        and its scalar AUC.

    Example:
        >>> y = pd.Series(["cat", "dog"])
        >>> probs = pd.DataFrame({"cat": [0.8, 0.3], "dog": [0.2, 0.7]})
        >>> fpr, tpr, a = micro_roc_curve(y, probs, ["cat", "dog"])
        >>> 0.0 <= a <= 1.0
        True
    """
    y_bin = _one_vs_rest_binary(true_classes, classes)
    scores = prob_df[classes].to_numpy()
    fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
    return fpr, tpr, float(auc(fpr, tpr))
