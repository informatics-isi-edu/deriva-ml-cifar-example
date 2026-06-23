"""Tests for the analyst join + metrics helpers (``src/scripts/analyst_join.py``).

These verify the pure data-shaping and metric logic an analyst relies on:
the ground-truth/prediction split (tk-003 dual-purpose feature convention),
the joined wide-table construction, per-run and per-class accuracy, confusion
matrices, and one-vs-rest ROC AUC. No catalog access — all fixtures are
in-memory DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyst_join import (
    build_joined_table,
    confusion,
    macro_micro_auc,
    micro_roc_curve,
    per_class_accuracy,
    per_run_accuracy,
    split_ground_truth_and_predictions,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Dual-purpose feature rows: 2 GT rows + 2 prediction rows for one run."""
    return pd.DataFrame(
        {
            "Image": ["IMG1", "IMG2", "IMG1", "IMG2"],
            "Image_Class": ["bird", "cat", "bird", "dog"],
            "Confidence": [None, None, 0.91, 0.55],
            "Execution": ["LOAD", "LOAD", "RUN1", "RUN1"],
        }
    )


def test_split_separates_gt_from_predictions(feature_df: pd.DataFrame) -> None:
    gt, pred = split_ground_truth_and_predictions(feature_df)
    assert set(gt["Execution"]) == {"LOAD"}
    assert set(pred["Execution"]) == {"RUN1"}
    assert len(gt) == 2 and len(pred) == 2


def test_split_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        split_ground_truth_and_predictions(pd.DataFrame({"Image": ["A"]}))


def test_build_joined_table_marks_correctness(feature_df: pd.DataFrame) -> None:
    gt, pred = split_ground_truth_and_predictions(feature_df)
    table = build_joined_table(gt, pred, {"RUN1": "run1"})
    assert list(table.columns) == [
        "Image",
        "true_class",
        "run1_pred",
        "run1_conf",
        "run1_correct",
    ]
    # IMG1: true bird, pred bird -> correct; IMG2: true cat, pred dog -> wrong
    by_image = table.set_index("Image")
    assert bool(by_image.loc["IMG1", "run1_correct"]) is True
    assert bool(by_image.loc["IMG2", "run1_correct"]) is False


def test_build_joined_table_multiple_runs() -> None:
    gt = pd.DataFrame({"Image": ["A", "B"], "Image_Class": ["cat", "dog"]})
    pred = pd.DataFrame(
        {
            "Image": ["A", "B", "A", "B"],
            "Image_Class": ["cat", "cat", "dog", "dog"],
            "Confidence": [0.9, 0.4, 0.3, 0.8],
            "Execution": ["R1", "R1", "R2", "R2"],
        }
    )
    table = build_joined_table(gt, pred, {"R1": "r1", "R2": "r2"})
    assert per_run_accuracy(table, ["r1", "r2"]) == {"r1": 0.5, "r2": 0.5}


def test_build_joined_table_empty_run_raises() -> None:
    gt = pd.DataFrame({"Image": ["A"], "Image_Class": ["cat"]})
    pred = pd.DataFrame(
        {
            "Image": ["A"],
            "Image_Class": ["cat"],
            "Confidence": [0.9],
            "Execution": ["R1"],
        }
    )
    with pytest.raises(ValueError, match="no prediction rows"):
        build_joined_table(gt, pred, {"R2": "missing"})


def test_per_run_accuracy() -> None:
    table = pd.DataFrame({"r_correct": [True, True, False, True]})
    assert per_run_accuracy(table, ["r"]) == {"r": 0.75}


def test_per_class_accuracy() -> None:
    table = pd.DataFrame(
        {
            "true_class": ["cat", "cat", "dog", "dog"],
            "r_correct": [True, False, True, True],
        }
    )
    acc = per_class_accuracy(table, "r", ["cat", "dog"])
    assert acc["cat"] == 0.5
    assert acc["dog"] == 1.0


def test_per_class_accuracy_missing_class_is_nan() -> None:
    table = pd.DataFrame({"true_class": ["cat"], "r_correct": [True]})
    acc = per_class_accuracy(table, "r", ["cat", "dog"])
    assert acc["cat"] == 1.0
    assert np.isnan(acc["dog"])


def test_confusion_matrix_counts() -> None:
    table = pd.DataFrame(
        {
            "true_class": ["cat", "cat", "dog"],
            "r_pred": ["cat", "dog", "dog"],
        }
    )
    cm = confusion(table, "r", ["cat", "dog"])
    assert cm.loc["cat", "cat"] == 1
    assert cm.loc["cat", "dog"] == 1
    assert cm.loc["dog", "dog"] == 1


def test_macro_micro_auc_perfect_separation() -> None:
    y = pd.Series(["cat", "cat", "dog", "dog"])
    probs = pd.DataFrame(
        {
            "cat": [0.9, 0.8, 0.2, 0.1],
            "dog": [0.1, 0.2, 0.8, 0.9],
        }
    )
    out = macro_micro_auc(y, probs, ["cat", "dog"])
    assert out["macro_auc"] == pytest.approx(1.0)
    assert out["micro_auc"] == pytest.approx(1.0)


def test_macro_auc_skips_absent_class() -> None:
    # 'frog' never appears in ground truth -> skipped in the macro average,
    # but 'cat' and 'dog' both have positives and negatives so their
    # per-class AUC is well-defined, leaving macro finite.
    y = pd.Series(["cat", "dog", "cat", "dog"])
    probs = pd.DataFrame(
        {
            "cat": [0.8, 0.2, 0.7, 0.3],
            "dog": [0.2, 0.8, 0.3, 0.7],
            "frog": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = macro_micro_auc(y, probs, ["cat", "dog", "frog"])
    assert not np.isnan(out["macro_auc"])
    assert out["macro_auc"] == pytest.approx(1.0)


def test_micro_roc_curve_returns_monotone_arrays() -> None:
    y = pd.Series(["cat", "dog", "cat", "dog"])
    probs = pd.DataFrame({"cat": [0.7, 0.3, 0.6, 0.4], "dog": [0.3, 0.7, 0.4, 0.6]})
    fpr, tpr, auc_val = micro_roc_curve(y, probs, ["cat", "dog"])
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert 0.0 <= auc_val <= 1.0
