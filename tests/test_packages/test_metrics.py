"""
test_metrics.py - Unit tests for core_ml.metrics.

Each function is tested with:
  - A known correct case
  - Edge cases (all-correct, all-wrong, empty input guard)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages"))

from core_ml.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix_counts,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)


# -- accuracy -----------------------------------------------------------------

class TestAccuracy:
    def test_perfect_prediction(self):
        assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert accuracy([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0

    def test_half_correct(self):
        assert accuracy([0, 1, 0, 1], [0, 1, 1, 0]) == 0.5

    def test_single_element(self):
        assert accuracy([1], [1]) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy([0, 1], [0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy([], [])


# -- confusion matrix ----------------------------------------------------------

class TestConfusionMatrix:
    def test_all_true_positives(self):
        counts = confusion_matrix_counts([1, 1, 1], [1, 1, 1])
        assert counts == {"tp": 3, "fp": 0, "fn": 0, "tn": 0}

    def test_mixed(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        counts = confusion_matrix_counts(y_true, y_pred)
        assert counts["tp"] == 1
        assert counts["fp"] == 1
        assert counts["fn"] == 1
        assert counts["tn"] == 1


# -- precision / recall / f1 ---------------------------------------------------

class TestPrecisionRecallF1:
    def test_precision_perfect(self):
        assert precision([1, 1], [1, 1]) == 1.0

    def test_precision_zero_division(self):
        # No positive predictions -> precision = 0
        assert precision([1, 1], [0, 0]) == 0.0

    def test_recall_perfect(self):
        assert recall([1, 1], [1, 1]) == 1.0

    def test_f1_balanced(self):
        # precision=0.5, recall=1.0 -> f1 = 2/3 ~= 0.667
        score = f1_score([1, 1], [0, 1])
        assert abs(score - 2 / 3) < 1e-9


# -- regression metrics --------------------------------------------------------

class TestRegressionMetrics:
    def test_mse_zero(self):
        assert mean_squared_error([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_mse_known_value(self):
        # MSE([0,4], [2,2]) = (4 + 4) / 2 = 4.0
        assert mean_squared_error([0.0, 4.0], [2.0, 2.0]) == 4.0

    def test_rmse(self):
        assert root_mean_squared_error([0.0, 4.0], [2.0, 2.0]) == 2.0

    def test_mae(self):
        assert mean_absolute_error([0.0, 4.0], [2.0, 2.0]) == 2.0
