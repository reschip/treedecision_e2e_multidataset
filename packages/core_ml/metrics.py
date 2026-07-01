"""
metrics.py - Pure mathematical functions for model evaluation.

All functions operate on plain Python lists or NumPy arrays and have
no side effects, making them trivially testable and reusable across
every domain package.

No sklearn wrappers here - these are first-principles implementations
that make the math explicit and auditable.
"""
from __future__ import annotations

import math
from typing import Sequence


# Helpers

def _validate(y_true: Sequence, y_pred: Sequence) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_pred length ({len(y_pred)})"
        )
    if len(y_true) == 0:
        raise ValueError("Input sequences must not be empty.")


# Classification metrics

def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Fraction of correct predictions.

    accuracy = (TP + TN) / N
    """
    _validate(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix_counts(
    y_true: Sequence, y_pred: Sequence, positive_label=1
) -> dict[str, int]:
    """Return TP, FP, FN, TN counts for a binary classifier."""
    _validate(y_true, y_pred)
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_label and p == positive_label:
            tp += 1
        elif t != positive_label and p == positive_label:
            fp += 1
        elif t == positive_label and p != positive_label:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision(y_true: Sequence, y_pred: Sequence, positive_label=1) -> float:
    """Precision = TP / (TP + FP).  Returns 0.0 when denominator is zero."""
    counts = confusion_matrix_counts(y_true, y_pred, positive_label)
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else 0.0


def recall(y_true: Sequence, y_pred: Sequence, positive_label=1) -> float:
    """Recall = TP / (TP + FN).  Returns 0.0 when denominator is zero."""
    counts = confusion_matrix_counts(y_true, y_pred, positive_label)
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else 0.0


def f1_score(y_true: Sequence, y_pred: Sequence, positive_label=1) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    denom = p + r
    return 2 * p * r / denom if denom else 0.0


# Regression metrics (bonus - used in future regression models)

def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
