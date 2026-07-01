"""
core_ml - Shared ML foundations.

Exports the abstract BaseModel interface and pure metric functions
so every domain package builds on the same contract.
"""
from .base_model import BaseModel
from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix_counts,
)

__all__ = [
    "BaseModel",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix_counts",
]
