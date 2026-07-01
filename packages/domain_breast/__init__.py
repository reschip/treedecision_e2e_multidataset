"""
domain_breast - Isolated library for Breast Cancer Wisconsin classification.

Exports:
    BreastFeatures   - feature loading from the WDBC dataset
    BreastClassifier - SVM wrapper (wdbc_model.joblib)
"""
from .features import BreastFeatures
from .classifier import BreastClassifier

__all__ = ["BreastFeatures", "BreastClassifier"]
