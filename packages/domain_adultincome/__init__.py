"""
domain_adultincome - Isolated library for Adult Income classification.

Predicts whether a person earns >50K/yr based on census data.

Exports:
    AdultDataset       - data loading and preprocessing
    AdultClassifier    - Decision Tree wrapper
"""
from .dataset import AdultDataset
from .classifier import AdultClassifier

__all__ = ["AdultDataset", "AdultClassifier"]
