"""
domain_wine - Isolated library for wine quality classification.

Exports:
    WineDataset     - data loading and preprocessing
    WineClassifier  - production model wrapper (vino_blanco / vino_tinto)
"""
from .dataset import WineDataset
from .architecture import WineClassifier

__all__ = ["WineDataset", "WineClassifier"]
