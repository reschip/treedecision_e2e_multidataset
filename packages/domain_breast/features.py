"""
features.py - Feature extraction for Breast Cancer Wisconsin (WDBC) dataset.

The WDBC dataset contains 30 real-valued features computed from a
digitized image of a fine needle aspirate (FNA) of a breast mass.
They describe characteristics of the cell nuclei present in the image.

Data source:
    data/breast/raw/wdbc.data          (original UCI format, no header)
    data/breast/processed/wdbc_cleaned.csv  (cleaned, labeled columns)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data" / "breast"

# Column names for the raw wdbc.data file (no header row)
_RAW_COLUMNS = [
    "id",
    "diagnosis",  # M = Malignant, B = Benign
    # Ten real-valued features computed for each nucleus x 3 stats (mean, se, worst)
    *[f"{feat}_{stat}"
      for stat in ("mean", "se", "worst")
      for feat in (
          "radius", "texture", "perimeter", "area", "smoothness",
          "compactness", "concavity", "concave_points", "symmetry", "fractal_dim"
      )],
]

FEATURE_COLS = [c for c in _RAW_COLUMNS if c not in ("id", "diagnosis")]
TARGET_COL = "diagnosis"


class BreastFeatures:
    """Loads and preprocesses the WDBC dataset.

    Parameters
    ----------
    processed : bool
        Load the cleaned CSV (True) or parse the raw .data file (False).
    test_size : float
    random_state : int
    """

    def __init__(
        self,
        processed: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.processed = processed
        self.test_size = test_size
        self.random_state = random_state
        self._df: pd.DataFrame | None = None
        self._scaler: StandardScaler | None = None

    # Public API

    def load(self) -> "BreastFeatures":
        if self.processed:
            path = _DATA_DIR / "processed" / "wdbc_cleaned.csv"
            self._df = pd.read_csv(path)
        else:
            path = _DATA_DIR / "raw" / "wdbc.data"
            self._df = pd.read_csv(path, header=None, names=_RAW_COLUMNS)
            self._df["diagnosis"] = self._df["diagnosis"].map({"M": 1, "B": 0})
            self._df.drop(columns=["id"], inplace=True)
        return self

    def get_splits(
        self, scale: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test)."""
        if self._df is None:
            self.load()

        feature_cols = [c for c in self._df.columns if c != TARGET_COL]
        X = self._df[feature_cols].values.astype(np.float64)
        y = self._df[TARGET_COL].values

        # Encode diagnosis strings to int if needed
        if y.dtype == object:
            y = np.where(y == "M", 1, 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        if scale:
            self._scaler = StandardScaler()
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def __repr__(self) -> str:
        n = len(self._df) if self._df is not None else "not loaded"
        return f"<BreastFeatures rows={n}>"
