"""
dataset.py - Wine quality data loading and preprocessing.

Handles both red and white wine variants.
Raw CSVs are expected at:
    data/wine/raw/winequality-red.csv
    data/wine/raw/winequality-white.csv

Processed (scaled) CSVs live at:
    data/wine/processed/red_wine_clean.csv
    data/wine/processed/white_wine_clean.csv

!  FEATURE ENGINEERING NOTE
The saved .joblib models were trained with 2 engineered features ON TOP of
the original 11 physicochemical columns.  The notebook computed:

  RED wine:
    free_total_sulfur_dioxide = free_sulfur_dioxide - total_sulfur_dioxide
    ratio_density_alcohol     = density / alcohol

  WHITE wine:
    free_total_sulfur_dioxide = free_sulfur_dioxide - total_sulfur_dioxide
    ratio_alc_dens            = alcohol / density

  Then dropped:  free sulfur dioxide, total sulfur dioxide, density
  -> 11 - 3 + 2 = 10 engineered features fed to the tree.

This module replicates that exact pipeline so predictions are consistent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -- Feature columns AFTER engineering (must match training order exactly) ----
FEATURE_COLS_RED = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "sulphates",
    "alcohol",
    "free_total_sulfur_dioxide",   # engineered: free - total
    "ratio_density_alcohol",       # engineered: density / alcohol
]

FEATURE_COLS_WHITE = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "sulphates",
    "alcohol",
    "free_total_sulfur_dioxide",   # engineered: free - total
    "ratio_alc_dens",              # engineered: alcohol / density
]

TARGET_COL = "quality"

# Monorepo root is four levels up from this file
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data" / "wine"


def engineer_features(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Apply the exact feature engineering the notebook applied before training.

    Adds two derived columns and returns the dataframe with them included.
    The raw SO2 and density columns are NOT dropped here so the caller
    can still inspect them; the model only sees FEATURE_COLS_*.
    """
    df = df.copy()
    df["free_total_sulfur_dioxide"] = (
        df["free sulfur dioxide"] - df["total sulfur dioxide"]
    )
    if kind == "red":
        df["ratio_density_alcohol"] = df["density"] / df["alcohol"]
    else:
        df["ratio_alc_dens"] = df["alcohol"] / df["density"]
    return df


class WineDataset:
    """Loads, engineers and splits wine quality data.

    Parameters
    ----------
    kind : {"white", "red"}
        Which wine variety to load.
    processed : bool
        If True, load the cleaned/scaled CSV; otherwise load the raw CSV.
    test_size : float
        Fraction of samples reserved for the test split.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        kind: str = "white",
        processed: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if kind not in ("white", "red"):
            raise ValueError(f"kind must be 'white' or 'red', got '{kind}'")
        self.kind = kind
        self.processed = processed
        self.test_size = test_size
        self.random_state = random_state
        self._df: pd.DataFrame | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols = FEATURE_COLS_RED if kind == "red" else FEATURE_COLS_WHITE

    # Public API

    def load(self) -> "WineDataset":
        """Read the CSV from disk into memory."""
        if self.processed:
            fname = f"{'white' if self.kind == 'white' else 'red'}_wine_clean.csv"
            path = _DATA_DIR / "processed" / fname
            self._df = pd.read_csv(path)
        else:
            fname = f"winequality-{'white' if self.kind == 'white' else 'red'}.csv"
            path = _DATA_DIR / "raw" / fname
            self._df = pd.read_csv(path, sep=";")

        # Apply engineering regardless of raw vs processed source
        self._df = engineer_features(self._df, self.kind)
        return self

    def get_splits(
        self,
        scale: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test).

        Parameters
        ----------
        scale : bool
            If True, apply StandardScaler fitted on the training set.
        """
        if self._df is None:
            self.load()

        X = self._df[FEATURE_COLS].values.astype(np.float64)
        y = self._df[TARGET_COL].values.astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if scale:
            self._scaler = StandardScaler()
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    # Private helpers

    def _validate(self) -> None:
        missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")
        n_nulls = self._df.isnull().sum().sum()
        if n_nulls > 0:
            raise ValueError(f"Dataset contains {n_nulls} null values - clean it first.")

    def __repr__(self) -> str:
        n = len(self._df) if self._df is not None else "not loaded"
        return f"<WineDataset kind={self.kind!r} rows={n}>"
