"""
dataset.py - Adult Income data loading and preprocessing.

Raw data:    data/adultincome/raw/adult.csv
Processed:   data/adultincome/processed/adult_clean.csv

Target variable: income (">50K" -> 1, "<=50K" -> 0)

ENCODING NOTE
The notebook used pd.get_dummies() (One-Hot Encoding), NOT Label Encoding.
The model was trained on 41 binary columns (after dropping one dummy per
group to avoid multicollinearity) plus 5 numeric columns.

The exact feature order the saved .joblib expects is:

    NUMERIC (5):
        age, educational-num, capital-gain, capital-loss, hours-per-week

    ONE-HOT - marital-status (6 dummies, "Divorced" dropped as base):
        marital-status_Married-AF-spouse
        marital-status_Married-civ-spouse
        marital-status_Married-spouse-absent
        marital-status_Never-married
        marital-status_Separated
        marital-status_Widowed

    ONE-HOT - relationship (5 dummies, "Husband" dropped as base):
        relationship_Not-in-family
        relationship_Other-relative
        relationship_Own-child
        relationship_Unmarried
        relationship_Wife

    ONE-HOT - race (4 dummies, "Amer-Indian-Eskimo" dropped as base):
        race_Asian-Pac-Islander
        race_Black
        race_Other
        race_White

    ONE-HOT - workclass (6 dummies, "Federal-gov" dropped as base):
        workclass_Local-gov
        workclass_Private
        workclass_Self-emp-inc
        workclass_Self-emp-not-inc
        workclass_State-gov
        workclass_Without-pay

    ONE-HOT - occupation (13 dummies, "Adm-clerical" dropped as base):
        occupation_Armed-Forces
        occupation_Craft-repair
        occupation_Exec-managerial
        occupation_Farming-fishing
        occupation_Handlers-cleaners
        occupation_Machine-op-inspct
        occupation_Other-service
        occupation_Priv-house-serv
        occupation_Prof-specialty
        occupation_Protective-serv
        occupation_Sales
        occupation_Tech-support
        occupation_Transport-moving

    ONE-HOT - native-country (1 dummy):
        native-country_United-States

    ONE-HOT - gender (1 dummy):
        gender_Male

Total = 5 + 6 + 5 + 4 + 6 + 13 + 1 + 1 = 41 features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data" / "adultincome"

TARGET_COL = "income"

# Exact column order the saved model expects (41 features).
# This list must match the output of _apply_ohe() below.
FEATURE_COLS = [
    "age", "educational-num", "capital-gain", "capital-loss", "hours-per-week",
    # marital-status (base = "Divorced")
    "marital-status_Married-AF-spouse",
    "marital-status_Married-civ-spouse",
    "marital-status_Married-spouse-absent",
    "marital-status_Never-married",
    "marital-status_Separated",
    "marital-status_Widowed",
    # relationship (base = "Husband")
    "relationship_Not-in-family",
    "relationship_Other-relative",
    "relationship_Own-child",
    "relationship_Unmarried",
    "relationship_Wife",
    # race (base = "Amer-Indian-Eskimo")
    "race_Asian-Pac-Islander",
    "race_Black",
    "race_Other",
    "race_White",
    # workclass (base = "Federal-gov")
    "workclass_Local-gov",
    "workclass_Private",
    "workclass_Self-emp-inc",
    "workclass_Self-emp-not-inc",
    "workclass_State-gov",
    "workclass_Without-pay",
    # occupation (base = "Adm-clerical")
    "occupation_Armed-Forces",
    "occupation_Craft-repair",
    "occupation_Exec-managerial",
    "occupation_Farming-fishing",
    "occupation_Handlers-cleaners",
    "occupation_Machine-op-inspct",
    "occupation_Other-service",
    "occupation_Priv-house-serv",
    "occupation_Prof-specialty",
    "occupation_Protective-serv",
    "occupation_Sales",
    "occupation_Tech-support",
    "occupation_Transport-moving",
    # native-country (simplified to US vs other)
    "native-country_United-States",
    # gender
    "gender_Male",
]


def apply_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the notebook's pd.get_dummies() pipeline.

    Steps:
      1. Strip whitespace from string columns.
      2. Drop rows with '?' in workclass / occupation.
      3. Drop: fnlwgt, education (keep educational-num), workclass-raw string.
      4. One-hot encode categorical columns (drop_first=True matches notebook).
      5. Ensure all expected dummy columns exist (fill missing with 0).
      6. Return only FEATURE_COLS in exact order.
    """
    df = df.copy()

    # Normalise column names and strip whitespace
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Remove unknown values
    df = df[df["workclass"] != "?"]
    df = df[df["occupation"] != "?"]

    # Drop columns the notebook dropped
    drop_cols = [c for c in ["fnlwgt", "education", "ID"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Encode target if present
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({">50K": 1, "<=50K": 0})

    # One-hot encode (drop_first=True - alphabetically lowest category dropped)
    cat_cols = ["marital-status", "relationship", "race",
                "workclass", "occupation", "native-country", "gender"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure all expected feature columns exist (handles unseen categories in API)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df


class AdultDataset:
    """Loads and preprocesses the Adult Census Income dataset.

    Parameters
    ----------
    processed : bool
        If True, load cleaned CSV; otherwise raw CSV.
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

    def load(self) -> "AdultDataset":
        if self.processed:
            path = _DATA_DIR / "processed" / "adult_clean.csv"
        else:
            path = _DATA_DIR / "raw" / "adult.csv"
        df = pd.read_csv(path)
        self._df = apply_ohe(df)
        return self

    def get_splits(
        self,
        scale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test)."""
        if self._df is None:
            self.load()

        X = self._df[FEATURE_COLS].values.astype(np.float64)
        y = self._df[TARGET_COL].values.astype(np.int64)

        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def __repr__(self) -> str:
        n = len(self._df) if self._df is not None else "not loaded"
        return f"<AdultDataset rows={n} features={len(FEATURE_COLS)}>"
