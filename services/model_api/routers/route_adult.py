"""
route_adult.py - Adult income prediction endpoints.

POST /api/v1/adult/predict
    Body: AdultInput (raw census fields - strings + numbers)
    Returns: income_prediction (">50K" / "<=50K") and probability

The router applies the same One-Hot Encoding the notebook used before
calling the model. The user sends human-readable strings; the router
converts them into the 41-feature binary vector the model expects.

GET /api/v1/adult/info
"""

import logging
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import settings
from limiter import limiter
from domain_adultincome.classifier import AdultClassifier
from domain_adultincome.dataset import FEATURE_COLS, apply_ohe

log = logging.getLogger(__name__)
router = APIRouter()

_model: AdultClassifier | None = None
try:
    _model = AdultClassifier.load()
except FileNotFoundError as e:
    import warnings
    warnings.warn(f"Adult income model not found: {e}.")


# Literal types on every categorical field.
# Pydantic rejects unknown values with 422 before the request reaches the model.
class AdultInput(BaseModel):
    """Raw census fields with strict categorical validation."""
    age: int = Field(..., ge=17, le=90, example=39)
    educational_num: int = Field(..., ge=1, le=16, example=13,
                                  description="Numeric education level 1-16")
    capital_gain: int = Field(..., ge=0, le=99999, example=2174)
    capital_loss: int = Field(..., ge=0, le=4356, example=0)
    hours_per_week: int = Field(..., ge=1, le=99, example=40)

    workclass: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay"
    ] = Field(..., example="State-gov")

    marital_status: Literal[
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ] = Field(..., example="Never-married")

    occupation: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ] = Field(..., example="Adm-clerical")

    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ] = Field(..., example="Not-in-family")

    race: Literal[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ] = Field(..., example="White")

    gender: Literal["Male", "Female"] = Field(..., example="Male")

    native_country: str = Field(
        ..., example="United-States",
        description="Any country string. Unknown countries map to non-US silently."
    )


class AdultPrediction(BaseModel):
    income_prediction: str
    label: int
    probability_over_50k: float


def _build_feature_vector(data: AdultInput) -> np.ndarray:
    """
    Convert raw AdultInput into the 41-feature OHE vector the model expects.

    Builds a one-row DataFrame mirroring the notebook's column names,
    then calls apply_ohe() (the same function used during training).
    """
    row = {
        "age": data.age,
        "educational-num": data.educational_num,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "workclass": data.workclass,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "gender": data.gender,
        "native-country": data.native_country,
    }
    df = pd.DataFrame([row])
    df_ohe = apply_ohe(df)
    return df_ohe[FEATURE_COLS].values.astype(np.float64)


@router.post("/predict", response_model=AdultPrediction)
@limiter.limit(settings.predict_rate_limit)
def predict_adult(request: Request, data: AdultInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Adult income model not loaded.")

    X = _build_feature_vector(data)
    label = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    prob_over = round(float(proba[1]), 4)

    log.info(
        "adult_predict | income=%s label=%d prob_over_50k=%.3f age=%d occupation=%s",
        ">50K" if label == 1 else "<=50K", label, prob_over,
        data.age, data.occupation,
    )

    return AdultPrediction(
        income_prediction=">50K" if label == 1 else "<=50K",
        label=label,
        probability_over_50k=prob_over,
    )


@router.get("/info")
def adult_info():
    return {
        "model_loaded": _model is not None,
        "user_provides": "12 raw census fields (strings + numbers)",
        "model_sees": f"{len(FEATURE_COLS)} OHE binary features",
        "encoding": "pd.get_dummies(drop_first=True) - same as notebook",
        "target": "income (>50K=1, <=50K=0)",
        "algorithm": "Decision Tree",
        "dataset": "Adult Census Income (UCI / Kaggle)",
        "category_validation": "Strict - unknown categorical values return HTTP 422",
    }
