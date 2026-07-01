"""
route_wine.py - Wine quality prediction endpoints.

POST /api/v1/wine/predict
    Body: WineInput (11 raw physicochemical features + wine kind).
    The router applies the same feature engineering the notebook used
    before passing data to the model.
    Returns: predicted quality class and class probabilities.

GET /api/v1/wine/info

Feature engineering applied internally (matches the notebook exactly):
    RED wine:
        free_total_sulfur_dioxide = free_sulfur_dioxide - total_sulfur_dioxide
        ratio_density_alcohol     = density / alcohol
        Model receives 9 features.

    WHITE wine:
        free_total_sulfur_dioxide = free_sulfur_dioxide - total_sulfur_dioxide
        ratio_alc_dens            = alcohol / density
        Model receives 9 features.
"""

import logging
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import settings
from limiter import limiter
from domain_wine.architecture import WineClassifier

log = logging.getLogger(__name__)

# Maps the 3-class discretization the notebook used back to human labels
QUALITY_LABELS: dict[int, str] = {0: "baja", 1: "media", 2: "alta"}

router = APIRouter()

_models: dict[str, WineClassifier] = {}
try:
    _models["white"] = WineClassifier.load("white")
    _models["red"] = WineClassifier.load("red")
except FileNotFoundError as e:
    import warnings
    warnings.warn(f"Wine models not found: {e}. /predict will return 503.")


class WineInput(BaseModel):
    """
    Raw physicochemical inputs. Feature engineering is handled internally.
    The user does not need to compute derived features.
    """
    kind: Literal["white", "red"] = Field("white", description="'white' or 'red'")
    fixed_acidity: float = Field(..., ge=3.0, le=16.0, example=7.0)
    volatile_acidity: float = Field(..., ge=0.08, le=1.60, example=0.27)
    citric_acid: float = Field(..., ge=0.0, le=1.70, example=0.36)
    residual_sugar: float = Field(..., ge=0.6, le=66.0, example=20.7)
    chlorides: float = Field(..., ge=0.009, le=0.611, example=0.045)
    free_sulfur_dioxide: float = Field(..., ge=1.0, le=289.0, example=45.0)
    total_sulfur_dioxide: float = Field(..., ge=6.0, le=440.0, example=170.0)
    density: float = Field(..., ge=0.987, le=1.040, example=1.001)
    pH: float = Field(..., ge=2.72, le=4.01, example=3.0)
    sulphates: float = Field(..., ge=0.22, le=2.00, example=0.45)
    alcohol: float = Field(..., ge=8.0, le=15.0, example=8.8)


class WinePrediction(BaseModel):
    kind: str
    quality_class: int
    quality_label: str
    probabilities: dict[str, str]


def _build_feature_vector(data: WineInput) -> np.ndarray:
    """
    Replicates the notebook's preprocessing pipeline for a single sample.

    RED (9 features):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, sulphates, alcohol,
        free_total_sulfur_dioxide, ratio_density_alcohol

    WHITE (9 features):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, sulphates, alcohol,
        free_total_sulfur_dioxide, ratio_alc_dens
    """
    free_total = data.free_sulfur_dioxide - data.total_sulfur_dioxide

    if data.kind == "red":
        ratio = data.density / data.alcohol
    else:
        ratio = data.alcohol / data.density

    return np.array([[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.sulphates,
        data.alcohol,
        free_total,
        ratio,
    ]])


@router.post("/predict", response_model=WinePrediction)
@limiter.limit(settings.predict_rate_limit)
def predict_wine(request: Request, data: WineInput):
    if data.kind not in _models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{data.kind}' wine is not loaded."
        )

    clf = _models[data.kind]
    X = _build_feature_vector(data)

    prediction = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    classes = clf._model.classes_
    label = QUALITY_LABELS.get(prediction, str(prediction))

    log.info(
        "wine_predict | kind=%s quality_class=%d quality_label=%s confidence=%.3f",
        data.kind, prediction, label, float(proba.max()),
    )

    return WinePrediction(
        kind=data.kind,
        quality_class=prediction,
        quality_label=label,
        probabilities={
            QUALITY_LABELS.get(int(c), str(int(c))): f"{float(p):.4f}"
            for c, p in zip(classes, proba)
        },
    )


@router.get("/info")
def wine_info():
    return {
        "models_loaded": list(_models.keys()),
        "user_provides": "11 raw physicochemical features",
        "model_sees": "9 engineered features",
        "engineering": {
            "free_total_sulfur_dioxide": "free_sulfur_dioxide - total_sulfur_dioxide",
            "ratio_red": "density / alcohol",
            "ratio_white": "alcohol / density",
        },
        "target": "quality class (0=baja, 1=media, 2=alta)",
        "algorithm": "Decision Tree",
    }
