"""
route_breast.py - Breast cancer prediction endpoints.

POST /api/v1/breast/predict
    Body: BreastInput (30 WDBC features)
    Returns: diagnosis ("Malignant" / "Benign") and confidence

GET /api/v1/breast/info
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import settings
from limiter import limiter
from domain_breast.classifier import BreastClassifier

log = logging.getLogger(__name__)

router = APIRouter()

_model: BreastClassifier | None = None
try:
    _model = BreastClassifier.load()
except FileNotFoundError as e:
    import warnings
    warnings.warn(f"Breast cancer model not found: {e}.")


class BreastInput(BaseModel):
    """
    30 WDBC nuclear features in 3 groups: mean, SE, worst.
    Ranges derived from the training dataset statistics.
    """
    # Mean features (10)
    radius_mean: float = Field(..., ge=6.98, le=28.11, example=17.99)
    texture_mean: float = Field(..., ge=9.71, le=39.28, example=10.38)
    perimeter_mean: float = Field(..., ge=43.79, le=188.5, example=122.8)
    area_mean: float = Field(..., ge=143.5, le=2501.0, example=1001.0)
    smoothness_mean: float = Field(..., ge=0.0526, le=0.1634, example=0.1184)
    compactness_mean: float = Field(..., ge=0.0194, le=0.3454, example=0.2776)
    concavity_mean: float = Field(..., ge=0.0, le=0.4268, example=0.3001)
    concave_points_mean: float = Field(..., ge=0.0, le=0.2012, example=0.1471)
    symmetry_mean: float = Field(..., ge=0.106, le=0.304, example=0.2419)
    fractal_dim_mean: float = Field(..., ge=0.05, le=0.097, example=0.07871)
    # SE features (10)
    radius_se: float = Field(..., ge=0.11, le=2.87, example=1.095)
    texture_se: float = Field(..., ge=0.36, le=4.88, example=0.9053)
    perimeter_se: float = Field(..., ge=0.76, le=21.98, example=8.589)
    area_se: float = Field(..., ge=6.8, le=542.2, example=153.4)
    smoothness_se: float = Field(..., ge=0.002, le=0.031, example=0.006399)
    compactness_se: float = Field(..., ge=0.002, le=0.135, example=0.04904)
    concavity_se: float = Field(..., ge=0.0, le=0.396, example=0.05373)
    concave_points_se: float = Field(..., ge=0.0, le=0.053, example=0.01587)
    symmetry_se: float = Field(..., ge=0.008, le=0.079, example=0.03003)
    fractal_dim_se: float = Field(..., ge=0.001, le=0.030, example=0.006193)
    # Worst features (10)
    radius_worst: float = Field(..., ge=7.93, le=36.04, example=25.38)
    texture_worst: float = Field(..., ge=12.02, le=49.54, example=17.33)
    perimeter_worst: float = Field(..., ge=50.41, le=251.2, example=184.6)
    area_worst: float = Field(..., ge=185.2, le=4254.0, example=2019.0)
    smoothness_worst: float = Field(..., ge=0.071, le=0.223, example=0.1622)
    compactness_worst: float = Field(..., ge=0.027, le=1.058, example=0.6656)
    concavity_worst: float = Field(..., ge=0.0, le=1.252, example=0.7119)
    concave_points_worst: float = Field(..., ge=0.0, le=0.291, example=0.2654)
    symmetry_worst: float = Field(..., ge=0.157, le=0.664, example=0.4601)
    fractal_dim_worst: float = Field(..., ge=0.055, le=0.208, example=0.1189)


class BreastPrediction(BaseModel):
    diagnosis: str
    label: int
    confidence: float


def _build_feature_vector(data: BreastInput) -> "np.ndarray":
    """
    Build the 30-feature vector in the exact order the model was trained on.
    Order is explicit to avoid any dependency on dict ordering.
    """
    import numpy as np
    return np.array([[
        data.radius_mean, data.texture_mean, data.perimeter_mean, data.area_mean,
        data.smoothness_mean, data.compactness_mean, data.concavity_mean,
        data.concave_points_mean, data.symmetry_mean, data.fractal_dim_mean,
        data.radius_se, data.texture_se, data.perimeter_se, data.area_se,
        data.smoothness_se, data.compactness_se, data.concavity_se,
        data.concave_points_se, data.symmetry_se, data.fractal_dim_se,
        data.radius_worst, data.texture_worst, data.perimeter_worst, data.area_worst,
        data.smoothness_worst, data.compactness_worst, data.concavity_worst,
        data.concave_points_worst, data.symmetry_worst, data.fractal_dim_worst,
    ]])


@router.post("/predict", response_model=BreastPrediction)
@limiter.limit(settings.predict_rate_limit)
def predict_breast(request: Request, data: BreastInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Breast cancer model not loaded.")

    X = _build_feature_vector(data)

    label = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    confidence = round(float(proba[label]), 4)

    log.info(
        "breast_predict | diagnosis=%s label=%d confidence=%.3f",
        "Malignant" if label == 1 else "Benign", label, confidence,
    )

    return BreastPrediction(
        diagnosis="Malignant" if label == 1 else "Benign",
        label=label,
        confidence=confidence,
    )


@router.get("/info")
def breast_info():
    return {
        "model_loaded": _model is not None,
        "features": 30,
        "target": "diagnosis (Malignant=1, Benign=0)",
        "algorithm": "Decision Tree (criterion=entropy, max_depth=8, class_weight=balanced)",
        "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
    }
