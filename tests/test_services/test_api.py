"""
test_api.py - Integration tests for the FastAPI endpoints.

Uses httpx.AsyncClient with the ASGI transport to send real HTTP
requests without needing a running server.
"""
import sys
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "model_api"))

from main import app  # noqa: E402 - import after sys.path manipulation


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# -- Health & Root -------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200


# -- Wine endpoints ------------------------------------------------------------

@pytest.mark.asyncio
async def test_wine_info(client):
    response = await client.get("/api/v1/wine/info")
    assert response.status_code == 200
    data = response.json()
    assert "models_loaded" in data


@pytest.mark.asyncio
async def test_wine_predict_white(client):
    payload = {
        "kind": "white",
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45.0,
        "total_sulfur_dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
    }
    response = await client.post("/api/v1/wine/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_quality" in data
    assert 3 <= data["predicted_quality"] <= 9


# -- Breast endpoints ----------------------------------------------------------

@pytest.mark.asyncio
async def test_breast_info(client):
    response = await client.get("/api/v1/breast/info")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_breast_predict(client):
    # Use first WDBC sample (malignant)
    payload = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
        "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
        "concavity_mean": 0.3001, "concave_points_mean": 0.1471,
        "symmetry_mean": 0.2419, "fractal_dim_mean": 0.07871,
        "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589,
        "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904,
        "concavity_se": 0.05373, "concave_points_se": 0.01587,
        "symmetry_se": 0.03003, "fractal_dim_se": 0.006193,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6,
        "area_worst": 2019.0, "smoothness_worst": 0.1622, "compactness_worst": 0.6656,
        "concavity_worst": 0.7119, "concave_points_worst": 0.2654,
        "symmetry_worst": 0.4601, "fractal_dim_worst": 0.1189,
    }
    response = await client.post("/api/v1/breast/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["diagnosis"] in ("Malignant", "Benign")
    assert 0.0 <= data["confidence"] <= 1.0


# -- Adult endpoints -----------------------------------------------------------

@pytest.mark.asyncio
async def test_adult_info(client):
    response = await client.get("/api/v1/adult/info")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_adult_predict(client):
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    response = await client.post("/api/v1/adult/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["income_prediction"] in (">50K", "<=50K")
    assert 0.0 <= data["probability_over_50k"] <= 1.0
