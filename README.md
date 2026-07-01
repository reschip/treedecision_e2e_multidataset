# ML Monorepo

A production-grade monorepo for three independent ML classification systems, unified behind a single FastAPI service.

For a detailed technical guide on architectures, preprocessing steps (OHE, feature engineering), logging, and design decisions, see the [API Guide](file:///Users/reschip/Downloads/Projects/monorepo/api_guide.md).

## Domains

| Domain | Algorithm | Dataset | Target |
|---|---|---|---|
| Wine Quality | Decision Tree | UCI Wine Quality | quality class (baja/media/alta) |
| Breast Cancer | Decision Tree | WDBC | Malignant / Benign |
| Adult Income | Decision Tree | UCI Adult Census | >50K / ≤50K |

## Structure

```
monorepo/
├── .github/workflows/     # CI: tests + linting on every push
├── infrastructure/        # Kubernetes manifests (placeholder)
├── data/                  # Local data cloud — ignored by Git
│   ├── wine/
│   ├── breast/
│   └── adultincome/
├── notebooks/             # R&D notebooks (one folder per domain)
├── packages/              # Internal libraries
│   ├── core_ml/           # Shared interfaces + pure metrics
│   ├── domain_wine/       # Wine dataset + classifier
│   ├── domain_breast/     # WDBC features + classifier
│   └── domain_adultincome/# Adult income dataset + classifier
├── services/
│   ├── model_api/         # Unified FastAPI server
│   └── data_pipeline/     # Overnight retraining script
└── tests/
    ├── test_packages/     # Unit tests (metrics, model interfaces)
    └── test_services/     # Integration tests (HTTP endpoints)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r services/model_api/requirements.txt
```

### 2. Run the API locally

```bash
cd services/model_api
PYTHONPATH=../../packages uvicorn main:app --reload --port 8000
```

Then visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 3. Run tests

```bash
# Unit tests
pytest tests/test_packages/ -v

# Integration tests
pytest tests/test_services/ -v --asyncio-mode=auto
```

### 4. Retrain all models

```bash
PYTHONPATH=packages python services/data_pipeline/run_retraining.py
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/version` | Loaded model metadata and build timestamps |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/api/v1/wine/predict` | Wine quality class prediction (baja/media/alta) |
| `POST` | `/api/v1/breast/predict` | Breast cancer diagnosis (Malignant/Benign) |
| `POST` | `/api/v1/adult/predict` | Income bracket prediction (>50K/<=50K) |
| `GET` | `/api/v1/{domain}/info` | Model metadata |

## Architecture

```
packages/core_ml          ← Abstract interface + pure math
       ↑        ↑        ↑
domain_wine  domain_breast  domain_adultincome
       ↑        ↑        ↑
        services/model_api   ← FastAPI mounts all routers
              ↑
         services/data_pipeline  ← Batch retraining
```
