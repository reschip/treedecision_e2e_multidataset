"""
main.py - FastAPI server for the ML monorepo.

Mounts three domain routers:
  /api/v1/wine    ->  route_wine.py
  /api/v1/breast  ->  route_breast.py
  /api/v1/adult   ->  route_adult.py

Run locally:
  uvicorn main:app --reload --port 8000
"""
import logging
import logging.config
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "packages"))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from limiter import limiter
from routers import route_wine, route_breast, route_adult

# Logging
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"level": settings.log_level, "handlers": ["console"]},
    "loggers": {
        "uvicorn.access": {"level": "WARNING"},
        "uvicorn.error":  {"level": "INFO"},
    },
})

log = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    description=(
        "Single entry point for wine quality, breast cancer, "
        "and adult income classification models."
    ),
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - origins are read from ALLOWED_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Routers
app.include_router(route_wine.router,   prefix="/api/v1/wine",   tags=["Wine Quality"])
app.include_router(route_breast.router, prefix="/api/v1/breast", tags=["Breast Cancer"])
app.include_router(route_adult.router,  prefix="/api/v1/adult",  tags=["Adult Income"])


# Global error handler - prevents internal details from leaking to the client
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error(
        "Unhandled exception on %s %s | %s: %s",
        request.method, request.url.path,
        type(exc).__name__, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please contact the API maintainer."},
    )


@app.on_event("startup")
async def on_startup():
    log.info("ML Monorepo API starting up (v%s)", settings.api_version)
    log.info("CORS origins: %s", settings.origins_list)
    log.info("Predict rate limit: %s", settings.predict_rate_limit)
    log.info("Wine models loaded: %s", list(route_wine._models.keys()))
    log.info("Breast model loaded: %s", route_breast._model is not None)
    log.info("Adult model loaded: %s", route_adult._model is not None)


@app.on_event("shutdown")
async def on_shutdown():
    log.info("ML Monorepo API shutting down.")


@app.get("/health", tags=["Meta"])
def health_check():
    all_loaded = (
        bool(route_wine._models)
        and route_breast._model is not None
        and route_adult._model is not None
    )
    status = "ok" if all_loaded else "degraded"
    code = 200 if all_loaded else 503
    body = {
        "status": status,
        "version": settings.api_version,
        "models": {
            "wine":   list(route_wine._models.keys()),
            "breast": route_breast._model is not None,
            "adult":  route_adult._model is not None,
        },
    }
    return JSONResponse(content=body, status_code=code)


@app.get("/version", tags=["Meta"])
def model_versions():
    """Returns the saved_at timestamp and type of each loaded model."""
    return {
        "api_version": settings.api_version,
        "models": {
            "wine_white": getattr(route_wine._models.get("white"), "meta", {}),
            "wine_red":   getattr(route_wine._models.get("red"),   "meta", {}),
            "breast":     getattr(route_breast._model, "meta", {}),
            "adult":      getattr(route_adult._model,  "meta", {}),
        },
    }


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "ML Monorepo API is running. Visit /docs for the interactive UI.",
        "endpoints": ["/api/v1/wine", "/api/v1/breast", "/api/v1/adult"],
    }
