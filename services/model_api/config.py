"""
config.py - Settings loaded from environment variables.

Uses pydantic-settings to read from a .env file (development)
or from system environment variables (production / Docker / Kubernetes).

Available environment variables:
    LOG_LEVEL               INFO | DEBUG | WARNING | ERROR  (default: INFO)
    ALLOWED_ORIGINS         comma-separated list             (default: *)
    PREDICT_RATE_LIMIT      e.g. "60/minute"                 (default: 60/minute)
    INFO_RATE_LIMIT         e.g. "120/minute"                (default: 120/minute)
    API_VERSION             e.g. "0.1.0"                     (default: 0.1.0)
    MODELS_BASE_DIR         absolute path to /models         (default: package-relative)
    REQUEST_TIMEOUT_SECONDS informational - actual timeout set in Nginx
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API
    api_title: str = "ML Monorepo - Unified Prediction API"
    api_version: str = "0.1.0"

    # CORS - comma-separated list of allowed origins
    # In development you can use "*" but restrict in production.
    # Example: "https://app.example.com,https://admin.example.com"
    allowed_origins: str = "*"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    # Rate limiting
    predict_rate_limit: str = "60/minute"   # inference endpoints (expensive)
    info_rate_limit: str = "120/minute"     # info endpoints (cheap reads)

    # Logging
    log_level: str = "INFO"

    # Timeouts - informational only; the real timeout is enforced by Nginx
    request_timeout_seconds: int = 30

    # Optional: override model paths. Leave empty to use package-relative defaults.
    models_base_dir: str = ""


# Singleton - import `settings` from any module
settings = Settings()
