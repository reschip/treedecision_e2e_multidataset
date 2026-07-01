# Root Dockerfile — reuses the multi-stage build in services/model_api/
# This file exists so DigitalOcean App Platform detects the repo as a
# Docker Web Service instead of a Serverless Functions project.

# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

COPY services/model_api/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY packages/          ./packages/
COPY services/model_api/ ./services/model_api/

ENV PYTHONPATH="/app/packages:/app/services/model_api"

WORKDIR /app/services/model_api

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "5"]
