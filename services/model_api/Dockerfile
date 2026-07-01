# Stage 1: Build
# Install dependencies in a separate directory so the runtime image
# does not include pip, setuptools, or other build tools.
FROM python:3.11-slim AS builder

WORKDIR /app

COPY services/model_api/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages (no pip or compilers)
COPY --from=builder /install /usr/local

# Copy application source
COPY packages/          ./packages/
COPY services/model_api/ ./services/model_api/

# PYTHONPATH includes:
#   /app/packages           - domain_wine, domain_breast, domain_adultincome, core_ml
#   /app/services/model_api - config, limiter (imported by routers)
ENV PYTHONPATH="/app/packages:/app/services/model_api"

WORKDIR /app/services/model_api

# Run as non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# workers=1 keeps in-memory rate limiting consistent.
# To scale with multiple workers, configure Redis as the limiter backend (see limiter.py).
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "5"]
