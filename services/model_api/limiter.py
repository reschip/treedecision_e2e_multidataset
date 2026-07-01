"""
limiter.py - Rate limiting singleton (slowapi).

Exported to be shared by main.py (registers the error handler)
and by each router (applies per-endpoint limits).

In development: uses in-process memory (resets on restart).
In production with multiple workers: switch to Redis backend:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri="redis://redis:6379",
    )
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Limits are applied per client IP.
# With Nginx as reverse proxy, pass X-Real-IP so slowapi
# sees the real client IP instead of 127.0.0.1.
limiter = Limiter(key_func=get_remote_address)
