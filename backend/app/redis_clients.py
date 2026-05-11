"""Lazy Redis client singletons.

Two flavours:
- `get_async_redis()` — used inside FastAPI request handlers and async services
  (PU tracker, SSE service, OAuth token cache, etc.).
- `get_sync_redis()` — used inside RQ jobs, which run in a sync worker process.

They share the same Redis instance (configured by `REDIS_URL`) but are
independent connection pools; no cross-contamination.
"""
from __future__ import annotations

import redis as sync_redis
import redis.asyncio as async_redis

from app.config import settings

_async_client: async_redis.Redis | None = None
_sync_client: sync_redis.Redis | None = None


def get_async_redis() -> async_redis.Redis:
    global _async_client
    if _async_client is None:
        _async_client = async_redis.from_url(settings.redis_url, decode_responses=False)
    return _async_client


def get_sync_redis() -> sync_redis.Redis:
    global _sync_client
    if _sync_client is None:
        _sync_client = sync_redis.from_url(settings.redis_url, decode_responses=False)
    return _sync_client


async def close_async_redis() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
