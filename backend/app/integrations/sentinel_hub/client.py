"""Sentinel Hub API client.

Why a thin async wrapper instead of using `sentinelhub-py` directly:
- Their library is sync (would block our FastAPI event loop unless run in
  a thread pool — extra ceremony).
- We need fine-grained control over PU accounting, Redis token cache,
  and Ukrainian-friendly error messages.
- Smaller dependency surface; easier to mock in tests.

The library is still in pyproject.toml because some helper utilities
(BBox geometry, polygon-to-WKT) are convenient — but the HTTP plumbing
is ours.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from redis.asyncio import Redis
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

log = logging.getLogger(__name__)

OAUTH_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
STATISTICAL_URL = "https://services.sentinel-hub.com/api/v1/statistics"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"
TOKEN_REDIS_KEY = "sh:oauth_token"
TOKEN_TTL_SECONDS = 55 * 60  # access tokens live 1h; refresh 5min before expiry


class SentinelHubError(RuntimeError):
    """Raised when the Sentinel Hub API returns a non-recoverable error."""


class SentinelHubClient:
    """Async client with OAuth + Redis-cached token + retries on transient errors."""

    def __init__(self, redis: Redis, http: httpx.AsyncClient | None = None):
        self._redis = redis
        self._http = http or httpx.AsyncClient(timeout=60.0)
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ───── OAuth ─────────────────────────────────────────────────

    async def get_access_token(self) -> str:
        cached = await self._redis.get(TOKEN_REDIS_KEY)
        if cached:
            return cached.decode() if isinstance(cached, bytes) else str(cached)

        async with self._token_lock:
            # Double-check after acquiring the lock.
            cached = await self._redis.get(TOKEN_REDIS_KEY)
            if cached:
                return cached.decode() if isinstance(cached, bytes) else str(cached)

            if not settings.sh_client_id or not settings.sh_client_secret:
                raise SentinelHubError(
                    "Sentinel Hub credentials not configured (SH_CLIENT_ID/SH_CLIENT_SECRET)."
                )

            log.info("Sentinel Hub: requesting new OAuth token")
            resp = await self._http.post(
                OAUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.sh_client_id,
                    "client_secret": settings.sh_client_secret,
                },
            )
            if resp.status_code != 200:
                raise SentinelHubError(
                    f"OAuth failed: {resp.status_code} {resp.text[:200]}"
                )
            token = resp.json()["access_token"]
            await self._redis.setex(TOKEN_REDIS_KEY, TOKEN_TTL_SECONDS, token)
            return token

    # ───── HTTP ──────────────────────────────────────────────────

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Authenticated POST with retry on 429/5xx."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, _TransientHTTP)),
            reraise=True,
        ):
            with attempt:
                token = await self.get_access_token()
                resp = await self._http.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code in (429, 502, 503, 504):
                    raise _TransientHTTP(f"{resp.status_code}: {resp.text[:200]}")
                if resp.status_code == 401:
                    # Token was rejected — drop cache and retry on next iteration.
                    await self._redis.delete(TOKEN_REDIS_KEY)
                    raise _TransientHTTP("401 Unauthorized — token invalidated")
                if resp.status_code >= 400:
                    raise SentinelHubError(
                        f"Sentinel Hub error {resp.status_code}: {resp.text[:300]}"
                    )
                return resp.json()
        raise SentinelHubError("Sentinel Hub request exhausted retries")

    async def _post_bytes(self, url: str, payload: dict[str, Any]) -> bytes:
        """Same as _post_json but expects an image/raw response."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, _TransientHTTP)),
            reraise=True,
        ):
            with attempt:
                token = await self.get_access_token()
                resp = await self._http.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code in (429, 502, 503, 504):
                    raise _TransientHTTP(f"{resp.status_code}")
                if resp.status_code == 401:
                    await self._redis.delete(TOKEN_REDIS_KEY)
                    raise _TransientHTTP("401")
                if resp.status_code >= 400:
                    raise SentinelHubError(
                        f"Sentinel Hub error {resp.status_code}: {resp.text[:300]}"
                    )
                return resp.content
        raise SentinelHubError("Sentinel Hub request exhausted retries")

    # ───── Public methods (HTTP-level) ───────────────────────────

    async def post_statistical(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json(STATISTICAL_URL, payload)

    async def post_process(self, payload: dict[str, Any]) -> bytes:
        return await self._post_bytes(PROCESS_URL, payload)


class _TransientHTTP(Exception):
    """Internal marker for retryable HTTP responses (separate from network errors)."""
