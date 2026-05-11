"""Auth and ownership tests for observation endpoints.

These tests don't reach the Sentinel Hub or PostGIS — they verify that the
HTTP layer rejects unauthenticated and cross-user requests before any
expensive work happens.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_observations_requires_auth(client: AsyncClient):
    r = await client.get("/api/fields/1/observations")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_refresh_requires_auth(client: AsyncClient):
    r = await client.post("/api/fields/1/observations/refresh")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_heatmap_request_requires_auth(client: AsyncClient):
    r = await client.post("/api/fields/1/observations/2024-07-01/heatmap?index=ndvi")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_get_heatmap_png_requires_auth(client: AsyncClient):
    r = await client.get("/api/fields/1/heatmaps/2024-07-01_ndvi.png")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_jobs_status_requires_auth(client: AsyncClient):
    r = await client.get("/api/jobs/nonexistent")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_quota_requires_auth(client: AsyncClient):
    r = await client.get("/api/quota")
    assert r.status_code == 401
