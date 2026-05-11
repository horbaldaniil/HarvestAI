"""Auth gates for Week-4 endpoints (no DB-heavy logic exercised here)."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_latest_prediction_requires_auth(client: AsyncClient):
    r = await client.get("/api/fields/1/predictions/latest")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_recompute_prediction_requires_auth(client: AsyncClient):
    r = await client.post("/api/fields/1/predictions/recompute")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_list_alerts_requires_auth(client: AsyncClient):
    r = await client.get("/api/alerts")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_alerts_count_requires_auth(client: AsyncClient):
    r = await client.get("/api/alerts/count")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_ack_alert_requires_auth(client: AsyncClient):
    r = await client.post("/api/alerts/1/ack")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_alerts_stream_requires_auth(client: AsyncClient):
    r = await client.get("/api/alerts/stream")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_weather_requires_auth(client: AsyncClient):
    r = await client.get("/api/fields/1/weather")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_dashboard_requires_auth(client: AsyncClient):
    r = await client.get("/api/dashboard")
    assert r.status_code == 401
