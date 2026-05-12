"""Methodology endpoints — auth gates + graceful empty-state handling."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_methodology_overview_requires_auth(client: AsyncClient):
    r = await client.get("/api/methodology")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_methodology_metrics_requires_auth(client: AsyncClient):
    r = await client.get("/api/methodology/metrics")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_methodology_coverage_requires_auth(client: AsyncClient):
    r = await client.get("/api/methodology/coverage")
    assert r.status_code == 401
