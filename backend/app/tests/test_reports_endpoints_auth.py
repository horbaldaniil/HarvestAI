"""Auth gates for the PDF report endpoints."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_field_report_requires_auth(client: AsyncClient):
    r = await client.post("/api/fields/1/report")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_portfolio_report_requires_auth(client: AsyncClient):
    r = await client.post("/api/portfolio/report")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_builder_requires_auth(client: AsyncClient):
    r = await client.post("/api/reports/builder", json={"kind": "portfolio"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_list_reports_requires_auth(client: AsyncClient):
    r = await client.get("/api/reports")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_download_report_requires_auth(client: AsyncClient):
    r = await client.get("/api/reports/1/download")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_delete_report_requires_auth(client: AsyncClient):
    r = await client.delete("/api/reports/1")
    assert r.status_code == 401
