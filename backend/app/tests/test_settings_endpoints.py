"""Auth gates for /api/settings endpoints."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_crop_prices_requires_auth(client: AsyncClient):
    r = await client.get("/api/settings/crop-prices")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_update_crop_prices_requires_auth(client: AsyncClient):
    r = await client.put("/api/settings/crop-prices", json={"wheat": 8500})
    assert r.status_code == 401
