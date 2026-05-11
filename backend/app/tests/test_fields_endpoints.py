"""Endpoint-level tests for the fields router.

We don't exercise actual PostGIS persistence here — that requires a real
Postgres+PostGIS instance and is covered by integration tests (added in
week 6). These tests verify the contracts that don't depend on the DB:
- Unauthenticated calls return 401
- Validation rejects bad input before the DB roundtrip
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _register_and_get_token(client: AsyncClient) -> str:
    await client.post(
        "/api/auth/register",
        json={"email": "fieldsuser@example.com", "password": "very-strong-pass"},
    )
    r = await client.post(
        "/api/auth/login",
        json={"email": "fieldsuser@example.com", "password": "very-strong-pass"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_list_fields_requires_auth(client: AsyncClient):
    r = await client.get("/api/fields")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_create_field_requires_auth(client: AsyncClient):
    r = await client.post("/api/fields", json={})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_get_field_requires_auth(client: AsyncClient):
    r = await client.get("/api/fields/1")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_create_field_rejects_invalid_polygon_before_db(client: AsyncClient):
    """Polygon validation runs in the service before any DB I/O, so even
    without PostGIS we can verify that bad geometry returns 400."""
    token = await _register_and_get_token(client)

    # Too-small square (~0.01 ha)
    payload = {
        "name": "Тестове поле",
        "crop_type": "wheat",
        "season_year": 2026,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [24.030, 49.840],
                    [24.0301, 49.840],
                    [24.0301, 49.8401],
                    [24.030, 49.8401],
                    [24.030, 49.840],
                ]
            ],
        },
    }
    r = await client.post(
        "/api/fields",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "мал" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_field_rejects_invalid_crop_type(client: AsyncClient):
    token = await _register_and_get_token(client)
    payload = {
        "name": "Поле",
        "crop_type": "rice",  # not in CropType enum
        "season_year": 2026,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [24.0, 49.0],
                    [24.001, 49.0],
                    [24.001, 49.001],
                    [24.0, 49.001],
                    [24.0, 49.0],
                ]
            ],
        },
    }
    r = await client.post(
        "/api/fields",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_create_field_rejects_invalid_season_year(client: AsyncClient):
    token = await _register_and_get_token(client)
    payload = {
        "name": "Поле",
        "crop_type": "wheat",
        "season_year": 1500,  # out of range
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [24.0, 49.0],
                    [24.01, 49.0],
                    [24.01, 49.01],
                    [24.0, 49.01],
                    [24.0, 49.0],
                ]
            ],
        },
    }
    r = await client.post(
        "/api/fields",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 422
