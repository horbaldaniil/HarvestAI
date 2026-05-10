"""Integration tests for the auth flow: register → login → me → refresh → logout."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_then_login_then_me(client: AsyncClient):
    # Register
    r = await client.post(
        "/api/auth/register",
        json={
            "email": "farmer@example.com",
            "password": "very-strong-pass",
            "full_name": "Іван Фермер",
        },
    )
    assert r.status_code == 201, r.text
    user = r.json()
    assert user["email"] == "farmer@example.com"
    assert user["full_name"] == "Іван Фермер"
    assert user["locale"] == "uk"

    # Duplicate register → 409
    r2 = await client.post(
        "/api/auth/register",
        json={"email": "farmer@example.com", "password": "another-strong"},
    )
    assert r2.status_code == 409

    # Login
    r3 = await client.post(
        "/api/auth/login",
        json={"email": "farmer@example.com", "password": "very-strong-pass"},
    )
    assert r3.status_code == 200, r3.text
    tok = r3.json()
    assert tok["token_type"] == "bearer"
    assert tok["access_token"]
    assert "harvestai_refresh" in r3.cookies

    # /me
    r4 = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {tok['access_token']}"},
    )
    assert r4.status_code == 200
    assert r4.json()["email"] == "farmer@example.com"


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    await client.post(
        "/api/auth/register",
        json={"email": "x@example.com", "password": "actual-password-123"},
    )
    r = await client.post(
        "/api/auth/login",
        json={"email": "x@example.com", "password": "wrong-password"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_me_without_token_returns_401(client: AsyncClient):
    r = await client.get("/api/auth/me")
    assert r.status_code == 401
