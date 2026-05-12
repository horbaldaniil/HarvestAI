"""Auth + payload-shape tests for the Phase 5 v3 methodology endpoints.

We don't deeply verify metric values — that's covered by
`test_evaluate_models.py`. Here we just guard the HTTP contract:
auth required, payload structure usable by the frontend.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient


# ─── /api/methodology/evaluation_v3 ───────────────────────


async def test_evaluation_v3_requires_auth(client: AsyncClient):
    r = await client.get("/api/methodology/evaluation_v3")
    assert r.status_code == 401


async def test_evaluation_v3_returns_dict_when_authed(client: AsyncClient):
    """We need an authenticated session — register + login first."""
    await client.post("/api/auth/register", json={
        "email": "v3test@example.com", "password": "TestPassword!1",
    })
    login = await client.post("/api/auth/login", json={
        "email": "v3test@example.com", "password": "TestPassword!1",
    })
    token = login.json().get("access_token") or login.json().get("token")
    assert token, f"expected access_token in {login.json()}"

    r = await client.get(
        "/api/methodology/evaluation_v3",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "crops" in body
    assert "metadata" in body or "note" in body


# ─── /api/methodology/leaderboard ─────────────────────────


async def test_leaderboard_requires_auth(client: AsyncClient):
    r = await client.get("/api/methodology/leaderboard")
    assert r.status_code == 401


async def test_leaderboard_returns_rows_when_authed(client: AsyncClient):
    await client.post("/api/auth/register", json={
        "email": "lbtest@example.com", "password": "TestPassword!1",
    })
    login = await client.post("/api/auth/login", json={
        "email": "lbtest@example.com", "password": "TestPassword!1",
    })
    token = login.json().get("access_token") or login.json().get("token")
    assert token

    r = await client.get(
        "/api/methodology/leaderboard",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body
    assert isinstance(body["rows"], list)
    # If evaluation has been run, rows should be present and have the right shape.
    eval_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "evaluation_v3.json"
    if eval_path.exists():
        assert len(body["rows"]) > 0, "evaluation_v3.json exists but leaderboard is empty"
        row = body["rows"][0]
        # Headline columns the leaderboard UI needs.
        for col in ("crop", "family", "test_r2", "test_mae", "test_rmse", "test_mape"):
            assert col in row, f"missing column {col} in leaderboard row"


# ─── Eval JSON sanity — direct file inspection ────────────


def test_evaluation_v3_json_has_expected_structure_if_present():
    """When the file exists, it must have the structure the endpoints
    + the frontend depend on. If it's not generated yet, skip."""
    eval_path = (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "evaluation_v3.json"
    )
    if not eval_path.exists():
        pytest.skip("evaluation_v3.json not generated")

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    assert "crops" in data
    assert "metadata" in data
    metadata = data["metadata"]
    assert "features_origin" in metadata
    assert "feature_names" in metadata
    assert len(metadata["feature_names"]) == 17
    assert "families" in metadata


def test_evaluation_v3_has_per_oblast_residuals_when_evaluated():
    """Spot-check: at least one (crop, family) pair must include
    per_oblast_residuals — drives the choropleth map."""
    eval_path = (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "evaluation_v3.json"
    )
    if not eval_path.exists():
        pytest.skip("evaluation_v3.json not generated")
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    found = False
    for crop, by_family in data.get("crops", {}).items():
        for family, body in by_family.items():
            if not isinstance(body, dict) or body.get("skipped"):
                continue
            if "per_oblast_residuals" in body and body["per_oblast_residuals"]:
                found = True
                row = body["per_oblast_residuals"][0]
                # Schema check
                assert "iso_3166_2" in row
                assert "mean_abs_residual" in row
                return
    if not found:
        pytest.fail("No per_oblast_residuals found in any (crop, family)")
