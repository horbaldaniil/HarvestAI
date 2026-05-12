"""Methodology endpoints — reflect data sources, sample coverage, and
per-model evaluation metrics for the magistr-grade UI.

All read-only, all authenticated (no user-specific data — same response for
everyone, but we still gate on auth to avoid leaking infrastructure detail).

Files consumed:
- backend/data/processed/model_metrics_v2.json (RF/XGBoost/LSTM metrics)
- backend/data/processed/oblast_samples.geojson (coverage map polygons)
- ModelRegistry.list_available() (live, in-memory)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.config import settings
from app.deps import CurrentUser
from app.ml.registry import get_registry

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/methodology", tags=["methodology"])

# Files are relative to the working directory at runtime (typically `backend/`).
METRICS_PATH = Path("data/processed/model_metrics_v2.json")
SAMPLES_PATH = Path("data/processed/oblast_samples.geojson")
TRAINING_SET_PATH = Path("data/processed/training_set_v2.parquet")


# ─── Pydantic schemas ─────────────────────────────────────────


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    crop: str
    family: str
    version: str
    metrics: dict[str, Any] | None
    trained_at: str | None
    feature_count: int


class MethodologyOverview(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_sources: list[str]
    oblast_count: int
    sample_count_per_oblast: int | None
    year_range: list[int]
    train_val_test_split: str
    active_algorithm: str
    models: list[ModelInfo]


class CoverageEntry(BaseModel):
    oblast: str
    oblast_uk: str | None = None
    sample_count: int
    iso_3166_2: str | None = None


class CoverageResponse(BaseModel):
    oblasts: list[CoverageEntry]
    samples_geojson: dict[str, Any]


# ─── Endpoints ────────────────────────────────────────────────


@router.get("", response_model=MethodologyOverview)
async def overview(current_user: CurrentUser) -> MethodologyOverview:
    """High-level summary of how the platform's ML pipeline is built.

    Static fields are baked in; metrics/inventory come from the live registry.
    """
    _ = current_user  # auth-only — same payload for all users

    registry = get_registry()
    models = [ModelInfo(**m) for m in registry.list_available()]

    sample_count = None
    if SAMPLES_PATH.exists():
        try:
            data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
            features = data.get("features", [])
            # Same N per oblast by construction, so the modal value is fine.
            counts: dict[str, int] = {}
            for f in features:
                ob = f.get("properties", {}).get("oblast")
                if ob:
                    counts[ob] = counts.get(ob, 0) + 1
            sample_count = max(counts.values()) if counts else None
        except (json.JSONDecodeError, KeyError):
            pass

    active = settings.active_model_family or "xgboost (v2 if available)"

    return MethodologyOverview(
        data_sources=[
            "Sentinel-2 L2A (Sentinel Hub Statistical API)",
            "Open-Meteo (historical + 14-day forecast)",
            "USDA FAS Ukraine yield 2017-2023",
            "Natural Earth Vector (oblast boundaries)",
        ],
        oblast_count=24,
        sample_count_per_oblast=sample_count,
        year_range=[2019, 2023],
        train_val_test_split="2019-2021 / 2022 / 2023 (chronological)",
        active_algorithm=active,
        models=models,
    )


@router.get("/metrics")
async def metrics(current_user: CurrentUser) -> dict[str, Any]:
    """Raw model_metrics_v2.json — used by the UI for full metric tables.

    Returns an empty `{"models": {}}` if training hasn't run yet, rather than
    404, so the frontend can render an empty state gracefully.
    """
    _ = current_user
    if not METRICS_PATH.exists():
        return {"models": {}, "note": "Run scripts/train_yield_models_v2.py to populate."}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"metrics file unparseable: {exc}") from exc


@router.get("/coverage", response_model=CoverageResponse)
async def coverage(current_user: CurrentUser) -> CoverageResponse:
    """Per-oblast sample counts + the raw GeoJSON for the coverage map.

    GeoJSON is included verbatim so the frontend can drop it straight into
    Leaflet without a second round-trip.
    """
    _ = current_user
    if not SAMPLES_PATH.exists():
        # Empty payload — frontend shows "data not collected yet" state.
        return CoverageResponse(oblasts=[], samples_geojson={"type": "FeatureCollection", "features": []})

    try:
        geojson = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"samples geojson unparseable: {exc}") from exc

    counts: dict[str, dict[str, Any]] = {}
    for f in geojson.get("features", []):
        props = f.get("properties", {})
        ob = props.get("oblast")
        if not ob:
            continue
        if ob not in counts:
            counts[ob] = {
                "oblast": ob,
                "oblast_uk": props.get("oblast_uk"),
                "iso_3166_2": props.get("iso_3166_2"),
                "sample_count": 0,
            }
        counts[ob]["sample_count"] += 1

    return CoverageResponse(
        oblasts=[CoverageEntry(**v) for v in sorted(counts.values(), key=lambda x: x["oblast"])],
        samples_geojson=geojson,
    )
