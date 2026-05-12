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
# Phase 4 v3 deliverables — the scientific metric panel JSON + an
# overview powering the leaderboard / residual map / SHAP charts.
EVAL_V3_PATH = Path("data/processed/evaluation_v3.json")
EVAL_V4_PATH = Path("data/processed/evaluation_v4.json")
EVAL_V5_PATH = Path("data/processed/evaluation_v5.json")
EVAL_V5_REAL_PATH = Path("data/processed/evaluation_v5_real_only.json")
EVAL_V6_PATH = Path("data/processed/evaluation_v6.json")
EVAL_V7_HYBRID_PATH = Path("data/processed/evaluation_v7_hybrid.json")
METRICS_V3_PATH = Path("data/processed/model_metrics_v3.json")
METRICS_V4_PATH = Path("data/processed/model_metrics_v4.json")
METRICS_V5_PATH = Path("data/processed/model_metrics_v5.json")
METRICS_V6_PATH = Path("data/processed/model_metrics_v6.json")
TRAINING_V3_PATH = Path("data/processed/training_set_v3.parquet")


def _pick_eval_path() -> Path:
    """Prefer the v6 (real-only-trained) evaluation when present.

    Order:
      1. `evaluation_v6.json` — production models trained on real-only
         Держстат yields (2018-2021), real-to-real chronological eval.
      2. `evaluation_v5_real_only.json` — same numerical content as v6
         (it's where v6 metrics were first computed before formal rename).
      3. `evaluation_v5.json` / `evaluation_v4.json` / `evaluation_v3.json`
         — legacy fallbacks for back-compat.
    """
    if EVAL_V6_PATH.exists():
        return EVAL_V6_PATH
    if EVAL_V5_REAL_PATH.exists():
        return EVAL_V5_REAL_PATH
    if EVAL_V5_PATH.exists():
        return EVAL_V5_PATH
    if EVAL_V4_PATH.exists():
        return EVAL_V4_PATH
    return EVAL_V3_PATH


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


# ─── Phase 4 v3 endpoints ─────────────────────────────────────


@router.get("/evaluation_v3")
async def evaluation_v3(current_user: CurrentUser) -> dict[str, Any]:
    """Full Phase-4 scientific metric panel: LOOCV-R²/RMSE, RepeatedKFold
    variance, MAPE, pinball loss, per-oblast residuals, pred-vs-actual
    scatter, global SHAP, permutation importance and learning curves —
    per (crop, family).

    Sourced from `data/processed/evaluation_v4.json` (preferred) or v3
    fallback, written by `scripts/evaluate_models.py`. The frontend
    slices this single payload into multiple charts; one round-trip is
    faster than five smaller endpoints, and the file is small (≈1 MB)."""
    _ = current_user
    path = _pick_eval_path()
    if not path.exists():
        return {
            "crops": {},
            "metadata": {},
            "note": "Run scripts/evaluate_models.py --version v4 to populate.",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{path.name} unparseable: {exc}") from exc


@router.get("/v7_hybrid")
async def v7_hybrid(current_user: CurrentUser) -> dict[str, Any]:
    """v7-hybrid thesis headline — best of {v6, v7+SoilGrids, v7h-hierarchical}
    per crop.

    Each crop is served by its empirically-best model architecture:
    cereals (wheat/barley) usually win on v7 (NDVI + weather + SoilGrids);
    high-variance crops (corn, sunflower, buckwheat, corn-silage) win on
    v7h (hierarchical oblast-offset model); narrow-variance crops (oats)
    fall back to plain v7 because hierarchical residual modelling
    overfits their tiny within-oblast spread.

    Schema is **flat per crop** — `crops[crop] = {selected_version,
    selected_family, test_r2/mae/rmse/mape, candidates: {v6, v7, v7h}}`.
    This differs from `/evaluation_v3` (nested by family) which is why
    it lives on its own endpoint instead of going through
    `_pick_eval_path()`. Source: `data/processed/evaluation_v7_hybrid.json`
    written by `scripts/evaluate_v7_hybrid.py`.
    """
    _ = current_user
    if not EVAL_V7_HYBRID_PATH.exists():
        return {
            "crops": {},
            "metadata": {},
            "note": "Run scripts/evaluate_v7_hybrid.py to populate.",
        }
    try:
        return json.loads(EVAL_V7_HYBRID_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            500, f"{EVAL_V7_HYBRID_PATH.name} unparseable: {exc}"
        ) from exc


@router.get("/leaderboard")
async def leaderboard(current_user: CurrentUser) -> dict[str, Any]:
    """Flattened cross-crop leaderboard for the AlgorithmLeaderboard UI.

    Each row is one (crop, family) entry with the headline metrics:
    test R², test MAE, test RMSE, MAPE, LOOCV-R², repeated-K-fold mean ± σ.
    The frontend sorts and filters this client-side.

    Reading from the Phase-4 eval JSON keeps a single source of truth
    rather than recomputing summaries in the router.
    """
    _ = current_user
    path = _pick_eval_path()
    if not path.exists():
        return {"rows": [], "note": "Run scripts/evaluate_models.py to populate."}
    try:
        eval_payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{path.name} unparseable: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for crop, by_family in eval_payload.get("crops", {}).items():
        for family, body in by_family.items():
            if not isinstance(body, dict) or body.get("skipped"):
                continue
            test = body.get("test") or {}
            loocv = body.get("loocv_oblast") or {}
            kfold = body.get("repeated_kfold_5x3") or {}
            rows.append({
                "crop": crop,
                "family": family,
                "n_test": body.get("n_test"),
                "n_train": body.get("n_train"),
                "test_r2": test.get("r2"),
                "test_rmse": test.get("rmse"),
                "test_mae": test.get("mae"),
                "test_mape": test.get("mape"),
                "loocv_r2": loocv.get("r2"),
                "loocv_rmse": loocv.get("rmse"),
                "kfold_r2_mean": kfold.get("r2_mean"),
                "kfold_r2_std": kfold.get("r2_std"),
                "pinball_q05": body.get("pinball_loss_q05"),
                "pinball_q95": body.get("pinball_loss_q95"),
                "interval_coverage_90pct": body.get("interval_coverage_90pct"),
            })
    return {
        "rows": rows,
        "metadata": eval_payload.get("metadata", {}),
    }


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
