"""Unit tests for dashboard analytics helpers (no DB / no PostGIS)."""
from __future__ import annotations

from datetime import date

import pytest

from app.services.dashboard_analytics import (
    compute_risk_score,
    phenology_calendar,
    phenology_phase,
    pick_best_worst,
)


# ─── Phenology ──────────────────────────────────────────────


def test_phenology_unknown_crop_returns_none():
    assert phenology_phase("bananas") is None


def test_phenology_wheat_summer_is_harvest():
    assert phenology_phase("wheat", date(2026, 8, 15)) == "harvest"


def test_phenology_wheat_winter_is_dormancy():
    """Wheat dormancy spans November–March — wraparound month range."""
    assert phenology_phase("wheat", date(2026, 1, 15)) == "dormancy"
    assert phenology_phase("wheat", date(2026, 12, 5)) == "dormancy"


def test_phenology_corn_july_is_flowering():
    assert phenology_phase("corn", date(2026, 7, 10)) == "flowering"


def test_phenology_calendar_lists_all_phases():
    cal = phenology_calendar("sunflower")
    phases = [p["phase"] for p in cal]
    assert "sowing" in phases
    assert "harvest" in phases
    assert all(1 <= p["start_month"] <= 12 for p in cal)
    assert all(1 <= p["end_month"] <= 12 for p in cal)


# ─── Risk score ─────────────────────────────────────────────


def test_risk_score_zero_when_clean():
    score, factors = compute_risk_score(
        current_ndvi=0.65,
        oblast_avg_ndvi=0.62,
        alerts_by_severity={},
    )
    assert score == 0
    assert factors == []


def test_risk_score_ndvi_deficit_penalises():
    score, factors = compute_risk_score(
        current_ndvi=0.30,
        oblast_avg_ndvi=0.65,
        alerts_by_severity={},
    )
    assert score > 0
    assert any("NDVI" in f for f in factors)


def test_risk_score_critical_alert_heavy_weight():
    score, factors = compute_risk_score(
        current_ndvi=0.60,
        oblast_avg_ndvi=0.60,
        alerts_by_severity={"critical": 1},
    )
    # One critical alone should land roughly 30 points.
    assert 25 <= score <= 50
    assert any("Alert" in f or "критичних" in f for f in factors)


def test_risk_score_caps_at_100():
    score, _ = compute_risk_score(
        current_ndvi=0.10,
        oblast_avg_ndvi=0.80,
        alerts_by_severity={"warning": 5, "critical": 5},
        drought_days_recent=14,
        heat_days_recent=10,
    )
    assert score == 100


def test_risk_score_low_ndvi_standalone_still_flags():
    """Even without an oblast baseline, a very low NDVI alone is a flag."""
    score, factors = compute_risk_score(
        current_ndvi=0.20,
        oblast_avg_ndvi=None,
        alerts_by_severity={},
    )
    assert score > 0
    assert any("NDVI" in f for f in factors)


# ─── Best/worst picker ──────────────────────────────────────


def test_pick_best_worst_returns_none_for_empty():
    best, worst = pick_best_worst([])
    assert best is None
    assert worst is None


def test_pick_best_worst_picks_highest_ndvi_and_riskiest():
    rows = [
        {"field_id": 1, "name": "A", "crop_type": "wheat",
         "current_ndvi": 0.40, "predicted_tha": 4.0, "risk_score": 20,
         "risk_factors": ["Low NDVI"]},
        {"field_id": 2, "name": "B", "crop_type": "corn",
         "current_ndvi": 0.78, "predicted_tha": 7.0, "risk_score": 5,
         "risk_factors": []},
        {"field_id": 3, "name": "C", "crop_type": "sunflower",
         "current_ndvi": 0.55, "predicted_tha": None, "risk_score": 70,
         "risk_factors": ["2 критичних"]},
    ]
    best, worst = pick_best_worst(rows)
    assert best is not None and best["field_id"] == 2
    assert worst is not None and worst["field_id"] == 3
    assert "критичних" in worst["reason"] or "NDVI" in worst["reason"]


def test_pick_best_worst_worst_none_when_no_risk():
    rows = [
        {"field_id": 1, "name": "A", "crop_type": "wheat",
         "current_ndvi": 0.7, "predicted_tha": 5.0, "risk_score": 0,
         "risk_factors": []},
    ]
    best, worst = pick_best_worst(rows)
    assert best is not None
    assert worst is None
