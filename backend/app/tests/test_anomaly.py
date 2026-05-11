"""Unit tests for rule-based anomaly detectors.

We construct lightweight stand-ins for ORM rows so the tests don't need
a DB. The detectors only access plain attributes — they don't care about
SQLAlchemy state.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.db.models.enums import CropType
from app.ml.anomaly import (
    DROUGHT_EXPECTED_PRECIP_MM,
    HEAT_CRITICAL_DAYS,
    HEAT_WARNING_DAYS,
    detect_drought,
    detect_heat_stress,
    detect_ndvi_anomalies,
)


@dataclass
class _Obs:
    observed_on: date
    ndvi_mean: float | None


@dataclass
class _Weather:
    observed_on: date
    temp_max_c: float | None
    precip_mm: float | None
    is_forecast: bool = False


NORMS = {
    "wheat": {
        "26": {"mean": 0.85, "std": 0.07},
        "27": {"mean": 0.86, "std": 0.07},
        "28": {"mean": 0.85, "std": 0.07},
    }
}


# ───── NDVI drop ─────────────────────────────────────────


def test_ndvi_within_norm_emits_nothing():
    obs = [
        _Obs(date(2024, 6, 24), 0.84),  # week 26 norm
        _Obs(date(2024, 7, 1), 0.86),
    ]
    assert detect_ndvi_anomalies(obs, CropType.WHEAT, NORMS) == []


def test_ndvi_warning_at_minus_1_5_sigma():
    # z = -2 → warning territory (between -1.5 and -2.5)
    obs = [_Obs(date(2024, 6, 24), 0.85 - 2 * 0.07)]
    out = detect_ndvi_anomalies(obs, CropType.WHEAT, NORMS)
    assert len(out) == 1
    assert out[0].severity == "warning"
    assert out[0].type == "ndvi_drop"


def test_ndvi_critical_at_minus_3_sigma():
    obs = [_Obs(date(2024, 6, 24), 0.85 - 3 * 0.07)]
    out = detect_ndvi_anomalies(obs, CropType.WHEAT, NORMS)
    assert len(out) == 1
    assert out[0].severity == "critical"


def test_ndvi_missing_norm_skips():
    """Week 1 has no norm in our fixture → no alert even if NDVI is bad."""
    obs = [_Obs(date(2024, 1, 1), 0.05)]
    assert detect_ndvi_anomalies(obs, CropType.WHEAT, NORMS) == []


def test_ndvi_unknown_crop_returns_empty():
    obs = [_Obs(date(2024, 6, 24), 0.10)]
    out = detect_ndvi_anomalies(obs, CropType.CORN, NORMS)
    assert out == []


# ───── Drought ──────────────────────────────────────────


def test_drought_normal_rain_no_alert():
    today = date(2024, 7, 15)
    # ~30 mm spread over 14 days
    weather = [
        _Weather(today - timedelta(days=i), 25, 2.0) for i in range(14)
    ]
    assert detect_drought(weather, today) is None


def test_drought_critical_when_almost_no_rain():
    today = date(2024, 7, 15)
    # 2 mm over 14 days = ratio 0.066 < 0.10
    weather = [
        _Weather(today - timedelta(days=i), 25, 0.15) for i in range(14)
    ]
    out = detect_drought(weather, today)
    assert out is not None
    assert out.severity == "critical"
    assert out.type == "drought_stress"


def test_drought_warning_in_middle_band():
    today = date(2024, 7, 15)
    # 6 mm over 14 days = ratio 0.20, between 0.10 and 0.30 → warning
    weather = [
        _Weather(today - timedelta(days=i), 25, 0.43) for i in range(14)
    ]
    out = detect_drought(weather, today)
    assert out is not None
    assert out.severity == "warning"


def test_drought_ignores_forecast_rows():
    today = date(2024, 7, 15)
    weather = [
        _Weather(today - timedelta(days=i), 25, 0.1, is_forecast=False)
        for i in range(14)
    ] + [
        _Weather(today + timedelta(days=i), 25, 10.0, is_forecast=True)
        for i in range(1, 8)
    ]
    out = detect_drought(weather, today)
    assert out is not None  # forecast rain doesn't save us from past drought


# ───── Heat stress ──────────────────────────────────────


def test_heat_no_hot_days_no_alert():
    today = date(2024, 7, 15)
    weather = [_Weather(today - timedelta(days=i), 25, 1.0) for i in range(14)]
    assert detect_heat_stress(weather, today) is None


def test_heat_warning_threshold():
    today = date(2024, 7, 15)
    # exactly HEAT_WARNING_DAYS hot days → warning
    weather = [
        _Weather(today - timedelta(days=i), 32 if i < HEAT_WARNING_DAYS else 25, 1.0)
        for i in range(14)
    ]
    out = detect_heat_stress(weather, today)
    assert out is not None
    assert out.severity == "warning"


def test_heat_critical_threshold():
    today = date(2024, 7, 15)
    weather = [
        _Weather(today - timedelta(days=i), 35 if i < HEAT_CRITICAL_DAYS else 25, 1.0)
        for i in range(14)
    ]
    out = detect_heat_stress(weather, today)
    assert out is not None
    assert out.severity == "critical"
