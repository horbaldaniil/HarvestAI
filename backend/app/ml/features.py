"""Feature builder — converts DB rows to the XGBoost input vector.

The same FEATURE_NAMES used during training (see scripts/train_yield_models.py)
are produced here so that train-time and inference-time vectors match.

Inputs are pulled from the satellite_observations and weather_observations
tables for one field. Missing data → NaN, which XGBoost handles natively.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Field, SatelliteObservation, WeatherObservation

# Must match scripts/train_yield_models.py FEATURE_NAMES.
FEATURE_NAMES: tuple[str, ...] = (
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
)


def build_feature_vector(session: Session, field_id: int) -> dict[str, float | None]:
    """Build the inference feature vector for one field.

    Reads recent satellite + weather observations from the DB. Returns a
    plain dict (key → number); the caller converts to numpy array in the
    XGBoost input order via FEATURE_NAMES.
    """
    field = session.get(Field, field_id)
    if field is None:
        raise ValueError(f"Field {field_id} not found")

    # Centroid → (lat, lon)
    # The PostGIS centroid is stored as GEOGRAPHY(POINT, 4326). Easiest:
    # query ST_X/ST_Y on the centroid.
    from geoalchemy2.functions import ST_X, ST_Y

    lat, lon = session.execute(
        select(ST_Y(field.centroid), ST_X(field.centroid))
    ).one()

    obs = session.scalars(
        select(SatelliteObservation)
        .where(SatelliteObservation.field_id == field_id)
        .order_by(SatelliteObservation.observed_on.asc())
    ).all()
    weather = session.scalars(
        select(WeatherObservation)
        .where(WeatherObservation.field_id == field_id)
        .where(WeatherObservation.is_forecast.is_(False))
        .order_by(WeatherObservation.observed_on.asc())
    ).all()

    veg = _veg_features(obs)
    wx = _weather_features(weather)

    return {
        **veg,
        **wx,
        "centroid_lat": float(lat) if lat is not None else None,
        "centroid_lon": float(lon) if lon is not None else None,
    }


def to_xgb_input(features: dict[str, float | None]) -> np.ndarray:
    """Convert dict → (1, n_features) numpy array in FEATURE_NAMES order.

    Missing values become NaN; XGBoost handles them natively.
    """
    row = [features.get(name) for name in FEATURE_NAMES]
    arr = np.array([[(x if x is not None else np.nan) for x in row]], dtype=float)
    return arr


# ─── Internals ────────────────────────────────────────────


def _veg_features(observations: list[SatelliteObservation]) -> dict[str, float | None]:
    """Aggregate per-week observations into per-season summary features.

    Uses observations from the most recent full season (Apr-Sep of the
    latest year present in data).
    """
    if not observations:
        return {k: None for k in (
            "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
            "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
            "evi_peak", "ndwi_min", "savi_peak",
        )}

    latest_year = max(o.observed_on.year for o in observations)
    season = [
        o for o in observations
        if o.observed_on.year == latest_year and 4 <= o.observed_on.month <= 9
    ]
    if not season:
        season = observations

    def _mm(month: int) -> float | None:
        vals = [float(o.ndvi_mean) for o in season
                if o.ndvi_mean is not None and o.observed_on.month == month]
        return round(sum(vals) / len(vals), 3) if vals else None

    ndvi_values = [(o.observed_on, float(o.ndvi_mean))
                   for o in season if o.ndvi_mean is not None]

    if not ndvi_values:
        ndvi_peak = None
        ndvi_peak_week = None
        ndvi_std = None
        ndvi_integral = None
    else:
        peak_obs = max(ndvi_values, key=lambda x: x[1])
        ndvi_peak = round(peak_obs[1], 3)
        ndvi_peak_week = int(peak_obs[0].isocalendar()[1])
        arr = np.array([v for _, v in ndvi_values])
        ndvi_std = round(float(arr.std()), 3) if len(arr) >= 2 else 0.0
        ndvi_integral = round(float(arr.sum() * 7 / 30), 2)  # ~weekly→month-scaled

    def _peak(attr: str) -> float | None:
        vals = [float(getattr(o, attr)) for o in season
                if getattr(o, attr) is not None]
        return round(max(vals), 3) if vals else None

    def _floor(attr: str) -> float | None:
        vals = [float(getattr(o, attr)) for o in season
                if getattr(o, attr) is not None]
        return round(min(vals), 3) if vals else None

    return {
        "ndvi_peak": ndvi_peak,
        "ndvi_peak_week": ndvi_peak_week,
        "ndvi_mean_may": _mm(5),
        "ndvi_mean_june": _mm(6),
        "ndvi_mean_july": _mm(7),
        "ndvi_mean_august": _mm(8),
        "ndvi_integral": ndvi_integral,
        "ndvi_std": ndvi_std,
        "evi_peak": _peak("evi_mean"),
        "ndwi_min": _floor("ndwi_mean"),
        "savi_peak": _peak("savi_mean"),
    }


def _weather_features(observations: list[WeatherObservation]) -> dict[str, float | None]:
    """April-July aggregates from the latest year in the data."""
    if not observations:
        return {
            "precip_sum_apr_jul": None,
            "temp_mean_apr_jul": None,
            "heat_stress_days": None,
            "drought_dryspells": None,
        }
    latest_year = max(o.observed_on.year for o in observations)
    season = [
        o for o in observations
        if o.observed_on.year == latest_year and 4 <= o.observed_on.month <= 7
    ]
    if not season:
        return _weather_features([])

    precip = sum(float(o.precip_mm or 0) for o in season)
    temps = [float(o.temp_mean_c) for o in season if o.temp_mean_c is not None]
    heat = sum(
        1 for o in season
        if o.temp_max_c is not None and float(o.temp_max_c) > 30
    )

    # Longest dry-spell across the season.
    longest, current = 0, 0
    for o in season:
        if (o.precip_mm or 0) < 1.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return {
        "precip_sum_apr_jul": round(precip, 1),
        "temp_mean_apr_jul": round(sum(temps) / len(temps), 2) if temps else None,
        "heat_stress_days": heat,
        "drought_dryspells": longest,
    }
