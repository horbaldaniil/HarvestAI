"""Feature builder — converts DB rows to the model input vector.

Produces the full **v7 schema (30 features)** in one pass; downstream
callers pick the subset their model expects via the model's
`payload["features"]` list.

Schema layout (matches `FEATURE_NAMES` in `scripts/train_yield_models_v7.py`):

  1. v3 base (17) — vegetation indices + Apr-Jul weather aggregates +
     field centroid coordinates. Computed from raw DB rows.
  2. v7 SoilGrids (7) — bulk density, CEC, clay/sand/silt %, pH, SOC.
     Looked up per oblast (via field centroid → oblast polygon → ISO).
  3. v5 crop-specific (6) — overlap with Apr-Jul reference window, GDD
     proxy, weighted precip/heat/drought, growing-season length.
     Derived deterministically from `crop_calendar.py` + base weather.

Older models (v3 17-feat, v4/v5/v6 23-feat) just slice the first 17 or
23 entries.

Inputs are pulled from the `satellite_observations` and
`weather_observations` tables for one field. Missing data → NaN, which
XGBoost handles natively. RandomForest does NOT handle NaN, so the
yield_model layer chooses a model whose features the builder can
actually fill (SoilGrids defaults to national-median if oblast lookup
fails — see `data_reference/soilgrids.py`).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_reference.crop_calendar import (
    CROP_CALENDAR,
    REFERENCE_WINDOW_MONTHS,
    flowering_in_reference,
    gdd_proxy as _gdd_proxy_helper,
    overlap_with_reference,
)
from app.data_reference.oblast_names import iso_from_any_name
from app.data_reference.soilgrids import SOIL_FEATURES, soilgrids_for_oblast
from app.db.models import Field, SatelliteObservation, WeatherObservation
from app.db.models.enums import CropType

# ─── Canonical schemas per model generation ───────────────

# v3 base: vegetation indices + Apr-Jul weather + field centroid.
# Same 17 features the v1/v2 LSTM and XGBoost models were trained on.
V3_BASE_FEATURES: tuple[str, ...] = (
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
)

# v5/v6: v3 + crop-specific weather (overlap, GDD, weighted precip/heat/
# drought, season length). Same set used by v4/v5/v6 trainers — 23 cols.
V5_CROP_FEATURES: tuple[str, ...] = (
    "crop_season_overlap_aprjul",
    "gdd_proxy",
    "precip_crop_weighted",
    "heat_stress_crop_weighted",
    "drought_crop_weighted",
    "growing_season_length_months",
)

# v7: v6 23-feat + 7 SoilGrids per-oblast soil features. Order matches
# the trainer's FEATURE_NAMES tuple. Final width = 30.
V7_FEATURE_NAMES: tuple[str, ...] = (
    *V3_BASE_FEATURES,
    *SOIL_FEATURES,
    *V5_CROP_FEATURES,
)

# Kept as the historical name so existing call sites (yield_model.py,
# tests) still import `FEATURE_NAMES`. Points to the widest schema we
# currently know how to build; per-model slicing happens in
# `select_features()`.
FEATURE_NAMES: tuple[str, ...] = V7_FEATURE_NAMES


# Heat-weight applied when the crop's flowering peak lands inside the
# Apr-Jul reference window — empirically derived from the v5 trainer's
# feature-engineering step (heat at flowering damages reproductive
# organs disproportionately). Stored as constants so changing the
# multiplier later only edits one place.
_HEAT_W_IN_REF: float = 1.5
_HEAT_W_OUT_REF: float = 0.5


def build_feature_vector(
    session: Session,
    field_id: int,
    crop: CropType | None = None,
) -> dict[str, float | None]:
    """Build the inference feature dict for one field.

    Returns the full 30-feature dict. The caller picks which subset
    their model needs via `select_features(dict, model_feature_list)`.

    `crop` is optional — when None, the v5 crop-specific features are
    left as NaN. Pass the crop (typically `field.crop_type`) to fill
    them; required for v6/v7 models.
    """
    field = session.get(Field, field_id)
    if field is None:
        raise ValueError(f"Field {field_id} not found")

    # Centroid → (lat, lon). Stored as GEOGRAPHY(POINT, 4326); query
    # ST_X / ST_Y to extract degrees.
    from geoalchemy2.functions import ST_X, ST_Y

    lat, lon = session.execute(
        select(ST_Y(field.centroid), ST_X(field.centroid))
    ).one()
    lat_f = float(lat) if lat is not None else None
    lon_f = float(lon) if lon is not None else None

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
    soil = _soilgrids_for_centroid(lat_f, lon_f)
    crop_features = _crop_features(crop, wx) if crop is not None else _crop_features_none()

    return {
        **veg,
        **wx,
        "centroid_lat": lat_f,
        "centroid_lon": lon_f,
        **soil,
        **crop_features,
    }


def select_features(
    features: dict[str, float | None],
    feature_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Convert dict → (1, N) numpy array in the order required by a
    specific model.

    The model's expected feature order lives in its joblib payload as
    `payload["features"]` (set by every v3+ trainer); pass that here.
    Missing keys become NaN — RandomForest will choke on those, so the
    caller must ensure the builder produces every required name (or
    pick a different model).
    """
    row = [features.get(name) for name in feature_names]
    arr = np.array(
        [[(x if x is not None else np.nan) for x in row]],
        dtype=float,
    )
    return arr


def to_xgb_input(features: dict[str, float | None]) -> np.ndarray:
    """Legacy helper — converts to the **v3 base** 17-feature vector.
    Retained for callers that haven't been migrated to
    `select_features(payload["features"])` yet.
    """
    return select_features(features, V3_BASE_FEATURES)


# ─── Internals ────────────────────────────────────────────


def _veg_features(observations: list[SatelliteObservation]) -> dict[str, float | None]:
    """Aggregate per-week observations into per-season summary features.

    Uses observations from the most recent full season (Apr-Sep of the
    latest year present in data).
    """
    keys = (
        "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
        "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
        "evi_peak", "ndwi_min", "savi_peak",
    )
    if not observations:
        return {k: None for k in keys}

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
        if o.observed_on.year == latest_year and o.observed_on.month in REFERENCE_WINDOW_MONTHS
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


def _soilgrids_for_centroid(
    lat: float | None, lon: float | None,
) -> dict[str, float]:
    """Look up the 7 SoilGrids features for the oblast containing this
    centroid. Falls back to `NATIONAL_MEDIAN` if the point doesn't hit
    any oblast polygon (e.g. Kyiv City, points in lakes/rivers near
    boundaries).

    Always returns a real dict — never NaN — because RandomForest
    cannot consume NaN in its v7 retrofit.
    """
    # Local import — `dashboard_analytics` pulls in shapely / geojson,
    # and we want yield prediction not to incur that cost when soil
    # features aren't needed (i.e. when the caller picks a v3 model).
    from app.services.dashboard_analytics import find_oblast_for_centroid

    if lat is None or lon is None:
        return soilgrids_for_oblast(None)
    natural_earth_name = find_oblast_for_centroid(lat, lon)
    iso = iso_from_any_name(natural_earth_name)
    return soilgrids_for_oblast(iso)


def _crop_features(
    crop: CropType, weather: dict[str, float | None],
) -> dict[str, float | None]:
    """v5 crop-specific weather-derivation features.

    Mirrors the formulas in the v5 build script:
      - `crop_season_overlap_aprjul = overlap_with_reference(crop)`
      - `gdd_proxy = (temp_mean - gdd_base) × season_months × 30`
      - `precip_crop_weighted = precip_sum_apr_jul × overlap`
      - `heat_stress_crop_weighted = heat_stress_days ×
                  (1.5 if flowering in Apr-Jul else 0.5)`
      - `drought_crop_weighted = drought_dryspells × overlap`
      - `growing_season_length_months = len(growing_season_months)`

    Deterministic; no DB read required beyond the base weather already
    computed by `_weather_features`.
    """
    crop_slug = crop.value
    cal = CROP_CALENDAR.get(crop_slug)
    if cal is None:
        return _crop_features_none()

    overlap = overlap_with_reference(crop_slug)
    season_months = cal.growing_season_length_months
    temp_mean = weather.get("temp_mean_apr_jul")
    precip = weather.get("precip_sum_apr_jul")
    heat_days = weather.get("heat_stress_days")
    drought = weather.get("drought_dryspells")

    heat_w = _HEAT_W_IN_REF if flowering_in_reference(crop_slug) else _HEAT_W_OUT_REF

    return {
        "crop_season_overlap_aprjul": round(overlap, 3),
        "gdd_proxy": _gdd_proxy_helper(crop_slug, temp_mean, season_months),
        "precip_crop_weighted": (
            round(float(precip) * overlap, 1) if precip is not None else None
        ),
        "heat_stress_crop_weighted": (
            round(float(heat_days) * heat_w, 1) if heat_days is not None else None
        ),
        "drought_crop_weighted": (
            round(float(drought) * overlap, 2) if drought is not None else None
        ),
        "growing_season_length_months": season_months,
    }


def _crop_features_none() -> dict[str, float | None]:
    """Empty (None-filled) crop-specific dict, used when `crop` is not
    passed to the builder. Lets v3 models still get a full 30-key dict
    (just with NaN in the crop-specific positions, which they slice
    out anyway via `select_features`)."""
    return {name: None for name in V5_CROP_FEATURES}
