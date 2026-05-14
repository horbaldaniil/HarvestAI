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

import functools
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
# Post-phenology-audit extension (32-feature v7 schema): two crop-specific
# NDVI features added — they read the right monthly NDVI slot based on
# the crop's actual peak month, and quantify timing anomalies vs the
# expected peak week. Both derive from data already present in v3 base
# features + crop_calendar; no new DB reads.
V5_CROP_FEATURES: tuple[str, ...] = (
    "crop_season_overlap_aprjul",
    "gdd_proxy",
    "precip_crop_weighted",
    "heat_stress_crop_weighted",
    "drought_crop_weighted",
    "growing_season_length_months",
    # NEW (phenology audit): crop-aligned NDVI signal + timing anomaly.
    "ndvi_at_crop_peak_month",
    "ndvi_peak_timing_offset_weeks",
)

# 5 Ukrainian agroclimatic zones — values mirror the `OblastRef.zone`
# literal in `app/data_reference/oblast_names.py`. Encoded as 5 one-hot
# integer features so every base learner in the stack (rf / xgb / lgbm)
# consumes them uniformly. Added in the "regional-baseline lift" iteration
# after the audit identified that the model lacked categorical regional
# context — Lviv-zone barley typically yields 4.7 т/га, Steppe-zone
# barley typically yields ~2.5 т/га, but the continuous centroid_lat/lon
# features couldn't represent that crisp split.
_ZONES: tuple[str, ...] = (
    "polissia",
    "forest_steppe",
    "steppe_north",
    "steppe_south",
    "transcarpathia",
)
ZONE_FEATURES: tuple[str, ...] = tuple(f"zone_{z}" for z in _ZONES)

# Crop one-hot encoding — 13 features, one per supported crop. Used
# **only by the multi-task v8 schema** (`V8_FEATURE_NAMES` below).
# Per-crop v7 / v7h trainers ignore these columns; they're injected
# at `build_feature_vector` time so v8 inference has the right input
# shape, but per-crop callers slice them out via their stored
# `payload["features"]` list — they don't appear in V7_FEATURE_NAMES.
#
# Why one-hot and not native categorical: same reasoning as zones
# — RF + Stack don't take native categorical (only XGBoost / LightGBM
# do). One-hot keeps all four base learners consuming the input
# uniformly.
from app.data_reference.crop_zones import ALL_CROPS as _ALL_CROPS_FOR_ONEHOT
CROP_ONEHOT_FEATURES: tuple[str, ...] = tuple(
    f"crop_{c}" for c in _ALL_CROPS_FOR_ONEHOT
)

# Lagged oblast yield baselines — four continuous features giving the
# model a direct regional + crop-specific anchor that the categorical
# zone one-hots only approximate.
#
# At training time each row gets (year-1), (year-2), (year-3) Держстат
# yields for its (oblast, crop) plus their 3-year arithmetic mean.
# Tree models pick whichever horizon carries the most signal (SHAP
# usually shows lag_mean3 dominating for stable cereals and lag1 for
# crops with year-to-year volatility).
#
# At inference time `oblast_avg_yield(name, crop, year=None)` falls
# back to the most-recent published year (2025 once XLS bulletins are
# ingested), so all four features pull from the same row at predict
# time — the variation only matters during training where each row
# legitimately has a different "previous year". Same parquet-patcher
# rule the regional-baseline lift introduced: missing year-N values
# fall back to the crop's 2018-2019 train-mean (idempotent + no
# leakage from val/test).
OBLAST_LAG_FEATURES: tuple[str, ...] = (
    "oblast_yield_lag1",
    "oblast_yield_lag2",
    "oblast_yield_lag3",
    "oblast_yield_lag_mean3",
)

# Phase-D Option B — cross-crop annual signal. For each row
# (iso, crop, year), the mean yield across **all other 12 crops** in
# the same oblast for `year-1`. Captures regional weather/economic
# conditions that propagate across crops; complements the same-crop
# lag (`oblast_yield_lag1`) which only captures same-crop history.
# Patched in by `scripts/patch_parquet_neighbor_yield.py` after the
# yield-lag patcher. See Schauberger & Gornott 2017 panel-data
# yield models.
NEIGHBOR_YIELD_FEATURES: tuple[str, ...] = (
    "neighbor_yield_lag1_mean",
)

# ─── Phase-A round-3 feature additions (Oct 2025) ─────────────
#
# Three feature blocks added to address the post-Round-2 audit's
# identified gaps for crops still below R² 0.7.

# Winter NDVI (October — November of yield_year-1 + March of yield_year).
# Captures winter-crop establishment and spring regrowth. The
# `spring_regrowth_ndvi_delta` is `ndvi_march - ndvi_october` — a high
# positive value signals successful overwintering, near-zero or
# negative signals winter-kill. Computed by `_veg_features` from the
# extended Sentinel-2 winter collection (`scripts/collect_oblast_s2.py
# --window winter`).
WINTER_NDVI_FEATURES: tuple[str, ...] = (
    "ndvi_mean_october",
    "ndvi_mean_november",
    "ndvi_mean_march",
    "spring_regrowth_ndvi_delta",
)

# Soil-moisture + winter-kill features (from Open-Meteo's daily
# `soil_moisture_0_to_10cm_mean` + `temperature_2m_min`, both already
# fetched and stored in `WeatherObservation` since Week 8). Soil
# moisture during Jun-Jul drives tuber bulking for potato + sugar_beet
# — biggest expected lift for those two crops. `winter_kill_days`
# complements the winter NDVI signals for cereal cold-shock damage.
WINTER_WEATHER_FEATURES: tuple[str, ...] = (
    "sm_jun_jul_mean",
    "sm_drydown_days",
    "winter_kill_days",
)

# BBCH-aligned weather windows — 9 features (3 weather aggregates ×
# 3 phenological phases per crop). Each phase derived from
# `CropCalendar.early_veg_months / flowering_months / grain_fill_months`.
# This is the per-crop replacement for the one-size-fits-all
# Apr-Jul aggregates: a heat-stress day in May matters massively for
# wheat (flowering) but barely for sunflower (still vegetative).
BBCH_PHASE_FEATURES: tuple[str, ...] = (
    "precip_early_veg",
    "precip_flowering",
    "precip_grain_fill",
    "temp_mean_flowering",
    "temp_mean_grain_fill",
    "heat_days_flowering",
    "heat_days_grain_fill",
    "drought_days_flowering",
    "drought_days_grain_fill",
)


# v7: v6 (25 feat with phenology-audit NDVI) + 7 SoilGrids + 5 zone
# one-hots + 4 lagged-yield baselines + 4 winter NDVI + 3 winter
# weather + 9 BBCH-phase = **57 features** (was 41 before Phase-A
# round-3, 38 before multi-year lag).
#
# Phase-D Option B tried adding `NEIGHBOR_YIELD_FEATURES` (cross-crop
# lag1 mean) — net regressed wheat by 0.08 R² because Optuna's
# per-crop hyperparameters were tuned on the 57-feature schema, not
# the wider 58-feature one. The feature still has potential
# (correlation with own-lag1 was only -0.139), but unlocking it
# requires a fresh ~4 h Optuna sweep on the 58-feature schema —
# deferred. Patcher script + dashboard_analytics helper kept in
# place for that follow-up.
V7_FEATURE_NAMES: tuple[str, ...] = (
    *V3_BASE_FEATURES,
    *SOIL_FEATURES,
    *ZONE_FEATURES,
    *OBLAST_LAG_FEATURES,
    *WINTER_NDVI_FEATURES,
    *WINTER_WEATHER_FEATURES,
    *BBCH_PHASE_FEATURES,
    *V5_CROP_FEATURES,
)

# v8: multi-task schema — v7's 57 features + 13 crop one-hots = **70
# features**. Used by `scripts/train_yield_models_v8.py` which trains
# ONE model per family on all 13 crops jointly. The crop one-hots let
# tree splits learn per-crop biases on top of the shared NDVI/weather/
# soil signal — effective n grows from 48 (per-crop v7) to 624 (all
# crops jointly), dropping the d/n ratio from 1.19 (overfit territory)
# to 70/624 ≈ 0.11 (safely below the rule-of-thumb 0.3 boundary).
#
# Backward compatibility: v7 and v7h models stay on `V7_FEATURE_NAMES`.
# `select_features(features_dict, payload["features"])` slices to the
# columns the model was trained on, so v7 callers naturally skip the
# 13 crop one-hots even when `build_feature_vector` emits them. v8
# callers (payload version="v8") receive them.
V8_FEATURE_NAMES: tuple[str, ...] = (
    *V7_FEATURE_NAMES,
    *CROP_ONEHOT_FEATURES,
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
    # _zone_features resolves the same centroid → oblast → zone path
    # as soil; both share the geocode round-trip. Five one-hot ints
    # for the agroclimatic zones.
    zone = _zone_features(lat_f, lon_f)
    # Oblast×crop most-recent yield from Держстат. One float. The model
    # uses this as a direct regional anchor; at training time the same
    # column is computed with strict year-1 anti-leakage.
    lag = _oblast_yield_lag(lat_f, lon_f, crop)
    # Phase-D Option B (deferred): `_neighbor_yield_lag(...)` helper +
    # `neighbor_yield_lag1_mean` parquet column remain available for a
    # future re-attempt that includes a fresh Optuna sweep on the
    # widened 58-feature schema. The current schema stays at 57 to
    # preserve Phase-D Option C's headline R²>0.5: 8/13.
    neighbor: dict[str, float | None] = {}
    # _crop_features needs both `wx` (for crop-weighted weather metrics)
    # AND `veg` (for the two new NDVI-at-peak / peak-timing-offset
    # features). All three inputs derive from already-loaded data.
    crop_features = (
        _crop_features(crop, wx, veg) if crop is not None
        else _crop_features_none()
    )
    # Phase-A round-3: BBCH-aligned weather windows (precip / temp /
    # heat / drought split per phase per crop). Replaces the
    # one-size-fits-all Apr-Jul aggregate with phase-resolved signals.
    # For crops we don't have a calendar for (or when crop=None at
    # inference) returns 9 None values — RF/Stack fall back via the
    # standard imputation path.
    bbch = _bbch_phase_weather(weather, crop)
    # Phase-B: crop one-hot encoding for the multi-task v8 schema.
    # Always emitted (13 keys); per-crop v7/v7h callers slice them
    # out via their stored `payload["features"]` list, v8 callers
    # consume them.
    crop_onehots = _crop_onehot_features(crop)

    return {
        **veg,
        **wx,
        "centroid_lat": lat_f,
        "centroid_lon": lon_f,
        **soil,
        **zone,
        **lag,
        **neighbor,
        **bbch,
        **crop_features,
        **crop_onehots,
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
    """Aggregate per-week observations into per-cropping-year features.

    **Cropping-year semantic** (Phase-A round-3): the "season" now spans
    Oct(target_year - 1) → Sep(target_year), matching how the training
    aggregator groups winter S2 rows alongside summer ones (winter rows
    tagged with `year=yield_year` by the collector). This lets winter
    crops (wheat / barley / rye / rapeseed) see the Oct/Nov
    establishment + March regrowth signals that were previously
    inaccessible.

    Returns 15 features now (was 11):
      - existing peak/std/integral + monthly may/june/july/august
        (+ evi_peak / ndwi_min / savi_peak)
      - **new**: `ndvi_mean_october`, `ndvi_mean_november`,
        `ndvi_mean_march`, `spring_regrowth_ndvi_delta`

    Choice of `target_year`: the **latest year that has Apr-Sep
    observations**. This is the year the model treats as "current" for
    inference. If only winter observations exist for the latest year
    (e.g. user just added a field in November), we fall back to the
    overall latest year — degraded but doesn't crash.
    """
    keys = (
        "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
        "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
        "evi_peak", "ndwi_min", "savi_peak",
        # Phase-A round-3 winter NDVI additions
        "ndvi_mean_october", "ndvi_mean_november", "ndvi_mean_march",
        "spring_regrowth_ndvi_delta",
    )
    if not observations:
        return {k: None for k in keys}

    # Anchor on the latest year that has summer observations — that's
    # the "current" cropping year for prediction purposes.
    summer_years = [
        o.observed_on.year for o in observations
        if 4 <= o.observed_on.month <= 9
    ]
    target_year = max(summer_years) if summer_years else max(
        o.observed_on.year for o in observations
    )
    # Season = Oct(target_year-1) through Sep(target_year). Captures
    # winter-crop establishment + dormancy + spring regrowth + main
    # growing season in one cohesive window.
    season = [
        o for o in observations
        if (o.observed_on.year == target_year and o.observed_on.month <= 9)
        or (o.observed_on.year == target_year - 1 and o.observed_on.month >= 10)
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

    # Phase-A round-3 winter NDVI block.
    ndvi_oct = _mm(10)
    ndvi_nov = _mm(11)
    ndvi_mar = _mm(3)
    spring_regrowth = (
        round(ndvi_mar - ndvi_oct, 3)
        if ndvi_mar is not None and ndvi_oct is not None
        else None
    )

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
        # Phase-A round-3 winter NDVI additions
        "ndvi_mean_october": ndvi_oct,
        "ndvi_mean_november": ndvi_nov,
        "ndvi_mean_march": ndvi_mar,
        "spring_regrowth_ndvi_delta": spring_regrowth,
    }


def _weather_features(observations: list[WeatherObservation]) -> dict[str, float | None]:
    """Per-cropping-year weather aggregates from DB observations.

    Phase-A round-3 adds 3 features beyond the legacy 4 Apr-Jul cells:
      - `sm_jun_jul_mean` — soil moisture 0-10 cm avg across Jun + Jul
      - `sm_drydown_days` — Jun-Aug days with SM < 0.15 m³/m³
      - `winter_kill_days` — Dec(prev) + Jan-Feb(curr) days with T_min < -15°C

    Anchors on the latest year with Apr-Jul data (the inference
    "current year"), then crosses the year boundary for winter_kill.
    Missing observations (older fields without year-round weather
    backfill) yield `None` for that feature; tree models with NaN-
    tolerance still operate.
    """
    legacy_none = {
        "precip_sum_apr_jul": None,
        "temp_mean_apr_jul": None,
        "heat_stress_days": None,
        "drought_dryspells": None,
        "sm_jun_jul_mean": None,
        "sm_drydown_days": None,
        "winter_kill_days": None,
    }
    if not observations:
        return legacy_none

    summer_years = [
        o.observed_on.year for o in observations
        if o.observed_on.month in REFERENCE_WINDOW_MONTHS
    ]
    target_year = max(summer_years) if summer_years else max(
        o.observed_on.year for o in observations
    )
    season = [
        o for o in observations
        if o.observed_on.year == target_year and o.observed_on.month in REFERENCE_WINDOW_MONTHS
    ]
    if not season:
        return legacy_none

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

    # Phase-A round-3: soil-moisture aggregates. Jun-Jul mean covers
    # tuber-bulking (potato) + root expansion (sugar_beet) — the
    # phases where soil-water shortage caps yield. Jun-Aug drydown
    # count picks up extended dry spells that DON'T register in the
    # Apr-Jul precip aggregate (because precip can be normal in May
    # but a dry July still kills tuber yield).
    sm_jun_jul_vals: list[float] = []
    sm_drydown_days = 0
    for o in observations:
        if o.observed_on.year != target_year:
            continue
        sm = getattr(o, "soil_moisture_0_10cm", None)
        if sm is None:
            continue
        sm_float = float(sm)
        if o.observed_on.month in (6, 7):
            sm_jun_jul_vals.append(sm_float)
        if o.observed_on.month in (6, 7, 8) and sm_float < 0.15:
            sm_drydown_days += 1
    sm_jun_jul_mean: float | None = (
        round(sum(sm_jun_jul_vals) / len(sm_jun_jul_vals), 3)
        if sm_jun_jul_vals else None
    )

    # Phase-A round-3: winter_kill — Dec(target_year-1) + Jan/Feb(target_year)
    # days with T_min < -15°C. Captures cold-shock damage to winter
    # crops (Larcher 2003 cold-tolerance threshold for Triticum and
    # Hordeum sub-zero injury). Spring crops sown in April are
    # unaffected; the feature is informative-but-near-zero for them
    # (and tree models can ignore it via low SHAP).
    winter_kill_days = sum(
        1 for o in observations
        if (
            (o.observed_on.year == target_year - 1 and o.observed_on.month == 12)
            or (o.observed_on.year == target_year and o.observed_on.month in (1, 2))
        )
        and o.temp_min_c is not None
        and float(o.temp_min_c) < -15.0
    )

    return {
        "precip_sum_apr_jul": round(precip, 1),
        "temp_mean_apr_jul": round(sum(temps) / len(temps), 2) if temps else None,
        "heat_stress_days": heat,
        "drought_dryspells": longest,
        # Phase-A round-3 additions
        "sm_jun_jul_mean": sm_jun_jul_mean,
        "sm_drydown_days": sm_drydown_days,
        "winter_kill_days": winter_kill_days,
    }


def _bbch_phase_weather(
    observations: list[WeatherObservation],
    crop: CropType | None,
) -> dict[str, float | int | None]:
    """Per-(crop, phase) weather aggregates — replaces generic Apr-Jul
    with BBCH-aligned splits.

    Returns 9 features keyed:
      - `precip_early_veg` / `precip_flowering` / `precip_grain_fill`
      - `temp_mean_flowering` / `temp_mean_grain_fill`
      - `heat_days_flowering` / `heat_days_grain_fill`
      - `drought_days_flowering` / `drought_days_grain_fill`

    Phase month tuples come from `CropCalendar.early_veg_months`,
    `.flowering_months`, `.grain_fill_months` — see crop_calendar.py
    for derivation rules. When crop is None or unknown, returns all
    9 keys as None — RF/Stack fall back via the standard NaN
    imputation path used by other crop-conditional features.

    Temperature mean isn't reported for early_veg (it's the
    establishment phase; mean temp is highly correlated with month
    boundaries and adds little signal). Heat-stress and drought
    counts also skipped for early_veg — the threshold-based stress
    metrics only matter once the crop is reproductive.
    """
    keys = BBCH_PHASE_FEATURES
    none_result: dict[str, float | int | None] = {k: None for k in keys}
    if crop is None or not observations:
        return none_result

    cal = CROP_CALENDAR.get(crop.value)
    if cal is None:
        return none_result

    # Cropping-year anchor: latest year with Apr-Jul observations, same
    # rule as _weather_features. For winter crops, BBCH months can
    # legitimately cross the year boundary (sow=Oct(N-1)), so we look
    # at both years and let the phase month-tuples decide membership.
    summer_years = [
        o.observed_on.year for o in observations
        if o.observed_on.month in REFERENCE_WINDOW_MONTHS
    ]
    target_year = max(summer_years) if summer_years else max(
        o.observed_on.year for o in observations
    )

    early_set = set(cal.early_veg_months)
    flower_set = set(cal.flowering_months)
    grain_set = set(cal.grain_fill_months)

    def _in_phase(o: WeatherObservation, months: set[int]) -> bool:
        """Membership-check with cropping-year wraparound. The phase
        month-tuple comes from `CropCalendar` and may include both
        autumn months (assigned to year N-1) and spring/summer months
        (assigned to year N). We tag any wraparound month <= sow_month
        as belonging to the prior year."""
        m = o.observed_on.month
        if m not in months:
            return False
        # Determine if this observation belongs to the right year.
        # Heuristic: for winter crops (sow_month > peak_month), months
        # >= sow_month belong to (target_year - 1); months <
        # sow_month belong to target_year. Spring crops: all months
        # belong to target_year.
        if cal.sow_month <= cal.harvest_month:  # spring crop
            return o.observed_on.year == target_year
        # winter crop:
        if m >= cal.sow_month:
            return o.observed_on.year == target_year - 1
        return o.observed_on.year == target_year

    def _aggregate(
        months_set: set[int], *, mean_temp: bool, count_heat: bool, count_dry: bool,
    ) -> dict[str, float | int | None]:
        sub = [o for o in observations if _in_phase(o, months_set)]
        if not sub:
            return {}
        precip_total = sum(float(o.precip_mm or 0) for o in sub)
        out: dict[str, float | int | None] = {"precip": round(precip_total, 1)}
        if mean_temp:
            temps = [float(o.temp_mean_c) for o in sub if o.temp_mean_c is not None]
            out["temp_mean"] = round(sum(temps) / len(temps), 2) if temps else None
        if count_heat:
            out["heat_days"] = sum(
                1 for o in sub
                if o.temp_max_c is not None and float(o.temp_max_c) > 30
            )
        if count_dry:
            # Longest dry-spell (continuity within the phase only).
            longest, current = 0, 0
            for o in sorted(sub, key=lambda x: x.observed_on):
                if (o.precip_mm or 0) < 1.0:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            out["drought_days"] = longest
        return out

    early = _aggregate(early_set, mean_temp=False, count_heat=False, count_dry=False)
    flower = _aggregate(flower_set, mean_temp=True, count_heat=True, count_dry=True)
    grain = _aggregate(grain_set, mean_temp=True, count_heat=True, count_dry=True)

    return {
        "precip_early_veg": early.get("precip"),
        "precip_flowering": flower.get("precip"),
        "precip_grain_fill": grain.get("precip"),
        "temp_mean_flowering": flower.get("temp_mean"),
        "temp_mean_grain_fill": grain.get("temp_mean"),
        "heat_days_flowering": flower.get("heat_days"),
        "heat_days_grain_fill": grain.get("heat_days"),
        "drought_days_flowering": flower.get("drought_days"),
        "drought_days_grain_fill": grain.get("drought_days"),
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


def _oblast_yield_lag(
    lat: float | None, lon: float | None, crop: CropType | None,
) -> dict[str, float | None]:
    """Resolve centroid → oblast → lagged Держстат yields, returns four
    feature values keyed `oblast_yield_lag1/2/3/mean3`.

    At inference time `oblast_avg_yield(name, crop, year=None)` returns
    the most-recent published yield (currently 2025 once XLS bulletins
    are ingested). All four features pull from the SAME most-recent row
    — the multi-year variation is only meaningful at training time,
    where each parquet row legitimately has a distinct (year-N) lag.
    Returning the same value across the four keys means the model gets
    a "fresh-as-possible" regional anchor for whichever lag horizon
    its SHAP-importance ranking finds informative.

    Falls back to the crop's national median when the centroid is
    outside Ukraine or the oblast has no Держстат row for this crop
    — RF and Stack don't tolerate NaN, so a numerical fallback is
    required for production inference.
    """
    keys = ("oblast_yield_lag1", "oblast_yield_lag2",
            "oblast_yield_lag3", "oblast_yield_lag_mean3")
    if crop is None:
        return {k: None for k in keys}

    # Local imports — same rationale as `_zone_features` (avoid pulling
    # shapely / pandas into v3-legacy paths that don't need geocoding).
    from app.services.dashboard_analytics import (
        find_oblast_for_centroid,
        oblast_avg_yield,
    )

    value: float | None = None
    if lat is not None and lon is not None:
        oblast_name = find_oblast_for_centroid(lat, lon)
        if oblast_name is not None:
            avg, _year = oblast_avg_yield(oblast_name, crop.value, None)
            value = avg

    # Numerical fallback for RF / Stack — crop's national median across
    # the real-yield corpus. Same logical role as the zone-all-zeros
    # reference category: "field with no known regional context".
    if value is None:
        value = _crop_national_median(crop.value)

    # Broadcast the single most-recent value to all four lag features.
    # Tree models that learned a meaningful split on lag_mean3 during
    # training will use it; the others stay near-redundant at inference
    # time. This is the cost of single-row inference vs the training
    # rows which have genuinely distinct year-N values.
    return {k: value for k in keys}


def _neighbor_yield_lag(
    lat: float | None, lon: float | None, crop: CropType | None,
) -> dict[str, float | None]:
    """Resolve centroid → oblast → mean yield across OTHER crops for
    the most-recent year. Returns `{"neighbor_yield_lag1_mean": ...}`.

    Mirror of `_oblast_yield_lag` but aggregates across the 12 crops
    NOT equal to `crop`, matching the patch-time
    `scripts/patch_parquet_neighbor_yield.py` rule. At inference time
    the most-recent published year stands in for "year-1" (the patcher
    enforces strict year-1 at training time — for the autumn-2025
    inference window, the most-recent publication is the 2025 partial
    bulletin, which is "previous-year-ish" relative to the field's
    full-season prediction date).

    Falls back to the cached global crop median (across all real-yield
    rows) when:
      - centroid → oblast resolution fails
      - the oblast has no neighbor-crop data in the parquet
      - crop=None at the v3-legacy code-path (registry override path)
    """
    key = "neighbor_yield_lag1_mean"
    if crop is None:
        return {key: None}

    from app.services.dashboard_analytics import (
        find_oblast_for_centroid,
        oblast_neighbor_yield_mean,
    )

    value: float | None = None
    if lat is not None and lon is not None:
        oblast_name = find_oblast_for_centroid(lat, lon)
        if oblast_name is not None:
            value = oblast_neighbor_yield_mean(oblast_name, crop.value)

    if value is None:
        # Fall back to the overall median yield across all crops — same
        # numerical-fallback role the own-crop national-median plays in
        # `_oblast_yield_lag`. Not crop-specific because the neighbor
        # signal is by-design cross-crop.
        medians = _crop_national_medians()
        if medians:
            value = sum(medians.values()) / len(medians)

    return {key: value}


@functools.lru_cache(maxsize=1)
def _crop_national_medians() -> dict[str, float]:
    """`{crop: median yield_tha}` over all real-yield rows in the
    training parquet. Used as a fallback by `_oblast_yield_lag` when
    the centroid can't be resolved to a Держстат-covered oblast.

    Cached because the parquet read is ~50 ms and this is called on
    every inference. Independent of which crop is being predicted —
    the whole table is materialised once.
    """
    from pathlib import Path

    import pandas as pd

    # Path mirrors `dashboard_analytics._load_oblast_yield_baseline`.
    path = Path("data/processed/training_set_v3.parquet")
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(
            path, columns=["crop", "yield_tha", "is_real_yield"],
        )
    except Exception:  # noqa: BLE001
        return {}
    real = df[df["is_real_yield"].astype(bool)]
    return real.groupby("crop")["yield_tha"].median().to_dict()


def _crop_national_median(crop_slug: str) -> float | None:
    """Convenience accessor — returns None when the parquet is missing
    or the crop has no real-yield rows. Caller (`_oblast_yield_lag`)
    falls back to None which is RF-incompatible, so this should only
    really return None during unit tests that mock out the parquet."""
    return _crop_national_medians().get(crop_slug)


def _crop_onehot_features(crop: CropType | None) -> dict[str, int]:
    """13 one-hot features `crop_{slug}` — exactly one set to 1, rest 0.

    Returns all zeros when `crop` is None — degrades gracefully for
    callers that don't have a crop assigned (e.g. exploratory data
    builds). The v8 trainer requires a crop to be set on every row;
    the inference path always knows the field's crop_type and passes
    it through `predict_yield`.

    Symmetric to `_zone_features` — same one-hot pattern, just keyed
    by `app.data_reference.crop_zones.ALL_CROPS` instead of the
    agroclimatic zone enum.
    """
    crop_slug = crop.value if crop is not None else None
    return {f"crop_{c}": int(c == crop_slug) for c in _ALL_CROPS_FOR_ONEHOT}


def _zone_features(lat: float | None, lon: float | None) -> dict[str, int]:
    """Resolve centroid → oblast → agroclimatic zone → 5 one-hot ints.

    Falls back to all-zeros when the centroid is outside Ukraine (no
    polygon match) or when the ISO lookup fails. The model treats
    all-zeros as the implicit "unknown zone" reference category — its
    own global mean prediction. That's the right behaviour: a field
    outside Ukraine has no Ukrainian agroclimatic context.
    """
    # Local imports for the same reason `_soilgrids_for_centroid` does
    # them: avoid pulling shapely / geojson into callers that don't
    # need geocoding (e.g. v3-model legacy paths).
    from app.data_reference.oblast_names import oblast_by_iso
    from app.services.dashboard_analytics import find_oblast_for_centroid

    zone: str | None = None
    if lat is not None and lon is not None:
        natural_earth = find_oblast_for_centroid(lat, lon)
        iso = iso_from_any_name(natural_earth)
        ref = oblast_by_iso(iso)
        if ref is not None:
            zone = ref.zone
    return {f"zone_{z}": int(z == zone) for z in _ZONES}


def _crop_features(
    crop: CropType,
    weather: dict[str, float | None],
    veg: dict[str, float | None],
) -> dict[str, float | None]:
    """v5 crop-specific weather-derivation features + 2 NDVI-aligned
    extensions added in the post-v7 phenology audit.

    Weather-derived (v5 originals):
      - `crop_season_overlap_aprjul = overlap_with_reference(crop)`
      - `gdd_proxy = (temp_mean - gdd_base) × season_months × 30`
      - `precip_crop_weighted = precip_sum_apr_jul × overlap`
      - `heat_stress_crop_weighted = heat_stress_days ×
                  (1.5 if flowering in Apr-Jul else 0.5)`
      - `drought_crop_weighted = drought_dryspells × overlap`
      - `growing_season_length_months = len(growing_season_months)`

    NDVI-aligned (new):
      - `ndvi_at_crop_peak_month` — pulls the right monthly NDVI from
        `veg["ndvi_mean_{may|june|july|august}"]` based on the crop's
        `peak_month`. For wheat (peak=5) returns ndvi_mean_may; for
        corn (peak=7) returns ndvi_mean_july; for sugar_beet (peak=8)
        returns ndvi_mean_august. Falls back to `ndvi_peak` when the
        peak month is outside our pre-computed May-August window
        (covers edge cases like winter-rapeseed peak=May, which IS
        in window — but defensive coding is cheap).
      - `ndvi_peak_timing_offset_weeks` — signed integer diff
        `observed_ndvi_peak_week - expected_peak_week`. Negative
        means crop peaked early (drought stress / unusually warm
        spring); positive means late (cold spring / replanting).
        Magnitude > 3 weeks is agronomically significant.

    Deterministic; no DB read beyond the already-loaded `weather`
    + `veg` dicts.
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

    # NDVI-at-peak month. Map peak_month integer to the corresponding
    # monthly NDVI slot. May/June/July/August are precomputed in
    # `_veg_features`; outside that range we fall back to the
    # season-wide `ndvi_peak`.
    ndvi_month_keys = {
        5: "ndvi_mean_may",
        6: "ndvi_mean_june",
        7: "ndvi_mean_july",
        8: "ndvi_mean_august",
    }
    peak_key = ndvi_month_keys.get(cal.peak_month)
    ndvi_at_peak = (
        veg.get(peak_key) if peak_key is not None
        else veg.get("ndvi_peak")
    )

    # Peak-timing offset in weeks. Approximate "expected peak week" as
    # the ISO week of the 15th of `peak_month` — close enough for an
    # anomaly signal (we don't need calendar precision here).
    expected_peak_week = _iso_week_for_mid_month(cal.peak_month)
    observed_peak_week = veg.get("ndvi_peak_week")
    timing_offset = (
        int(observed_peak_week) - expected_peak_week
        if observed_peak_week is not None else None
    )

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
        "ndvi_at_crop_peak_month": (
            round(float(ndvi_at_peak), 3) if ndvi_at_peak is not None else None
        ),
        "ndvi_peak_timing_offset_weeks": timing_offset,
    }


# Approximate ISO week numbers for the 15th of each calendar month.
# Used as the "expected peak week" target for the timing-offset feature.
# Values are baseline 2024 (non-leap-year); seasonal shifts vs leap
# years are sub-week and don't matter for an anomaly signal.
_ISO_WEEK_OF_MID_MONTH: dict[int, int] = {
    1: 3, 2: 7, 3: 11, 4: 16, 5: 20, 6: 24,
    7: 29, 8: 33, 9: 37, 10: 42, 11: 46, 12: 50,
}


def _iso_week_for_mid_month(month: int) -> int:
    """Return ISO week number for the 15th of `month`. Approximate but
    stable — gives the timing-offset feature a consistent reference."""
    return _ISO_WEEK_OF_MID_MONTH.get(month, 20)


def _crop_features_none() -> dict[str, float | None]:
    """Empty (None-filled) crop-specific dict, used when `crop` is not
    passed to the builder. Lets v3 models still get a full 30-key dict
    (just with NaN in the crop-specific positions, which they slice
    out anyway via `select_features`)."""
    return {name: None for name in V5_CROP_FEATURES}
