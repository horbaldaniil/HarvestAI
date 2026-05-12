"""Prediction + alerts + weather + dashboard schemas."""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


# ───── Predictions ────────────────────────────────────────


class ShapBar(BaseModel):
    name: str
    value: float | None
    contribution: float


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field_id: int
    model_name: str
    model_version: str
    value_tha: float
    confidence: float | None
    shap_top_json: list[ShapBar]
    predicted_at: datetime


# ───── Alerts ─────────────────────────────────────────────


class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field_id: int
    # Name of the field this alert is about — denormalised at read time so
    # the bell dropdown and dashboard list can show it without an extra
    # round-trip.
    field_name: str | None = None
    severity: str
    type: str
    message_uk: str
    metric_value: float | None
    threshold: float | None
    acknowledged: bool
    created_at: datetime


class AlertsCountResponse(BaseModel):
    unread: int


# ───── Weather ────────────────────────────────────────────


class WeatherDayRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    observed_on: date
    temp_min_c: float | None
    temp_max_c: float | None
    temp_mean_c: float | None
    precip_mm: float | None
    humidity_pct: float | None
    radiation_mj: float | None
    # Week 8 — extra metrics powering the dedicated /weather page. Older
    # cached rows have NULL here; the UI degrades gracefully.
    wind_speed_max_ms: float | None = None
    cloud_cover_pct: float | None = None
    soil_moisture_0_10cm: float | None = None
    is_forecast: bool


# ───── Dashboard ──────────────────────────────────────────


class YearOverYear(BaseModel):
    current_year: int
    current_year_avg_ndvi: float | None
    prev_year_avg_ndvi: float | None
    diff_pct: float | None


class DashboardKpis(BaseModel):
    total_fields: int
    total_area_ha: float
    predicted_total_yield_t: float | None
    avg_ndvi_current: float | None
    avg_ndwi_current: float | None = None  # Week 7: portfolio-wide moisture proxy
    active_alerts_count: int


class DashboardFieldRow(BaseModel):
    field_id: int
    name: str
    crop_type: str
    area_ha: float
    current_ndvi: float | None
    current_ndwi: float | None = None
    predicted_tha: float | None
    has_alerts: bool
    # Composite 0–100 risk score (higher = worse) plus the contributing
    # factors as short tags so the UI can show a tooltip.
    risk_score: int = 0
    risk_factors: list[str] = []
    # Resolved oblast (Ukrainian display name) the field's centroid lands in.
    # None if the centroid is outside Ukraine or the oblast geojson is missing.
    oblast_name: str | None = None
    # Optional oblast-level NDVI mean used for the "your field vs oblast"
    # comparison column. Present only when training_set_v2 covers this oblast.
    oblast_avg_ndvi: float | None = None
    # Which year the baseline NDVI was sourced from (most-recent year in
    # the Week 6 parquet). Lets the UI label the comparison honestly.
    oblast_baseline_year: int | None = None
    # GeoJSON Polygon for the mini-map on the dashboard. Allows the UI to
    # render all fields in one round-trip without hitting /api/fields.
    geometry: dict | None = None


class CropBreakdownItem(BaseModel):
    crop_type: str
    field_count: int
    area_ha: float


class FieldYoYDelta(BaseModel):
    """One row in the dashboard's "Top-3 NDVI movers" card. Replaces the
    legacy portfolio-average YoY widget — names a specific field, shows
    a signed % delta, and lets the UI link straight to the field page."""
    field_id: int
    name: str
    crop_type: str
    current_year_ndvi: float
    prev_year_ndvi: float
    diff_pct: float


class FieldWeather(BaseModel):
    """Per-field 7-day weather summary. Replaces the old averaged
    `WeatherSummary` — much clearer where "this weather" applies, especially
    when fields are spread across oblasts.

    The `temp_min_7d` and `temp_avg_7d` aggregates are surfaced for the
    per-field row on the dashboard's WeatherSummaryCard ("Серед. NN°C,
    ↑max°/↓min°"). `crop_type` is denormalised here so the frontend can
    render the crop chip + label without joining back against the
    `fields` collection.
    """
    field_id: int
    field_name: str
    crop_type: str
    centroid_lat: float | None
    centroid_lon: float | None
    days: list[WeatherDayRead] = []
    temp_max_7d: float | None = None
    temp_min_7d: float | None = None
    temp_avg_7d: float | None = None
    precip_sum_7d: float | None = None
    heat_stress_days_7d: int = 0


class DashboardResponse(BaseModel):
    kpis: DashboardKpis
    fields: list[DashboardFieldRow]
    yoy: YearOverYear
    # Top-3 fields with the biggest |YoY% change|. Replaces the
    # portfolio-average YoY widget.
    top_movers: list[FieldYoYDelta] = []
    crops_breakdown: list[CropBreakdownItem] = []
    # Per-field 7-day forecast list; UI picks one via dropdown.
    weather_by_field: list[FieldWeather] = []
    # Echo back what filters were applied so the UI can verify state.
    applied_crops: list[str] = []
