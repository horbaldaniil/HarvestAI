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
    active_alerts_count: int


class DashboardFieldRow(BaseModel):
    field_id: int
    name: str
    crop_type: str
    area_ha: float
    current_ndvi: float | None
    predicted_tha: float | None
    has_alerts: bool


class DashboardResponse(BaseModel):
    kpis: DashboardKpis
    fields: list[DashboardFieldRow]
    yoy: YearOverYear
