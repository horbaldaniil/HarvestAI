"""Schemas for satellite observations + job + quota responses."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ObservationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    observed_on: date
    ndvi_mean: float | None
    ndvi_min: float | None
    ndvi_max: float | None
    ndvi_std: float | None
    evi_mean: float | None
    ndwi_mean: float | None
    savi_mean: float | None
    cloud_cover: float | None
    raster_uri: str | None


class JobHandleResponse(BaseModel):
    job_id: str
    queue: str
    status_url: str  # SSE stream URL


class JobStatusResponse(BaseModel):
    job_id: str
    state: Literal["queued", "started", "running", "done", "failed", "unknown"]
    result: dict | None = None


class QuotaResponse(BaseModel):
    units_used: float
    units_limit: float
    units_remaining: float
    percent_used: float
    period_start: date
    period_end: date
    refresh_at: datetime
