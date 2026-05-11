"""Field-related request/response schemas.

Wire format for geometry is GeoJSON — the only widely-supported, human-readable,
JS-friendly format. The service layer converts GeoJSON → WKT for PostGIS and
back. Clients never see WKT/WKB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field as PydField

from app.db.models.enums import CropType


# ───── GeoJSON primitives ────────────────────────────────────────


class PolygonGeometry(BaseModel):
    """GeoJSON Polygon.

    `coordinates` is a list of linear rings. The first ring is the outer
    boundary, subsequent rings (if any) are holes. Each ring is a list of
    [lon, lat] pairs; the first and last point of a ring MUST be equal.
    For our use case we always send a single ring (no holes).
    """

    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class PointGeometry(BaseModel):
    type: Literal["Point"]
    coordinates: list[float]  # [lon, lat]


# ───── Field DTOs ────────────────────────────────────────────────


class FieldCreate(BaseModel):
    name: str = PydField(min_length=1, max_length=255)
    crop_type: CropType
    season_year: int = PydField(ge=2000, le=2100)
    geometry: PolygonGeometry
    color: str | None = PydField(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class FieldUpdate(BaseModel):
    name: str | None = PydField(default=None, min_length=1, max_length=255)
    crop_type: CropType | None = None
    season_year: int | None = PydField(default=None, ge=2000, le=2100)
    geometry: PolygonGeometry | None = None
    color: str | None = PydField(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class FieldRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    crop_type: CropType
    season_year: int
    geometry: PolygonGeometry
    centroid: PointGeometry
    area_ha: float
    color: str
    created_at: datetime
    updated_at: datetime
