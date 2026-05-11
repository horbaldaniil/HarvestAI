"""Field business logic: CRUD + validation + GeoJSON <-> PostGIS conversions.

Authorization model is simple and centralized: every read/write path filters by
`Field.user_id == current_user.id`. Cross-user reads return 404 (not 403) to
avoid leaking existence of fields.
"""
from __future__ import annotations

import json
import logging

from fastapi import HTTPException, status
from geoalchemy2.functions import ST_AsGeoJSON, ST_GeogFromText
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Field
from app.db.models.enums import CropType
from app.schemas.field import (
    FieldCreate,
    FieldRead,
    FieldUpdate,
    PointGeometry,
    PolygonGeometry,
)
from app.utils.geo import polygon_to_wkt, validate_polygon

log = logging.getLogger(__name__)


def _try_enqueue_observations(field_id: int) -> str | None:
    """Best-effort hand-off to the full post-create pipeline.

    Enqueues the chained jobs:
      fetch_sentinel → fetch_weather → predict_yield → check_anomalies

    The returned job_id is the LAST job in the chain. Front-end subscribes
    to its SSE stream — when it reports "done", all four upstream jobs
    have already completed (RQ depends_on guarantees ordering).
    """
    try:
        from app.workers.dispatcher import enqueue_full_pipeline

        handle = enqueue_full_pipeline(field_id)
        return handle.job_id
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not enqueue post-create pipeline for field=%s: %s. "
            "User can trigger refresh manually.",
            field_id, exc,
        )
        return None


async def _to_read(
    db: AsyncSession, field: Field, pending_job_id: str | None = None
) -> FieldRead:
    """Build a FieldRead DTO, fetching PostGIS geometries as GeoJSON.

    PostGIS stores WKB. ST_AsGeoJSON returns a JSON string we can json.loads
    then feed into PolygonGeometry/PointGeometry.
    """
    geom_json, centroid_json = (
        await db.execute(
            select(
                ST_AsGeoJSON(field.geom),
                ST_AsGeoJSON(field.centroid),
            )
        )
    ).one()

    crop = (
        field.crop_type
        if isinstance(field.crop_type, CropType)
        else CropType(field.crop_type)
    )
    return FieldRead(
        id=field.id,
        name=field.name,
        crop_type=crop,
        season_year=field.season_year,
        geometry=PolygonGeometry(**json.loads(geom_json)),
        centroid=PointGeometry(**json.loads(centroid_json)),
        area_ha=float(field.area_ha),
        color=field.color or crop.default_color,
        created_at=field.created_at,
        updated_at=field.updated_at,
        pending_job_id=pending_job_id,
    )


async def create_field(db: AsyncSession, user_id: int, data: FieldCreate) -> FieldRead:
    validate_polygon(data.geometry)

    wkt = polygon_to_wkt(data.geometry)
    field = Field(
        user_id=user_id,
        name=data.name,
        crop_type=data.crop_type,
        season_year=data.season_year,
        color=data.color,
        geom=ST_GeogFromText(f"SRID=4326;{wkt}"),
    )
    db.add(field)
    await db.flush()
    await db.refresh(field)  # pull back generated columns

    # Kick off the 2-year Sentinel time-series fetch in the background and
    # surface the job_id so the frontend can subscribe to SSE for progress.
    pending_job_id = _try_enqueue_observations(field.id)

    return await _to_read(db, field, pending_job_id=pending_job_id)


async def list_fields(db: AsyncSession, user_id: int) -> list[FieldRead]:
    result = await db.execute(
        select(Field).where(Field.user_id == user_id).order_by(Field.created_at.desc())
    )
    fields = list(result.scalars().all())
    return [await _to_read(db, f) for f in fields]


async def get_field(db: AsyncSession, user_id: int, field_id: int) -> FieldRead:
    field = await _get_owned_or_404(db, user_id, field_id)
    return await _to_read(db, field)


async def update_field(
    db: AsyncSession, user_id: int, field_id: int, data: FieldUpdate
) -> FieldRead:
    field = await _get_owned_or_404(db, user_id, field_id)

    geometry_changed = False
    if data.geometry is not None:
        validate_polygon(data.geometry)
        wkt = polygon_to_wkt(data.geometry)
        field.geom = ST_GeogFromText(f"SRID=4326;{wkt}")
        geometry_changed = True

    if data.name is not None:
        field.name = data.name
    if data.crop_type is not None:
        field.crop_type = data.crop_type
    if data.season_year is not None:
        field.season_year = data.season_year
    if data.color is not None:
        field.color = data.color

    await db.flush()
    await db.refresh(field)

    # If geometry changed, prior observations don't match the new shape →
    # re-fetch from Sentinel.
    pending_job_id: str | None = None
    if geometry_changed:
        pending_job_id = _try_enqueue_observations(field.id)

    return await _to_read(db, field, pending_job_id=pending_job_id)


async def delete_field(db: AsyncSession, user_id: int, field_id: int) -> None:
    field = await _get_owned_or_404(db, user_id, field_id)
    await db.delete(field)
    await db.flush()


# ───── Internal ──────────────────────────────────────────────────


async def _get_owned_or_404(db: AsyncSession, user_id: int, field_id: int) -> Field:
    field = await db.get(Field, field_id)
    if field is None or field.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Поле не знайдено.",
        )
    return field
