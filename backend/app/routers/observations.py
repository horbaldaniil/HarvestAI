"""Satellite observation endpoints: list time-series, trigger refresh + heatmap, serve PNG.

All endpoints scope by the current user via the field's ownership.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.config import settings
from app.db.models import Field, SatelliteObservation
from app.deps import CurrentUser, DbSession
from app.schemas.observation import JobHandleResponse, ObservationRead
from app.workers.dispatcher import (
    enqueue_fetch_heatmap,
    enqueue_fetch_observations,
)

router = APIRouter(prefix="/api", tags=["observations"])


async def _get_owned_field(db, user_id: int, field_id: int) -> Field:
    field = await db.get(Field, field_id)
    if field is None or field.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Поле не знайдено.",
        )
    return field


@router.get(
    "/fields/{field_id}/observations",
    response_model=list[ObservationRead],
)
async def list_observations(
    field_id: int,
    current_user: CurrentUser,
    db: DbSession,
    since: date | None = Query(default=None),
    until: date | None = Query(default=None),
) -> list[ObservationRead]:
    await _get_owned_field(db, current_user.id, field_id)

    stmt = (
        select(SatelliteObservation)
        .where(SatelliteObservation.field_id == field_id)
        .order_by(SatelliteObservation.observed_on.asc())
    )
    if since is not None:
        stmt = stmt.where(SatelliteObservation.observed_on >= since)
    if until is not None:
        stmt = stmt.where(SatelliteObservation.observed_on <= until)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    return [ObservationRead.model_validate(r) for r in rows]


@router.post(
    "/fields/{field_id}/observations/refresh",
    response_model=JobHandleResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def refresh_observations(
    field_id: int,
    current_user: CurrentUser,
    db: DbSession,
    years_back: int = Query(default=2, ge=1, le=5),
) -> JobHandleResponse:
    await _get_owned_field(db, current_user.id, field_id)
    handle = enqueue_fetch_observations(field_id, years_back)
    return JobHandleResponse(
        job_id=handle.job_id,
        queue=handle.queue,
        status_url=f"/api/jobs/{handle.job_id}/stream",
    )


@router.post(
    "/fields/{field_id}/observations/{date_iso}/heatmap",
    response_model=JobHandleResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_heatmap(
    field_id: int,
    date_iso: str,
    current_user: CurrentUser,
    db: DbSession,
    index: str = Query(default="ndvi", pattern=r"^(ndvi|evi|ndwi|savi)$"),
) -> JobHandleResponse:
    await _get_owned_field(db, current_user.id, field_id)
    # Validate date format
    try:
        date.fromisoformat(date_iso)
    except ValueError as exc:
        raise HTTPException(400, detail="Невірний формат дати, очікую YYYY-MM-DD.") from exc

    handle = enqueue_fetch_heatmap(field_id, date_iso, index)
    return JobHandleResponse(
        job_id=handle.job_id,
        queue=handle.queue,
        status_url=f"/api/jobs/{handle.job_id}/stream",
    )


@router.get("/fields/{field_id}/heatmaps/{date_iso}_{index}.png")
async def get_heatmap_png(
    field_id: int,
    date_iso: str,
    index: str,
    current_user: CurrentUser,
    db: DbSession,
):
    await _get_owned_field(db, current_user.id, field_id)
    if index not in {"ndvi", "evi", "ndwi", "savi"}:
        raise HTTPException(400, detail="Невідомий індекс.")
    rel = f"{field_id}/{date_iso}_{index}.png"
    path = settings.rasters_dir / rel
    if not path.exists():
        raise HTTPException(404, detail="Heatmap не знайдено — спробуйте згенерувати.")
    return FileResponse(path, media_type="image/png")
