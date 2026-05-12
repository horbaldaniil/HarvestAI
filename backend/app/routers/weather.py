"""Weather endpoints — Dashboard forecast card + dedicated /weather page."""
from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.db.models import Field, WeatherObservation
from app.deps import CurrentUser, DbSession
from app.schemas.prediction import WeatherDayRead
from app.services.weather_advice import build_advices

router = APIRouter(prefix="/api/fields", tags=["weather"])


# ─── Forecast list (existing — used by BottomPanel, etc.) ──────


@router.get(
    "/{field_id}/weather",
    response_model=list[WeatherDayRead],
)
async def list_weather(
    field_id: int,
    current_user: CurrentUser,
    db: DbSession,
    days: int = Query(default=14, ge=1, le=60),
    forecast_only: bool = Query(default=False),
) -> list[WeatherDayRead]:
    f = await db.get(Field, field_id)
    if f is None or f.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Поле не знайдено.")

    today = date.today()
    until = today + timedelta(days=days)
    stmt = (
        select(WeatherObservation)
        .where(WeatherObservation.field_id == field_id)
        .where(WeatherObservation.observed_on >= today)
        .where(WeatherObservation.observed_on <= until)
        .order_by(WeatherObservation.observed_on.asc())
    )
    if forecast_only:
        stmt = stmt.where(WeatherObservation.is_forecast.is_(True))

    result = await db.execute(stmt)
    return [WeatherDayRead.model_validate(w) for w in result.scalars().all()]


# ─── Detailed weather page payload ─────────────────────────────


class WeatherAdviceRead(BaseModel):
    severity: str
    title: str
    detail: str


class WeatherDetail(BaseModel):
    field_id: int
    field_name: str
    crop_type: str
    centroid_lat: float | None
    centroid_lon: float | None
    days: list[WeatherDayRead]
    advices: list[WeatherAdviceRead]


@router.get(
    "/{field_id}/weather/detail",
    response_model=WeatherDetail,
)
async def weather_detail(
    field_id: int,
    current_user: CurrentUser,
    db: DbSession,
    days: int = Query(default=14, ge=1, le=16),
) -> WeatherDetail:
    """Everything the /weather page needs for one field in one call:
    upcoming-N-days forecast + computed rule-based advisories."""
    f = await db.get(Field, field_id)
    if f is None or f.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Поле не знайдено.")

    today = date.today()
    until = today + timedelta(days=days)
    rows = list((await db.execute(
        select(WeatherObservation)
        .where(WeatherObservation.field_id == field_id)
        .where(WeatherObservation.is_forecast.is_(True))
        .where(WeatherObservation.observed_on >= today)
        .where(WeatherObservation.observed_on <= until)
        .order_by(WeatherObservation.observed_on.asc())
    )).scalars().all())

    day_payloads = [WeatherDayRead.model_validate(r) for r in rows]
    crop_val = f.crop_type.value if hasattr(f.crop_type, "value") else str(f.crop_type)
    advices = build_advices(
        [d.model_dump() for d in day_payloads],
        crop=crop_val,
    )

    # Centroid via PostGIS WKB → shapely. Falls back to None on parse failure.
    centroid_lat = centroid_lon = None
    try:
        from shapely import wkb

        pt = wkb.loads(bytes(f.centroid.data))
        centroid_lat, centroid_lon = round(pt.y, 5), round(pt.x, 5)
    except Exception:  # noqa: BLE001
        pass

    return WeatherDetail(
        field_id=f.id,
        field_name=f.name,
        crop_type=crop_val,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        days=day_payloads,
        advices=[
            WeatherAdviceRead(severity=a.severity, title=a.title, detail=a.detail)
            for a in advices
        ],
    )
