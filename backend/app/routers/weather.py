"""Weather endpoints for Dashboard forecast card."""
from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from app.db.models import Field, WeatherObservation
from app.deps import CurrentUser, DbSession
from app.schemas.prediction import WeatherDayRead

router = APIRouter(prefix="/api/fields", tags=["weather"])


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
