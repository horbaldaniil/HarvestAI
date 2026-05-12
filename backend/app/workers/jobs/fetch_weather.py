"""RQ job: fetch Open-Meteo daily history (2y) + 14-day forecast for one field."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta

from rq import get_current_job
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from geoalchemy2.functions import ST_X, ST_Y

from app.config import settings
from app.db.models import Field, WeatherObservation
from app.integrations.openmeteo.client import OpenMeteoClient
from app.redis_clients import get_sync_redis
from app.workers.pubsub import publish_progress

log = logging.getLogger(__name__)


def fetch_field_weather(field_id: int, years_back: int = 2) -> dict:
    """RQ entrypoint."""
    redis = get_sync_redis()
    job = get_current_job()
    job_id = job.id if job else "no-job"
    publish_progress(redis, job_id, "running", progress=0.1)

    engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
    try:
        with Session(engine, expire_on_commit=False) as s:
            field = s.get(Field, field_id)
            if field is None:
                publish_progress(redis, job_id, "failed", error="Field not found")
                return {"ok": False, "reason": "not_found"}
            lat, lon = s.execute(
                select(ST_Y(field.centroid), ST_X(field.centroid))
            ).one()
            lat, lon = float(lat), float(lon)

        publish_progress(redis, job_id, "running", progress=0.2)

        end_hist = date.today() - timedelta(days=1)
        start_hist = end_hist - timedelta(days=365 * years_back)
        hist, fc = asyncio.run(_fetch(lat, lon, start_hist, end_hist))

        publish_progress(redis, job_id, "running", progress=0.7)

        with Session(engine) as s:
            _upsert(s, field_id, hist, is_forecast=False)
            _upsert(s, field_id, fc, is_forecast=True)
            s.commit()

        publish_progress(redis, job_id, "done", progress=1.0,
                         data={"hist_days": len(hist), "forecast_days": len(fc)})
        return {"ok": True, "hist_days": len(hist), "forecast_days": len(fc)}

    except Exception as exc:  # noqa: BLE001
        log.exception("fetch_weather failed for field=%s", field_id)
        publish_progress(redis, job_id, "failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        engine.dispose()


def _upsert(session: Session, field_id: int, rows, *, is_forecast: bool) -> None:
    for r in rows:
        stmt = insert(WeatherObservation).values(
            field_id=field_id,
            observed_on=r.observed_on,
            source="open-meteo",
            temp_min_c=r.temp_min_c,
            temp_max_c=r.temp_max_c,
            temp_mean_c=r.temp_mean_c,
            precip_mm=r.precip_mm,
            humidity_pct=r.humidity_pct,
            radiation_mj=r.radiation_mj,
            wind_speed_max_ms=r.wind_speed_max_ms,
            cloud_cover_pct=r.cloud_cover_pct,
            soil_moisture_0_10cm=r.soil_moisture_0_10cm,
            is_forecast=is_forecast,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_weather_field_date_source",
            set_={
                "temp_min_c": stmt.excluded.temp_min_c,
                "temp_max_c": stmt.excluded.temp_max_c,
                "temp_mean_c": stmt.excluded.temp_mean_c,
                "precip_mm": stmt.excluded.precip_mm,
                "humidity_pct": stmt.excluded.humidity_pct,
                "radiation_mj": stmt.excluded.radiation_mj,
                "wind_speed_max_ms": stmt.excluded.wind_speed_max_ms,
                "cloud_cover_pct": stmt.excluded.cloud_cover_pct,
                "soil_moisture_0_10cm": stmt.excluded.soil_moisture_0_10cm,
                "is_forecast": stmt.excluded.is_forecast,
                "updated_at": datetime.now(UTC),
            },
        )
        session.execute(stmt)


async def _fetch(lat: float, lon: float, start: date, end: date):
    client = OpenMeteoClient()
    try:
        hist = await client.fetch_historical(lat, lon, start, end)
        fc = await client.fetch_forecast(lat, lon, days=14)
        return hist, fc
    finally:
        await client.aclose()
