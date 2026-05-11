"""RQ job: fetch 2-year Statistical-API time-series for one field.

This runs in a sync RQ worker process, so we use:
- a sync DB session via psycopg2 + SQLAlchemy sync engine
- a sync Redis client
- `asyncio.run()` to call the async Sentinel HTTP wrappers
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta

from rq import get_current_job
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from geoalchemy2.functions import ST_AsGeoJSON

from app.config import settings
from app.db.models import Field, SatelliteObservation
from app.integrations.sentinel_hub.service import SentinelService
from app.redis_clients import get_sync_redis
from app.workers.pubsub import publish_progress

log = logging.getLogger(__name__)


def fetch_field_observations(field_id: int, years_back: int = 2) -> dict:
    """RQ entrypoint. Returns a small summary dict (recorded as job result)."""
    redis = get_sync_redis()
    job = get_current_job()
    job_id = job.id if job else "no-job"

    publish_progress(redis, job_id, "running", progress=0.05)

    engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
    try:
        with Session(engine, expire_on_commit=False) as session:
            field = session.get(Field, field_id)
            if field is None:
                publish_progress(redis, job_id, "failed", error="Field not found")
                return {"ok": False, "reason": "not_found"}
            geometry_geojson = session.scalar(
                select(ST_AsGeoJSON(field.geom))
            )

        publish_progress(redis, job_id, "running", progress=0.15)

        import json
        geometry = json.loads(geometry_geojson)
        end = date.today()
        start = end - timedelta(days=365 * years_back)

        log.info(
            "fetch_sentinel: field=%s start=%s end=%s", field_id, start, end
        )

        rows, pu_spent = asyncio.run(_fetch(geometry, start, end))

        publish_progress(redis, job_id, "running", progress=0.75)

        # UPSERT rows into satellite_observations.
        with Session(engine) as session:
            for row in rows:
                stmt = insert(SatelliteObservation).values(
                    field_id=field_id,
                    observed_on=row.observed_on,
                    source="sentinel-2-l2a",
                    ndvi_mean=row.ndvi_mean,
                    ndvi_min=row.ndvi_min,
                    ndvi_max=row.ndvi_max,
                    ndvi_std=row.ndvi_std,
                    evi_mean=row.evi_mean,
                    ndwi_mean=row.ndwi_mean,
                    savi_mean=row.savi_mean,
                    cloud_cover=row.cloud_cover,
                    processing_units=0,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_observation_field_date_source",
                    set_={
                        "ndvi_mean": stmt.excluded.ndvi_mean,
                        "ndvi_min": stmt.excluded.ndvi_min,
                        "ndvi_max": stmt.excluded.ndvi_max,
                        "ndvi_std": stmt.excluded.ndvi_std,
                        "evi_mean": stmt.excluded.evi_mean,
                        "ndwi_mean": stmt.excluded.ndwi_mean,
                        "savi_mean": stmt.excluded.savi_mean,
                        "cloud_cover": stmt.excluded.cloud_cover,
                        "updated_at": datetime.now(UTC),
                    },
                )
                session.execute(stmt)
            session.commit()

        publish_progress(
            redis, job_id, "done",
            progress=1.0,
            data={"rows": len(rows), "pu_spent": float(pu_spent)},
        )
        return {"ok": True, "rows": len(rows), "pu_spent": float(pu_spent)}

    except Exception as exc:  # noqa: BLE001
        log.exception("fetch_sentinel failed for field=%s", field_id)
        publish_progress(redis, job_id, "failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        engine.dispose()


async def _fetch(geometry: dict, start: date, end: date):
    """Create a fresh async Redis client bound to THIS event loop.

    We can't reuse the FastAPI process singleton: each `asyncio.run()` inside
    an RQ worker spins up a new event loop and the old loop's transports
    become invalid. Creating per-call avoids the dreaded
    "RuntimeError: Event loop is closed" trace.
    """
    import redis.asyncio as async_redis

    from app.config import settings

    r = async_redis.from_url(settings.redis_url, decode_responses=False)
    svc = SentinelService(r)
    try:
        return await svc.fetch_timeseries(geometry, start, end)
    finally:
        await svc.aclose()
        await r.aclose()
