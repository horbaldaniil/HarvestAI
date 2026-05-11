"""RQ job: render a single-date Process-API heatmap for one (field, index)."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path

from rq import get_current_job
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from geoalchemy2.functions import ST_AsGeoJSON

from app.config import settings
from app.db.models import Field, SatelliteObservation
from app.integrations.sentinel_hub.service import SentinelService
from app.redis_clients import get_sync_redis
from app.workers.pubsub import publish_progress

log = logging.getLogger(__name__)


def fetch_field_heatmap(field_id: int, target_date_iso: str, index: str) -> dict:
    redis = get_sync_redis()
    job = get_current_job()
    job_id = job.id if job else "no-job"

    target_date = date.fromisoformat(target_date_iso)
    publish_progress(redis, job_id, "running", progress=0.1)

    engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
    try:
        with Session(engine, expire_on_commit=False) as session:
            field = session.get(Field, field_id)
            if field is None:
                publish_progress(redis, job_id, "failed", error="Field not found")
                return {"ok": False, "reason": "not_found"}
            geometry_geojson = session.scalar(select(ST_AsGeoJSON(field.geom)))

        geometry = json.loads(geometry_geojson)
        publish_progress(redis, job_id, "running", progress=0.3)

        png_bytes, pu_spent = asyncio.run(_fetch(geometry, target_date, index))
        publish_progress(redis, job_id, "running", progress=0.8)

        # Persist PNG to RASTERS_DIR/{field_id}/{date}_{index}.png
        out_dir: Path = settings.rasters_dir / str(field_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_path = f"{field_id}/{target_date.isoformat()}_{index}.png"
        full_path = settings.rasters_dir / rel_path
        full_path.write_bytes(png_bytes)

        # Record the raster_uri against the corresponding observation row
        # (we keep one raster per observation row even if it's per-index).
        with Session(engine) as session:
            session.execute(
                update(SatelliteObservation)
                .where(
                    SatelliteObservation.field_id == field_id,
                    SatelliteObservation.observed_on == target_date,
                )
                .values(raster_uri=rel_path)
            )
            session.commit()

        publish_progress(
            redis, job_id, "done",
            progress=1.0,
            data={"raster_uri": rel_path, "pu_spent": float(pu_spent), "index": index},
        )
        return {
            "ok": True,
            "raster_uri": rel_path,
            "pu_spent": float(pu_spent),
            "index": index,
        }

    except Exception as exc:  # noqa: BLE001
        log.exception("fetch_heatmap failed: field=%s date=%s", field_id, target_date)
        publish_progress(redis, job_id, "failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        engine.dispose()


async def _fetch(geometry: dict, target_date: date, index: str):
    """Per-call async Redis client — see note in fetch_sentinel._fetch."""
    import redis.asyncio as async_redis

    from app.config import settings

    r = async_redis.from_url(settings.redis_url, decode_responses=False)
    svc = SentinelService(r)
    try:
        return await svc.fetch_heatmap(geometry, target_date, index)
    finally:
        await svc.aclose()
        await r.aclose()
