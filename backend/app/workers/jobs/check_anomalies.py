"""RQ job: run all 3 anomaly detectors for one field, insert new alerts."""
from __future__ import annotations

import logging
from datetime import date

from rq import get_current_job
from sqlalchemy import and_, create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Alert, Field, SatelliteObservation, WeatherObservation
from app.db.models.enums import CropType
from app.ml.anomaly import (
    AnomalySignal,
    detect_drought,
    detect_heat_stress,
    detect_ndvi_anomalies,
)
from app.ml.registry import get_registry
from app.redis_clients import get_sync_redis
from app.workers.pubsub import publish_progress

log = logging.getLogger(__name__)


def check_field_anomalies(field_id: int) -> dict:
    redis = get_sync_redis()
    job = get_current_job()
    job_id = job.id if job else "no-job"
    publish_progress(redis, job_id, "running", progress=0.2)

    engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
    try:
        get_registry().load_all()
        norms = get_registry().seasonal_norms

        with Session(engine, expire_on_commit=False) as s:
            field = s.get(Field, field_id)
            if field is None:
                publish_progress(redis, job_id, "failed", error="Field not found")
                return {"ok": False, "reason": "not_found"}
            crop = (
                field.crop_type
                if isinstance(field.crop_type, CropType)
                else CropType(field.crop_type)
            )
            user_id = field.user_id

            obs = list(s.scalars(
                select(SatelliteObservation)
                .where(SatelliteObservation.field_id == field_id)
                .order_by(SatelliteObservation.observed_on.asc())
            ).all())
            weather = list(s.scalars(
                select(WeatherObservation)
                .where(WeatherObservation.field_id == field_id)
                .order_by(WeatherObservation.observed_on.asc())
            ).all())

            publish_progress(redis, job_id, "running", progress=0.5)

            signals: list[AnomalySignal] = []
            signals.extend(detect_ndvi_anomalies(obs, crop, norms))
            d = detect_drought(weather, date.today())
            if d is not None:
                signals.append(d)
            h = detect_heat_stress(weather, date.today())
            if h is not None:
                signals.append(h)

            # Dedup against existing unacknowledged alerts.
            existing = set(s.scalars(
                select(Alert.type).where(and_(
                    Alert.field_id == field_id,
                    Alert.acknowledged.is_(False),
                ))
            ).all())
            inserted = 0
            for sig in signals:
                if sig.type in existing:
                    continue  # already alerted, not yet ack'd
                s.add(Alert(
                    field_id=field_id,
                    user_id=user_id,
                    severity=sig.severity,
                    type=sig.type,
                    message_uk=sig.message_uk,
                    metric_value=sig.metric_value,
                    threshold=sig.threshold,
                ))
                existing.add(sig.type)
                inserted += 1
            s.commit()

        # Notify the user-scoped alerts stream so the bell icon updates live.
        from app.workers.pubsub import channel_for
        import json
        redis.publish(
            f"user:{user_id}:alerts",
            json.dumps({"new": inserted}, ensure_ascii=False),
        )

        publish_progress(redis, job_id, "done", progress=1.0,
                         data={"new_alerts": inserted, "checked": len(signals)})
        return {"ok": True, "new_alerts": inserted}

    except Exception as exc:  # noqa: BLE001
        log.exception("check_anomalies failed for field=%s", field_id)
        publish_progress(redis, job_id, "failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        engine.dispose()
