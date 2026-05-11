"""RQ job: compute a yield prediction for one field + persist to DB."""
from __future__ import annotations

import logging

from rq import get_current_job
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Field, Prediction
from app.db.models.enums import CropType
from app.ml.registry import get_registry
from app.ml.yield_model import predict_yield
from app.redis_clients import get_sync_redis
from app.workers.pubsub import publish_progress

log = logging.getLogger(__name__)


def predict_field_yield(field_id: int) -> dict:
    redis = get_sync_redis()
    job = get_current_job()
    job_id = job.id if job else "no-job"
    publish_progress(redis, job_id, "running", progress=0.2)

    engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
    try:
        # ModelRegistry loads on first use; idempotent.
        get_registry().load_all()

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

            publish_progress(redis, job_id, "running", progress=0.5)
            result = predict_yield(s, field_id, crop)

            pred = Prediction(
                field_id=field_id,
                model_name=result.model_name,
                model_version=result.model_version,
                value_tha=result.value_tha,
                confidence=result.confidence,
                features_json={k: v for k, v in result.features.items()},
                shap_top_json=result.shap_top,
            )
            s.add(pred)
            s.commit()

        publish_progress(
            redis, job_id, "done", progress=1.0,
            data={"value_tha": result.value_tha, "confidence": result.confidence,
                  "model_version": result.model_version},
        )
        return {
            "ok": True, "value_tha": result.value_tha,
            "confidence": result.confidence,
            "model_version": result.model_version,
        }

    except Exception as exc:  # noqa: BLE001
        log.exception("predict_yield failed for field=%s", field_id)
        publish_progress(redis, job_id, "failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
    finally:
        engine.dispose()
