"""RQ job: compute a yield prediction for one field + persist to DB.

After the sync `predict_yield()` returns, we make a best-effort async
call to gpt-4o-mini to generate a 2-3 sentence narrative explanation
(see `app/ml/explanation.py`). The narrative is cached on the
Prediction row so page-views never trigger an OpenAI call — only the
one "Перерахувати" click that originated this job does.

Failures of the LLM step never abort the job: a prediction without a
summary is still useful (range / SHAP / number all render). The user
just sees no italic narrative under the value.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from rq import get_current_job
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Field, Prediction
from app.db.models.enums import CropType
from app.ml.explanation import generate_explanation_via_llm
from app.ml.registry import get_registry
from app.ml.yield_model import predict_yield
from app.redis_clients import get_sync_redis
from app.services.dashboard_analytics import (
    find_oblast_for_centroid,
    oblast_avg_ndvi,
)
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

            # Best-effort LLM narrative. Reads the field's oblast for
            # comparison context (NDVI baseline proxy → translates to a
            # "vs oblast" framing in the prompt). Failures here never
            # abort the job — summary_text stays None and the UI hides
            # that block.
            publish_progress(redis, job_id, "running", progress=0.75)
            summary = _try_generate_summary(field, result, crop)
            if summary is not None:
                result = replace(result, summary_text=summary)

            pred = Prediction(
                field_id=field_id,
                model_name=result.model_name,
                model_version=result.model_version,
                value_tha=result.value_tha,
                confidence=result.confidence,
                value_tha_q05=result.value_tha_q05,
                value_tha_q95=result.value_tha_q95,
                explainer_source=result.explainer_source,
                features_json={k: v for k, v in result.features.items()},
                shap_top_json=result.shap_top,
                summary_text=result.summary_text,
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


def _try_generate_summary(field: Field, result, crop: CropType) -> str | None:
    """Run the async LLM call from inside this sync worker. Returns
    None on any failure — including OpenAI not configured, network
    errors, or malformed responses. The prediction is always saved;
    the summary is purely additive."""
    try:
        # Oblast NDVI baseline serves as a "typical" comparison anchor —
        # the LLM mentions whether this field is above/below the
        # regional norm. Falls back to None if the field's centroid
        # doesn't land in any known oblast or the parquet isn't loaded.
        centroid_lat = centroid_lon = None
        try:
            from shapely import wkb

            pt = wkb.loads(bytes(field.centroid.data))
            centroid_lat, centroid_lon = pt.y, pt.x
        except Exception:  # noqa: BLE001
            pass

        oblast_baseline: float | None = None
        if centroid_lat is not None and centroid_lon is not None:
            oblast_name = find_oblast_for_centroid(centroid_lat, centroid_lon)
            if oblast_name:
                # NDVI baseline is a proxy — same value the dashboard's
                # risk-score uses to compare a field to its oblast.
                # Year argument: use latest available (None → most recent).
                oblast_baseline = oblast_avg_ndvi(oblast_name, None)

        return asyncio.run(generate_explanation_via_llm(
            crop=crop.value,
            value_tha=result.value_tha,
            oblast_baseline=oblast_baseline,
            q_low=result.value_tha_q05,
            q_high=result.value_tha_q95,
            top_shap=result.shap_top,
        ))
    except Exception as exc:  # noqa: BLE001 — never let summary break the job
        log.info("LLM summary skipped for field=%s crop=%s: %s",
                 field.id, crop.value, exc)
        return None
