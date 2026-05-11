"""Yield prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.db.models import Field, Prediction
from app.deps import CurrentUser, DbSession
from app.schemas.observation import JobHandleResponse
from app.schemas.prediction import PredictionRead
from app.workers.dispatcher import enqueue_predict_yield

router = APIRouter(prefix="/api/fields", tags=["predictions"])


async def _owned(db, user_id: int, field_id: int) -> Field:
    f = await db.get(Field, field_id)
    if f is None or f.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Поле не знайдено.")
    return f


@router.get(
    "/{field_id}/predictions/latest",
    response_model=PredictionRead | None,
)
async def latest_prediction(
    field_id: int, current_user: CurrentUser, db: DbSession,
) -> PredictionRead | None:
    await _owned(db, current_user.id, field_id)
    result = await db.execute(
        select(Prediction)
        .where(Prediction.field_id == field_id)
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
    )
    pred = result.scalar_one_or_none()
    return PredictionRead.model_validate(pred) if pred else None


@router.post(
    "/{field_id}/predictions/recompute",
    response_model=JobHandleResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def recompute_prediction(
    field_id: int, current_user: CurrentUser, db: DbSession,
) -> JobHandleResponse:
    await _owned(db, current_user.id, field_id)
    handle = enqueue_predict_yield(field_id)
    return JobHandleResponse(
        job_id=handle.job_id,
        queue=handle.queue,
        status_url=f"/api/jobs/{handle.job_id}/stream",
    )
