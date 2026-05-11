"""Alerts endpoints + SSE stream for live bell-icon updates."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, update

from app.db.models import Alert, Field
from app.deps import CurrentUser, DbSession
from app.redis_clients import get_async_redis
from app.schemas.prediction import AlertRead, AlertsCountResponse

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertRead])
async def list_alerts(
    current_user: CurrentUser,
    db: DbSession,
    acknowledged: bool = Query(default=False),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[AlertRead]:
    # LEFT JOIN with fields so the row also carries the field name, used
    # by the bell dropdown and the dashboard "Recent alerts" widget.
    result = await db.execute(
        select(Alert, Field.name)
        .join(Field, Field.id == Alert.field_id, isouter=True)
        .where(Alert.user_id == current_user.id)
        .where(Alert.acknowledged.is_(acknowledged))
        .order_by(Alert.created_at.desc())
        .limit(limit)
    )
    out: list[AlertRead] = []
    for alert, field_name in result.all():
        data = AlertRead.model_validate(alert).model_dump()
        data["field_name"] = field_name
        out.append(AlertRead(**data))
    return out


@router.get("/count", response_model=AlertsCountResponse)
async def alerts_count(
    current_user: CurrentUser, db: DbSession,
) -> AlertsCountResponse:
    n = await db.scalar(
        select(func.count()).select_from(Alert)
        .where(Alert.user_id == current_user.id)
        .where(Alert.acknowledged.is_(False))
    )
    return AlertsCountResponse(unread=int(n or 0))


@router.post("/{alert_id}/ack", status_code=status.HTTP_204_NO_CONTENT)
async def ack_alert(
    alert_id: int, current_user: CurrentUser, db: DbSession,
) -> None:
    alert = await db.get(Alert, alert_id)
    if alert is None or alert.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Сповіщення не знайдено.")
    await db.execute(
        update(Alert).where(Alert.id == alert_id).values(acknowledged=True)
    )


@router.get("/stream")
async def stream(current_user: CurrentUser) -> StreamingResponse:
    """SSE stream: pushed events when new alerts are created for the user.

    Front-end uses this to invalidate the bell counter without polling.
    """
    return StreamingResponse(
        _stream_user_alerts(current_user.id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_user_alerts(user_id: int):
    redis = get_async_redis()
    pubsub = redis.pubsub()
    channel = f"user:{user_id}:alerts"
    await pubsub.subscribe(channel)
    try:
        yield "event: status\ndata: {\"state\": \"connected\"}\n\n"
        import asyncio
        while True:
            try:
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=20,
                )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if msg is None:
                continue
            try:
                data = msg["data"].decode("utf-8") if isinstance(msg["data"], bytes) else msg["data"]
                payload = json.loads(data)
            except Exception:  # noqa: BLE001
                continue
            yield f"event: alert\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
