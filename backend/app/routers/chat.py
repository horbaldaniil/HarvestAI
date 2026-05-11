"""Chat-bot endpoints (CRUD sessions + send message + SSE stream).

The flow on the frontend is:
  1. POST /api/chat/sessions[?field_id=N]      → returns session id
  2. GET  /api/chat/sessions/{id}/stream       → SSE subscription
  3. POST /api/chat/sessions/{id}/messages     → kicks off bot turn, 202
  4. Frontend reads streamed `delta` events from (2) until `done`.

The "send" and "stream" are split so a user can re-open a session (in a new
tab, for example) and the EventSource keeps working without re-sending the
prompt. Persistence in `chat_messages` is the source of truth — the stream
is purely a transport for tokens-as-they-arrive.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select

from app.db.models import ChatMessage, ChatSession, Field
from app.db.session import AsyncSessionLocal
from app.deps import CurrentUser, DbSession
from app.integrations.openai.client import OpenAIClient, OpenAIError
from app.redis_clients import get_async_redis
from app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionRead,
)
from app.services.chat_service import run_streaming_response
from app.services.rate_limiter import RateLimitExceeded, check_chat_rate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# How long the SSE handler is willing to hold the connection open while
# waiting for new pub/sub messages. Re-published as a `keepalive` so
# proxies don't drop the idle socket.
HEARTBEAT_INTERVAL_S = 15
# Hard ceiling on a single subscription. The frontend reconnects via
# EventSource's built-in retry if it gets cut off mid-answer.
MAX_STREAM_DURATION_S = 300


# ─── Helpers ──────────────────────────────────────────────────


async def _owned_session(db, user_id: int, session_id: int) -> ChatSession:
    sess = await db.get(ChatSession, session_id)
    if sess is None or sess.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Розмову не знайдено.")
    return sess


async def _default_title(db, field_id: int | None) -> str:
    if field_id is None:
        return "Загальна розмова"
    field = await db.get(Field, field_id)
    if field is None:
        return "Розмова"
    return f"{field.name}"


# ─── Session CRUD ─────────────────────────────────────────────


@router.post("/sessions", response_model=ChatSessionRead, status_code=201)
async def create_session(
    payload: ChatSessionCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> ChatSessionRead:
    # Field-ownership check (if specified). Foreign sessions auto-strip to None.
    if payload.field_id is not None:
        f = await db.get(Field, payload.field_id)
        if f is None or f.user_id != current_user.id:
            raise HTTPException(404, "Поле не знайдено.")
    title = payload.title or await _default_title(db, payload.field_id)
    sess = ChatSession(
        user_id=current_user.id,
        field_id=payload.field_id,
        title=title[:255],
    )
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    return ChatSessionRead.model_validate(sess)


@router.get("/sessions", response_model=list[ChatSessionRead])
async def list_sessions(
    current_user: CurrentUser, db: DbSession,
) -> list[ChatSessionRead]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(20)
    )
    return [ChatSessionRead.model_validate(s) for s in result.scalars().all()]


@router.get(
    "/sessions/{session_id}/messages", response_model=list[ChatMessageRead]
)
async def list_messages(
    session_id: int, current_user: CurrentUser, db: DbSession,
) -> list[ChatMessageRead]:
    await _owned_session(db, current_user.id, session_id)
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    )
    return [ChatMessageRead.model_validate(m) for m in result.scalars().all()]


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: int, current_user: CurrentUser, db: DbSession,
) -> None:
    await _owned_session(db, current_user.id, session_id)
    await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
    await db.commit()


# ─── Send message + background bot turn ───────────────────────


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def send_message(
    session_id: int,
    payload: ChatMessageCreate,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DbSession,
) -> ChatMessageRead:
    sess = await _owned_session(db, current_user.id, session_id)
    redis = get_async_redis()
    try:
        await check_chat_rate(current_user.id, redis)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Перевищено ліміт {exc.limit} запитів на годину.",
        ) from exc

    msg = ChatMessage(
        session_id=sess.id, role="user", content=payload.content,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    # Kick off the bot turn AFTER the request returns. We open our own DB
    # session inside the task because `db` (request-scoped) is gone by then.
    background_tasks.add_task(_run_turn_in_background, session_id, current_user.id)
    return ChatMessageRead.model_validate(msg)


async def _run_turn_in_background(session_id: int, user_id: int) -> None:
    """Open fresh DB session + OpenAI client per bot turn.

    Lives outside the request lifecycle so the HTTP 202 returns immediately.
    Any exception is logged and pushed to the chat SSE channel by
    `run_streaming_response` itself.
    """
    redis = get_async_redis()
    openai = OpenAIClient()
    try:
        async with AsyncSessionLocal() as db:
            from app.db.models import User
            user = await db.get(User, user_id)
            if user is None:
                log.warning("Background chat turn: user %s missing", user_id)
                return
            await run_streaming_response(
                session_id=session_id, db=db, user=user,
                redis=redis, openai=openai,
            )
    except OpenAIError as exc:
        log.error("OpenAI failure in background chat turn: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("Background chat turn crashed: %s", exc)
    finally:
        await openai.aclose()


# ─── SSE stream ───────────────────────────────────────────────


@router.get("/sessions/{session_id}/stream")
async def stream_session(
    session_id: int, current_user: CurrentUser, db: DbSession,
) -> StreamingResponse:
    """Subscribe to the chat:{session_id} Redis channel.

    Each event is one delta/tool_call/tool_result/done line in SSE format.
    The handler closes itself after `done`/`failed` or MAX_STREAM_DURATION_S.
    """
    await _owned_session(db, current_user.id, session_id)
    return StreamingResponse(
        _stream_chat_events(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_chat_events(session_id: int):
    redis = get_async_redis()
    pubsub = redis.pubsub()
    channel = f"chat:{session_id}"
    await pubsub.subscribe(channel)

    started = asyncio.get_event_loop().time()
    try:
        yield "event: status\ndata: {\"state\": \"connected\"}\n\n"

        while True:
            elapsed = asyncio.get_event_loop().time() - started
            if elapsed > MAX_STREAM_DURATION_S:
                yield "event: status\ndata: {\"state\": \"timeout\"}\n\n"
                return

            try:
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=HEARTBEAT_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if msg is None:
                continue

            raw = msg["data"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                payload = json.loads(raw)
            except ValueError:
                continue

            event = payload.get("type", "message")
            body = json.dumps(payload, ensure_ascii=False)
            yield f"event: {event}\ndata: {body}\n\n"

            if event in ("done", "failed"):
                return
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
