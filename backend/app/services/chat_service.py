"""Chat orchestration: history → OpenAI streaming → tool dispatch → publish to Redis.

Flow per user message:
  1. Load the last N messages from DB → build OpenAI-shaped `messages` list.
  2. Stream a completion from OpenAI. Forward content deltas to the SSE
     channel `chat:{session_id}` as they arrive.
  3. If the assistant finishes with `tool_calls`, run each tool, append the
     tool result as a `role="tool"` message, and loop. Bounded to
     `MAX_TOOL_ITERATIONS` to avoid infinite think-loops.
  4. Persist the final assistant message (with its `tool_calls` JSONB) and
     publish a terminal `done` event.

Errors at any step publish a `failed` event and the frontend rolls back the
optimistic placeholder.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession, Field, User
from app.integrations.openai.client import OpenAIClient, OpenAIError
from app.services.chat_tools import TOOL_SCHEMAS, dispatch_tool

log = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20
MAX_TOOL_ITERATIONS = 3

SYSTEM_PROMPT_UK = (
    "Ти — AI-помічник аграрної платформи HarvestAI. Ти спілкуєшся з фермером "
    "українською мовою. Твоє завдання — допомогти йому зрозуміти стан його "
    "полів на основі супутникових даних (NDVI/EVI/NDWI/SAVI), прогнозів "
    "врожайності, погоди та виявлених аномалій.\n\n"
    "Правила:\n"
    "• ЗАВЖДИ викликай інструменти (get_field_info, get_observations, "
    "get_prediction, get_alerts, get_weather, list_user_fields) щоб отримати "
    "реальні дані перед тим, як давати відповідь. Не вигадуй цифри.\n"
    "• Якщо контекст поля надано в системному повідомленні — починай з нього. "
    "Якщо ні — попроси користувача уточнити, або виклич list_user_fields.\n"
    "• Відповідай конкретно: посилайся на дати, значення індексів, конкретні "
    "альерти. Уникай загальних фраз.\n"
    "• Якщо інструмент повернув `error` або порожні дані — чесно скажи про це.\n"
    "• Якщо рекомендуєш дії (полив, добриво) — додай застереження, що це "
    "не агрономічна порада, а допоміжний інсайт на основі даних.\n"
    "• Використовуй короткі речення. Можна форматувати маркдауном "
    "(списки, **жирний**), але не зловживай."
)


# ─── Public entrypoint ────────────────────────────────────────


async def run_streaming_response(
    *,
    session_id: int,
    db: AsyncSession,
    user: User,
    redis: Redis,
    openai: OpenAIClient,
) -> None:
    """Generate one assistant turn for a session, publishing to Redis.

    Caller already persisted the user's message. This function does NOT
    commit DB transactions of its own — caller is responsible.
    """
    channel = f"chat:{session_id}"
    try:
        session = await _load_session(db, session_id, user.id)
        history = await _load_history(db, session_id)
        messages = await _build_messages(db, session, history)

        await _publish(redis, channel, {"type": "start"})

        assistant_text_parts: list[str] = []
        finish_reason = "stop"
        accumulated_tool_calls: list[dict[str, Any]] = []

        for _iteration in range(MAX_TOOL_ITERATIONS + 1):
            content_parts: list[str] = []
            tool_call_buffers: dict[int, dict[str, Any]] = {}
            finish_reason = "stop"
            usage_payload: dict | None = None

            async for chunk in openai.stream_completion(
                messages, tools=TOOL_SCHEMAS
            ):
                ctype = chunk.get("type")
                if ctype == "delta":
                    content_parts.append(chunk["content"])
                    await _publish(
                        redis, channel,
                        {"type": "delta", "content": chunk["content"]},
                    )
                elif ctype == "tool_call_delta":
                    idx = chunk.get("index", 0)
                    buf = tool_call_buffers.setdefault(
                        idx, {"id": None, "name": None, "args": ""},
                    )
                    if chunk.get("id"):
                        buf["id"] = chunk["id"]
                    if chunk.get("name"):
                        buf["name"] = chunk["name"]
                    if chunk.get("args_delta"):
                        buf["args"] += chunk["args_delta"]
                elif ctype == "done":
                    finish_reason = chunk.get("finish_reason", "stop")
                elif ctype == "usage":
                    usage_payload = chunk.get("usage")

            assistant_text_parts.extend(content_parts)

            if finish_reason != "tool_calls":
                # Plain text answer — break out and persist.
                if tool_call_buffers:
                    accumulated_tool_calls.extend(
                        _materialise(tool_call_buffers).values()
                    )
                break

            # Tool-call iteration: append the assistant turn (with calls)
            # to `messages`, then execute and append tool results.
            tool_calls = _materialise(tool_call_buffers)
            accumulated_tool_calls.extend(tool_calls.values())
            messages.append(_assistant_with_tools(content_parts, tool_calls))

            for idx in sorted(tool_calls):
                tc = tool_calls[idx]
                args = _safe_json_loads(tc["args"]) or {}
                await _publish(
                    redis, channel,
                    {"type": "tool_call", "name": tc["name"], "args": args},
                )
                result = await dispatch_tool(tc["name"], args, db=db, user=user)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"] or f"call_{idx}",
                        "name": tc["name"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                await _publish(
                    redis, channel,
                    {
                        "type": "tool_result",
                        "name": tc["name"],
                        "ok": "error" not in result,
                    },
                )
        else:
            # Loop guard tripped — surface to user instead of looping forever.
            warn = "\n\n_(Перевищено ліміт викликів інструментів. Спробуйте уточнити запит.)_"
            assistant_text_parts.append(warn)
            await _publish(redis, channel, {"type": "delta", "content": warn})

        final_text = "".join(assistant_text_parts).strip()
        await _persist_assistant(
            db, session, final_text, accumulated_tool_calls or None, usage_payload
        )
        await db.commit()
        await _publish(redis, channel, {"type": "done"})
    except OpenAIError as exc:
        log.error("OpenAI error in chat session %s: %s", session_id, exc)
        await _publish(
            redis, channel,
            {"type": "failed", "error": "openai_error", "detail": str(exc)[:200]},
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Chat service crashed for session %s", session_id)
        await _publish(
            redis, channel,
            {"type": "failed", "error": "internal_error", "detail": str(exc)[:200]},
        )


# ─── Helpers ──────────────────────────────────────────────────


async def _load_session(
    db: AsyncSession, session_id: int, user_id: int
) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        raise ValueError(f"chat session {session_id} not found for user {user_id}")
    return session


async def _load_history(db: AsyncSession, session_id: int) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(MAX_HISTORY_MESSAGES)
    )
    return list(reversed(result.scalars().all()))


async def _build_messages(
    db: AsyncSession, session: ChatSession, history: list[ChatMessage]
) -> list[dict[str, Any]]:
    """Convert DB messages → OpenAI's message-list format with a system prompt."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT_UK}]

    if session.field_id is not None:
        field = await db.get(Field, session.field_id)
        if field is not None:
            msgs.append(
                {
                    "role": "system",
                    "content": (
                        f"Поточний контекст: поле id={field.id}, "
                        f"назва={field.name!r}, культура={field.crop_type.value}, "
                        f"сезон={field.season_year}, площа={float(field.area_ha):.2f} га. "
                        f"Якщо користувач питає про 'моє поле', використовуй цей id."
                    ),
                }
            )

    for m in history:
        if m.role == "user":
            msgs.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            # IMPORTANT: strip `tool_calls` when rehydrating history. OpenAI
            # requires every assistant turn that carries tool_calls to be
            # followed by matching `role="tool"` rows (one per call_id).
            # We don't persist those intermediate tool rows — only the final
            # text answer — so re-sending the tool_calls would cause an
            # "OpenAI 400" on the very next user turn. The final assistant
            # text already encodes the tool-derived facts, so the bot keeps
            # full context without the call metadata.
            if m.content:
                msgs.append({"role": "assistant", "content": m.content})
        # role == "tool": skipped. We don't persist tool rows (see note above).
    return msgs


def _materialise(buffers: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Drop incomplete tool-call slots (no name) before dispatch."""
    return {
        idx: buf for idx, buf in buffers.items()
        if buf.get("name")
    }


def _assistant_with_tools(
    content_parts: list[str], tool_calls: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "".join(content_parts),
        "tool_calls": [
            {
                "id": tc["id"] or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["args"] or "{}",
                },
            }
            for idx, tc in sorted(tool_calls.items())
        ],
    }


def _safe_json_loads(s: str | None) -> dict[str, Any] | None:
    if not s:
        return {}
    try:
        return json.loads(s)
    except ValueError:
        return None


async def _persist_assistant(
    db: AsyncSession,
    session: ChatSession,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    usage: dict | None,
) -> None:
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=text,
            tool_calls=tool_calls,
            tokens_in=(usage or {}).get("prompt_tokens"),
            tokens_out=(usage or {}).get("completion_tokens"),
        )
    )
    # Bump session.updated_at so it sorts to the top of the sessions list.
    session.updated_at = datetime.now(tz=timezone.utc)


async def _publish(redis: Redis, channel: str, payload: dict[str, Any]) -> None:
    await redis.publish(channel, json.dumps(payload, ensure_ascii=False))


__all__ = ["SYSTEM_PROMPT_UK", "run_streaming_response"]
