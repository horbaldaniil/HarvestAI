"""Unit tests for chat_service helper functions (no DB / OpenAI involved)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from app.services.chat_service import (
    _assistant_with_tools,
    _build_messages,
    _materialise,
    _safe_json_loads,
)


def test_safe_json_loads_returns_empty_for_blank():
    assert _safe_json_loads("") == {}
    assert _safe_json_loads(None) == {}


def test_safe_json_loads_returns_dict():
    assert _safe_json_loads('{"x": 1}') == {"x": 1}


def test_safe_json_loads_returns_none_on_invalid():
    assert _safe_json_loads("{not json") is None


def test_materialise_drops_incomplete_calls():
    buffers = {
        0: {"id": "call_a", "name": "get_field_info", "args": '{"field_id":1}'},
        1: {"id": None, "name": None, "args": ""},  # incomplete, drop
        2: {"id": "call_c", "name": "get_alerts", "args": "{}"},
    }
    result = _materialise(buffers)
    assert set(result.keys()) == {0, 2}


def test_assistant_with_tools_shapes_tool_calls():
    parts = ["Let me ", "check."]
    tools = {
        0: {"id": "call_x", "name": "get_field_info", "args": '{"field_id": 5}'},
    }
    msg = _assistant_with_tools(parts, tools)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me check."
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_x"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_field_info"
    assert tc["function"]["arguments"] == '{"field_id": 5}'


def test_assistant_with_tools_synthesises_missing_call_id():
    tools = {3: {"id": None, "name": "get_alerts", "args": "{}"}}
    msg = _assistant_with_tools([], tools)
    assert msg["tool_calls"][0]["id"] == "call_3"


# ─── _build_messages regression test for OpenAI 400 bug ─────


@dataclass
class _FakeSession:
    """Stand-in for ChatSession that doesn't require DB / PostGIS."""
    id: int = 1
    user_id: int = 1
    field_id: int | None = None
    title: str = "test"


@dataclass
class _FakeMsg:
    """Stand-in for ChatMessage rows in `history`."""
    role: str
    content: str = ""
    tool_calls: Any = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    created_at: datetime = datetime(2026, 5, 11)


@pytest.mark.asyncio
async def test_build_messages_strips_tool_calls_from_history():
    """Regression: a stored assistant turn with tool_calls but no following
    tool rows would cause OpenAI to return 400 on the next user turn.
    We must rehydrate it as a plain text assistant message.
    """
    session = _FakeSession()
    history = [
        _FakeMsg(role="user", content="Чому NDVI впав?"),
        _FakeMsg(
            role="assistant",
            content="NDVI впав 12 травня через heat stress.",
            tool_calls=[
                {"id": "call_x", "name": "get_observations", "args": '{"field_id":1,"index":"ndvi"}'},
                {"id": "call_y", "name": "get_weather", "args": '{"field_id":1}'},
            ],
        ),
        _FakeMsg(role="user", content="А прогноз?"),
    ]
    msgs = await _build_messages(db=None, session=session, history=history)  # type: ignore[arg-type]
    # System prompt is first, then no other system rows (field_id is None).
    assert msgs[0]["role"] == "system"
    # The assistant turn must NOT carry tool_calls — it would orphan
    # without subsequent role="tool" rows and cause OpenAI 400.
    assistant_rows = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert "tool_calls" not in assistant_rows[0]
    assert assistant_rows[0]["content"] == "NDVI впав 12 травня через heat stress."
    # tool rows shouldn't be in the output either.
    assert all(m["role"] != "tool" for m in msgs)


@pytest.mark.asyncio
async def test_build_messages_skips_empty_assistant_rows():
    """If an assistant row has neither content nor (after stripping) tool_calls,
    don't emit it — OpenAI rejects fully empty assistant turns in history.
    """
    session = _FakeSession()
    history = [
        _FakeMsg(role="user", content="hi"),
        _FakeMsg(role="assistant", content="", tool_calls=[
            {"id": "x", "name": "n", "args": "{}"},
        ]),
        _FakeMsg(role="user", content="hi again"),
    ]
    msgs = await _build_messages(db=None, session=session, history=history)  # type: ignore[arg-type]
    assert [m["role"] for m in msgs] == ["system", "user", "user"]


@pytest.mark.asyncio
async def test_build_messages_preserves_plain_turns():
    session = _FakeSession()
    history = [
        _FakeMsg(role="user", content="Що це за NDVI?"),
        _FakeMsg(role="assistant", content="NDVI — індекс рослинності..."),
        _FakeMsg(role="user", content="А EVI?"),
    ]
    msgs = await _build_messages(db=None, session=session, history=history)  # type: ignore[arg-type]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "NDVI — індекс рослинності..."
