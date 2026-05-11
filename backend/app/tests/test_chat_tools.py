"""Smoke-tests for the chat tool surface.

We can't easily exercise tool implementations end-to-end without PostgreSQL/
PostGIS (most tools touch fields/observations/predictions), so this module
covers the *contract* level instead:

- TOOL_SCHEMAS is well-shaped and OpenAI-compliant.
- dispatch_tool handles unknown tool names + arg-shape errors gracefully.

The PostGIS-backed paths are exercised in integration tests that bring up
a real Postgres+PostGIS instance (added later in CI).
"""
from __future__ import annotations

import pytest

from app.services.chat_tools import TOOL_SCHEMAS, ToolError, dispatch_tool


def test_tool_schemas_well_formed():
    expected_names = {
        "get_field_info",
        "get_observations",
        "get_prediction",
        "get_alerts",
        "get_weather",
        "list_user_fields",
    }
    actual = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert actual == expected_names

    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert isinstance(fn["name"], str)
        assert isinstance(fn["description"], str) and len(fn["description"]) > 20
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error():
    result = await dispatch_tool("nope", {}, db=None, user=None)  # type: ignore[arg-type]
    assert "error" in result
    assert "unknown_tool" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_wraps_tool_errors(monkeypatch):
    """A ToolError inside an impl should surface as `{"error": "..."}` to the bot."""
    from app.services import chat_tools

    async def _boom(**kwargs):  # noqa: ARG001
        raise ToolError("field_not_found: id=999")

    monkeypatch.setitem(chat_tools._DISPATCH, "get_field_info", _boom)  # noqa: SLF001
    result = await dispatch_tool(
        "get_field_info", {"field_id": 999}, db=None, user=None,  # type: ignore[arg-type]
    )
    assert result == {"error": "field_not_found: id=999"}


@pytest.mark.asyncio
async def test_dispatch_wraps_type_errors(monkeypatch):
    """Bad argument shape (TypeError) should not crash the assistant turn."""
    from app.services import chat_tools

    async def _expects_x(*, db, user, x):  # noqa: ARG001
        return {"x": x}

    monkeypatch.setitem(chat_tools._DISPATCH, "tester", _expects_x)  # noqa: SLF001
    result = await dispatch_tool("tester", {"y": 1}, db=None, user=None)  # type: ignore[arg-type]
    assert "error" in result
    assert "bad_arguments" in result["error"]
