"""OpenAI client unit tests — mocked httpx, exercise SSE parsing."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.integrations.openai.client import OpenAIClient, OpenAIError, _parse_sse_stream


class _MockResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"


@pytest.mark.asyncio
async def test_sse_parses_content_delta():
    lines = [
        _sse_data(
            {
                "choices": [
                    {"delta": {"content": "Привіт"}, "finish_reason": None}
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {"delta": {"content": ", фермере"}, "finish_reason": None}
                ]
            }
        ),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    chunks = [c async for c in _parse_sse_stream(_MockResponse(lines))]
    assert chunks == [
        {"type": "delta", "content": "Привіт"},
        {"type": "delta", "content": ", фермере"},
        {"type": "done", "finish_reason": "stop"},
    ]


@pytest.mark.asyncio
async def test_sse_parses_tool_calls():
    lines = [
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {
                                        "name": "get_field_info",
                                        "arguments": '{"fie',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'ld_id":1}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse_data(
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        ),
    ]
    chunks = [c async for c in _parse_sse_stream(_MockResponse(lines))]
    # We expect two tool_call_deltas (id+name+args_delta on first, args_delta on second).
    tool_deltas = [c for c in chunks if c["type"] == "tool_call_delta"]
    assert len(tool_deltas) == 2
    assert tool_deltas[0]["id"] == "call_abc"
    assert tool_deltas[0]["name"] == "get_field_info"
    assert tool_deltas[0]["args_delta"] == '{"fie'
    assert tool_deltas[1]["args_delta"] == 'ld_id":1}'
    assert chunks[-1] == {"type": "done", "finish_reason": "tool_calls"}


@pytest.mark.asyncio
async def test_sse_parses_usage_frame():
    lines = [
        _sse_data(
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        ),
    ]
    chunks = [c async for c in _parse_sse_stream(_MockResponse(lines))]
    assert chunks == [
        {"type": "usage", "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    ]


@pytest.mark.asyncio
async def test_sse_handles_invalid_json_lines():
    lines = [
        "data: not-json",
        _sse_data({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
    ]
    chunks = [c async for c in _parse_sse_stream(_MockResponse(lines))]
    assert chunks == [
        {"type": "delta", "content": "ok"},
        {"type": "done", "finish_reason": "stop"},
    ]


@pytest.mark.asyncio
async def test_client_raises_without_api_key(monkeypatch):
    from app.config import settings as app_settings
    monkeypatch.setattr(app_settings, "openai_api_key", "")
    client = OpenAIClient(api_key=None)
    # Force the lazy property to evaluate — `_headers` raises if no key.
    with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
        _ = client._headers  # noqa: SLF001 (intentional unit test)
