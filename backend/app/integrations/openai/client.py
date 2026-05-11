"""OpenAI Chat Completion async client.

Two entry points:
- `stream_completion(messages, tools=None)` — yields parsed SSE chunks for
  the chat panel. Each chunk is `{type, ...}`:
    * `{"type": "delta", "content": "tok"}` — content token
    * `{"type": "tool_call_delta", "index": N, "id"?, "name"?, "args_delta"?}` —
      partial tool-call (OpenAI streams these incrementally per index)
    * `{"type": "done", "finish_reason": "stop"|"tool_calls", "usage"?}` —
      end of one assistant turn
- `completion(messages, tools=None)` — non-streaming, one parsed dict
  (raw response shape from OpenAI). Used by the PDF report builder.

Retry policy mirrors OpenMeteoClient: tenacity exponential 1→8s, 3 attempts,
retries on transient HTTP (429/5xx + transport errors). 4xx (besides 429) is
permanent and surfaces as `OpenAIError`.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

log = logging.getLogger(__name__)

CHAT_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIError(RuntimeError):
    pass


class OpenAIClient:
    """Async client for OpenAI chat completions with retry + streaming."""

    def __init__(
        self,
        http: httpx.AsyncClient | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        # Streaming requests can legitimately take 60+ seconds for long
        # answers, so the per-request timeout is generous. Connect timeout
        # is tight so we fail fast if the network is dead.
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._max_tokens = max_tokens or settings.openai_max_tokens

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise OpenAIError("OPENAI_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def completion(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the parsed JSON response."""
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, stream=False,
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TransportError, _TransientHTTP)),
            reraise=True,
        ):
            with attempt:
                resp = await self._http.post(
                    CHAT_URL, headers=self._headers, json=payload
                )
                await self._raise_for_status(resp)
                return resp.json()
        raise OpenAIError("Exhausted retries")

    async def stream_completion(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat completion. Yields parsed chunks (see module docstring)."""
        payload = self._build_payload(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, stream=True,
        )
        # OpenAI's streaming endpoint is hard to retry mid-stream (we'd
        # re-emit duplicate tokens), so the retry only covers connection
        # establishment. Once the stream is open, errors bubble immediately.
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TransportError, _TransientHTTP)),
            reraise=True,
        ):
            with attempt:
                async with self._http.stream(
                    "POST", CHAT_URL, headers=self._headers, json=payload,
                ) as resp:
                    await self._raise_for_status(resp)
                    async for chunk in _parse_sse_stream(resp):
                        yield chunk
                return
        raise OpenAIError("Exhausted retries")

    def _build_payload(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    async def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code in (429, 500, 502, 503, 504):
            raise _TransientHTTP(f"OpenAI {resp.status_code}")
        if resp.status_code >= 400:
            # On the streaming code path the body hasn't been pulled yet
            # (`resp.text` raises ResponseNotRead), so we read it explicitly
            # here. We cap at 500 chars to keep logs tidy — OpenAI errors
            # are concise JSON anyway (e.g. `{"error": {"message": "..."}}`).
            body = "<could not read body>"
            try:
                await resp.aread()
                body = resp.text[:500]
            except Exception:  # noqa: BLE001
                pass
            raise OpenAIError(f"OpenAI {resp.status_code}: {body}")


async def _parse_sse_stream(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Translate OpenAI's `data: {...}` SSE frames into our internal chunk shape.

    Buffers partial tool-call argument deltas: OpenAI streams them char-by-char
    per (choice, tool_call_index), so we forward each delta plus terminal id/name
    metadata as it arrives. The chat_service is responsible for accumulating.
    """
    async for raw_line in resp.aiter_lines():
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data == "[DONE]":
            # OpenAI sends a trailing [DONE] sentinel. Stream end is also
            # signalled by `finish_reason` on the last meaningful chunk;
            # we just exit here.
            return
        try:
            obj = json.loads(data)
        except ValueError:
            log.warning("Invalid JSON in OpenAI SSE: %s", data[:120])
            continue

        choices = obj.get("choices") or []
        if not choices:
            # Final usage frame from `stream_options.include_usage`.
            usage = obj.get("usage")
            if usage:
                yield {"type": "usage", "usage": usage}
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        if "content" in delta and delta["content"] is not None:
            yield {"type": "delta", "content": delta["content"]}

        for tc in delta.get("tool_calls") or []:
            yield {
                "type": "tool_call_delta",
                "index": tc.get("index", 0),
                "id": tc.get("id"),
                "name": (tc.get("function") or {}).get("name"),
                "args_delta": (tc.get("function") or {}).get("arguments"),
            }

        if finish is not None:
            yield {"type": "done", "finish_reason": finish}


class _TransientHTTP(Exception):
    """Raised for HTTP statuses we want tenacity to retry."""


__all__ = ["OpenAIClient", "OpenAIError"]
