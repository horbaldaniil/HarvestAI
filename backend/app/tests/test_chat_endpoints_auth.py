"""Auth gates for chat-bot endpoints.

DB-heavy logic (session/message ownership, OpenAI call orchestration) is
covered separately by unit tests against mocked OpenAI + Redis; these
tests just verify the routes require authentication.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_session_requires_auth(client: AsyncClient):
    r = await client.post("/api/chat/sessions", json={})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_list_sessions_requires_auth(client: AsyncClient):
    r = await client.get("/api/chat/sessions")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_list_messages_requires_auth(client: AsyncClient):
    r = await client.get("/api/chat/sessions/1/messages")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_send_message_requires_auth(client: AsyncClient):
    r = await client.post(
        "/api/chat/sessions/1/messages", json={"content": "hi"}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_delete_session_requires_auth(client: AsyncClient):
    r = await client.delete("/api/chat/sessions/1")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_stream_session_requires_auth(client: AsyncClient):
    r = await client.get("/api/chat/sessions/1/stream")
    assert r.status_code == 401
