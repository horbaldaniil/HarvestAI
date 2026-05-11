"""Pydantic schemas for the chat-bot API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    field_id: int | None
    title: str
    created_at: datetime
    updated_at: datetime


class ChatSessionCreate(BaseModel):
    field_id: int | None = None
    title: str | None = Field(default=None, max_length=255)


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    created_at: datetime


class ChatMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
