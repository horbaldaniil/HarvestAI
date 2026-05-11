"""Chat ORM models — persisted conversations with the AI assistant.

We split the conversation across two tables (sessions + messages) so the
sidebar can list "last 20 conversations" with one cheap query, and message
history loads on demand only when the user re-opens a session.

A session is optionally bound to a field (`field_id`). When the user clicks
the floating chat icon from a field page, we open a field-scoped session so
the system prompt can inject "current field is X" without the user typing it.

The tool-call payload is stored as JSONB on the assistant turn that produced
it. We also persist a separate "tool" role row per tool call so message
history can be re-fed to OpenAI verbatim on a follow-up turn.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    # Field-scoped sessions get auto-titled with the field name; portfolio
    # sessions get "Загальна розмова" by default.
    field_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("fields.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)


class ChatMessage(Base):
    """One turn in a conversation. Mirrors the OpenAI chat-completion API shape.

    `role` is one of: "user", "assistant", "tool", "system". We don't usually
    persist "system" — it's injected on every call from chat_service — but the
    column allows it so the schema is forward-compatible.

    `tool_calls` (JSONB) is populated on assistant turns that triggered tools;
    `tool_call_id` is set on `role="tool"` rows linking back to a specific call.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")

    tool_calls: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
