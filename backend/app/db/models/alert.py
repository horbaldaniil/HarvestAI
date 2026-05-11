"""Alert ORM model — output of the anomaly detector.

`acknowledged=False` rows are what the bell-icon counter shows. Ack-ing
an alert (POST /api/alerts/{id}/ack) hides it from the badge but keeps
the row for history.
"""
from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    field_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("fields.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    # Denormalised user_id so the /api/alerts list query doesn't need a join.
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    severity: Mapped[str] = mapped_column(String(16), nullable=False)  # info|warning|critical
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Human-readable Ukrainian message — what the user sees in the bell dropdown.
    message_uk: Mapped[str] = mapped_column(String(512), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Numeric(8, 3), nullable=True)
    threshold: Mapped[float | None] = mapped_column(Numeric(8, 3), nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
