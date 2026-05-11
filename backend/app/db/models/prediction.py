"""Yield prediction ORM model.

One row per (field, model_version, predicted_at). Multiple predictions can
exist for a field over time as the season progresses and observations
accumulate. The latest one wins in the UI, but history is preserved so a
user can see how the forecast evolved.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    field_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("fields.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # Point estimate (t/ha) + ±confidence (one standard deviation).
    value_tha: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    # JSON for inspection + reproducibility (good for the thesis report).
    features_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Top-N SHAP contributions [{name, value, contribution}, ...]
    shap_top_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
