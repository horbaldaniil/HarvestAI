"""Generated PDF report ORM model.

One row per persisted PDF. The actual bytes live under
`settings.reports_dir / <user_id> / <id>.pdf`; the DB row carries
metadata so the History UI can list/filter without touching disk.

Auto-pruning: we cap at 50 reports per user; the router deletes the
oldest row + file when the user crosses the limit (see reports router).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    # "field" — single-field report; "portfolio" — full user summary;
    # "compare" — multi-field comparison.
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # Params used at generation time so re-runs are reproducible. Includes
    # field_ids list, date range, included sections, etc.
    params_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Relative path under settings.reports_dir. Stored as relative so disk
    # moves don't break old rows.
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
