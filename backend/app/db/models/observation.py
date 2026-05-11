"""SatelliteObservation: one row per (field, scene-date) with index aggregates.

We store only summary statistics from Statistical API (mean/min/max/std per
index) — that's cheap in PU and trivially queryable for the time-series chart.
Per-pixel raster lives as a flat PNG file under RASTERS_DIR (populated by the
Process API path, on demand for heatmap).
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import (
    BigInteger,
    Date,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class SatelliteObservation(Base, TimestampMixin):
    __tablename__ = "satellite_observations"
    __table_args__ = (
        UniqueConstraint(
            "field_id", "observed_on", "source", name="uq_observation_field_date_source"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    field_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("fields.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    observed_on: Mapped[date] = mapped_column(Date, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="sentinel-2-l2a", nullable=False)

    # NULL when the bucket was too cloudy (we keep the row to record absence).
    ndvi_mean: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    ndvi_min: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    ndvi_max: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    ndvi_std: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    evi_mean: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    ndwi_mean: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)
    savi_mean: Mapped[float | None] = mapped_column(Numeric(5, 3), nullable=True)

    cloud_cover: Mapped[float | None] = mapped_column(Numeric(4, 1), nullable=True)
    processing_units: Mapped[float] = mapped_column(Numeric(8, 3), default=0, nullable=False)

    # Relative path under RASTERS_DIR; NULL until Process API materialises.
    raster_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return f"<SatelliteObservation field={self.field_id} date={self.observed_on}>"


class PuUsageMonthly(Base, TimestampMixin):
    """Persisted snapshot of monthly PU usage.

    Redis (key `pu:YYYY-MM`) is the live source of truth for accounting; this
    table is a periodic dump so we don't lose history across restarts.
    """

    __tablename__ = "pu_usage_monthly"

    year_month: Mapped[str] = mapped_column(String(7), primary_key=True)  # "2026-05"
    units_used: Mapped[float] = mapped_column(Numeric(10, 3), default=0, nullable=False)
