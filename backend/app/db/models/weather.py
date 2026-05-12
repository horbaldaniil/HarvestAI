"""Weather observation ORM model — cached daily weather from Open-Meteo.

One row per (field, day, source). We cache 2 years of history + 14 days
of forecast per field, refreshed on field create/refresh.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import BigInteger, Date, ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class WeatherObservation(Base, TimestampMixin):
    __tablename__ = "weather_observations"
    __table_args__ = (
        UniqueConstraint(
            "field_id", "observed_on", "source", name="uq_weather_field_date_source"
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
    source: Mapped[str] = mapped_column(String(32), default="open-meteo", nullable=False)

    temp_min_c: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    temp_max_c: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    temp_mean_c: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    precip_mm: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    radiation_mj: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    # Migration 0008 — extra fields for the dedicated /weather page.
    wind_speed_max_ms: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    cloud_cover_pct: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    soil_moisture_0_10cm: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    # is_forecast=true rows live in the 14-day future window
    is_forecast: Mapped[bool] = mapped_column(default=False, nullable=False)
