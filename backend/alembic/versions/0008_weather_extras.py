"""weather_observations: wind_speed, cloud_cover, soil_moisture

Revision ID: 0008_weather_extras
Revises: 0007_user_settings
Create Date: 2026-05-12

Three new optional columns to power the dedicated /weather page:
- wind_speed_max_ms — agronomic: spraying advised against above 6 m/s
- cloud_cover_pct — drives the day-strip sky icons + S2 acquisition planning
- soil_moisture_0_10cm — Open-Meteo model estimate (cubic m / cubic m)
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_weather_extras"
down_revision: str | Sequence[str] | None = "0007_user_settings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "weather_observations",
        sa.Column("wind_speed_max_ms", sa.Numeric(5, 2), nullable=True),
    )
    op.add_column(
        "weather_observations",
        sa.Column("cloud_cover_pct", sa.Numeric(5, 2), nullable=True),
    )
    op.add_column(
        "weather_observations",
        sa.Column("soil_moisture_0_10cm", sa.Numeric(6, 4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("weather_observations", "soil_moisture_0_10cm")
    op.drop_column("weather_observations", "cloud_cover_pct")
    op.drop_column("weather_observations", "wind_speed_max_ms")
