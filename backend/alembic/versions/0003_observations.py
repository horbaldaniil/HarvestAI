"""satellite_observations + pu_usage_monthly tables

Revision ID: 0003_observations
Revises: 0002_fields
Create Date: 2026-05-11

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_observations"
down_revision: str | Sequence[str] | None = "0002_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "satellite_observations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("field_id", sa.BigInteger(), nullable=False),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column(
            "source",
            sa.String(length=32),
            server_default="sentinel-2-l2a",
            nullable=False,
        ),
        sa.Column("ndvi_mean", sa.Numeric(5, 3), nullable=True),
        sa.Column("ndvi_min", sa.Numeric(5, 3), nullable=True),
        sa.Column("ndvi_max", sa.Numeric(5, 3), nullable=True),
        sa.Column("ndvi_std", sa.Numeric(5, 3), nullable=True),
        sa.Column("evi_mean", sa.Numeric(5, 3), nullable=True),
        sa.Column("ndwi_mean", sa.Numeric(5, 3), nullable=True),
        sa.Column("savi_mean", sa.Numeric(5, 3), nullable=True),
        sa.Column("cloud_cover", sa.Numeric(4, 1), nullable=True),
        sa.Column(
            "processing_units",
            sa.Numeric(8, 3),
            server_default="0",
            nullable=False,
        ),
        sa.Column("raster_uri", sa.String(length=512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_satellite_observations")),
        sa.ForeignKeyConstraint(
            ["field_id"],
            ["fields.id"],
            name=op.f("fk_satellite_observations_field_id_fields"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "field_id",
            "observed_on",
            "source",
            name="uq_observation_field_date_source",
        ),
    )
    op.create_index(
        op.f("ix_satellite_observations_field_id"),
        "satellite_observations",
        ["field_id"],
        unique=False,
    )
    op.create_index(
        "ix_observation_field_observed_desc",
        "satellite_observations",
        ["field_id", sa.text("observed_on DESC")],
        unique=False,
    )

    op.create_table(
        "pu_usage_monthly",
        sa.Column("year_month", sa.String(length=7), nullable=False),
        sa.Column(
            "units_used",
            sa.Numeric(10, 3),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("year_month", name=op.f("pk_pu_usage_monthly")),
    )


def downgrade() -> None:
    op.drop_table("pu_usage_monthly")
    op.drop_index(
        "ix_observation_field_observed_desc", table_name="satellite_observations"
    )
    op.drop_index(
        op.f("ix_satellite_observations_field_id"),
        table_name="satellite_observations",
    )
    op.drop_table("satellite_observations")
