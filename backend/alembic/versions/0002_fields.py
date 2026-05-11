"""fields table with PostGIS geometry and generated columns

Revision ID: 0002_fields
Revises: 0001_initial
Create Date: 2026-05-11

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography

revision: str = "0002_fields"
down_revision: str | Sequence[str] | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLAlchemy emits CREATE TYPE crop_type_enum AS ENUM (...) automatically
    # when the Enum column is first encountered, then references it in the
    # column DDL. We keep the column declared with the named Enum so PostgreSQL
    # enforces value integrity at the row level.
    op.create_table(
        "fields",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "crop_type",
            sa.Enum("wheat", "corn", "sunflower", name="crop_type_enum"),
            nullable=False,
        ),
        sa.Column("season_year", sa.Integer(), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=True),
        sa.Column(
            "geom",
            Geography(geometry_type="POLYGON", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column(
            "area_ha",
            sa.Numeric(10, 2),
            sa.Computed("ST_Area(geom) / 10000.0", persisted=True),
            nullable=False,
        ),
        sa.Column(
            "centroid",
            Geography(geometry_type="POINT", srid=4326, spatial_index=False),
            sa.Computed("ST_Centroid(geom::geometry)::geography", persisted=True),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fields")),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_fields_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "season_year BETWEEN 2000 AND 2100",
            name=op.f("ck_fields_season_year_range"),
        ),
    )
    op.create_index(op.f("ix_fields_user_id"), "fields", ["user_id"], unique=False)
    # GIST spatial index on geometry — essential for any geo-query later.
    op.execute("CREATE INDEX ix_fields_geom ON fields USING GIST (geom)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_fields_geom")
    op.drop_index(op.f("ix_fields_user_id"), table_name="fields")
    op.drop_table("fields")
    op.execute("DROP TYPE IF EXISTS crop_type_enum")
