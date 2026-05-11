"""predictions, alerts, weather_observations

Revision ID: 0004_pred_alerts_weather
Revises: 0003_observations
Create Date: 2026-05-11

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_pred_alerts_weather"
down_revision: str | Sequence[str] | None = "0003_observations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ───── predictions ────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("field_id", sa.BigInteger(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("value_tha", sa.Numeric(6, 3), nullable=False),
        sa.Column("confidence", sa.Numeric(6, 3), nullable=True),
        sa.Column("features_json", postgresql.JSONB(), nullable=False),
        sa.Column("shap_top_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "predicted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.ForeignKeyConstraint(
            ["field_id"], ["fields.id"],
            name=op.f("fk_predictions_field_id_fields"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        op.f("ix_predictions_field_id"), "predictions", ["field_id"], unique=False
    )
    op.create_index(
        "ix_predictions_field_predicted_desc",
        "predictions",
        ["field_id", sa.text("predicted_at DESC")],
        unique=False,
    )

    # ───── alerts ─────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("field_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("message_uk", sa.String(length=512), nullable=False),
        sa.Column("metric_value", sa.Numeric(8, 3), nullable=True),
        sa.Column("threshold", sa.Numeric(8, 3), nullable=True),
        sa.Column(
            "acknowledged",
            sa.Boolean(),
            server_default=sa.text("false"),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
        sa.ForeignKeyConstraint(
            ["field_id"], ["fields.id"],
            name=op.f("fk_alerts_field_id_fields"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name=op.f("fk_alerts_user_id_users"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(op.f("ix_alerts_field_id"), "alerts", ["field_id"], unique=False)
    op.create_index(op.f("ix_alerts_user_id"), "alerts", ["user_id"], unique=False)
    op.create_index(
        "ix_alerts_user_acked_created_desc",
        "alerts",
        ["user_id", "acknowledged", sa.text("created_at DESC")],
        unique=False,
    )

    # ───── weather_observations ───────────────────────────────
    op.create_table(
        "weather_observations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("field_id", sa.BigInteger(), nullable=False),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column(
            "source",
            sa.String(length=32),
            server_default="open-meteo",
            nullable=False,
        ),
        sa.Column("temp_min_c", sa.Numeric(5, 2), nullable=True),
        sa.Column("temp_max_c", sa.Numeric(5, 2), nullable=True),
        sa.Column("temp_mean_c", sa.Numeric(5, 2), nullable=True),
        sa.Column("precip_mm", sa.Numeric(6, 2), nullable=True),
        sa.Column("humidity_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("radiation_mj", sa.Numeric(6, 2), nullable=True),
        sa.Column(
            "is_forecast",
            sa.Boolean(),
            server_default=sa.text("false"),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_weather_observations")),
        sa.ForeignKeyConstraint(
            ["field_id"], ["fields.id"],
            name=op.f("fk_weather_observations_field_id_fields"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "field_id", "observed_on", "source",
            name="uq_weather_field_date_source",
        ),
    )
    op.create_index(
        op.f("ix_weather_observations_field_id"),
        "weather_observations", ["field_id"], unique=False,
    )
    op.create_index(
        "ix_weather_field_observed_desc",
        "weather_observations",
        ["field_id", sa.text("observed_on DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_weather_field_observed_desc", table_name="weather_observations")
    op.drop_index(op.f("ix_weather_observations_field_id"), table_name="weather_observations")
    op.drop_table("weather_observations")

    op.drop_index("ix_alerts_user_acked_created_desc", table_name="alerts")
    op.drop_index(op.f("ix_alerts_user_id"), table_name="alerts")
    op.drop_index(op.f("ix_alerts_field_id"), table_name="alerts")
    op.drop_table("alerts")

    op.drop_index("ix_predictions_field_predicted_desc", table_name="predictions")
    op.drop_index(op.f("ix_predictions_field_id"), table_name="predictions")
    op.drop_table("predictions")
