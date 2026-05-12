"""users.settings_json JSONB column

Revision ID: 0007_user_settings
Revises: 0006_generated_reports
Create Date: 2026-05-12

Adds a free-form JSONB settings column to `users` so the dashboard can
persist user-defined crop prices (and any future preferences) without
adding a per-feature table. Keys we currently consume:

  settings_json["crop_prices"] = {
      "wheat":     {"price_per_ton": 8500, "currency": "UAH"},
      "corn":      {...},
      "sunflower": {...},
  }
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_user_settings"
down_revision: str | Sequence[str] | None = "0006_generated_reports"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "settings_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "settings_json")
