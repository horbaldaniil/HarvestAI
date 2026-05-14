"""Add empirical-Bayes shrinkage diagnostics to predictions.

Revision ID: 0011_prediction_shrinkage
Revises: 0010_prediction_extras
Create Date: 2026-05-14

Adds two nullable columns so each yield prediction can carry the
Phase-C empirical-Bayes shrinkage trail:

  - shrinkage_weight  — share of `value_tha` that came from the
                        regional Держстат baseline (0 = pure ML,
                        1 = pure baseline). Drives optional UI badge
                        ("X % regional baseline").
  - raw_ml_value_tha  — the model's pre-shrinkage point estimate.
                        Lets the UI surface "model said X, shrunk
                        to Y" and lets us recompute alternate-weight
                        shrinkage post-hoc without re-running the
                        model.

Both nullable so pre-migration rows (predictions persisted before
this lift landed) continue to render — the UI hides each diagnostic
when its column is None.

`ALTER TABLE … ADD COLUMN nullable` is metadata-only on PostgreSQL
≥ 11, no row rewrite needed. Safe to apply live.

Downgrade drops both columns. No data preserved — these are
diagnostics, not source-of-truth values.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_prediction_shrinkage"
down_revision: str | Sequence[str] | None = "0010_prediction_extras"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("shrinkage_weight", sa.Numeric(5, 3), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("raw_ml_value_tha", sa.Numeric(6, 3), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("predictions", "raw_ml_value_tha")
    op.drop_column("predictions", "shrinkage_weight")
