"""generated_reports

Revision ID: 0006_generated_reports
Revises: 0005_chat
Create Date: 2026-05-12

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_generated_reports"
down_revision: str | Sequence[str] | None = "0005_chat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generated_reports",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("params_json", postgresql.JSONB(), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generated_reports")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name=op.f("fk_generated_reports_user_id_users"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        op.f("ix_generated_reports_user_id"),
        "generated_reports", ["user_id"], unique=False,
    )
    op.create_index(
        "ix_generated_reports_user_generated_desc",
        "generated_reports",
        ["user_id", sa.text("generated_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_generated_reports_user_generated_desc", table_name="generated_reports")
    op.drop_index(op.f("ix_generated_reports_user_id"), table_name="generated_reports")
    op.drop_table("generated_reports")
