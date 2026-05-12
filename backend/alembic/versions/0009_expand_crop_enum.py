"""Expand crop_type_enum from 3 to 13 values

Revision ID: 0009_expand_crop_enum
Revises: 0008_weather_extras
Create Date: 2026-05-13

Adds ten new crop types to the PostgreSQL `crop_type_enum`:
soybean, rapeseed, barley, rye, oats, buckwheat, peas, sugar_beet,
potato, corn_silage. See `backend/app/db/models/enums.py:CropType`
for the canonical list and the rationale.

PostgreSQL note: `ALTER TYPE … ADD VALUE` is non-transactional in
older PG versions and the new values are not visible inside the same
transaction in some cases — Alembic 1.18 handles this by executing
each `ADD VALUE` outside the implicit migration transaction. We use
`IF NOT EXISTS` so re-running the migration on a partially-upgraded
DB doesn't error out.

Downgrade is intentionally a no-op. PostgreSQL has no `DROP VALUE`
clause for enums; the only safe rollback is to dump/restore the DB
without those values, which is out of scope here. Any row inserted
with one of the new values would block downgrade anyway — better to
fail loudly than to silently leave the DB in an inconsistent state.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009_expand_crop_enum"
down_revision: str | Sequence[str] | None = "0008_weather_extras"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_NEW_VALUES: tuple[str, ...] = (
    "soybean",
    "rapeseed",
    "barley",
    "rye",
    "oats",
    "buckwheat",
    "peas",
    "sugar_beet",
    "potato",
    "corn_silage",
)


def upgrade() -> None:
    # ADD VALUE must run outside a transaction in PostgreSQL < 12; modern
    # versions allow it inside, but COMMIT first to be safe across envs.
    # IF NOT EXISTS keeps re-runs idempotent.
    bind = op.get_bind()
    bind.exec_driver_sql("COMMIT")
    for value in _NEW_VALUES:
        bind.exec_driver_sql(
            f"ALTER TYPE crop_type_enum ADD VALUE IF NOT EXISTS '{value}'"
        )


def downgrade() -> None:
    # No safe downgrade — see module docstring.
    pass
