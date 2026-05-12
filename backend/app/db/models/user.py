"""User ORM model."""
from __future__ import annotations

from sqlalchemy import BigInteger, String, text
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    locale: Mapped[str] = mapped_column(String(8), default="uk", nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # Free-form preferences (Alembic 0007). Currently stores `crop_prices`,
    # consumed by the dashboard's income-projection card and the PDF
    # portfolio report. Keys are documented in the settings router.
    settings_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        # `'{}'` (without an explicit `::jsonb` cast) compiles cleanly on
        # SQLite too — PostgreSQL still casts to JSONB on insert because the
        # column type drives the literal coercion.
        server_default=text("'{}'"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"
