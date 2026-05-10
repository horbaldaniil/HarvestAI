"""Auth business logic.

Domain rules:
- Email is case-insensitive (stored in CITEXT column).
- Password ≥10 chars (validated at schema level), no other rules — we don't
  want to false-block users; entropy + brute-force rate-limiting is the
  defense, not arbitrary "must contain symbol" rules.
- Refresh tokens are stored as sha256 hashes; rotation happens on every refresh.
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RefreshToken, User
from app.schemas.auth import RegisterRequest
from app.utils.security import (
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_token_expiry,
    verify_password,
)


async def register_user(db: AsyncSession, data: RegisterRequest) -> User:
    existing = await db.scalar(select(User).where(User.email == data.email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Користувач із такою електронною адресою вже існує.",
        )
    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        full_name=data.full_name,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(password, user.password_hash):
        # Same error for both cases — prevents email enumeration.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невірний email або пароль.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Обліковий запис деактивовано.",
        )
    return user


async def issue_refresh_token(db: AsyncSession, user_id: int) -> str:
    plain, digest = generate_refresh_token()
    token = RefreshToken(
        user_id=user_id,
        token_hash=digest,
        expires_at=refresh_token_expiry(),
    )
    db.add(token)
    await db.flush()
    return plain


async def rotate_refresh_token(db: AsyncSession, plain_token: str) -> tuple[User, str]:
    """Validate an incoming refresh token, revoke it, and issue a new one."""
    digest = hash_refresh_token(plain_token)
    record = await db.scalar(select(RefreshToken).where(RefreshToken.token_hash == digest))
    if record is None or record.revoked or record.expires_at < datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалідний або застарілий refresh token.",
        )
    user = await db.get(User, record.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Користувач недоступний.",
        )
    record.revoked = True
    new_plain = await issue_refresh_token(db, user.id)
    return user, new_plain


async def revoke_refresh_token(db: AsyncSession, plain_token: str) -> None:
    digest = hash_refresh_token(plain_token)
    record = await db.scalar(select(RefreshToken).where(RefreshToken.token_hash == digest))
    if record is not None and not record.revoked:
        record.revoked = True
