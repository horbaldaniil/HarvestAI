"""Cryptographic utilities: password hashing and JWT token handling.

Why argon2id over bcrypt: memory-hard, OWASP-recommended default for new applications.
Why HS256: shared-secret symmetric — backend is the only verifier, so asymmetric keys
add complexity without security benefit at our scale.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jose import JWTError, jwt

from app.config import settings

_hasher = PasswordHasher()


# ───── Passwords ─────────────────────────────────────────────────


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _hasher.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False


# ───── Access tokens (JWT) ───────────────────────────────────────


def create_access_token(subject: int | str, extra_claims: dict[str, Any] | None = None) -> str:
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.jwt_access_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT. Raises `JWTError` if invalid/expired."""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    if payload.get("type") != "access":
        raise JWTError("Wrong token type")
    return payload


# ───── Refresh tokens (opaque + DB-stored hash) ──────────────────


def generate_refresh_token() -> tuple[str, str]:
    """Generate a fresh refresh token. Returns (plain_token, sha256_hash)."""
    plain = secrets.token_urlsafe(48)  # ~64 chars, 384 bits of entropy
    digest = hashlib.sha256(plain.encode("utf-8")).hexdigest()
    return plain, digest


def hash_refresh_token(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def refresh_token_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(days=settings.jwt_refresh_expire_days)


__all__ = [
    "JWTError",
    "create_access_token",
    "decode_access_token",
    "generate_refresh_token",
    "hash_password",
    "hash_refresh_token",
    "refresh_token_expiry",
    "verify_password",
]
