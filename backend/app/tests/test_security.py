"""Unit tests for password hashing and JWT helpers."""
from __future__ import annotations

import time

import pytest
from jose import JWTError

from app.utils.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


def test_hash_password_returns_different_hash_each_call():
    h1 = hash_password("correct horse battery staple")
    h2 = hash_password("correct horse battery staple")
    assert h1 != h2  # argon2 includes random salt
    assert verify_password("correct horse battery staple", h1)
    assert verify_password("correct horse battery staple", h2)


def test_verify_password_rejects_wrong():
    h = hash_password("super-secret-passphrase")
    assert not verify_password("super-secret-passphras", h)
    assert not verify_password("", h)


def test_create_and_decode_access_token_roundtrip():
    token = create_access_token(subject=42, extra_claims={"role": "farmer"})
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "farmer"
    assert payload["type"] == "access"
    assert payload["exp"] > payload["iat"]


def test_decode_rejects_tampered_token():
    token = create_access_token(subject=1)
    tampered = token[:-4] + "AAAA"
    with pytest.raises(JWTError):
        decode_access_token(tampered)


def test_generate_refresh_token_returns_plain_and_matching_hash():
    plain, digest = generate_refresh_token()
    assert plain and digest
    assert len(plain) >= 40
    assert hash_refresh_token(plain) == digest
    # Two generations must differ.
    other_plain, _ = generate_refresh_token()
    assert plain != other_plain
