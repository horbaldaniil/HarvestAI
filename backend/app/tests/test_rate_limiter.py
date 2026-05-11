"""Rate-limiter unit tests using fakeredis."""
from __future__ import annotations

import pytest
import fakeredis.aioredis

from app.config import settings
from app.services.rate_limiter import RateLimitExceeded, check_chat_rate


@pytest.mark.asyncio
async def test_under_limit_passes(monkeypatch):
    monkeypatch.setattr(settings, "chat_rate_limit_per_hour", 5)
    redis = fakeredis.aioredis.FakeRedis()
    for _ in range(5):
        await check_chat_rate(user_id=42, redis=redis)
    await redis.aclose()


@pytest.mark.asyncio
async def test_over_limit_raises(monkeypatch):
    monkeypatch.setattr(settings, "chat_rate_limit_per_hour", 3)
    redis = fakeredis.aioredis.FakeRedis()
    for _ in range(3):
        await check_chat_rate(user_id=7, redis=redis)
    with pytest.raises(RateLimitExceeded) as exc_info:
        await check_chat_rate(user_id=7, redis=redis)
    assert exc_info.value.limit == 3
    assert exc_info.value.count == 4
    await redis.aclose()


@pytest.mark.asyncio
async def test_zero_limit_disables(monkeypatch):
    """Setting limit to 0 should be a no-op even after many calls."""
    monkeypatch.setattr(settings, "chat_rate_limit_per_hour", 0)
    redis = fakeredis.aioredis.FakeRedis()
    for _ in range(100):
        await check_chat_rate(user_id=1, redis=redis)
    await redis.aclose()


@pytest.mark.asyncio
async def test_separate_users_independent(monkeypatch):
    monkeypatch.setattr(settings, "chat_rate_limit_per_hour", 2)
    redis = fakeredis.aioredis.FakeRedis()
    await check_chat_rate(user_id=1, redis=redis)
    await check_chat_rate(user_id=1, redis=redis)
    # user 2 has its own bucket.
    await check_chat_rate(user_id=2, redis=redis)
    await check_chat_rate(user_id=2, redis=redis)
    with pytest.raises(RateLimitExceeded):
        await check_chat_rate(user_id=1, redis=redis)
    with pytest.raises(RateLimitExceeded):
        await check_chat_rate(user_id=2, redis=redis)
    await redis.aclose()
