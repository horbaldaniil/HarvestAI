"""Tests for the Redis-backed PU tracker. Uses fakeredis — no real Redis required."""
from __future__ import annotations

from decimal import Decimal

import fakeredis.aioredis
import pytest
from fastapi import HTTPException

from app.integrations.sentinel_hub.pu_tracker import PuTracker


@pytest.fixture
async def tracker():
    redis = fakeredis.aioredis.FakeRedis()
    yield PuTracker(redis, monthly_limit=100)
    await redis.aclose()


@pytest.mark.asyncio
async def test_initial_usage_is_zero(tracker):
    assert await tracker.current_usage() == Decimal("0")
    assert await tracker.remaining() == Decimal("100")


@pytest.mark.asyncio
async def test_record_accumulates(tracker):
    await tracker.record(Decimal("2.5"))
    await tracker.record(Decimal("1.5"))
    used = await tracker.current_usage()
    assert used == Decimal("4")


@pytest.mark.asyncio
async def test_assert_can_spend_allows_under_limit(tracker):
    await tracker.record(50)
    # 30 more would total 80 ≤ 100 → fine.
    await tracker.assert_can_spend(30)


@pytest.mark.asyncio
async def test_assert_can_spend_refuses_when_would_exceed(tracker):
    await tracker.record(80)
    with pytest.raises(HTTPException) as ei:
        await tracker.assert_can_spend(30)  # 80 + 30 = 110 > 100
    assert ei.value.status_code == 429
    assert "ліміт" in ei.value.detail.lower()


@pytest.mark.asyncio
async def test_record_returns_new_total(tracker):
    t1 = await tracker.record(10)
    t2 = await tracker.record(5.5)
    assert t1 == Decimal("10")
    assert t2 == Decimal("15.5")
