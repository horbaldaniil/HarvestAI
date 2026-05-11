"""Processing-Unit (PU) accounting against the Sentinel Hub free-tier quota.

Why Redis as the source of truth (not Postgres):
- Atomic INCRBYFLOAT means no transaction races between concurrent RQ workers.
- Reads are fast enough to gate every Sentinel call with negligible overhead.
- Postgres `pu_usage_monthly` is just a periodic snapshot (cron flush) for
  historical/durability purposes — losing it doesn't break enforcement.

Key shape: `pu:YYYY-MM` (one counter per calendar month, UTC).
TTL: 35 days so old months auto-expire without explicit cleanup.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from fastapi import HTTPException, status
from redis.asyncio import Redis

from app.config import settings


def _current_period_key(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"pu:{now.strftime('%Y-%m')}"


class PuTracker:
    """Thin wrapper over a Redis connection for atomic PU accounting."""

    def __init__(self, redis: Redis, monthly_limit: int | None = None):
        self._redis = redis
        self._limit = monthly_limit if monthly_limit is not None else settings.sh_monthly_pu_limit

    async def current_usage(self) -> Decimal:
        raw = await self._redis.get(_current_period_key())
        return Decimal(raw.decode() if isinstance(raw, bytes) else raw or "0")

    async def remaining(self) -> Decimal:
        return Decimal(self._limit) - await self.current_usage()

    @property
    def limit(self) -> Decimal:
        return Decimal(self._limit)

    async def assert_can_spend(self, estimate: Decimal | float | int) -> None:
        """Pre-flight check: refuse the request if `estimate` would breach the cap.

        We refuse *before* the SH call, so the user gets a clean 429 instead
        of silently going over budget.
        """
        est = Decimal(str(estimate))
        used = await self.current_usage()
        if used + est > self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Місячний ліміт супутникових запитів вичерпано "
                    f"({used:.2f} / {self.limit} PU). "
                    f"Скидається 1 числа наступного місяця."
                ),
            )

    async def record(self, amount: Decimal | float | int) -> Decimal:
        """Increment the live counter and return the new total."""
        key = _current_period_key()
        # 35-day TTL covers any month length; refreshed each increment.
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrbyfloat(key, float(amount))
            pipe.expire(key, 35 * 24 * 60 * 60)
            results = await pipe.execute()
        new_total = results[0]
        return Decimal(str(new_total))


_singleton: PuTracker | None = None


def get_tracker(redis: Redis) -> PuTracker:
    """Convenience accessor — keeps a single tracker instance per process."""
    global _singleton
    if _singleton is None or _singleton._redis is not redis:
        _singleton = PuTracker(redis)
    return _singleton
