"""Per-user rate limiting for the chat assistant.

Approach: hourly fixed-window counter in Redis. One `INCR` + one `EXPIRE`
per request, no extra round-trips. Fixed windows can over-permit at the
boundary (the user could send 20 at 09:59 and 20 at 10:00), but for a
budget guard (not abuse mitigation) that's acceptable and dead-simple.

Key shape: `rl:chat:{user_id}:{YYYYMMDDHH}`. The hour is in UTC and is
encoded into the key so it expires naturally — no GC needed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from redis.asyncio import Redis

from app.config import settings


class RateLimitExceeded(Exception):
    """Raised by `check_chat_rate` when the user is over budget.

    Caller (the chat router) translates this to HTTP 429.
    """

    def __init__(self, limit: int, count: int):
        super().__init__(f"chat rate limit exceeded: {count}/{limit} per hour")
        self.limit = limit
        self.count = count


async def check_chat_rate(user_id: int, redis: Redis) -> None:
    """Increment the user's hourly counter; raise if over limit.

    Limit comes from `settings.chat_rate_limit_per_hour`. A value of 0 (or
    less) disables the limiter entirely.
    """
    limit = settings.chat_rate_limit_per_hour
    if limit <= 0:
        return

    now = datetime.now(tz=timezone.utc)
    bucket = now.strftime("%Y%m%d%H")
    key = f"rl:chat:{user_id}:{bucket}"

    # INCR returns the new count. EXPIRE is idempotent — setting it every
    # time is fine and ensures the key dies even if the first increment
    # was lost (e.g. Redis restart mid-window).
    count = await redis.incr(key)
    if count == 1:
        # First call in this window — set TTL to slightly more than an
        # hour so the key outlives the window before lazy expiry.
        await redis.expire(key, 3700)

    if count > limit:
        raise RateLimitExceeded(limit=limit, count=count)


__all__ = ["RateLimitExceeded", "check_chat_rate"]
