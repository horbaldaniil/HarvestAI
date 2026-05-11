"""Sentinel Hub PU quota visibility endpoint.

Source of truth: the Redis counter maintained by `PuTracker`. We surface it
to the UI so the user can see how much budget remains before refreshing.
"""
from __future__ import annotations

import calendar
from datetime import UTC, date, datetime

from fastapi import APIRouter

from app.config import settings
from app.deps import CurrentUser
from app.integrations.sentinel_hub.pu_tracker import PuTracker
from app.redis_clients import get_async_redis
from app.schemas.observation import QuotaResponse

router = APIRouter(prefix="/api/quota", tags=["quota"])


@router.get("", response_model=QuotaResponse)
async def get_quota(current_user: CurrentUser) -> QuotaResponse:
    tracker = PuTracker(get_async_redis())
    used = float(await tracker.current_usage())
    limit = float(tracker.limit)
    remaining = max(0.0, limit - used)
    percent = (used / limit * 100.0) if limit > 0 else 0.0

    now = datetime.now(UTC)
    period_start = date(now.year, now.month, 1)
    last_day = calendar.monthrange(now.year, now.month)[1]
    period_end = date(now.year, now.month, last_day)
    refresh_at = datetime(
        now.year + (1 if now.month == 12 else 0),
        1 if now.month == 12 else now.month + 1,
        1,
        tzinfo=UTC,
    )

    return QuotaResponse(
        units_used=round(used, 3),
        units_limit=limit,
        units_remaining=round(remaining, 3),
        percent_used=round(percent, 2),
        period_start=period_start,
        period_end=period_end,
        refresh_at=refresh_at,
    )
