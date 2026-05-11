"""High-level orchestration layer over the Sentinel Hub primitives.

This is what routers and RQ jobs call — they don't need to know about
OAuth, PU accounting, or response shapes.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

from redis.asyncio import Redis

from app.integrations.sentinel_hub.client import SentinelHubClient
from app.integrations.sentinel_hub.process_api import fetch_heatmap_png
from app.integrations.sentinel_hub.pu_tracker import PuTracker
from app.integrations.sentinel_hub.statistical_api import (
    IndexAggregates,
    fetch_indices_timeseries,
)

log = logging.getLogger(__name__)


class SentinelService:
    def __init__(self, redis: Redis):
        self._redis = redis
        self._client = SentinelHubClient(redis)
        self._tracker = PuTracker(redis)

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def tracker(self) -> PuTracker:
        return self._tracker

    async def fetch_timeseries(
        self, geometry: dict, start: date, end: date, max_cloud_cover: int = 30
    ) -> tuple[list[IndexAggregates], Decimal]:
        """Quota-gated Statistical API call. Returns (rows, actual_pu_used)."""
        # Pre-flight: pessimistic estimate (covers 2 years × weekly × 4 indices).
        weeks = max(1, (end - start).days // 7)
        await self._tracker.assert_can_spend(weeks * 0.02 * 4)

        rows, estimated_pu = await fetch_indices_timeseries(
            self._client, geometry, start, end, max_cloud_cover
        )
        actual = await self._tracker.record(estimated_pu)
        log.info(
            "Sentinel timeseries: %d rows, %.3f PU spent (total this month: %.2f)",
            len(rows), estimated_pu, float(actual),
        )
        return rows, Decimal(str(estimated_pu))

    async def fetch_heatmap(
        self, geometry: dict, target_date: date, index: str, max_cloud_cover: int = 50
    ) -> tuple[bytes, Decimal]:
        """Quota-gated Process API call. Returns (png_bytes, actual_pu_used)."""
        await self._tracker.assert_can_spend(3.0)  # conservative upper bound
        png, estimated_pu = await fetch_heatmap_png(
            self._client, geometry, target_date, index, max_cloud_cover
        )
        actual = await self._tracker.record(estimated_pu)
        log.info(
            "Sentinel heatmap: %s %s, %.3f PU spent (total this month: %.2f)",
            target_date, index, estimated_pu, float(actual),
        )
        return png, Decimal(str(estimated_pu))
