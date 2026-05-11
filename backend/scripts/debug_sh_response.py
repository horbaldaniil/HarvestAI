"""Debug: dump the raw Sentinel Hub Statistical API response so we can
verify the actual structure (where are the stats keyed?).

Usage: uv run python scripts/debug_sh_response.py <field_id>
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, timedelta

import redis.asyncio as async_redis
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from geoalchemy2.functions import ST_AsGeoJSON

from app.config import settings
from app.db.models import Field
from app.integrations.sentinel_hub.client import SentinelHubClient
from app.integrations.sentinel_hub.statistical_api import _build_request


async def main(field_id: int) -> None:
    engine = create_engine(settings.database_url_sync)
    with Session(engine, expire_on_commit=False) as session:
        field = session.get(Field, field_id)
        if field is None:
            print(f"Field {field_id} not found")
            return
        geom_json = session.scalar(select(ST_AsGeoJSON(field.geom)))

    geometry = json.loads(geom_json)
    print("=== GEOMETRY ===")
    print(json.dumps(geometry, indent=2))

    # July 2024 — peak growing season; relax cloud filter to confirm coverage.
    start = date(2024, 7, 1)
    end = date(2024, 8, 1)

    r = async_redis.from_url(settings.redis_url, decode_responses=False)
    client = SentinelHubClient(r)
    try:
        payload = _build_request(geometry, start, end, max_cloud_cover=100)
        print("\n=== REQUEST (first 3000 chars) ===")
        print(json.dumps(payload, indent=2)[:3000])
        print("\n=== RESPONSE (first bucket) ===")
        response = await client.post_statistical(payload)
        if response.get("data"):
            print(json.dumps(response["data"][0], indent=2, default=str)[:3000])
            # Summary
            total = len(response["data"])
            with_data = sum(
                1 for d in response["data"]
                if d.get("outputs", {}).get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {}).get("mean") not in (None, "NaN")
            )
            print(f"\n=== SUMMARY: {total} buckets, {with_data} with NDVI values ===")
        else:
            print("Empty response.data")
    finally:
        await client.aclose()
        await r.aclose()


if __name__ == "__main__":
    field_id = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    asyncio.run(main(field_id))
