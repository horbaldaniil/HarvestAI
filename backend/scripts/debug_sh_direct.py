"""Direct Sentinel Hub probe with a known-good simple NDVI evalscript.

Hits the API by hand to isolate whether the issue is in our wrapper code
or in the request itself. If this returns NaN too, the polygon is bad
or there's an account-level coverage issue.
"""
from __future__ import annotations

import asyncio
import json

import redis.asyncio as async_redis

from app.config import settings
from app.integrations.sentinel_hub.client import SentinelHubClient


SIMPLE_NDVI = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  return {
    ndvi: [(s.B08 - s.B04) / (s.B08 + s.B04)],
    dataMask: [s.dataMask]
  };
}
"""


async def main() -> None:
    # Polygon from user's field (Krasne near Lviv).
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [24.427607, 49.876661],
            [24.428122, 49.874269],
            [24.419603, 49.875085],
            [24.420247, 49.876689],
            [24.427607, 49.876661],
        ]],
    }

    payload = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2024-07-01T00:00:00Z",
                        "to": "2024-08-01T00:00:00Z",
                    },
                    "maxCloudCoverage": 30,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "aggregation": {
            "timeRange": {
                "from": "2024-07-01T00:00:00Z",
                "to": "2024-08-01T00:00:00Z",
            },
            "aggregationInterval": {"of": "P10D"},
            "resx": 0.0001,
            "resy": 0.0001,
            "evalscript": SIMPLE_NDVI,
        },
    }

    r = async_redis.from_url(settings.redis_url, decode_responses=False)
    client = SentinelHubClient(r)
    try:
        resp = await client.post_statistical(payload)
        print(json.dumps(resp, indent=2, default=str)[:3000])
    finally:
        await client.aclose()
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
