"""High-level Statistical API wrapper.

Statistical API is the right tool for time-series ingestion: it accepts a
polygon + date range + aggregation period and returns aggregate statistics
(mean / stDev / min / max / percentiles) without ever shipping the raster
back. That's ~10x cheaper PU than Process API for our use case.

Response shape (relevant slice):
{
  "data": [
    {
      "interval": { "from": "2024-07-01T00:00:00Z", "to": "2024-07-08T00:00:00Z" },
      "outputs": {
        "ndvi": {"bands": {"B0": {"stats": {"mean": 0.73, "stDev": 0.05, ...}}}},
        "evi":  {"bands": {"B0": {"stats": {...}}}},
        ...
      }
    },
    ...
  ]
}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from app.integrations.sentinel_hub.client import SentinelHubClient
from app.integrations.sentinel_hub.evalscripts import INDICES_EVALSCRIPT

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexAggregates:
    """Per-week aggregate stats for one field. Any field may be None when cloudy."""

    observed_on: date
    ndvi_mean: float | None
    ndvi_min: float | None
    ndvi_max: float | None
    ndvi_std: float | None
    evi_mean: float | None
    ndwi_mean: float | None
    savi_mean: float | None
    cloud_cover: float | None


def _build_request(
    geometry: dict, start: date, end: date, max_cloud_cover: int
) -> dict:
    return {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "maxCloudCoverage": max_cloud_cover,
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": f"{start.isoformat()}T00:00:00Z",
                "to": f"{end.isoformat()}T23:59:59Z",
            },
            "aggregationInterval": {"of": "P7D"},  # weekly buckets
            # resx/resy units follow the CRS. We're in CRS84 (EPSG:4326),
            # so units are DEGREES — NOT meters. 0.0001° is roughly
            # 7-11m in Ukraine (depending on lat), matching Sentinel-2's
            # native 10m bands. Using "10" would sample at 10° resolution
            # and give a single pixel for the entire field.
            "resx": 0.0001,
            "resy": 0.0001,
            "evalscript": INDICES_EVALSCRIPT,
        },
        "calculations": {
            "ndvi": {
                "statistics": {
                    "default": {
                        "percentiles": {"k": [25, 50, 75]}
                    }
                }
            }
        },
    }


def _extract_stat(outputs: dict, output_id: str, key: str) -> float | None:
    try:
        stats = outputs[output_id]["bands"]["B0"]["stats"]
        v = stats.get(key)
        if v is None:
            return None
        # Sentinel Hub uses "NaN" string sentinel for masked aggregates.
        if isinstance(v, str):
            return None
        return float(v)
    except (KeyError, TypeError):
        return None


def parse_statistical_response(payload: dict) -> list[IndexAggregates]:
    rows: list[IndexAggregates] = []
    for entry in payload.get("data", []):
        interval = entry.get("interval", {})
        iso_from = interval.get("from")
        if not iso_from:
            continue
        observed_on = date.fromisoformat(iso_from[:10])
        outputs = entry.get("outputs", {})

        # If "ndvi" is missing entirely → no valid pixels in the bucket.
        if "ndvi" not in outputs:
            rows.append(
                IndexAggregates(
                    observed_on=observed_on,
                    ndvi_mean=None, ndvi_min=None, ndvi_max=None, ndvi_std=None,
                    evi_mean=None, ndwi_mean=None, savi_mean=None,
                    cloud_cover=None,
                )
            )
            continue

        rows.append(
            IndexAggregates(
                observed_on=observed_on,
                ndvi_mean=_extract_stat(outputs, "ndvi", "mean"),
                ndvi_min=_extract_stat(outputs, "ndvi", "min"),
                ndvi_max=_extract_stat(outputs, "ndvi", "max"),
                ndvi_std=_extract_stat(outputs, "ndvi", "stDev"),
                evi_mean=_extract_stat(outputs, "evi", "mean"),
                ndwi_mean=_extract_stat(outputs, "ndwi", "mean"),
                savi_mean=_extract_stat(outputs, "savi", "mean"),
                cloud_cover=None,  # Statistical API doesn't report it per interval directly
            )
        )
    return rows


async def fetch_indices_timeseries(
    client: SentinelHubClient,
    geometry: dict,
    start: date,
    end: date,
    max_cloud_cover: int = 30,
) -> tuple[list[IndexAggregates], float]:
    """Fetch weekly NDVI/EVI/NDWI/SAVI aggregates for a polygon over a date range.

    Returns (rows, estimated_pu_used). The PU cost is reported by Sentinel Hub
    in the `x-processingunits-spent` response header — we do not currently
    parse that here (httpx response headers are surfaced by the client
    layer when needed). For accounting we record a conservative estimate.
    """
    payload = _build_request(geometry, start, end, max_cloud_cover)
    response = await client.post_statistical(payload)
    rows = parse_statistical_response(response)
    # Rough estimate: ~0.02 PU per bucket × 4 outputs. Conservative upper bound
    # used by the tracker until SH exposes the real cost on this code path.
    estimated_pu = round(len(rows) * 0.02 * 4, 3)
    return rows, estimated_pu
