"""Process API wrapper — used only when the user requests a heatmap overlay.

We don't use Process API for time series because Statistical API is ~10x
cheaper in PU per query. Process API gives us the per-pixel raster, which
we want exclusively for the visual overlay on the map.

The Process API response is binary (the requested image format), so we
return raw bytes from this layer and let the caller persist them.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from app.integrations.sentinel_hub.client import SentinelHubClient
from app.integrations.sentinel_hub.evalscripts import (
    HEATMAP_EVALSCRIPTS,
    SUPPORTED_INDICES,
)

log = logging.getLogger(__name__)

# Hard cap to keep PU cost bounded — generated image is at most 1024x1024.
MAX_DIM_PX = 1024


def _bbox_from_geometry(geometry: dict) -> tuple[float, float, float, float]:
    """Compute [minLon, minLat, maxLon, maxLat] from a GeoJSON polygon."""
    if geometry.get("type") != "Polygon":
        raise ValueError("Only Polygon geometries are supported for heatmap.")
    ring = geometry["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return min(lons), min(lats), max(lons), max(lats)


def _dims_for_bbox(bbox: tuple[float, float, float, float], resolution_m: float) -> tuple[int, int]:
    """Approximate pixel dimensions for an EPSG:4326 bbox at given metric resolution."""
    import math

    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(mid_lat_rad)
    width_m = (max_lon - min_lon) * meters_per_deg_lon
    height_m = (max_lat - min_lat) * meters_per_deg_lat
    width_px = max(64, min(MAX_DIM_PX, int(round(width_m / resolution_m))))
    height_px = max(64, min(MAX_DIM_PX, int(round(height_m / resolution_m))))
    return width_px, height_px


def _build_request(
    geometry: dict, target_date: date, index: str, max_cloud_cover: int
) -> tuple[dict, tuple[int, int]]:
    if index not in SUPPORTED_INDICES:
        raise ValueError(f"Unsupported index: {index}. Use one of {SUPPORTED_INDICES}.")

    bbox = _bbox_from_geometry(geometry)
    # 5m output — 2x oversampling vs Sentinel-2's native 10m. Doesn't add
    # information (SH bilinearly interpolates), but gives 4x more visible
    # cells in the rendered PNG so the user sees a clearly tiled heatmap
    # after the browser stretches it. PU cost grows ~linearly with pixel
    # count, but ~0.25 PU per heatmap is well within budget.
    width, height = _dims_for_bbox(bbox, resolution_m=5.0)

    # Match the full Statistical-API weekly bucket (chart points are the
    # bucket-START date). With `mosaickingOrder: leastCC` SH then picks the
    # clearest scene from the same 7-day window the chart aggregated over,
    # which guarantees the heatmap "matches" the chart point the user
    # clicked. Without this, a ±1 day window often missed the actual scene
    # (Sentinel-2 has a 5-day revisit, so the scene that contributed to
    # week's NDVI mean might be from day 3-5 of the bucket).
    date_from = target_date
    date_to = target_date + timedelta(days=6)

    payload = {
        "input": {
            "bounds": {
                # Use the polygon geometry (not just its bbox) so SH sets
                # dataMask=0 for pixels outside the field outline. Our
                # heatmap evalscript then returns [0,0,0,0] for those
                # pixels, giving a PNG that visually follows the field's
                # shape rather than its rectangular envelope.
                "geometry": geometry,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_from.isoformat()}T00:00:00Z",
                            "to": f"{date_to.isoformat()}T23:59:59Z",
                        },
                        "maxCloudCoverage": max_cloud_cover,
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": HEATMAP_EVALSCRIPTS[index],
    }
    return payload, (width, height)


async def fetch_heatmap_png(
    client: SentinelHubClient,
    geometry: dict,
    target_date: date,
    index: str,
    # 70% lets through scenes that have some clouds but still have valid
    # pixels over the field. Time-series ingestion uses 30% (we want
    # *clean* aggregates), but heatmap visualization tolerates more cloud
    # since masked pixels just render transparent.
    max_cloud_cover: int = 70,
) -> tuple[bytes, float]:
    """Fetch a coloured PNG heatmap for one (field, date, index).

    Returns (png_bytes, estimated_pu).
    """
    payload, (w, h) = _build_request(geometry, target_date, index, max_cloud_cover)
    png_bytes = await client.post_process(payload)
    # PU cost ~ samples × bands × area / (3 × 512²). Width × height × 4 bands.
    estimated_pu = round((w * h * 4) / (3 * 512 * 512), 3)
    return png_bytes, estimated_pu
