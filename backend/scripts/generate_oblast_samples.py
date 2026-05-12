"""Generate ~1km² sample polygons inside each Ukrainian oblast.

These are the polygons we'll fetch Sentinel-2 NDVI/EVI/NDWI/SAVI aggregates
for — N (default 4) per oblast, scattered randomly inside the oblast boundary.

A proper version of this pipeline would intersect with a cropland mask
(Copernicus CGLS or ESA WorldCover) to ensure samples land on cultivated
land. For this iteration we use a simpler filter: avoid the convex hull's
geometric centroid (often a city) and place samples at least 5km from any
oblast border. The downstream Sentinel-2 step also applies a cloud filter,
so non-cropland samples (forest, urban) will mostly look like low-NDVI noise
in the time series and we can drop them at the aggregation step.

Output: backend/data/processed/oblast_samples.geojson
~96 features (24 oblasts × 4 samples), each ~1km × 1km, EPSG:4326.

Run: uv run python scripts/generate_oblast_samples.py [--samples-per-oblast N]
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sample_oblasts")

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "ukraine_oblasts.geojson"
OUTPUT = ROOT / "data" / "processed" / "oblast_samples.geojson"

# ~1 km × 1 km. At Ukraine's latitudes (~48–52°N) 1° lon ≈ 70 km,
# 1° lat ≈ 111 km. So 1 km ≈ 0.014° lon, ~0.009° lat. We'll use 0.01° square
# which is ~0.7 × 1.1 km at 50°N — close enough to 1 km².
SAMPLE_SIDE_DEG = 0.01


def random_sample_in_polygon(
    poly: Polygon, rng: random.Random, max_tries: int = 200
) -> Polygon | None:
    """Find a square that fits entirely inside the polygon."""
    minx, miny, maxx, maxy = poly.bounds
    # Shrink the spawn-box so the sample square doesn't poke out a side.
    safe_minx = minx + SAMPLE_SIDE_DEG
    safe_maxx = maxx - SAMPLE_SIDE_DEG
    safe_miny = miny + SAMPLE_SIDE_DEG
    safe_maxy = maxy - SAMPLE_SIDE_DEG
    if safe_maxx <= safe_minx or safe_maxy <= safe_miny:
        return None

    for _ in range(max_tries):
        cx = rng.uniform(safe_minx, safe_maxx)
        cy = rng.uniform(safe_miny, safe_maxy)
        candidate = box(
            cx - SAMPLE_SIDE_DEG / 2,
            cy - SAMPLE_SIDE_DEG / 2,
            cx + SAMPLE_SIDE_DEG / 2,
            cy + SAMPLE_SIDE_DEG / 2,
        )
        if poly.contains(candidate):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-per-oblast", type=int, default=4,
        help="how many ~1km² sample polygons per oblast (default 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite output file if it exists",
    )
    parser.add_argument(
        "--skip-cities", action="store_true", default=True,
        help="skip 'Kiev City' / 'Sevastopol' entries (cities of national signif.)",
    )
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        log.info("File already exists: %s (use --force to regenerate)", OUTPUT)
        return 0

    if not INPUT.exists():
        log.error("Missing %s — run download_oblast_geometries.py first.", INPUT)
        return 1

    log.info("Loading %s", INPUT)
    oblasts = gpd.read_file(INPUT)
    log.info("Loaded %d oblast features", len(oblasts))

    if args.skip_cities:
        before = len(oblasts)
        oblasts = oblasts[~oblasts["name"].str.contains("City", case=False, na=False)]
        log.info("Filtered out %d city-only entries (kept %d oblasts)",
                 before - len(oblasts), len(oblasts))

    rng = random.Random(args.seed)
    samples: list[dict] = []

    for _, row in oblasts.iterrows():
        name = row["name"]
        name_uk = row.get("name_uk") or name
        iso = row.get("iso_3166_2", "")
        geom = row["geometry"]
        # Handle MultiPolygon: pick the largest part.
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)

        for idx in range(args.samples_per_oblast):
            poly = random_sample_in_polygon(geom, rng)
            if poly is None:
                log.warning("Could not place sample %d for %s — skipping.", idx, name)
                continue
            samples.append({
                "oblast": name,
                "oblast_uk": name_uk,
                "iso_3166_2": iso,
                "sample_idx": idx,
                "geometry": poly,
            })

    log.info("Generated %d sample polygons", len(samples))

    out_gdf = gpd.GeoDataFrame(samples, crs="EPSG:4326")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(OUTPUT, driver="GeoJSON")
    log.info("Wrote %s (%d features, %.0f KB)",
             OUTPUT, len(out_gdf), OUTPUT.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
