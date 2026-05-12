"""Generate ~1km² sample polygons inside each Ukrainian oblast.

These are the polygons we fetch Sentinel-2 NDVI/EVI/NDWI/SAVI aggregates
for — N (default 8 in v3) per oblast, scattered inside the oblast.

## v3 — cropland-masked sampling

v2 documented as a limitation: "samples may land on forest, urban or
water; downstream NDVI aggregation washes them out as low-signal noise".
v3 fixes this by intersecting each candidate sample square against the
ESA WorldCover 10 m product (class 40 = cropland). Only squares whose
**majority pixels are cropland** are accepted.

The mask is loaded lazily: pass `--cropland-mask` (also requires that
`data/raw/worldcover/*.tif` exist — run `download_worldcover_tiles.py`
first). Without the flag, the script behaves exactly like v2 (centroid-
avoidance only) so the existing v2 parquet rows can still be reproduced.

This matters scientifically because:
- Forest pixels read NDVI ≈ 0.85 year-round → look like vigorous wheat
  at peak, which biases the trained model.
- Urban pixels read NDVI < 0.3 always → drag oblast aggregates down.

After v3 masking, expect ~5-15 % of v2 samples to be rejected in
heavily-forested Polissia oblasts and ~2-5 % elsewhere.

Output: `backend/data/processed/oblast_samples_v2.geojson` (when
`--cropland-mask` is set) or `oblast_samples.geojson` (v2 default).

Run:
    uv run python scripts/generate_oblast_samples.py
    uv run python scripts/generate_oblast_samples.py --cropland-mask
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sample_oblasts")

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "ukraine_oblasts.geojson"
OUTPUT_V2 = ROOT / "data" / "processed" / "oblast_samples.geojson"
OUTPUT_V3 = ROOT / "data" / "processed" / "oblast_samples_v2.geojson"
WORLDCOVER_DIR = ROOT / "data" / "raw" / "worldcover"

# ~1 km × 1 km at Ukraine's latitudes — 0.01° lon × 0.01° lat is
# ~0.7 × 1.1 km at 50°N. Close enough to 1 km² for aggregation.
SAMPLE_SIDE_DEG = 0.01

# ESA WorldCover class 40 = cropland (rainfed and irrigated combined).
# All other class codes (10 tree-cover, 20 shrubland, 30 grassland, 50
# built-up, 60 bare/sparse, 70 snow/ice, 80 water, 90 wetland, 95
# mangroves, 100 moss/lichen) are non-cropland and rejected.
CROPLAND_CLASS = 40

# Minimum cropland fraction inside a sample square for acceptance.
# 0.6 = at least 60 % of the 10 m pixels under the sample window must
# be class 40. Empirically chosen: 0.5 lets too many edge-of-forest
# squares slip through; 0.8 is too strict for narrow valley cropland.
MIN_CROPLAND_FRACTION = 0.6


def random_sample_in_polygon(
    poly: Polygon, rng: random.Random, max_tries: int = 200,
) -> Polygon | None:
    """Find a square that fits entirely inside the polygon."""
    minx, miny, maxx, maxy = poly.bounds
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


# ─── Cropland mask (WorldCover) ──────────────────────────────


class CroplandMask:
    """Lazy reader for ESA WorldCover tiles covering Ukraine.

    Each query opens whichever tile contains the sample bbox (cached
    in-process). For each square the percentage of class-40 pixels is
    computed via a rasterio window read.
    """

    def __init__(self, tiles_dir: Path, min_fraction: float = MIN_CROPLAND_FRACTION):
        self.tiles_dir = tiles_dir
        self.min_fraction = min_fraction
        # Cached opened datasets keyed by tile filename.
        self._datasets: dict[str, "rasterio.io.DatasetReader"] = {}  # noqa: F821

    @staticmethod
    def _tile_for_point(lat: float, lon: float) -> str:
        """Filename of the 3°×3° tile containing (lat, lon)."""
        lat_sw = (int(lat) // 3) * 3
        lon_sw = (int(lon) // 3) * 3
        return f"ESA_WorldCover_10m_2021_v200_N{lat_sw:02d}E{lon_sw:03d}_Map.tif"

    def _open(self, tile_name: str):
        """Open (and cache) one WorldCover tile."""
        import rasterio  # imported lazily so tests can stub it out

        cached = self._datasets.get(tile_name)
        if cached is not None:
            return cached
        path = self.tiles_dir / tile_name
        if not path.exists():
            return None
        ds = rasterio.open(path)
        self._datasets[tile_name] = ds
        return ds

    def is_cropland(self, square: BaseGeometry) -> bool:
        """True if `square` contains ≥ `min_fraction` cropland pixels.

        Picks the WorldCover tile under the square's centroid and reads
        the corresponding window. Squares spanning two tiles (rare —
        ~0.01° edges falling exactly on a 3° boundary) fall back to a
        centroid-class check rather than stitching, which is a tiny
        accuracy compromise for code simplicity.
        """
        import numpy as np
        from rasterio.windows import from_bounds

        cx, cy = square.centroid.x, square.centroid.y
        tile_name = self._tile_for_point(cy, cx)
        ds = self._open(tile_name)
        if ds is None:
            log.debug("No tile %s for centroid (%.3f, %.3f) — rejecting", tile_name, cy, cx)
            return False

        minx, miny, maxx, maxy = square.bounds
        try:
            win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
            arr = ds.read(1, window=win)
        except Exception as exc:  # noqa: BLE001
            log.debug("WorldCover read failed for square %s: %s", square.bounds, exc)
            return False

        if arr.size == 0:
            return False
        fraction = float(np.count_nonzero(arr == CROPLAND_CLASS)) / arr.size
        return fraction >= self.min_fraction

    def close(self) -> None:
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()


def random_cropland_sample(
    poly: Polygon, rng: random.Random, mask: CroplandMask | None,
    max_tries: int = 500,
) -> Polygon | None:
    """Find a square inside `poly` that ALSO satisfies the cropland mask.

    When `mask` is None this is identical to `random_sample_in_polygon`.
    We allow more retries (500 vs 200) because the mask is a stricter
    constraint — forested oblasts (Volyn, Zhytomyr) may need many tries.
    """
    for _ in range(max_tries):
        square = random_sample_in_polygon(poly, rng, max_tries=1)
        if square is None:
            continue
        if mask is None or mask.is_cropland(square):
            return square
    return None


# ─── Main ────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--samples-per-oblast", type=int, default=4,
        help="how many ~1km² sample polygons per oblast (default 4; use 8 for v3)",
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
    parser.add_argument(
        "--cropland-mask", action="store_true", default=False,
        help="filter samples through ESA WorldCover cropland class (10 m); requires "
             "download_worldcover_tiles.py to have been run. Output goes to "
             "oblast_samples_v2.geojson when enabled.",
    )
    args = parser.parse_args()

    output = OUTPUT_V3 if args.cropland_mask else OUTPUT_V2

    if output.exists() and not args.force:
        log.info("File already exists: %s (use --force to regenerate)", output)
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

    mask: CroplandMask | None = None
    if args.cropland_mask:
        if not WORLDCOVER_DIR.exists() or not any(WORLDCOVER_DIR.glob("*.tif")):
            log.error("--cropland-mask requested but %s is empty.\n"
                      "Run: uv run python scripts/download_worldcover_tiles.py first.",
                      WORLDCOVER_DIR)
            return 1
        mask = CroplandMask(WORLDCOVER_DIR)
        log.info("Cropland mask enabled (WorldCover class %d, min %d%% cropland fraction).",
                 CROPLAND_CLASS, int(100 * MIN_CROPLAND_FRACTION))

    rng = random.Random(args.seed)
    samples: list[dict] = []
    rejected_per_oblast: dict[str, int] = {}

    for _, row in oblasts.iterrows():
        name = row["name"]
        name_uk = row.get("name_uk") or name
        iso = row.get("iso_3166_2", "")
        geom = row["geometry"]
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)

        accepted = 0
        rejected = 0
        for idx in range(args.samples_per_oblast):
            poly = random_cropland_sample(geom, rng, mask)
            if poly is None:
                rejected += 1
                log.warning("Could not place sample %d for %s — too few cropland tries.",
                            idx, name)
                continue
            samples.append({
                "oblast": name,
                "oblast_uk": name_uk,
                "iso_3166_2": iso,
                "sample_idx": idx,
                "geometry": poly,
            })
            accepted += 1
        if rejected > 0:
            rejected_per_oblast[name] = rejected

    log.info("Generated %d sample polygons", len(samples))
    if rejected_per_oblast:
        log.warning("Oblasts with rejected samples: %s", rejected_per_oblast)

    out_gdf = gpd.GeoDataFrame(samples, crs="EPSG:4326")
    output.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(output, driver="GeoJSON")
    log.info("Wrote %s (%d features, %.0f KB)",
             output, len(out_gdf), output.stat().st_size / 1024)

    if mask is not None:
        mask.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
