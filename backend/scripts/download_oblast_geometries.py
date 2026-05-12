"""Download Ukrainian oblast (admin-1) boundaries from Natural Earth.

Natural Earth Vector is public domain. We pull the 1:10m
admin-1 (`states_provinces`) layer, filter to Ukraine (`admin == "Ukraine"`),
drop the de-facto / disputed status columns we don't need, and write the
result as a simplified GeoJSON.

Output: backend/data/raw/ukraine_oblasts.geojson
~24 features (24 oblasts + sometimes Sevastopol/Crimea as separate entries —
we keep them all; the consumer can filter).

Run: uv run python scripts/download_oblast_geometries.py

Idempotent: skips network fetch if the file already exists and `--force`
not passed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_oblasts")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "ukraine_oblasts.geojson"

# Natural Earth Vector — 1:10m admin-1 layer. Cached at GitHub mirror because
# the official S3 bucket sometimes 502s. Public domain.
NE_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_1_states_provinces.zip"
)


def download(url: str) -> bytes:
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def extract_ukraine(zip_bytes: bytes) -> gpd.GeoDataFrame:
    with ZipFile(BytesIO(zip_bytes)) as zf:
        # The zip ships shapefiles — geopandas reads them through a virtual FS.
        shp_name = next(n for n in zf.namelist() if n.endswith(".shp"))
        log.info("Reading %s", shp_name)
        gdf = gpd.read_file(f"zip://{shp_name}", vfs=f"/vsizip/{{{zf.filename}}}/")
    raise RuntimeError(
        "Direct in-memory shapefile read needs the zip on disk in geopandas "
        "1.x — see fallback path in main()."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch even if file exists")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        log.info("File already exists: %s (use --force to refetch)", OUTPUT)
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Save the zip to a temp file because geopandas's in-memory zip path is
    # finicky on Windows; fiona's `zip+file:` protocol works most reliably with
    # an actual file path.
    import tempfile

    zip_bytes = download(NE_URL)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = Path(tmp.name)

    try:
        # geopandas can read a shapefile bundled inside a zip via fiona vfs.
        gdf = gpd.read_file(f"zip://{tmp_path}!ne_10m_admin_1_states_provinces.shp")
        log.info("Loaded %d global admin-1 features", len(gdf))

        ukraine = gdf[gdf["admin"] == "Ukraine"].copy()
        log.info("Filtered to %d Ukrainian features", len(ukraine))

        if len(ukraine) == 0:
            log.error("No Ukrainian features found — Natural Earth schema may have changed.")
            return 1

        # Keep only the columns we care about. `name` is the English name,
        # `name_uk` is Ukrainian if Natural Earth carries it (not always).
        keep_cols = [c for c in ("name", "name_uk", "iso_3166_2", "type_en", "geometry") if c in ukraine.columns]
        ukraine = ukraine[keep_cols]

        # Normalise the schema so the consumer doesn't have to branch on
        # whether name_uk is present.
        if "name_uk" not in ukraine.columns:
            ukraine["name_uk"] = ukraine["name"]

        # Project to WGS84 if it isn't already (NE comes in EPSG:4326 by default).
        if ukraine.crs is None or ukraine.crs.to_epsg() != 4326:
            ukraine = ukraine.to_crs(4326)

        ukraine.to_file(OUTPUT, driver="GeoJSON")
        log.info("Wrote %s (%d features, %.0f KB)",
                 OUTPUT, len(ukraine), OUTPUT.stat().st_size / 1024)
        return 0
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
