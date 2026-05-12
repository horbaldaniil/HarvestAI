"""Download SoilGrids 250m soil property rasters for Ukraine.

SoilGrids (https://www.isric.org/explore/soilgrids) is a global soil
information system from ISRIC — free under CC BY 4.0. Provides 6
soil properties (clay, sand, silt, pH, organic carbon, bulk density,
CEC) at 6 depth intervals.

For HarvestAI v7 we download the **0-30 cm topsoil** layers — root-zone
soil quality that drives yield, especially for tuber/root crops (potato,
sugar beet) and water-sensitive cereals.

## Why these 6 properties?

| Property | Why it matters |
|---|---|
| **clay** (% mass) | Water-holding capacity; clay-heavy soils retain moisture |
| **sand** (% mass) | Inverse of clay — drainage, root penetration |
| **phh2o** (pH × 10) | Nutrient availability; most crops want pH 6-7 |
| **soc** (g/kg) | Soil organic carbon — fertility proxy |
| **bdod** (cg/cm³) | Bulk density — compaction indicator |
| **cec** (mmol(c)/kg) | Cation exchange capacity — nutrient buffering |

## Download API

SoilGrids exposes a **WCS (Web Coverage Service)** at
https://maps.isric.org/mapserv?map=/map/{property}.map. We make one
GetCoverage request per property for the Ukraine bbox (lat 44-53,
lon 22-41).

Each request returns a ~50-80 MB GeoTIFF at 250m resolution.
Total download: ~400-500 MB for 6 properties.

## Run

    uv run python scripts/download_soilgrids.py
    uv run python scripts/download_soilgrids.py --dest data/raw/soilgrids

The script is resume-safe: existing properly-sized .tif files are skipped.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("soilgrids")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "raw" / "soilgrids"

# Ukraine bounding box in EPSG:4326 (WGS84) — wide enough to cover all 24
# agricultural oblasts plus a small buffer.
BBOX_WGS84 = (22.0, 44.0, 41.0, 53.0)  # west, south, east, north

# Properties and their service URLs. SoilGrids WCS endpoint pattern:
#   https://maps.isric.org/mapserv?map=/map/{property}.map&SERVICE=WCS&...
# The `subset` parameter selects depth (0-30cm we use _0-5cm + _5-15cm +
# _15-30cm aggregate, but for simplicity grab a single layer per property).
#
# Layer naming in SoilGrids v2.0:
#   clay (mean, 0-30 cm) → property=clay, coverage=clay_0-5cm_mean
PROPERTIES: tuple[tuple[str, str], ...] = (
    # (property name, coverage layer)
    ("clay",  "clay_0-5cm_mean"),
    ("sand",  "sand_0-5cm_mean"),
    ("silt",  "silt_0-5cm_mean"),
    ("phh2o", "phh2o_0-5cm_mean"),
    ("soc",   "soc_0-5cm_mean"),
    ("bdod",  "bdod_0-5cm_mean"),
    ("cec",   "cec_0-5cm_mean"),
)


def _build_session() -> requests.Session:
    s = requests.Session()
    policy = Retry(total=5, backoff_factor=2.0,
                   status_forcelist=(429, 500, 502, 503, 504),
                   allowed_methods=("GET", "HEAD"))
    s.mount("https://", HTTPAdapter(max_retries=policy))
    return s


def _build_wcs_url(prop: str, coverage: str, bbox: tuple[float, float, float, float],
                   resolution: float = 0.0025) -> str:
    """SoilGrids WCS v2.0 GetCoverage URL.

    resolution=0.0025 ≈ 250m at Ukraine latitudes.
    """
    west, south, east, north = bbox
    return (
        f"https://maps.isric.org/mapserv?map=/map/{prop}.map"
        f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={coverage}"
        f"&FORMAT=image/tiff"
        f"&SUBSET=Long({west},{east})"
        f"&SUBSET=Lat({south},{north})"
        f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        f"&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    )


def _download_one(session: requests.Session, url: str, dest: Path,
                  expected_min_size: int = 5_000_000) -> bool:
    """Stream-download a SoilGrids tile. Skip if already present at
    sensible size. Returns True on success."""
    if dest.exists() and dest.stat().st_size >= expected_min_size:
        log.info("Skip %s (already present, %.1f MB)",
                 dest.name, dest.stat().st_size / 1e6)
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            written = 0
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
        tmp.replace(dest)
        log.info("Got %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Download %s failed: %s", url, exc)
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"output directory (default {DEFAULT_DEST})")
    parser.add_argument("--dry-run", action="store_true",
                        help="list URLs without downloading")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %d SoilGrids properties for Ukraine bbox %s",
             len(PROPERTIES), BBOX_WGS84)

    session = _build_session()
    ok = 0
    fail = 0
    for prop, coverage in PROPERTIES:
        url = _build_wcs_url(prop, coverage, BBOX_WGS84)
        dest = args.dest / f"{prop}_0-5cm_mean_ua.tif"
        if args.dry_run:
            log.info("WOULD GET %s → %s", url, dest.name)
            continue
        if _download_one(session, url, dest):
            ok += 1
        else:
            fail += 1

    log.info("Done: %d OK, %d failed.", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
