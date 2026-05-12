"""Download SRTM 30m elevation tiles for Ukraine.

NASA Shuttle Radar Topography Mission — global 30m DEM, free via USGS.

For HarvestAI v7 we use SRTM to derive **topographic features**:
  - elevation (mean per sample polygon)
  - slope (derivative — affects runoff)
  - aspect (NSEW orientation — affects insolation)
  - TWI (topographic wetness index — proxy for water availability)

These features especially help **water-sensitive crops** (potato,
sugar beet, soybean) where water distribution across a field/oblast
is yield-critical.

## Download approach

We use the OpenTopography S3 mirror via the `elevation` Python package
(https://github.com/bopen/elevation), which wraps GDAL to assemble
SRTM tiles seamlessly into a single GeoTIFF for any bbox.

Alternative: direct NASA EarthData (needs login). The `elevation`
package is simpler and the OpenTopography mirror is unauthenticated.

## Run

    uv run python scripts/download_srtm.py

Output: `data/raw/srtm/ua_srtm_30m.tif` (~150 MB for Ukraine bbox).
Derived slope/aspect/TWI rasters are computed in `build_features_v7.py`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("srtm")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "raw" / "srtm"
BBOX_UA = (22.0, 44.0, 41.0, 53.0)  # west, south, east, north (WGS84)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--product", choices=("SRTM1", "SRTM3"),
                        default="SRTM1",
                        help="SRTM1=30m (preferred), SRTM3=90m (smaller, faster)")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    out = args.dest / f"ua_srtm_{args.product.lower()}.tif"
    if out.exists() and out.stat().st_size > 50_000_000:
        log.info("Skip %s (already present, %.1f MB)",
                 out.name, out.stat().st_size / 1e6)
        return 0

    try:
        import elevation
    except ImportError:
        log.error("`elevation` package not installed. Run: uv pip install elevation")
        return 1

    log.info("Downloading %s for Ukraine bbox %s → %s",
             args.product, BBOX_UA, out)
    try:
        elevation.clip(bounds=BBOX_UA, output=str(out.absolute()),
                       product=args.product)
        if out.exists():
            log.info("Got %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
            return 0
        log.error("Download did not produce file at %s", out)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Download failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
