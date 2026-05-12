"""Download ESA WorldCover 10 m land-cover tiles covering Ukraine.

ESA WorldCover is a free 10 m global land-cover product derived from
Sentinel-1 + Sentinel-2 fusion. Class **40 = cropland**, which is the
mask we use to ensure our oblast sampling lands on cultivated land
rather than forest/urban/water (see `generate_oblast_samples.py
--cropland-mask`).

## Why WorldCover and not Copernicus CGLS?

- WorldCover is 10 m; CGLS is 100 m. With 1 km × 1 km sample windows
  we want fine-grained class precision so we don't accept a window
  whose centroid happens to be cropland but whose majority is urban.
- WorldCover v200 (2021) is the most recent product as of 2024.
- Hosted on AWS S3 free-tier, no auth required.

## Tiling convention

WorldCover splits the globe into 3°×3° tiles named by the SW corner:

    ESA_WorldCover_10m_2021_v200_N{LAT}E{LON}_Map.tif

So `N45E024_Map.tif` covers (45°N..48°N, 24°E..27°E). For Ukraine
(roughly 44°N..53°N, 22°E..41°E) we need 4 × 7 = **28 tiles**, each
~80 MB compressed → ~2.2 GB total. Tiles are Cloud-Optimised GeoTIFFs
so downstream rasterio reads only the windows we need.

## Run

    uv run python scripts/download_worldcover_tiles.py
    # or with a specific destination
    uv run python scripts/download_worldcover_tiles.py --dest data/raw/worldcover

The script is resume-safe: a partially-downloaded tile is detected
(size mismatch vs HEAD) and re-fetched from byte 0. Tiles already
present at full size are skipped.

## Citation (thesis)

Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., ...
(2022). ESA WorldCover 10 m 2021 v200. Zenodo.
https://doi.org/10.5281/zenodo.7254221
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
log = logging.getLogger("worldcover")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "raw" / "worldcover"

# AWS S3 bucket hosting the WorldCover v200 tiles. The URL pattern is
# stable as of 2024 — if it changes upstream, the failure is loud
# (404s) rather than silent (corrupt downloads).
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# Tile naming uses 3° steps with the southwest corner. Ukraine's bbox
# is roughly (44 N..53 N, 22 E..41 E). We include 42 N..51 N (× 3° step)
# and 21 E..39 E to be safe at the edges.
LAT_STARTS: tuple[int, ...] = (42, 45, 48, 51)
LON_STARTS: tuple[int, ...] = (21, 24, 27, 30, 33, 36, 39)


def _tile_name(lat: int, lon: int) -> str:
    """`N48E024_Map.tif` style. Lat/lon are zero-padded to two digits."""
    return f"ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif"


def _build_session(retries: int = 5) -> requests.Session:
    """HTTP session with exponential backoff for flaky downloads."""
    s = requests.Session()
    policy = Retry(
        total=retries, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    s.mount("https://", HTTPAdapter(max_retries=policy))
    return s


def _remote_size(session: requests.Session, url: str) -> int | None:
    """HEAD-request Content-Length; returns None if HEAD is unsupported."""
    try:
        r = session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        v = r.headers.get("Content-Length")
        return int(v) if v else None
    except Exception as exc:  # noqa: BLE001
        log.warning("HEAD %s failed: %s", url, exc)
        return None


def _download_one(session: requests.Session, url: str, dest: Path,
                  chunk_size: int = 4 * 1024 * 1024) -> bool:
    """Stream-download one tile. Skips if already present at full size.

    Returns True on success, False if the tile didn't exist (404) or
    permanently failed.
    """
    expected_size = _remote_size(session, url)
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        log.info("Skip %s (already present, %.1f MB)", dest.name, expected_size / 1e6)
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=120) as r:
            if r.status_code == 404:
                log.warning("Tile not on S3: %s (404) — water-only tile?", dest.name)
                return False
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0") or 0)
            written = 0
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    if total and written % (50 * chunk_size) == 0:
                        log.info("  %s: %.0f%% (%.1f / %.1f MB)",
                                 dest.name, 100 * written / total,
                                 written / 1e6, total / 1e6)
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
                        help="list the URLs that would be fetched without downloading")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path]] = []
    for lat in LAT_STARTS:
        for lon in LON_STARTS:
            name = _tile_name(lat, lon)
            targets.append((f"{BASE_URL}/{name}", args.dest / name))

    log.info("%d tile(s) targeted → %s", len(targets), args.dest)
    if args.dry_run:
        for url, _ in targets:
            print(url)
        return 0

    session = _build_session()
    ok = 0
    fail = 0
    for url, dest in targets:
        if _download_one(session, url, dest):
            ok += 1
        else:
            fail += 1

    log.info("Done: %d OK, %d failed/missing.", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
