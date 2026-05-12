"""Bulk-fetch Sentinel-2 weekly NDVI/EVI/NDWI/SAVI for every (sample, year)
combo in `oblast_samples.geojson`.

Per-sample × per-year is ONE Statistical API call (P7D weekly buckets, April-
September window). Result is a parquet file we can re-aggregate downstream.

Resumable: existing rows in the output parquet are skipped on re-run, so you
can split execution across days/months to stay under the monthly PU budget.

Output: backend/data/processed/oblast_s2_observations.parquet

Run:
  uv run python scripts/collect_oblast_s2.py                   # full run
  uv run python scripts/collect_oblast_s2.py --year 2023       # single year
  uv run python scripts/collect_oblast_s2.py --max-calls 50    # PU-budget guard
  uv run python scripts/collect_oblast_s2.py --dry-run         # show plan only
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

from app.integrations.sentinel_hub.client import SentinelHubClient
from app.integrations.sentinel_hub.statistical_api import fetch_indices_timeseries
from app.redis_clients import get_async_redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_s2")

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "processed" / "oblast_samples.geojson"
OUTPUT = ROOT / "data" / "processed" / "oblast_s2_observations.parquet"

DEFAULT_YEARS = (2019, 2020, 2021, 2022, 2023)
# Window: April through September covers the full growing season in Ukraine.
# Bookending wider would waste PU on dormant-season buckets where NDVI is noise.
WINDOW_START_MMDD = (4, 1)
WINDOW_END_MMDD = (9, 30)


def _load_existing() -> set[tuple[str, int, int]]:
    """Return (oblast, sample_idx, year) tuples already collected."""
    if not OUTPUT.exists():
        return set()
    try:
        df = pd.read_parquet(OUTPUT, columns=["oblast", "sample_idx", "year"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read existing parquet (%s) — starting fresh.", exc)
        return set()
    return set(map(tuple, df.drop_duplicates().values.tolist()))


def _append(rows: list[dict]) -> None:
    """Append rows to the output parquet (concat-rewrite — fine at this scale)."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)


async def _fetch_one(
    client: SentinelHubClient,
    geometry: dict,
    year: int,
) -> tuple[list[dict], float]:
    """Fetch one (sample, year) of weekly aggregates. Returns (rows, pu_estimate)."""
    start = date(year, *WINDOW_START_MMDD)
    end = date(year, *WINDOW_END_MMDD)
    aggregates, pu_used = await fetch_indices_timeseries(
        client, geometry, start, end, max_cloud_cover=40,
    )
    return [
        {
            "observed_on": a.observed_on.isoformat() if a.observed_on else None,
            "iso_week": a.observed_on.isocalendar().week if a.observed_on else None,
            "ndvi_mean": a.ndvi_mean,
            "ndvi_min": a.ndvi_min,
            "ndvi_max": a.ndvi_max,
            "ndvi_std": a.ndvi_std,
            "evi_mean": a.evi_mean,
            "ndwi_mean": a.ndwi_mean,
            "savi_mean": a.savi_mean,
            "cloud_cover": a.cloud_cover,
        }
        for a in aggregates
    ], pu_used


async def run(args: argparse.Namespace) -> int:
    if not SAMPLES.exists():
        log.error("Missing %s — run generate_oblast_samples.py first.", SAMPLES)
        return 1

    samples = gpd.read_file(SAMPLES)
    log.info("Loaded %d sample polygons", len(samples))

    years = [args.year] if args.year else list(DEFAULT_YEARS)
    log.info("Years to collect: %s", years)

    existing = _load_existing()
    log.info("Already collected: %d (sample, year) buckets", len(existing))

    plan: list[tuple[str, int, int, dict]] = []
    for _, row in samples.iterrows():
        for year in years:
            key = (row["oblast"], int(row["sample_idx"]), year)
            if key in existing:
                continue
            plan.append((row["oblast"], int(row["sample_idx"]), year, row["geometry"].__geo_interface__))

    log.info("Plan: %d (sample, year) calls remaining", len(plan))
    if args.max_calls:
        plan = plan[: args.max_calls]
        log.info("Capped at --max-calls=%d", args.max_calls)

    if args.dry_run:
        for oblast, idx, year, _ in plan[:20]:
            log.info("  WOULD FETCH: %s sample=%d year=%d", oblast, idx, year)
        if len(plan) > 20:
            log.info("  ... and %d more", len(plan) - 20)
        log.info("Dry-run estimated PU: ~%.1f (at ~3 PU per call)", len(plan) * 3.0)
        return 0

    if not plan:
        log.info("Nothing to do — exiting.")
        return 0

    redis = get_async_redis()
    client = SentinelHubClient(redis)
    total_pu = 0.0
    buffer_rows: list[dict] = []
    try:
        for i, (oblast, idx, year, geometry) in enumerate(plan, 1):
            try:
                aggregates, pu_used = await _fetch_one(client, geometry, year)
                total_pu += pu_used
                for row in aggregates:
                    row.update(oblast=oblast, sample_idx=idx, year=year)
                    buffer_rows.append(row)
                log.info("(%d/%d) %s sample=%d year=%d → %d rows, %.2f PU (running: %.1f)",
                         i, len(plan), oblast, idx, year, len(aggregates),
                         pu_used, total_pu)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed for %s sample=%d year=%d: %s", oblast, idx, year, exc)

            # Flush every 10 calls so a Ctrl-C doesn't lose work.
            if len(buffer_rows) >= 200 or i % 10 == 0:
                _append(buffer_rows)
                buffer_rows.clear()
    finally:
        if buffer_rows:
            _append(buffer_rows)
        await client.aclose()
        await redis.aclose()

    log.info("DONE. Total PU spent (estimate): %.1f", total_pu)
    if OUTPUT.exists():
        df = pd.read_parquet(OUTPUT)
        log.info("Final parquet: %d rows, %d unique (sample, year) buckets",
                 len(df), df[["oblast", "sample_idx", "year"]].drop_duplicates().shape[0])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="single year (default: all)")
    parser.add_argument("--max-calls", type=int, default=None, help="cap to N API calls")
    parser.add_argument("--dry-run", action="store_true", help="show plan, no fetching")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
