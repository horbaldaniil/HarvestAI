"""Bulk-fetch Sentinel-2 weekly NDVI/EVI/NDWI/SAVI for every (sample, year)
combo in an oblast samples geojson.

Per-sample × per-year is ONE Statistical API call (P7D weekly buckets,
April–September window). Result is a parquet file we can re-aggregate
downstream.

Resumable: existing rows in the output parquet are skipped on re-run,
so you can split execution across days/months to stay under the monthly
PU budget if needed. The v3 plan operates inside a 30 000 PU/month
budget, so resumability matters less, but the safety is free.

## v3 changes vs v2
- `--samples-file` selects which sample geojson to use (default
  `oblast_samples_v2.geojson` — the cropland-masked v3 file when
  present, otherwise the legacy v2 file).
- `--output-file` mirrors the samples file naming (v3 writes to
  `oblast_s2_observations_v2.parquet` to keep v1 intact for ablation).
- `--years` accepts either a single year (`--years 2023`) or a range
  (`--years 2017-2023`) or a comma list (`--years 2018,2021,2023`).
- PU-budget pre-flight: a `--dry-run` now reports the **estimated
  monthly-budget impact** alongside the call count.

Run:
    uv run python scripts/collect_oblast_s2.py --years 2017-2023
    uv run python scripts/collect_oblast_s2.py --year 2023
    uv run python scripts/collect_oblast_s2.py --max-calls 50
    uv run python scripts/collect_oblast_s2.py --dry-run
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

# v2 (legacy) → no cropland mask, 4 samples × 8 oblasts × 5 years.
# v3 → cropland-masked, 8 samples × 24 oblasts × 7 years (2017-2023).
SAMPLES_V2 = ROOT / "data" / "processed" / "oblast_samples.geojson"
SAMPLES_V3 = ROOT / "data" / "processed" / "oblast_samples_v2.geojson"

OUTPUT_V2 = ROOT / "data" / "processed" / "oblast_s2_observations.parquet"
OUTPUT_V3 = ROOT / "data" / "processed" / "oblast_s2_observations_v2.parquet"

DEFAULT_YEARS_V3 = (2017, 2018, 2019, 2020, 2021, 2022, 2023)
DEFAULT_YEARS_V2 = (2019, 2020, 2021, 2022, 2023)

# Statistical API typically costs ~3 PU per (sample, year) call covering
# April–September weekly buckets. Measured across 480 v2 calls — actual
# per-call PU varies 2.5–3.5 depending on cloud-mask complexity.
PU_PER_CALL_ESTIMATE = 3.0

# Sentinel Hub monthly budget (units: PU). Used purely for the dry-run
# advisory; the API itself enforces nothing here.
DEFAULT_MONTHLY_BUDGET_PU = 30_000

WINDOW_START_MMDD = (4, 1)
WINDOW_END_MMDD = (9, 30)


def _parse_years(spec: str | None) -> tuple[int, ...] | None:
    """Accept `2023`, `2017-2023`, or `2018,2021,2023`. None → default range."""
    if not spec:
        return None
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return tuple(range(int(a), int(b) + 1))
    if "," in spec:
        return tuple(int(p) for p in spec.split(",") if p.strip())
    return (int(spec),)


def _resolve_samples_path(explicit: Path | None) -> Path:
    """Prefer the user-provided path; else prefer v3 (cropland-masked) if it
    exists; else fall back to v2 (legacy)."""
    if explicit is not None:
        return explicit
    return SAMPLES_V3 if SAMPLES_V3.exists() else SAMPLES_V2


def _resolve_output_path(explicit: Path | None, samples_path: Path) -> Path:
    """If samples is the v3 file → write to v3 parquet, else v2 parquet."""
    if explicit is not None:
        return explicit
    return OUTPUT_V3 if samples_path == SAMPLES_V3 else OUTPUT_V2


def _load_existing(output_path: Path) -> set[tuple[str, int, int]]:
    """Return (oblast, sample_idx, year) tuples already collected."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["oblast", "sample_idx", "year"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read existing parquet (%s) — starting fresh.", exc)
        return set()
    return set(map(tuple, df.drop_duplicates().values.tolist()))


def _append(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)


async def _fetch_one(
    client: SentinelHubClient,
    geometry: dict,
    year: int,
) -> tuple[list[dict], float]:
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
    samples_path = _resolve_samples_path(args.samples_file)
    output_path = _resolve_output_path(args.output_file, samples_path)

    if not samples_path.exists():
        log.error("Missing %s — run generate_oblast_samples.py first.", samples_path)
        return 1

    samples = gpd.read_file(samples_path)
    log.info("Loaded %d sample polygons from %s", len(samples), samples_path.name)

    # Year selection precedence: --years > --year > defaults
    years: tuple[int, ...]
    if args.years:
        parsed = _parse_years(args.years)
        years = parsed if parsed else ()
    elif args.year is not None:
        years = (args.year,)
    else:
        years = DEFAULT_YEARS_V3 if samples_path == SAMPLES_V3 else DEFAULT_YEARS_V2

    log.info("Years to collect: %s", list(years))
    log.info("Output parquet: %s", output_path)

    existing = _load_existing(output_path)
    log.info("Already collected: %d (sample, year) buckets", len(existing))

    plan: list[tuple[str, int, int, dict]] = []
    for _, row in samples.iterrows():
        for year in years:
            key = (row["oblast"], int(row["sample_idx"]), year)
            if key in existing:
                continue
            plan.append(
                (row["oblast"], int(row["sample_idx"]), year, row["geometry"].__geo_interface__)
            )

    log.info("Plan: %d (sample, year) calls remaining", len(plan))

    if args.max_calls:
        plan = plan[: args.max_calls]
        log.info("Capped at --max-calls=%d", args.max_calls)

    pu_estimate = len(plan) * PU_PER_CALL_ESTIMATE
    budget_pct = 100.0 * pu_estimate / DEFAULT_MONTHLY_BUDGET_PU
    log.info("Estimated PU: ~%.1f (at ~%.1f PU/call) — %.1f%% of %d-PU monthly budget",
             pu_estimate, PU_PER_CALL_ESTIMATE, budget_pct, DEFAULT_MONTHLY_BUDGET_PU)
    if budget_pct > 100:
        log.warning("Plan exceeds default monthly budget; consider --max-calls "
                    "and a multi-month rollout.")

    if args.dry_run:
        for oblast, idx, year, _ in plan[:20]:
            log.info("  WOULD FETCH: %s sample=%d year=%d", oblast, idx, year)
        if len(plan) > 20:
            log.info("  ... and %d more", len(plan) - 20)
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

            if len(buffer_rows) >= 200 or i % 10 == 0:
                _append(buffer_rows, output_path)
                buffer_rows.clear()
    finally:
        if buffer_rows:
            _append(buffer_rows, output_path)
        await client.aclose()
        await redis.aclose()

    log.info("DONE. Total PU spent (estimate): %.1f", total_pu)
    if output_path.exists():
        df = pd.read_parquet(output_path)
        log.info("Final parquet: %d rows, %d unique (sample, year) buckets",
                 len(df), df[["oblast", "sample_idx", "year"]].drop_duplicates().shape[0])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-file", type=Path, default=None,
                        help="path to the samples geojson; default picks v3 if present, else v2")
    parser.add_argument("--output-file", type=Path, default=None,
                        help="path to the output parquet; default mirrors samples file naming")
    parser.add_argument("--year", type=int, default=None,
                        help="single year (legacy alias for --years <Y>)")
    parser.add_argument("--years", type=str, default=None,
                        help="year selection: 'YYYY', 'YYYY-YYYY', or 'YYYY,YYYY,YYYY'")
    parser.add_argument("--max-calls", type=int, default=None,
                        help="cap to N API calls (useful for PU-budget guard)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show plan + estimated PU; no fetching")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
