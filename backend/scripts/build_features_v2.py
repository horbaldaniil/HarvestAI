"""Aggregate sample-level Sentinel-2 observations into per-(oblast, year, crop)
feature vectors compatible with the v1 17-feature schema in
`scripts/train_yield_models.py`.

For each (oblast, year):
  1. Read all weekly rows from oblast_s2_observations.parquet for that
     (oblast, year) — typically 4 samples × ~26 weeks = ~100 rows.
  2. Per ISO-week, take the MEDIAN across samples (robust to outliers).
  3. Compute the 11 vegetation-index features (peak, peak_week, monthly
     means, integral, std, evi/ndwi/savi summaries) from the aggregated
     weekly series.
  4. Pull weather stats from Open-Meteo per oblast-centroid + year (same
     interface as v1).
  5. Join with USDA yield labels per (oblast, year, crop).

Output: backend/data/processed/training_set_v2.parquet
~360 rows (24 oblasts × 5 years × 3 crops).

Run: uv run python scripts/build_features_v2.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from app.integrations.openmeteo.client import OpenMeteoClient

# Reuse the v1 weather aggregation helpers — they're already tested.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yield_models import WeatherStats, fetch_weather_for_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_features_v2")

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_GEOJSON = ROOT / "data" / "processed" / "oblast_samples.geojson"
S2_PARQUET = ROOT / "data" / "processed" / "oblast_s2_observations.parquet"
YIELD_CSV = ROOT / "data" / "raw" / "usda_ukraine_yield_2017_2023.csv"
OUTPUT = ROOT / "data" / "processed" / "training_set_v2.parquet"


def aggregate_oblast_year(rows: pd.DataFrame) -> dict[str, float | int | None]:
    """Roll up the per-sample weekly NDVI/EVI/NDWI/SAVI into 11 features.

    `rows` must be filtered to one (oblast, year) — multiple samples are
    averaged via per-week median to keep robust to one-off cloud spikes.
    """
    if rows.empty:
        return {}

    # Per ISO-week median across samples.
    weekly = rows.groupby("iso_week").agg(
        ndvi_mean=("ndvi_mean", "median"),
        evi_mean=("evi_mean", "median"),
        ndwi_mean=("ndwi_mean", "median"),
        savi_mean=("savi_mean", "median"),
        observed_on=("observed_on", "min"),
    ).reset_index()
    weekly = weekly.sort_values("iso_week")
    weekly["observed_on"] = pd.to_datetime(weekly["observed_on"])
    weekly["month"] = weekly["observed_on"].dt.month

    def month_mean(m: int) -> float | None:
        sub = weekly[weekly["month"] == m]["ndvi_mean"].dropna()
        return float(sub.mean()) if len(sub) else None

    ndvi_series = weekly["ndvi_mean"].dropna()
    if ndvi_series.empty:
        return {}

    peak_idx = ndvi_series.idxmax()
    peak_value = float(ndvi_series.loc[peak_idx])
    peak_week = int(weekly.loc[peak_idx, "iso_week"]) if peak_idx in weekly.index else 0

    evi_series = weekly["evi_mean"].dropna()
    ndwi_series = weekly["ndwi_mean"].dropna()
    savi_series = weekly["savi_mean"].dropna()

    return {
        "ndvi_peak": round(peak_value, 3),
        "ndvi_peak_week": int(peak_week),
        "ndvi_mean_may": _r3(month_mean(5)),
        "ndvi_mean_june": _r3(month_mean(6)),
        "ndvi_mean_july": _r3(month_mean(7)),
        "ndvi_mean_august": _r3(month_mean(8)),
        "ndvi_integral": round(float(ndvi_series.sum()), 2),
        "ndvi_std": round(float(ndvi_series.std()), 3) if len(ndvi_series) > 1 else 0.0,
        "evi_peak": round(float(evi_series.max()), 3) if len(evi_series) else 0.0,
        "ndwi_min": round(float(ndwi_series.min()), 3) if len(ndwi_series) else 0.0,
        "savi_peak": round(float(savi_series.max()), 3) if len(savi_series) else 0.0,
    }


def _r3(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


async def main() -> int:
    if not S2_PARQUET.exists():
        log.error(
            "Missing %s — run scripts/collect_oblast_s2.py first.\n"
            "If you want a placeholder run for plumbing tests, use --synthesise.",
            S2_PARQUET,
        )
        return 1

    log.info("Loading S2 observations from %s", S2_PARQUET)
    s2 = pd.read_parquet(S2_PARQUET)
    log.info("S2 rows: %d, covering %d (oblast, sample, year) buckets",
             len(s2),
             s2[["oblast", "sample_idx", "year"]].drop_duplicates().shape[0])

    samples = gpd.read_file(SAMPLES_GEOJSON)
    # Centroid per oblast from the sample polygon centroids (avoids re-reading
    # the oblast geojson + matching names; samples are inside oblast bounds).
    oblast_centroids = (
        samples.assign(
            lat=samples.geometry.centroid.y,
            lon=samples.geometry.centroid.x,
        )
        .groupby("oblast")[["lat", "lon"]]
        .mean()
        .reset_index()
    )
    log.info("Oblast centroids: %d", len(oblast_centroids))

    log.info("Loading yield labels from %s", YIELD_CSV)
    yield_df = pd.read_csv(YIELD_CSV)
    log.info("Yield rows: %d", len(yield_df))

    # Build (oblast, year) feature rows from S2.
    feature_rows: list[dict] = []
    for (oblast, year), grp in s2.groupby(["oblast", "year"]):
        agg = aggregate_oblast_year(grp)
        if not agg:
            log.warning("No usable S2 data for %s %d — skipping.", oblast, year)
            continue
        agg["oblast"] = oblast
        agg["year"] = int(year)
        feature_rows.append(agg)

    feat_df = pd.DataFrame(feature_rows)
    log.info("S2-derived feature rows: %d", len(feat_df))

    # Merge centroids.
    feat_df = feat_df.merge(oblast_centroids.rename(columns={"lat": "centroid_lat", "lon": "centroid_lon"}),
                            on="oblast", how="left")

    # Weather: same as v1 — Open-Meteo historical per oblast centroid × year.
    weather_keys = feat_df[["centroid_lat", "centroid_lon", "year"]].to_dict("records")
    weather_input = [
        {"centroid_lat": k["centroid_lat"], "centroid_lon": k["centroid_lon"], "year": int(k["year"])}
        for k in weather_keys
    ]
    log.info("Fetching weather for %d (lat, lon, year) keys...", len(weather_input))
    weather_map = await fetch_weather_for_rows(weather_input)

    def _weather_for_row(r: pd.Series) -> dict[str, float | int]:
        ws: WeatherStats = weather_map.get(
            (float(r["centroid_lat"]), float(r["centroid_lon"]), int(r["year"])),
            WeatherStats(0.0, 0.0, 0, 0),
        )
        return {
            "precip_sum_apr_jul": ws.precip_sum_apr_jul,
            "temp_mean_apr_jul": ws.temp_mean_apr_jul,
            "heat_stress_days": ws.heat_stress_days,
            "drought_dryspells": ws.drought_dryspells,
        }

    weather_cols = feat_df.apply(_weather_for_row, axis=1, result_type="expand")
    feat_df = pd.concat([feat_df, weather_cols], axis=1)

    # Cross with yields — produces (oblast, year, crop) rows.
    final = feat_df.merge(yield_df[["oblast", "year", "crop", "yield_tha"]], on=["oblast", "year"], how="inner")
    log.info("Final training rows: %d (oblast × year × crop)", len(final))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT, index=False)
    log.info("Wrote %s (%d rows, %d cols)", OUTPUT, len(final), final.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
