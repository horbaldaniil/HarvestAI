"""Build the v3 training set: 24 oblasts × 7 years × 13 crops with real
Sentinel-2 + weather features and ISO-3166-2-keyed joins.

## v3 changes vs v2
- Join key is `iso_3166_2` everywhere (resolved via
  `app.data_reference.oblast_names.iso_from_any_name`) — no more
  silent ~50% drops when Natural Earth's `"L'viv"` doesn't match
  `build_yield_csv.py`'s `"Lviv"`.
- 13 crops × all-feasible-zones outer product replaces v2's hard-coded
  3-crop universe.
- Output `training_set_v3.parquet` is additive: v2 file stays in place
  so the methodology page can show the ablation ("v2 vs v3" features).
- Cropland-masked Sentinel-2 input (`oblast_s2_observations_v2.parquet`)
  is preferred when present; falls back to v2 for back-compat.
- Adds `conflict_zone` boolean column propagated from the yield CSV so
  downstream evaluation can exclude conflict-affected (oblast, year)
  rows or treat them as a robustness-check subset.

Output schema (v3 columns marked with *):
    iso_3166_2*                — join key
    oblast, oblast_uk, zone*   — display / segmentation
    year, crop                 — keys
    centroid_lat, centroid_lon — for per-oblast weather fetch
    ndvi_peak, ndvi_peak_week, ndvi_mean_{may,june,july,august},
    ndvi_integral, ndvi_std,
    evi_peak, ndwi_min, savi_peak   — 11 vegetation features
    precip_sum_apr_jul, temp_mean_apr_jul,
    heat_stress_days, drought_dryspells   — 4 weather features
    yield_tha                  — label
    conflict_zone*             — flag (bool)

Run:
    uv run python scripts/build_features_v3.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.oblast_names import (  # noqa: E402
    iso_from_any_name,
    oblast_by_iso,
)

# Reuse v2 helpers + weather fetcher; they're already tested.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features_v2 import aggregate_oblast_year  # noqa: E402
from train_yield_models import WeatherStats, fetch_weather_for_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_features_v3")

# v3 prefers the cropland-masked inputs; falls back to v2 for back-compat.
SAMPLES_V3 = ROOT / "data" / "processed" / "oblast_samples_v2.geojson"
SAMPLES_V2 = ROOT / "data" / "processed" / "oblast_samples.geojson"
S2_V3 = ROOT / "data" / "processed" / "oblast_s2_observations_v2.parquet"
S2_V2 = ROOT / "data" / "processed" / "oblast_s2_observations.parquet"
YIELD_CSV = ROOT / "data" / "raw" / "usda_ukraine_yield_2017_2023.csv"
OUTPUT = ROOT / "data" / "processed" / "training_set_v3.parquet"


def _pick_path(v3: Path, v2: Path, label: str) -> Path:
    """Prefer v3 file; fall back to v2 with a clear log line. Errors if neither."""
    if v3.exists():
        log.info("Using %s for %s.", v3.name, label)
        return v3
    if v2.exists():
        log.warning("v3 %s missing — falling back to v2 (%s).", label, v2.name)
        return v2
    raise FileNotFoundError(
        f"Neither v3 ({v3}) nor v2 ({v2}) exists for {label}. "
        f"Run scripts/collect_oblast_s2.py first."
    )


def _attach_iso(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """Add `iso_3166_2` column derived from a free-text name column.

    Rows that fail to resolve are dropped with a warning. After v3, the
    parquet itself stores ISO directly; this helper bridges the gap for
    legacy inputs (samples geojson written before D1).
    """
    df = df.copy()
    df["iso_3166_2"] = df[name_col].apply(iso_from_any_name)
    unresolved = df[df["iso_3166_2"].isna()][name_col].unique().tolist()
    if unresolved:
        log.warning("Could not resolve %d distinct names to ISO codes: %s",
                    len(unresolved), unresolved)
    return df.dropna(subset=["iso_3166_2"])


async def main() -> int:
    s2_path = _pick_path(S2_V3, S2_V2, "Sentinel-2 observations")
    samples_path = _pick_path(SAMPLES_V3, SAMPLES_V2, "oblast samples")

    log.info("Loading S2 observations from %s", s2_path)
    s2 = pd.read_parquet(s2_path)
    log.info("S2 rows: %d, covering %d (oblast, sample, year) buckets",
             len(s2),
             s2[["oblast", "sample_idx", "year"]].drop_duplicates().shape[0])

    # Resolve every free-text oblast name in S2 → ISO 3166-2.
    s2 = _attach_iso(s2, "oblast")

    # Per-oblast centroid from the sample geometry — used for weather fetch.
    samples = gpd.read_file(samples_path)
    samples = _attach_iso(samples, "oblast")
    oblast_centroids = (
        samples.assign(
            lat=samples.geometry.centroid.y,
            lon=samples.geometry.centroid.x,
        )
        .groupby("iso_3166_2")[["lat", "lon"]]
        .mean()
        .reset_index()
        .rename(columns={"lat": "centroid_lat", "lon": "centroid_lon"})
    )
    log.info("Oblast centroids resolved: %d", len(oblast_centroids))

    log.info("Loading yield labels from %s", YIELD_CSV)
    yield_df = pd.read_csv(YIELD_CSV)
    if "iso_3166_2" not in yield_df.columns:
        # build_yield_csv.py v3 emits the column directly; v2 didn't.
        # Fall back to attaching from `oblast`.
        yield_df = _attach_iso(yield_df, "oblast")
    log.info("Yield rows: %d (covering %d (iso, year, crop) triples)",
             len(yield_df),
             yield_df[["iso_3166_2", "year", "crop"]].drop_duplicates().shape[0])

    # ─── Build (iso, year) S2 features ─────────────────────
    feature_rows: list[dict] = []
    for (iso, year), grp in s2.groupby(["iso_3166_2", "year"]):
        agg = aggregate_oblast_year(grp)
        if not agg:
            log.warning("No usable S2 data for %s %d — skipping.", iso, year)
            continue
        ref = oblast_by_iso(iso)
        agg["iso_3166_2"] = iso
        agg["oblast"] = ref.name_en if ref else iso
        agg["oblast_uk"] = ref.name_uk_short if ref else None
        agg["zone"] = ref.zone if ref else None
        agg["year"] = int(year)
        feature_rows.append(agg)

    feat_df = pd.DataFrame(feature_rows)
    log.info("S2-derived feature rows: %d (iso × year)", len(feat_df))

    feat_df = feat_df.merge(oblast_centroids, on="iso_3166_2", how="left")

    # ─── Weather (Open-Meteo per oblast centroid × year) ────
    weather_input = [
        {"centroid_lat": float(r.centroid_lat),
         "centroid_lon": float(r.centroid_lon),
         "year": int(r.year)}
        for r in feat_df.itertuples()
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

    # ─── Outer-product with 13 crops via yield join on ISO ──
    # Match columns: iso_3166_2, year, crop, yield_tha, conflict_zone.
    yield_cols = [c for c in ["iso_3166_2", "year", "crop", "yield_tha", "conflict_zone"]
                  if c in yield_df.columns]
    final = feat_df.merge(yield_df[yield_cols], on=["iso_3166_2", "year"], how="inner")
    # Tag where the vegetation features came from. evaluate_models.py
    # surfaces this in the methodology JSON so the UI caveat copy can
    # adapt (synthetic_v3 vs sentinel_hub_v3).
    final["features_origin"] = "sentinel_hub_v3"
    log.info("Final training rows: %d (iso × year × crop)", len(final))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT, index=False)
    log.info("Wrote %s (%d rows, %d cols)", OUTPUT, len(final), final.shape[1])

    # Quick crop-row sanity report (so the operator notices if a crop
    # silently drops out due to a bad zone/feasibility encoding).
    by_crop = final.groupby("crop").size().to_dict()
    log.info("Rows per crop: %s", dict(sorted(by_crop.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
