"""Patch `training_set_v3.parquet` з SoilGrids soil-property features.

Reads the 7 SoilGrids GeoTIFFs in `data/raw/soilgrids/`, aggregates
soil properties per-oblast (using the 8 sample polygons in
`oblast_samples_v2.geojson`), and joins as new columns to the
training set.

Result: training_set_v3.parquet grows from 36 cols → 43 cols
(adds 7 soil features: bdod, cec, clay, phh2o, sand, silt, soc).

For each (oblast, year) row in the training set, the SAME soil values
are used (soil doesn't change year-to-year). This is intentional —
soil is a "fixed effect" anchoring the model's understanding of WHERE
each row sits geographically.

## Rationale per crop

- **Potato, sugar beet**: tuber/root crops with HIGH soil sensitivity
  (clay/sand for drainage, pH for nutrient availability). Expected
  major R² improvement.
- **Cereals (wheat, corn, barley)**: secondary effect, but still
  meaningful (SOC predicts fertility).
- **Other**: small but consistent gain.

## Run

    uv run python scripts/add_soilgrids_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("soilgrids_features")

ROOT = Path(__file__).resolve().parents[1]
SOILGRIDS_DIR = ROOT / "data" / "raw" / "soilgrids"
SAMPLES_GEOJSON = ROOT / "data" / "processed" / "oblast_samples_v2.geojson"
TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"

# Soil properties we extract — matches the 7 TIFs we downloaded.
SOIL_PROPERTIES: tuple[str, ...] = (
    "bdod",   # bulk density (cg/cm³)
    "cec",    # cation exchange capacity (mmol(c)/kg)
    "clay",   # % mass
    "phh2o",  # soil pH (×10 in SoilGrids — divide on read)
    "sand",   # % mass
    "silt",   # % mass
    "soc",    # soil organic carbon (g/kg)
)

# SoilGrids stores values as int16 scaled — divide by conversion to get
# real units. Reference: https://www.isric.org/explore/soilgrids/faq-soilgrids
# Most properties × 10 (i.e. stored value 270 = real 27.0). pH × 10.
SOIL_SCALING: dict[str, float] = {
    "bdod": 100.0,   # → kg/dm³ (stored ×100)
    "cec":  10.0,    # → cmol(c)/kg (stored ×10)
    "clay": 10.0,    # → %
    "phh2o": 10.0,   # → pH units
    "sand": 10.0,    # → %
    "silt": 10.0,    # → %
    "soc":  10.0,    # → g/kg
}


def _load_oblast_centroids() -> pd.DataFrame:
    """Read sample polygons → 8 centroids per oblast → return as DataFrame
    with columns [oblast, iso_3166_2, sample_idx, lat, lon]."""
    import geopandas as gpd

    if not SAMPLES_GEOJSON.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLES_GEOJSON} — run "
            f"generate_oblast_samples.py --cropland-mask first."
        )
    gdf = gpd.read_file(SAMPLES_GEOJSON)
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x
    return gdf[["oblast", "iso_3166_2", "sample_idx", "lat", "lon"]].copy()


def _sample_soilgrids(
    centroids: pd.DataFrame,
    property_name: str,
) -> pd.DataFrame:
    """Look up SoilGrids raster value at each sample centroid.

    Returns DataFrame [oblast, iso_3166_2, sample_idx, <property>_value].
    """
    import rasterio
    from rasterio.errors import RasterioIOError

    tif = SOILGRIDS_DIR / f"{property_name}_0-5cm_mean_ua.tif"
    if not tif.exists():
        log.warning("Missing %s — skipping property %s", tif, property_name)
        return centroids.assign(**{property_name: np.nan})

    try:
        with rasterio.open(tif) as src:
            # Use `sample` to read pixel at each point.
            coords = list(zip(centroids["lon"], centroids["lat"], strict=False))
            values = [v[0] for v in src.sample(coords)]
    except RasterioIOError as exc:
        log.warning("Could not read %s: %s", tif, exc)
        return centroids.assign(**{property_name: np.nan})

    scaling = SOIL_SCALING.get(property_name, 1.0)
    real_values = [float(v) / scaling if v is not None and v > 0 else None
                   for v in values]
    return centroids.assign(**{property_name: real_values})


def _aggregate_per_oblast(samples_with_soil: pd.DataFrame) -> pd.DataFrame:
    """8 samples per oblast → 1 row per oblast with median soil properties.

    Median is robust to outliers (one sample on rocky/water soil won't
    distort the oblast's central value).
    """
    agg_dict = {p: "median" for p in SOIL_PROPERTIES}
    return (samples_with_soil
            .groupby(["oblast", "iso_3166_2"], as_index=False)
            .agg(agg_dict))


def main() -> int:
    if not TRAINING_SET.exists():
        log.error("Missing %s — run build_features_v3.py first.", TRAINING_SET)
        return 1
    if not SOILGRIDS_DIR.exists():
        log.error("Missing %s — run download_soilgrids.py first.", SOILGRIDS_DIR)
        return 1

    log.info("Loading oblast sample centroids …")
    centroids = _load_oblast_centroids()
    log.info("  %d centroids across %d oblasts",
             len(centroids), centroids["iso_3166_2"].nunique())

    log.info("Sampling SoilGrids rasters at centroids …")
    samples_with_soil = centroids
    for prop in SOIL_PROPERTIES:
        samples_with_soil = _sample_soilgrids(samples_with_soil, prop)
        non_nan = samples_with_soil[prop].notna().sum()
        log.info("  %s: %d / %d non-NaN", prop, non_nan, len(samples_with_soil))

    log.info("Aggregating per-oblast (median across 8 samples) …")
    per_oblast = _aggregate_per_oblast(samples_with_soil)
    log.info("  Final shape: %d oblasts × %d soil properties",
             len(per_oblast), len(SOIL_PROPERTIES))

    # Print sanity check
    log.info("Sample soil values per oblast (first 5):")
    print(per_oblast[["iso_3166_2"] + list(SOIL_PROPERTIES)].head().to_string())

    log.info("Joining to training_set_v3.parquet …")
    df = pd.read_parquet(TRAINING_SET)
    n_before = df.shape[1]
    # Drop existing soil columns if they're there from a previous run.
    df = df.drop(columns=[c for c in SOIL_PROPERTIES if c in df.columns])
    df = df.merge(
        per_oblast[["iso_3166_2"] + list(SOIL_PROPERTIES)],
        on="iso_3166_2",
        how="left",
    )
    log.info("  Parquet: %d rows × %d → %d cols", len(df), n_before, df.shape[1])

    # Verify no rows lost in merge
    if len(df) != pd.read_parquet(TRAINING_SET).shape[0]:
        log.error("Row count mismatch after merge — aborting.")
        return 1

    # Fill any NaN soil values with oblast-zone median (rare edge case)
    for prop in SOIL_PROPERTIES:
        if df[prop].isna().sum() > 0:
            zone_median = df.groupby("zone")[prop].transform("median")
            df[prop] = df[prop].fillna(zone_median)
            still_nan = df[prop].isna().sum()
            if still_nan > 0:
                global_med = df[prop].median()
                df[prop] = df[prop].fillna(global_med)
                log.warning("  %s: filled %d NaN with global median",
                            prop, still_nan)

    df.to_parquet(TRAINING_SET, index=False)
    log.info("Wrote %s (%d rows, %d cols)", TRAINING_SET, len(df), df.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
