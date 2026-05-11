"""Train one XGBoost yield regressor per crop and persist as joblib.

Methodology (honest about scope — it's a course project):
1. Real Ukrainian yield data (USDA-style, CSV) at oblast×year×crop level.
2. Real historical weather from Open-Meteo for each oblast centroid.
3. Vegetation-index features (NDVI/EVI/NDWI/SAVI summaries) are
   synthesised per row from (yield, crop, year, weather) with realistic
   per-crop phenology + noise. This is documented as a course-project
   simplification — for the master thesis we'd replace this synthesis
   with real Sentinel-2 oblast-aggregated values (504 SH calls ≈ 150 PU).

Splits:
  train: 2017-2021, val: 2022, test: 2023.

Outputs (committed to repo):
  models/yield_xgb_{crop}_v1.joblib   — XGBRegressor + feature schema
  models/yield_xgb_{crop}_v1.meta.json — RMSE/MAE/R² + train date
  data/processed/seasonal_norms.json  — per-crop weekly NDVI normals

Run: uv run python scripts/train_yield_models.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.integrations.openmeteo.client import OpenMeteoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train")

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/raw/usda_ukraine_yield_2017_2023.csv"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data/processed"

FEATURE_NAMES: tuple[str, ...] = (
    # Vegetation indices (synthesised at training time, real at runtime)
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    # Weather (real, from Open-Meteo)
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    # Field metadata
    "centroid_lat", "centroid_lon",
)

# Crop-specific phenology (peak month and typical NDVI peak range).
CROP_PROFILE: dict[str, dict[str, Any]] = {
    "wheat":     {"peak_month": 6, "peak_range": (0.65, 0.92), "ndwi_floor": -0.20},
    "corn":      {"peak_month": 8, "peak_range": (0.75, 0.95), "ndwi_floor": -0.15},
    "sunflower": {"peak_month": 7, "peak_range": (0.55, 0.80), "ndwi_floor": -0.25},
}


@dataclass(frozen=True, slots=True)
class WeatherStats:
    precip_sum_apr_jul: float
    temp_mean_apr_jul: float
    heat_stress_days: int
    drought_dryspells: int


# ────── Weather fetch ───────────────────────────────────────────


async def fetch_weather_for_rows(rows: list[dict]) -> dict[tuple[float, float, int], WeatherStats]:
    """Open-Meteo historical for each unique (lat, lon, year). Returns lookup map."""
    keys = {(r["centroid_lat"], r["centroid_lon"], r["year"]) for r in rows}
    client = OpenMeteoClient()
    out: dict[tuple[float, float, int], WeatherStats] = {}
    try:
        for i, (lat, lon, year) in enumerate(sorted(keys), 1):
            start = date(year, 4, 1)
            end = date(year, 7, 31)
            try:
                daily = await client.fetch_historical(lat, lon, start, end)
            except Exception as exc:  # noqa: BLE001
                log.warning("Open-Meteo failed for (%.2f, %.2f, %d): %s",
                            lat, lon, year, exc)
                out[(lat, lon, year)] = WeatherStats(0.0, 0.0, 0, 0)
                continue

            precip = sum(d.precip_mm for d in daily if d.precip_mm is not None)
            temps = [d.temp_mean_c for d in daily if d.temp_mean_c is not None]
            t_mean = float(np.mean(temps)) if temps else 0.0
            heat = sum(1 for d in daily if (d.temp_max_c or 0) > 30)
            dry = _max_consecutive_dry_days(daily)
            out[(lat, lon, year)] = WeatherStats(
                precip_sum_apr_jul=round(precip, 1),
                temp_mean_apr_jul=round(t_mean, 2),
                heat_stress_days=heat,
                drought_dryspells=dry,
            )
            if i % 25 == 0:
                log.info("Open-Meteo: fetched %d/%d", i, len(keys))
    finally:
        await client.aclose()
    return out


def _max_consecutive_dry_days(daily) -> int:
    longest = 0
    current = 0
    for d in daily:
        if (d.precip_mm or 0) < 1.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ────── Synthesise NDVI features ───────────────────────────────


def synthesize_indices(
    row: dict, weather: WeatherStats, rng: random.Random
) -> dict[str, float]:
    """Produce realistic-looking vegetation-index features.

    Higher yield ⇒ higher NDVI peak + smoother curve. Weather modulates:
    drought → lower NDWI, heat → earlier senescence (peak shifts back).

    These are realistic distributions; for the master's thesis they'd be
    replaced with real Sentinel-2 oblast aggregates.
    """
    crop = row["crop"]
    profile = CROP_PROFILE[crop]
    lo, hi = profile["peak_range"]
    # Yield → peak (linear stretch around national norm)
    yield_factor = max(0.5, min(1.5, row["yield_tha"] / _national_avg(crop)))
    peak_base = lo + (hi - lo) * (yield_factor - 0.5)
    peak = max(0.3, min(0.97, peak_base + rng.gauss(0, 0.04)))

    # Weather modulates
    drought_penalty = max(0.0, (weather.drought_dryspells - 7) * 0.01)
    heat_penalty = max(0.0, (weather.heat_stress_days - 5) * 0.005)
    peak -= (drought_penalty + heat_penalty)

    # Peak ISO-week (peak_month → week, with some jitter)
    peak_week = int(profile["peak_month"] * 4.33) + rng.randint(-2, 2)

    # Monthly means around the peak curve (Gaussian-ish)
    def month_mean(target_month: int) -> float:
        distance = abs(target_month - profile["peak_month"])
        v = peak * math.exp(-((distance / 1.8) ** 2)) + rng.gauss(0, 0.03)
        return max(0.1, min(0.95, v))

    return {
        "ndvi_peak": round(peak, 3),
        "ndvi_peak_week": int(peak_week),
        "ndvi_mean_may": round(month_mean(5), 3),
        "ndvi_mean_june": round(month_mean(6), 3),
        "ndvi_mean_july": round(month_mean(7), 3),
        "ndvi_mean_august": round(month_mean(8), 3),
        "ndvi_integral": round(
            sum(month_mean(m) for m in (5, 6, 7, 8)) * 4.33, 2
        ),
        "ndvi_std": round(rng.uniform(0.05, 0.14), 3),
        "evi_peak": round(max(0.1, min(0.9, peak - 0.08 + rng.gauss(0, 0.04))), 3),
        "ndwi_min": round(profile["ndwi_floor"] - drought_penalty * 2 + rng.gauss(0, 0.04), 3),
        "savi_peak": round(max(0.1, min(0.9, peak - 0.05 + rng.gauss(0, 0.04))), 3),
    }


def _national_avg(crop: str) -> float:
    return {"wheat": 4.07, "corn": 6.74, "sunflower": 2.27}[crop]


# ────── Training pipeline ──────────────────────────────────────


def load_yield_csv() -> list[dict]:
    rows: list[dict] = []
    with CSV_PATH.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "year": int(r["year"]),
                "oblast": r["oblast"],
                "crop": r["crop"],
                "centroid_lat": float(r["centroid_lat"]),
                "centroid_lon": float(r["centroid_lon"]),
                "yield_tha": float(r["yield_tha"]),
            })
    return rows


def build_dataset(
    rows: list[dict], weather: dict
) -> pd.DataFrame:
    rng = random.Random(42)
    records = []
    for r in rows:
        wkey = (r["centroid_lat"], r["centroid_lon"], r["year"])
        wstat = weather.get(wkey, WeatherStats(0.0, 0.0, 0, 0))
        idx = synthesize_indices(r, wstat, rng)
        records.append({
            **idx,
            "precip_sum_apr_jul": wstat.precip_sum_apr_jul,
            "temp_mean_apr_jul": wstat.temp_mean_apr_jul,
            "heat_stress_days": wstat.heat_stress_days,
            "drought_dryspells": wstat.drought_dryspells,
            "centroid_lat": r["centroid_lat"],
            "centroid_lon": r["centroid_lon"],
            "year": r["year"],
            "crop": r["crop"],
            "yield_tha": r["yield_tha"],
        })
    return pd.DataFrame(records)


def train_one_crop(df: pd.DataFrame, crop: str) -> dict:
    sub = df[df["crop"] == crop].copy()
    train = sub[sub["year"] <= 2021]
    val = sub[sub["year"] == 2022]
    test = sub[sub["year"] == 2023]
    log.info("crop=%s rows: train=%d val=%d test=%d",
             crop, len(train), len(val), len(test))

    X_train = train[list(FEATURE_NAMES)].values
    y_train = train["yield_tha"].values
    X_val = val[list(FEATURE_NAMES)].values
    y_val = val["yield_tha"].values
    X_test = test[list(FEATURE_NAMES)].values
    y_test = test["yield_tha"].values

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        early_stopping_rounds=30,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    metrics = {
        "rmse_test": float(math.sqrt(mean_squared_error(y_test, model.predict(X_test)))),
        "mae_test": float(mean_absolute_error(y_test, model.predict(X_test))),
        "r2_test": float(r2_score(y_test, model.predict(X_test))),
        "rmse_val": float(math.sqrt(mean_squared_error(y_val, model.predict(X_val)))),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
    }
    log.info("crop=%s test metrics: RMSE=%.3f MAE=%.3f R2=%.3f",
             crop, metrics["rmse_test"], metrics["mae_test"], metrics["r2_test"])

    return {"model": model, "metrics": metrics}


def save_artifact(crop: str, payload: dict) -> None:
    model = payload["model"]
    metrics = payload["metrics"]
    joblib_path = MODELS_DIR / f"yield_xgb_{crop}_v1.joblib"
    meta_path = MODELS_DIR / f"yield_xgb_{crop}_v1.meta.json"
    joblib.dump(
        {
            "model": model,
            "features": list(FEATURE_NAMES),
            "crop": crop,
            "version": "v1",
        },
        joblib_path,
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "crop": crop, "version": "v1",
            "metrics": metrics,
            "trained_at": datetime.now(UTC).isoformat(),
            "feature_names": list(FEATURE_NAMES),
        }, f, indent=2)
    log.info("Wrote %s and %s", joblib_path.name, meta_path.name)


def build_seasonal_norms(df: pd.DataFrame) -> dict:
    """Per-crop, per-ISO-week NDVI mean/std (used by anomaly detector)."""
    norms: dict[str, dict[str, dict[str, float]]] = {}
    for crop in CROP_PROFILE:
        sub = df[df["crop"] == crop]
        if sub.empty:
            continue
        peak_week = int(CROP_PROFILE[crop]["peak_month"] * 4.33)
        # Approximate per-week NDVI shape using a Gaussian around peak_week
        weeks = {}
        for w in range(15, 41):  # mid-Apr → end-Sep
            distance = abs(w - peak_week)
            # Mean follows Gaussian; std from training-set variability
            mean = float(np.mean(sub["ndvi_peak"])) * math.exp(-((distance / 6) ** 2)) * 0.85
            mean = max(0.1, min(0.92, mean))
            weeks[str(w)] = {"mean": round(mean, 3), "std": 0.07}
        norms[crop] = weeks
    return norms


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading USDA CSV from %s", CSV_PATH)
    rows = load_yield_csv()
    log.info("Loaded %d yield rows", len(rows))

    log.info("Fetching Open-Meteo historical (this takes a minute)...")
    weather = asyncio.run(fetch_weather_for_rows(rows))
    log.info("Got weather for %d (lat,lon,year) combos", len(weather))

    log.info("Building feature dataset (synthesising NDVI features)...")
    df = build_dataset(rows, weather)
    log.info("Dataset shape: %s", df.shape)
    df.to_csv(PROCESSED_DIR / "training_set.csv", index=False)

    for crop in ("wheat", "corn", "sunflower"):
        log.info("=== Training %s ===", crop)
        save_artifact(crop, train_one_crop(df, crop))

    log.info("Building seasonal NDVI norms...")
    norms = build_seasonal_norms(df)
    with (PROCESSED_DIR / "seasonal_norms.json").open("w", encoding="utf-8") as f:
        json.dump(norms, f, indent=2)
    log.info("Wrote seasonal_norms.json with %d crops", len(norms))

    log.info("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
