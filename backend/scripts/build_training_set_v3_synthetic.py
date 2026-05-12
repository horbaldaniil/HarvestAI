"""Generate a synthetic `training_set_v3.parquet` for offline development.

When real Sentinel-2 oblast data isn't available yet (Phase-3 wall-clock
collection in progress), this script produces a parquet with the **same
schema** as `build_features_v3.py` so the ML pipeline (Phase 4+) can be
developed and evaluated end-to-end. The vegetation indices are
synthesised per row from a documented yield → NDVI mapping that's
explicitly tagged in the output (`features_origin == "synthetic_v3"`)
so the evaluator never confuses synthetic results with real ones.

What's *real* in the output:
  - `yield_tha` from `usda_ukraine_yield_2017_2023.csv` (Phase-1
    Держстат-grounded values).
  - `iso_3166_2`, `oblast`, `zone`, `crop`, `year`, centroids.
  - `conflict_zone` flag.

What's *synthesised* with a documented formula:
  - 11 vegetation features (ndvi_*, evi_*, ndwi_*, savi_*).
  - 4 weather features (precip_sum_apr_jul, temp_mean_apr_jul,
    heat_stress_days, drought_dryspells) — deterministic from oblast
    centroid + year + a hash seed, so re-runs are reproducible without
    Open-Meteo network access.

When real S2 features become available, `build_features_v3.py` overwrites
this file with `features_origin == "sentinel_hub_v3"` and downstream
training scripts pick up the upgrade transparently — only the
methodology-page caveat copy needs updating.

Run:
    uv run python scripts/build_training_set_v3_synthetic.py
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402
from app.data_reference.oblast_names import (  # noqa: E402
    agricultural_oblasts,
    oblast_by_iso,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_training_set_v3_synthetic")

YIELD_CSV = ROOT / "data" / "raw" / "usda_ukraine_yield_2017_2023.csv"
OUTPUT = ROOT / "data" / "processed" / "training_set_v3.parquet"

# Per-crop phenology — drives the synthesised NDVI curves. Sourced from
# Ukrainian agronomy norms (Інститут рослинництва ім. Юр'єва, NAAS):
#   - peak_month: when ndvi_peak typically occurs
#   - peak_range: (lo, hi) of plausible ndvi_peak values t/ha-aware
#   - ndwi_floor: minimum NDWI under healthy water status
#   - national_avg_yield: anchor for the yield → NDVI mapping
CROP_PROFILE: dict[str, dict[str, float | tuple[float, float]]] = {
    "wheat":       {"peak_month": 6, "peak_range": (0.65, 0.92), "ndwi_floor": -0.20, "national_avg": 4.07},
    "corn":        {"peak_month": 8, "peak_range": (0.75, 0.95), "ndwi_floor": -0.15, "national_avg": 6.74},
    "sunflower":   {"peak_month": 7, "peak_range": (0.55, 0.80), "ndwi_floor": -0.25, "national_avg": 2.27},
    "soybean":     {"peak_month": 7, "peak_range": (0.65, 0.88), "ndwi_floor": -0.18, "national_avg": 2.29},
    "rapeseed":    {"peak_month": 5, "peak_range": (0.70, 0.92), "ndwi_floor": -0.18, "national_avg": 2.75},
    "barley":      {"peak_month": 6, "peak_range": (0.60, 0.88), "ndwi_floor": -0.22, "national_avg": 3.30},
    "rye":         {"peak_month": 5, "peak_range": (0.55, 0.80), "ndwi_floor": -0.22, "national_avg": 2.79},
    "oats":        {"peak_month": 6, "peak_range": (0.55, 0.80), "ndwi_floor": -0.22, "national_avg": 2.49},
    "buckwheat":   {"peak_month": 7, "peak_range": (0.50, 0.75), "ndwi_floor": -0.25, "national_avg": 1.17},
    "peas":        {"peak_month": 6, "peak_range": (0.60, 0.85), "ndwi_floor": -0.20, "national_avg": 2.32},
    "sugar_beet":  {"peak_month": 8, "peak_range": (0.80, 0.95), "ndwi_floor": -0.10, "national_avg": 48.0},
    "potato":      {"peak_month": 7, "peak_range": (0.70, 0.90), "ndwi_floor": -0.15, "national_avg": 17.7},
    "corn_silage": {"peak_month": 8, "peak_range": (0.75, 0.95), "ndwi_floor": -0.15, "national_avg": 24.0},
}


def _hash_seed(*parts: str) -> int:
    """Deterministic int from string parts — for per-row reproducibility."""
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _synthesise_weather(iso: str, year: int) -> dict[str, float | int]:
    """Per-oblast/year weather aggregates with realistic Ukrainian ranges.

    Range anchors per agro-climatic zone:
      - Polissia:        precip 220-380 mm, temp 16-19 °C, drought 3-9, heat 1-5
      - Forest-steppe:   precip 200-340 mm, temp 17-20 °C, drought 4-12, heat 2-8
      - Steppe north:    precip 170-300 mm, temp 18-21 °C, drought 6-15, heat 3-12
      - Steppe south:    precip 130-250 mm, temp 19-23 °C, drought 8-18, heat 5-18
      - Transcarpathia:  precip 280-460 mm, temp 16-19 °C, drought 1-6, heat 0-4
    """
    seed = _hash_seed("weather", iso, str(year))
    rng = random.Random(seed)
    ref = oblast_by_iso(iso)
    zone = ref.zone if ref else "forest_steppe"
    profile = {
        "polissia":       (300, 80,  17.5, 1.5, 6, 3, 3, 2),
        "forest_steppe":  (270, 70,  18.5, 1.5, 8, 4, 5, 3),
        "steppe_north":   (235, 65,  19.5, 1.5, 10, 4, 7, 4),
        "steppe_south":   (190, 60,  21.0, 2.0, 13, 5, 11, 6),
        "transcarpathia": (370, 90,  17.5, 1.5, 3, 2, 2, 1),
    }[zone]
    precip_mean, precip_sd, temp_mean, temp_sd, drought_mean, drought_sd, heat_mean, heat_sd = profile

    # 2017-2023 weather anomalies (very rough): 2018, 2020, 2022 drier-than-avg;
    # 2019, 2021, 2023 closer to normal.
    year_offset = {2017: 0, 2018: -25, 2019: 5, 2020: -15, 2021: 10, 2022: -20, 2023: 5}.get(year, 0)
    precip = max(80, rng.gauss(precip_mean + year_offset, precip_sd))
    temp = rng.gauss(temp_mean, temp_sd)
    drought = max(0, int(rng.gauss(drought_mean, drought_sd)))
    heat = max(0, int(rng.gauss(heat_mean, heat_sd)))

    return {
        "precip_sum_apr_jul": round(precip, 1),
        "temp_mean_apr_jul": round(temp, 2),
        "heat_stress_days": int(heat),
        "drought_dryspells": int(drought),
    }


def _synthesise_indices(crop: str, yield_tha: float, weather: dict,
                        iso: str, year: int) -> dict[str, float | int]:
    """Yield-conditioned NDVI synthesis. Same shape as build_features_v3.py."""
    profile = CROP_PROFILE[crop]
    rng = random.Random(_hash_seed("indices", iso, str(year), crop))

    lo, hi = profile["peak_range"]
    national = profile["national_avg"]
    # Yield → NDVI peak: linearly map yield_tha ∈ [0.5×, 1.5× national]
    # to NDVI ∈ [lo, hi], with noise.
    yield_factor = max(0.5, min(1.5, yield_tha / national))
    peak_base = lo + (hi - lo) * (yield_factor - 0.5)
    peak = max(0.30, min(0.97, peak_base + rng.gauss(0, 0.035)))

    # Weather penalties.
    drought_penalty = max(0.0, (weather["drought_dryspells"] - 7) * 0.010)
    heat_penalty = max(0.0, (weather["heat_stress_days"] - 5) * 0.005)
    peak -= drought_penalty + heat_penalty
    peak = max(0.30, peak)

    peak_month: int = int(profile["peak_month"])  # type: ignore[assignment]
    peak_week = int(peak_month * 4.33) + rng.randint(-2, 2)

    def month_mean(m: int) -> float:
        d = abs(m - peak_month)
        v = peak * math.exp(-((d / 1.8) ** 2)) + rng.gauss(0, 0.025)
        return max(0.10, min(0.95, v))

    ndvi_floor = profile["ndwi_floor"]
    return {
        "ndvi_peak": round(peak, 3),
        "ndvi_peak_week": int(peak_week),
        "ndvi_mean_may": round(month_mean(5), 3),
        "ndvi_mean_june": round(month_mean(6), 3),
        "ndvi_mean_july": round(month_mean(7), 3),
        "ndvi_mean_august": round(month_mean(8), 3),
        "ndvi_integral": round(sum(month_mean(m) for m in (5, 6, 7, 8)) * 4.33, 2),
        "ndvi_std": round(rng.uniform(0.05, 0.14), 3),
        "evi_peak": round(max(0.10, min(0.90, peak - 0.08 + rng.gauss(0, 0.03))), 3),
        "ndwi_min": round(float(ndvi_floor) - drought_penalty * 2 + rng.gauss(0, 0.03), 3),
        "savi_peak": round(max(0.10, min(0.90, peak - 0.05 + rng.gauss(0, 0.03))), 3),
    }


def main() -> int:
    if not YIELD_CSV.exists():
        log.error("Missing %s — run scripts/build_yield_csv.py first.", YIELD_CSV)
        return 1

    log.info("Reading yield CSV: %s", YIELD_CSV)
    yield_df = pd.read_csv(YIELD_CSV)
    log.info("Loaded %d yield rows", len(yield_df))

    # Build a per-(iso, year) weather cache so each oblast/year gets one
    # consistent weather record across all crops trained on it.
    weather_cache: dict[tuple[str, int], dict] = {}
    rows: list[dict] = []

    iso_set = {o.iso_3166_2 for o in agricultural_oblasts()}
    crops_set = set(ALL_CROPS)

    for _, r in yield_df.iterrows():
        iso = r["iso_3166_2"]
        year = int(r["year"])
        crop = r["crop"]
        if iso not in iso_set or crop not in crops_set:
            continue

        key = (iso, year)
        if key not in weather_cache:
            weather_cache[key] = _synthesise_weather(iso, year)
        weather = weather_cache[key]

        indices = _synthesise_indices(crop, float(r["yield_tha"]), weather, iso, year)

        ref = oblast_by_iso(iso)
        rows.append({
            "iso_3166_2": iso,
            "oblast": ref.name_en if ref else iso,
            "oblast_uk": ref.name_uk_short if ref else None,
            "zone": ref.zone if ref else None,
            "year": year,
            "crop": crop,
            "centroid_lat": ref.centroid_lat if ref else None,
            "centroid_lon": ref.centroid_lon if ref else None,
            **indices,
            **weather,
            "yield_tha": float(r["yield_tha"]),
            "conflict_zone": bool(r.get("conflict_zone", False)),
            "features_origin": "synthetic_v3",
        })

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    log.info("Wrote %s — %d rows × %d cols", OUTPUT, len(df), df.shape[1])

    # Per-crop summary so the operator notices catastrophic drops.
    by_crop = df.groupby("crop").size().to_dict()
    log.info("Rows per crop: %s", dict(sorted(by_crop.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
