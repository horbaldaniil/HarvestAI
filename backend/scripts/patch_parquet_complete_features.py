"""One-shot patcher: bring rebuilt `training_set_v3.parquet` up to the
v7+ schema after `build_features_v3.py` regenerates it.

`build_features_v3.py` produces a parquet with:
  - 11 NDVI (v3 base) + 4 winter NDVI (Phase-A round-3)
  - 4 weather (Apr-Jul) + 3 winter weather (Phase-A round-3)
  - 9 BBCH-phase weather (Phase-A round-3)
  - centroid, oblast metadata, yield label

But the v7+ trainer FEATURE_NAMES expects ALSO:
  - 5 agroclimatic zone one-hots (derived from `oblast → zone`)
  - 8 v5 crop features (overlap / GDD / weighted weather / season length
    / + 2 phenology-audit NDVI alignments)
  - `is_real_yield` / `weather_used` / `weather_factor` / `crop_flowering_in_aprjul`
    metadata flags joined from the yield CSV

This patcher fills all of those. Idempotent — if the columns already
exist, overwrites them with the canonical recomputation.

Run **after** `build_features_v3.py` and **before** `add_soilgrids_features.py`,
`patch_parquet_oblast_yield_lag.py`, and the augment scripts.

    uv run python scripts/build_features_v3.py
    uv run python scripts/patch_parquet_complete_features.py  # ← this
    uv run python scripts/add_soilgrids_features.py
    uv run python scripts/patch_parquet_oblast_yield_lag.py
    uv run python scripts/augment_parquet_lost_real_rows.py
    uv run python scripts/augment_parquet_year2025.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_calendar import (  # noqa: E402
    CROP_CALENDAR,
    flowering_in_reference,
    gdd_proxy as _gdd_proxy_helper,
    overlap_with_reference,
)
from app.data_reference.oblast_names import oblast_by_iso  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patch_parquet_complete")

PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"
YIELD_CSV = ROOT / "data" / "raw" / "usda_ukraine_yield_2017_2023.csv"

ZONES = ("polissia", "forest_steppe", "steppe_north", "steppe_south", "transcarpathia")

# Heat-stress weights — same constants used in
# `app/ml/features.py:_crop_features` and the augment scripts.
_HEAT_W_IN_REF = 1.5
_HEAT_W_OUT_REF = 0.5

# ISO week numbers for the 15th of each month (mirror of features.py).
_ISO_WEEK_OF_MID_MONTH = {
    1: 3, 2: 7, 3: 11, 4: 16, 5: 20, 6: 24,
    7: 29, 8: 33, 9: 37, 10: 42, 11: 46, 12: 50,
}


def _add_zone_onehots(df: pd.DataFrame) -> None:
    """Resolve iso → OblastRef.zone → 5 one-hot ints (in place)."""
    zone_map: dict[str, str] = {}
    for iso in df["iso_3166_2"].dropna().unique():
        ref = oblast_by_iso(iso)
        if ref is not None:
            zone_map[iso] = ref.zone

    df_zone = df["iso_3166_2"].map(zone_map)
    for z in ZONES:
        df[f"zone_{z}"] = (df_zone == z).astype(int)
    log.info(
        "Zone one-hots: %s",
        {z: int((df[f"zone_{z}"] == 1).sum()) for z in ZONES},
    )


def _add_crop_features(df: pd.DataFrame) -> None:
    """v5 crop features (8) + crop_flowering_in_aprjul bool. In place.

    Each crop has its own (peak_month, flowering_window_days, ...) from
    `CROP_CALENDAR`; per-row aggregates derive from the existing
    Apr-Jul weather cols.
    """
    def _row(r: pd.Series) -> dict:
        crop = r["crop"]
        cal = CROP_CALENDAR.get(crop)
        if cal is None:
            return {
                "crop_season_overlap_aprjul": None,
                "gdd_proxy": None,
                "precip_crop_weighted": None,
                "heat_stress_crop_weighted": None,
                "drought_crop_weighted": None,
                "growing_season_length_months": None,
                "ndvi_at_crop_peak_month": None,
                "ndvi_peak_timing_offset_weeks": None,
                "crop_flowering_in_aprjul": False,
            }

        overlap = overlap_with_reference(crop)
        season_months = cal.growing_season_length_months
        temp_mean = r.get("temp_mean_apr_jul")
        precip = r.get("precip_sum_apr_jul")
        heat_days = r.get("heat_stress_days")
        drought = r.get("drought_dryspells")
        heat_w = _HEAT_W_IN_REF if flowering_in_reference(crop) else _HEAT_W_OUT_REF

        ndvi_month_keys = {
            5: "ndvi_mean_may", 6: "ndvi_mean_june",
            7: "ndvi_mean_july", 8: "ndvi_mean_august",
        }
        peak_key = ndvi_month_keys.get(cal.peak_month)
        ndvi_at_peak = r.get(peak_key) if peak_key else r.get("ndvi_peak")

        expected_peak_week = _ISO_WEEK_OF_MID_MONTH.get(cal.peak_month, 20)
        observed_peak_week = r.get("ndvi_peak_week")
        timing_offset = (
            int(observed_peak_week) - expected_peak_week
            if observed_peak_week is not None and not pd.isna(observed_peak_week)
            else None
        )

        return {
            "crop_season_overlap_aprjul": round(overlap, 3),
            "gdd_proxy": _gdd_proxy_helper(crop, temp_mean, season_months),
            "precip_crop_weighted": (
                round(float(precip) * overlap, 1)
                if precip is not None and not pd.isna(precip) else None
            ),
            "heat_stress_crop_weighted": (
                round(float(heat_days) * heat_w, 1)
                if heat_days is not None and not pd.isna(heat_days) else None
            ),
            "drought_crop_weighted": (
                round(float(drought) * overlap, 2)
                if drought is not None and not pd.isna(drought) else None
            ),
            "growing_season_length_months": season_months,
            "ndvi_at_crop_peak_month": (
                round(float(ndvi_at_peak), 3)
                if ndvi_at_peak is not None and not pd.isna(ndvi_at_peak) else None
            ),
            "ndvi_peak_timing_offset_weeks": timing_offset,
            "crop_flowering_in_aprjul": bool(flowering_in_reference(crop)),
        }

    log.info("Computing v5 crop features for %d rows...", len(df))
    crop_cols = df.apply(_row, axis=1, result_type="expand")
    for col in crop_cols.columns:
        df[col] = crop_cols[col]


def _join_yield_metadata(df: pd.DataFrame) -> None:
    """Pull `is_real_yield`, `weather_used`, `weather_factor` from
    the yield CSV that `build_yield_csv.py` produces. Required for the
    trainer's real-only filter and for downstream evaluators.
    """
    if not YIELD_CSV.exists():
        log.warning("Yield CSV %s not found — leaving metadata flags at defaults", YIELD_CSV)
        df["is_real_yield"] = False
        df["weather_used"] = False
        df["weather_factor"] = 1.0
        return

    y = pd.read_csv(
        YIELD_CSV,
        usecols=["iso_3166_2", "year", "crop", "is_real_yield", "weather_used", "weather_factor"],
    )
    before = len(df.columns)
    merged = df.merge(
        y, on=["iso_3166_2", "year", "crop"], how="left",
        suffixes=("", "_csv"),
    )
    # Replace columns post-merge — merge may have created `_csv` suffix
    # cols when both sides already had them (e.g. fresh rebuild scenario).
    for col in ("is_real_yield", "weather_used", "weather_factor"):
        csv_col = f"{col}_csv"
        if csv_col in merged.columns:
            merged[col] = merged[csv_col]
            merged = merged.drop(columns=[csv_col])
    for col, default in (("is_real_yield", False), ("weather_used", False), ("weather_factor", 1.0)):
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)
    df.drop(df.index, inplace=True)
    for col in merged.columns:
        df[col] = merged[col].values
    log.info(
        "Joined yield metadata (is_real_yield, weather_used, weather_factor); col count %d → %d",
        before, len(df.columns),
    )


def main() -> int:
    if not PARQUET.exists():
        log.error("Missing parquet %s", PARQUET)
        return 1
    df = pd.read_parquet(PARQUET)
    log.info("Loaded %d rows, %d cols", len(df), len(df.columns))

    _add_zone_onehots(df)
    _add_crop_features(df)
    _join_yield_metadata(df)

    log.info("Final: %d rows, %d cols", len(df), len(df.columns))
    log.info(
        "Real-yield rows: %d / %d (%.1f %%)",
        int(df["is_real_yield"].astype(bool).sum()),
        len(df),
        100 * df["is_real_yield"].astype(bool).sum() / max(1, len(df)),
    )
    df.to_parquet(PARQUET, index=False)
    log.info("Wrote %s", PARQUET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
