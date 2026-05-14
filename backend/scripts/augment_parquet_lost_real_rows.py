"""Recover real Держстат rows that `build_yield_csv.py` silently dropped.

Discovered after R² ablation: ~71 real (crop, year, oblast) rows from
Держstat bulletins never made it into `training_set_v3.parquet` because
the old `build_yield_csv.py` rejected them when `crop_zones.is_grown()`
returned False — a flag intended to gate **synthesis** but mistakenly
also filtered **measurements**. Affected crops (with deltas vs the
parser's per-(crop,year) count):

    rye        −24 rows  (Steppe-South 6 oblasts × 4 years; published)
    buckwheat  −23 rows  (Steppe-South 6 × 4, some years partial)
    sunflower  −20 rows  (Polissia 4 + Transcarpathia 1 = 5 × 4)
    sugar_beet  −4 rows  (Steppe-South sparse coverage)

build_yield_csv.py is fixed for future rebuilds. This script catches up
the **existing parquet** without re-running the heavy S2-aggregation +
Open-Meteo fetch pipeline:

  1. Read `derzhstat_yield_2017_2023.yaml` → real (crop, year, iso, yield).
  2. Read `training_set_v3.parquet` → existing rows.
  3. For each YAML cell not already present as a real-yield row:
     a. Copy NDVI / weather / soil / zone columns from any existing parquet
        row at the same (iso, year). They're per-cell features and don't
        depend on crop, so any sibling row's values are correct.
     b. Compute the crop-specific features inline (crop_calendar lookups
        + `_crop_features` mirror).
     c. Compute `oblast_yield_lag1` via the same parquet lookup the lag
        patch uses.
     d. Mark `is_real_yield=True`, append.
  4. Write the augmented parquet back.

Idempotent: rerunning the script after it succeeds adds no new rows
because the existence check looks at the `(crop, year, iso_3166_2)`
key and `is_real_yield=True` filter.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_calendar import (  # noqa: E402
    CROP_CALENDAR,
    flowering_in_reference,
    gdd_proxy as _gdd_proxy_helper,
    overlap_with_reference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("augment_parquet_lost_real")

YAML_PATH = ROOT / "data" / "raw" / "derzhstat_yield_2017_2023.yaml"
PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"

TRAIN_YEARS = (2018, 2019)  # for lag1 fallback (mirrors patch_parquet_oblast_yield_lag)

_HEAT_W_IN_REF = 1.5
_HEAT_W_OUT_REF = 0.5

# Approximate ISO week for the 15th of each month — mirror of the
# constant in `app/ml/features.py`. Kept inline to avoid pulling the
# whole `features` module here (it does DB-y things).
_ISO_WEEK_OF_MID_MONTH = {
    1: 3, 2: 7, 3: 11, 4: 16, 5: 20, 6: 24,
    7: 29, 8: 33, 9: 37, 10: 42, 11: 46, 12: 50,
}


def _compute_crop_features(crop: str, weather: dict, veg: dict) -> dict:
    """Inline mirror of `app/ml/features.py:_crop_features`.

    Same arithmetic, but reads its inputs from a parquet-row dict instead
    of a SQLAlchemy/DB shape. Returns the 8 v5+phenology crop features.
    """
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
        }

    overlap = overlap_with_reference(crop)
    season_months = cal.growing_season_length_months
    temp_mean = weather.get("temp_mean_apr_jul")
    precip = weather.get("precip_sum_apr_jul")
    heat_days = weather.get("heat_stress_days")
    drought = weather.get("drought_dryspells")
    heat_w = _HEAT_W_IN_REF if flowering_in_reference(crop) else _HEAT_W_OUT_REF

    ndvi_month_keys = {
        5: "ndvi_mean_may",
        6: "ndvi_mean_june",
        7: "ndvi_mean_july",
        8: "ndvi_mean_august",
    }
    peak_key = ndvi_month_keys.get(cal.peak_month)
    ndvi_at_peak = (
        veg.get(peak_key) if peak_key is not None else veg.get("ndvi_peak")
    )

    expected_peak_week = _ISO_WEEK_OF_MID_MONTH.get(cal.peak_month, 20)
    observed_peak_week = veg.get("ndvi_peak_week")
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
    }


def main() -> int:
    if not YAML_PATH.exists():
        log.error("YAML missing: %s", YAML_PATH)
        return 1
    if not PARQUET.exists():
        log.error("Parquet missing: %s", PARQUET)
        return 1

    cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    real_yields: dict[str, dict[int, dict[str, float]]] = cfg.get("per_oblast_yield", {})
    log.info(
        "YAML cells: %d crop blocks, %d total (crop,year,iso) triples",
        len(real_yields),
        sum(len(by_year) for by_year in real_yields.values() for by_year in [by_year]),
    )

    df = pd.read_parquet(PARQUET)
    log.info("Parquet rows: %d", len(df))

    existing = set(
        zip(
            df.loc[df["is_real_yield"].astype(bool), "iso_3166_2"],
            df.loc[df["is_real_yield"].astype(bool), "crop"],
            df.loc[df["is_real_yield"].astype(bool), "year"].astype(int),
        )
    )
    log.info("Existing real-yield (iso, crop, year) keys: %d", len(existing))

    # Per-(iso, year) shared feature dict, sourced from existing parquet
    # rows. Any crop will do — these columns are crop-agnostic.
    shared_cols = [
        "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
        "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
        "evi_peak", "ndwi_min", "savi_peak",
        # Phase-A round-3: winter NDVI features. Per-(iso, year), so
        # donor-copy works — the rebuilt parquet's donor row has these
        # populated by `aggregate_oblast_year` after the winter S2
        # collection. For pre-2017 yield-years where S2A didn't exist,
        # the aggregator returns None for these — RF/Stack fall back
        # via the standard NaN imputation rule (crop-train-mean).
        "ndvi_mean_october", "ndvi_mean_november", "ndvi_mean_march",
        "spring_regrowth_ndvi_delta",
        "centroid_lat", "centroid_lon",
        "precip_sum_apr_jul", "temp_mean_apr_jul",
        "heat_stress_days", "drought_dryspells",
        # Phase-A round-3: soil-moisture + winter-kill features. Per-
        # (iso, year), so donor-copy works after build_features_v3
        # rebuild.
        "sm_jun_jul_mean", "sm_drydown_days", "winter_kill_days",
        "bdod", "cec", "clay", "phh2o", "sand", "silt", "soc",
        "zone_polissia", "zone_forest_steppe", "zone_steppe_north",
        "zone_steppe_south", "zone_transcarpathia",
        "oblast", "oblast_uk", "zone", "conflict_zone",
    ]
    # Phase-A round-3: BBCH-phase features (9 per crop). Cannot be
    # donor-copied because they're per-(iso, year, crop) and donor may
    # be a different crop. We leave them as None for the recovered
    # rows — RF/Stack fall back via crop-train-mean imputation. Same
    # rule as `oblast_yield_lag1` patch for pre-data years.
    from app.ml.features import BBCH_PHASE_FEATURES  # noqa: E402
    by_iso_year: dict[tuple[str, int], pd.Series] = {}
    for (iso, year), grp in df.groupby(["iso_3166_2", "year"]):
        by_iso_year[(iso, int(year))] = grp.iloc[0]  # any sibling row's shared cols
    log.info("Indexed %d (iso, year) shared-feature donors.", len(by_iso_year))

    # Lag1 lookup — mirror of patch_parquet_oblast_yield_lag.py.
    lookup = (
        df[df["is_real_yield"].astype(bool)]
        .groupby(["iso_3166_2", "crop", "year"])["yield_tha"]
        .mean()
        .to_dict()
    )
    train_only = df[df["is_real_yield"].astype(bool) & df["year"].isin(TRAIN_YEARS)]
    crop_mean = train_only.groupby("crop")["yield_tha"].mean().to_dict()
    global_mean = float(df[df["is_real_yield"].astype(bool)]["yield_tha"].mean())

    def _lag_n(iso: str, crop: str, year: int, n: int) -> float:
        """Same year-N + crop-train-mean fallback rule as
        `patch_parquet_oblast_yield_lag.py:_lag_n`."""
        key = (iso, crop, year - n)
        if key in lookup:
            return float(lookup[key])
        return float(crop_mean.get(crop, global_mean))

    def _lag_features(iso: str, crop: str, year: int) -> dict[str, float]:
        """Compute the 4-feature lag block (lag1 + lag2 + lag3 +
        lag_mean3) so each new row matches the schema produced by the
        full parquet patcher."""
        l1 = _lag_n(iso, crop, year, 1)
        l2 = _lag_n(iso, crop, year, 2)
        l3 = _lag_n(iso, crop, year, 3)
        return {
            "oblast_yield_lag1": l1,
            "oblast_yield_lag2": l2,
            "oblast_yield_lag3": l3,
            "oblast_yield_lag_mean3": (l1 + l2 + l3) / 3.0,
        }

    new_rows: list[dict] = []
    missing_donor = 0
    skipped_already_present = 0

    for crop, by_year in real_yields.items():
        for year, iso_to_yield in by_year.items():
            year_int = int(year)
            for iso, yield_val in iso_to_yield.items():
                if (iso, crop, year_int) in existing:
                    skipped_already_present += 1
                    continue
                donor = by_iso_year.get((iso, year_int))
                if donor is None:
                    # We could fall back to a different year's donor, but
                    # that would silently leak features from a wrong-year
                    # context — better to skip and report.
                    log.info(
                        "No (iso=%s, year=%d) donor in parquet — skipping "
                        "(crop=%s, yield=%.2f t/ha would have been added).",
                        iso, year_int, crop, yield_val,
                    )
                    missing_donor += 1
                    continue
                # Compose the new row from donor shared cols + computed
                # crop features + lag1 + the YAML yield value.
                row: dict = {col: donor[col] for col in shared_cols}
                row["crop"] = crop
                row["year"] = year_int
                row["iso_3166_2"] = iso
                row["yield_tha"] = float(yield_val)
                row["is_real_yield"] = True
                # Flag the synthetic-conditioning columns as
                # not-applicable for these recovered real rows.
                row["weather_used"] = False
                row["weather_factor"] = 1.0
                # Crop-specific features.
                row.update(_compute_crop_features(crop, donor.to_dict(), donor.to_dict()))
                # `crop_flowering_in_aprjul` lives in the parquet too —
                # mirror the boolean for downstream readers that read
                # it (the trainer doesn't, but reports might).
                row["crop_flowering_in_aprjul"] = bool(flowering_in_reference(crop))
                # Lag features (4): lag1 + lag2 + lag3 + lag_mean3.
                row.update(_lag_features(iso, crop, year_int))
                # Phase-A round-3: BBCH features as None (per-crop,
                # donor would be wrong-crop). Tree NaN-imputation in
                # RF/Stack handles via crop-train-mean.
                row.update({k: None for k in BBCH_PHASE_FEATURES})
                # `features_origin` — if the donor has it, mirror.
                if "features_origin" in df.columns:
                    row["features_origin"] = donor.get("features_origin")
                new_rows.append(row)

    log.info(
        "Augment summary: %d new rows to append (skipped already-present=%d, missing-donor=%d).",
        len(new_rows), skipped_already_present, missing_donor,
    )
    if not new_rows:
        log.info("Nothing to add — parquet already covers every YAML cell.")
        return 0

    new_df = pd.DataFrame(new_rows)
    # Match dtype of the existing parquet on columns where pandas might
    # silently promote int → object (year, *_oneHot, conflict_zone).
    for col in new_df.columns:
        if col in df.columns:
            try:
                new_df[col] = new_df[col].astype(df[col].dtype)
            except (TypeError, ValueError):
                pass

    out = pd.concat([df, new_df], ignore_index=True)
    out.to_parquet(PARQUET, index=False)

    # Breakdown of what we added, per crop.
    by_crop_added = new_df.groupby("crop").size().to_dict()
    log.info("Per-crop rows added: %s", by_crop_added)
    log.info(
        "Parquet now: %d total rows (was %d), real-yield rows: %d",
        len(out), len(df),
        int(out["is_real_yield"].sum()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
