"""Inline patch: add cross-crop "neighbor yield" feature.

For each row `(iso_3166_2, crop, year)` we compute
`neighbor_yield_lag1_mean` — the mean of `yield_tha` across **all
other 12 crops** in the same oblast for `year-1`.

## Why this feature

Per-row own-crop lag (`oblast_yield_lag1`) tells us "this oblast's
wheat was X last year". The neighbor-crop average tells us "this
oblast's *cereals/oilseeds/tubers in general* did Y last year".
The difference between own-crop lag and neighbor lag captures
**which crops the oblast specialises in** vs **which crops fail
when the year is hard**.

Concept from panel-data agronomy literature (Schauberger & Gornott
2017, Sci. Rep. — multi-crop yield panel models). They show that
including neighbor-crop signal lifts R² for narrow-variance crops
(rye, oats, buckwheat) by 0.03-0.08 because year-to-year regional
weather effects propagate across crops.

## Anti-leakage rules

  * Lookup table built ONLY from `is_real_yield=True` rows.
  * Strict `year-1` — no current-year cross-crop leakage. (The
    current-year row could see year-1 data for ALL crops without
    self-leakage since the row's own crop is excluded.)
  * **The row's own crop is excluded from the neighbor average.**
    Otherwise we'd just re-introduce own-crop lag1 with extra
    noise — that's already a separate feature.
  * When `(iso, year-1)` has no real rows at all (typical for
    2018 rows that need 2017 data, before the chronological split's
    train window), fall back to `global_mean` so downstream tree
    models with no NaN tolerance don't crash.

## Output

One new column: `neighbor_yield_lag1_mean`. Idempotent — running
the script twice yields the same parquet.

## Schema impact

  * V7_FEATURE_NAMES: 57 → 58
  * V8_FEATURE_NAMES: 70 → 71

## Run

    uv run python scripts/patch_parquet_neighbor_yield.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patch_parquet_neighbor_yield")

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"

# Same train-window as the lag patcher and v7 trainer so the fallback
# value doesn't pull from val/test years.
TRAIN_YEARS = (2018, 2019)


def main() -> int:
    if not PARQUET.exists():
        log.error("Missing %s", PARQUET)
        return 1

    df = pd.read_parquet(PARQUET)
    log.info("Loaded %d rows, %d cols", len(df), len(df.columns))

    required = {"is_real_yield", "iso_3166_2", "crop", "year", "yield_tha"}
    missing = required - set(df.columns)
    if missing:
        log.error("Parquet missing columns: %s", missing)
        return 1

    real = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Real-yield rows: %d", len(real))

    # Build (iso, year, crop) → yield lookup from real rows only. Used
    # twice below: once to compute the global per-(iso, year) crop list,
    # once to subtract out the row's own crop yield from the average.
    real_lookup: dict[tuple[str, int, str], float] = (
        real.groupby(["iso_3166_2", "year", "crop"])["yield_tha"]
        .mean()
        .to_dict()
    )

    # Per-(iso, year) → {crop: yield} aggregator. We don't precompute
    # a flat mean here because we need to exclude the row's own crop
    # at lookup time — that means the per-(iso, year-1) crop list is
    # the right granularity.
    by_iso_year: dict[tuple[str, int], dict[str, float]] = {}
    for (iso, year, crop), yld in real_lookup.items():
        by_iso_year.setdefault((iso, year), {})[crop] = float(yld)

    # Train-year-only global fallback. The most-likely missing case is
    # 2018 rows that need 2017 data — they fall back to the
    # train-window global mean (4-5 t/ha across crops).
    train_only = real[real["year"].isin(TRAIN_YEARS)]
    global_mean = float(train_only["yield_tha"].mean())
    log.info("Fallback global_mean (train-only): %.3f t/ha", global_mean)

    def _neighbor_mean(row: pd.Series) -> float:
        """Mean yield across all crops EXCEPT the row's own crop in
        `(iso, year-1)`. Fall back to global mean when no neighbor data."""
        key = (row["iso_3166_2"], int(row["year"]) - 1)
        crop_yields = by_iso_year.get(key) or {}
        # Drop the row's own crop — it's already a separate feature.
        # If the key isn't present (own crop never had a year-1 row),
        # this is a no-op.
        neighbor_yields = [
            v for c, v in crop_yields.items() if c != row["crop"]
        ]
        if not neighbor_yields:
            return global_mean
        return float(sum(neighbor_yields) / len(neighbor_yields))

    df["neighbor_yield_lag1_mean"] = df.apply(_neighbor_mean, axis=1).astype(float)

    col = "neighbor_yield_lag1_mean"
    log.info(
        "%-26s min=%.2f median=%.2f mean=%.2f max=%.2f std=%.3f",
        col, df[col].min(), df[col].median(),
        df[col].mean(), df[col].max(), df[col].std(),
    )

    # Sanity check: correlation with own-crop lag1 should be high but
    # not 1.0 — the neighbor mean is similar in years where ALL crops
    # do well, but diverges when one crop has crop-specific stress.
    if "oblast_yield_lag1" in df.columns:
        corr = df[["oblast_yield_lag1", "neighbor_yield_lag1_mean"]].corr().iloc[0, 1]
        log.info("Pearson corr(own_lag1, neighbor_lag1_mean) = %.3f", corr)
        if corr > 0.95:
            log.warning(
                "corr > 0.95 — neighbor feature may be redundant with own-lag1; "
                "watch SHAP attribution post-retrain",
            )

    df.to_parquet(PARQUET, index=False)
    log.info("Wrote %s (cols=%d)", PARQUET, len(df.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
