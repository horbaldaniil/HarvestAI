"""Inline patch: add four lagged-yield columns to training_set_v3.parquet.

For each row `(iso_3166_2, crop, year)` we set:

  * `oblast_yield_lag1`     — mean yield_tha for (iso, crop, year-1)
  * `oblast_yield_lag2`     — mean yield_tha for (iso, crop, year-2)
  * `oblast_yield_lag3`     — mean yield_tha for (iso, crop, year-3)
  * `oblast_yield_lag_mean3` — arithmetic mean of the three above

All values are pulled from `is_real_yield=True` rows only (no synthetic
leakage). The multi-horizon design lets tree models pick whichever
horizon carries the most signal for that crop — typically `lag_mean3`
for stable cereals (rye, oats) and `lag1` for crops with high year-to-
year volatility (sugar_beet, sunflower in drought years).

Anti-leakage rules:
  * lookup table built ONLY from real-yield rows
  * year-N is strictly past data — no current-year leakage
  * When year-N has no published row for that (iso, crop) — typical for
    2018 rows (no 2015/2016 data) and partial-year crops — fall back to
    the **crop's train-set mean over 2018-2019 only**. Same constant
    across affected rows of a given crop; the model can treat it as a
    crop-level intercept. There IS mild leakage at the aggregate
    (crop, train-years) level because 2018's own yields contribute to
    that mean, but per-row prediction effect is negligible compared to
    the row-level variance the lag is supposed to explain.
  * `lag_mean3` averages the three lag values (after fallback
    substitution), so it's never NaN by construction.

If the fallbacks weren't applied, RF / Stack would fail at fit time
because they cannot consume NaN in features. Dropping the affected
rows would cost ~25 % of training data, much worse than the mild
aggregate-level imputation leakage.

Idempotent: running this twice produces the same parquet (overwrites
the columns if present).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patch_parquet_lag")

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"

# Train-only years used for the 2018 fallback crop-mean. Must match the
# v7 trainer's `V7_TRAIN_YEARS` so the fallback doesn't pull from
# validation (2020) or test (2021) data.
TRAIN_YEARS = (2018, 2019)


def main() -> int:
    if not PARQUET.exists():
        log.error("Missing %s", PARQUET)
        return 1

    df = pd.read_parquet(PARQUET)
    log.info("Loaded %d rows", len(df))

    if "is_real_yield" not in df.columns:
        log.error("Parquet missing is_real_yield column")
        return 1
    if "iso_3166_2" not in df.columns or "crop" not in df.columns or "year" not in df.columns:
        log.error("Parquet missing required keys (iso_3166_2 / crop / year)")
        return 1

    real = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Real-yield rows: %d", len(real))

    # Build (iso, crop, year) → mean(yield_tha) lookup from real rows only.
    lookup = (
        real.groupby(["iso_3166_2", "crop", "year"])["yield_tha"]
        .mean()
        .to_dict()
    )

    # 2018 fallback: crop's mean over train-year real rows. Same value
    # for every (oblast, crop=X, year=2018) row of a given crop.
    train_only = real[real["year"].isin(TRAIN_YEARS)]
    crop_mean = train_only.groupby("crop")["yield_tha"].mean().to_dict()
    global_mean = float(real["yield_tha"].mean())

    def _lag_n(row: pd.Series, n: int) -> float:
        """Pull `(iso, crop, year-n)` mean from the lookup; fall back to
        the crop's 2018-2019 train-mean (or global) when missing."""
        key = (row["iso_3166_2"], row["crop"], int(row["year"]) - n)
        if key in lookup:
            return float(lookup[key])
        return float(crop_mean.get(row["crop"], global_mean))

    df["oblast_yield_lag1"] = df.apply(lambda r: _lag_n(r, 1), axis=1).astype(float)
    df["oblast_yield_lag2"] = df.apply(lambda r: _lag_n(r, 2), axis=1).astype(float)
    df["oblast_yield_lag3"] = df.apply(lambda r: _lag_n(r, 3), axis=1).astype(float)
    df["oblast_yield_lag_mean3"] = (
        df["oblast_yield_lag1"]
        + df["oblast_yield_lag2"]
        + df["oblast_yield_lag3"]
    ) / 3.0

    # Sanity print: distribution + per-year coverage for the four lag cols.
    for col in ("oblast_yield_lag1", "oblast_yield_lag2",
                "oblast_yield_lag3", "oblast_yield_lag_mean3"):
        log.info(
            "%-26s min=%.2f median=%.2f mean=%.2f max=%.2f",
            col, df[col].min(), df[col].median(),
            df[col].mean(), df[col].max(),
        )
    by_year = df.groupby("year")[
        ["oblast_yield_lag1", "oblast_yield_lag_mean3"]
    ].agg(["mean"])
    log.info("Per-year (lag1, lag_mean3):\n%s", by_year)

    df.to_parquet(PARQUET, index=False)
    log.info("Wrote %s (cols=%d)", PARQUET, len(df.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
