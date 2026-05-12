"""Train RandomForest + XGBoost v2 yield regressors on real Sentinel-2 features.

Reads `data/processed/training_set_v2.parquet` (produced by
`build_features_v2.py`). Uses the same 17-feature schema and the same
2017-2021 / 2022 / 2023 split as v1 — but the NDVI features are now real
sample-aggregated Sentinel-2 medians, not synthesised.

Per crop, trains:
  • RandomForestRegressor (baseline) → yield_rf_{crop}_v1.joblib
  • XGBRegressor (updated)            → yield_xgb_{crop}_v2.joblib

Quantile XGBoost (q=0.05, q=0.95) for confidence intervals matches the v1
contract so the API/UI don't need changes.

Run: uv run python scripts/train_yield_models_v2.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_v2")

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "processed" / "training_set_v2.parquet"
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "data" / "processed" / "model_metrics_v2.json"

FEATURE_NAMES: tuple[str, ...] = (
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
)

CROPS = ("wheat", "corn", "sunflower")


def split(df: pd.DataFrame, crop: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = df[df["crop"] == crop].copy()
    train = sub[sub["year"] <= 2021]
    val = sub[sub["year"] == 2022]
    test = sub[sub["year"] == 2023]
    return train, val, test


def metric_block(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(n),
    }


def fit_rf(train_X, train_y, val_X, val_y) -> RandomForestRegressor:
    """RandomForest baseline. Small grid for speed — bigger models don't help
    on ~120-row training sets."""
    best = None
    best_val = float("inf")
    for n_estimators in (200, 400):
        for max_depth in (6, None):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(train_X, train_y)
            val_rmse = math.sqrt(mean_squared_error(val_y, model.predict(val_X)))
            if val_rmse < best_val:
                best_val = val_rmse
                best = model
    return best


def fit_xgb(train_X, train_y, val_X, val_y, objective: str = "reg:squarederror",
            quantile_alpha: float | None = None) -> xgb.XGBRegressor:
    """XGBoost with optional quantile loss. Same hyperparams as v1 — what's
    different here is the data."""
    kwargs: dict = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        early_stopping_rounds=30, tree_method="hist",
    )
    if quantile_alpha is not None:
        kwargs["objective"] = "reg:quantileerror"
        kwargs["quantile_alpha"] = quantile_alpha
    model = xgb.XGBRegressor(**kwargs)
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
    return model


def main() -> int:
    if not PARQUET_PATH.exists():
        log.error("Missing %s — run build_features_v2.py first.", PARQUET_PATH)
        return 1

    df = pd.read_parquet(PARQUET_PATH)
    log.info("Loaded %d rows from %s", len(df), PARQUET_PATH)

    metrics_out: dict = {"trained_at": datetime.now(UTC).isoformat(), "models": {}}

    for crop in CROPS:
        train, val, test = split(df, crop)
        if train.empty or test.empty:
            log.warning("Crop %s: insufficient data (train=%d, test=%d) — skipping.",
                        crop, len(train), len(test))
            continue

        X_train = train[list(FEATURE_NAMES)].values
        y_train = train["yield_tha"].values
        X_val = val[list(FEATURE_NAMES)].values
        y_val = val["yield_tha"].values
        X_test = test[list(FEATURE_NAMES)].values
        y_test = test["yield_tha"].values

        log.info("=== %s: train=%d val=%d test=%d ===",
                 crop, len(train), len(val), len(test))

        # RF baseline.
        rf = fit_rf(X_train, y_train, X_val, y_val)
        rf_metrics = {
            "train": metric_block(y_train, rf.predict(X_train), len(train)),
            "val": metric_block(y_val, rf.predict(X_val), len(val)),
            "test": metric_block(y_test, rf.predict(X_test), len(test)),
        }
        log.info("  RF test:  R²=%.3f MAE=%.3f RMSE=%.3f",
                 rf_metrics["test"]["r2"], rf_metrics["test"]["mae"],
                 rf_metrics["test"]["rmse"])

        # XGBoost v2 (point + quantiles for ±confidence).
        xgb_med = fit_xgb(X_train, y_train, X_val, y_val)
        xgb_q05 = fit_xgb(X_train, y_train, X_val, y_val, quantile_alpha=0.05)
        xgb_q95 = fit_xgb(X_train, y_train, X_val, y_val, quantile_alpha=0.95)

        xgb_metrics = {
            "train": metric_block(y_train, xgb_med.predict(X_train), len(train)),
            "val": metric_block(y_val, xgb_med.predict(X_val), len(val)),
            "test": metric_block(y_test, xgb_med.predict(X_test), len(test)),
        }
        log.info("  XGBv2 test: R²=%.3f MAE=%.3f RMSE=%.3f",
                 xgb_metrics["test"]["r2"], xgb_metrics["test"]["mae"],
                 xgb_metrics["test"]["rmse"])

        # Persist.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": rf, "features": list(FEATURE_NAMES), "crop": crop, "version": "v1"},
            MODELS_DIR / f"yield_rf_{crop}_v1.joblib",
        )
        joblib.dump(
            {
                "model": xgb_med, "model_q05": xgb_q05, "model_q95": xgb_q95,
                "features": list(FEATURE_NAMES), "crop": crop, "version": "v2",
            },
            MODELS_DIR / f"yield_xgb_{crop}_v2.joblib",
        )
        # Per-crop meta.json for inspectability.
        for algo, payload in (("rf", rf_metrics), ("xgb", xgb_metrics)):
            tag = "v1" if algo == "rf" else "v2"
            meta_path = MODELS_DIR / f"yield_{algo}_{crop}_{tag}.meta.json"
            meta_path.write_text(json.dumps({
                "crop": crop, "version": tag, "algorithm": algo,
                "metrics": payload,
                "trained_at": metrics_out["trained_at"],
                "feature_names": list(FEATURE_NAMES),
            }, indent=2), encoding="utf-8")

        metrics_out["models"][f"rf_{crop}_v1"] = rf_metrics
        metrics_out["models"][f"xgb_{crop}_v2"] = xgb_metrics

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log.info("Wrote consolidated metrics to %s", METRICS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
