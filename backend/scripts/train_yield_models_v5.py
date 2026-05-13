"""Train 5 yield-regressor families per crop on `training_set_v3.parquet`.

Families: RF, XGBoost (point + q=0.05/q=0.95), LightGBM, and a
StackingRegressor combining all three with a Ridge meta-learner. LSTM is
kept at v1 because its 22 × 8 sequence schema is incompatible with the
tabular v3 feature matrix; the methodology UI surfaces it alongside.

## Splits

Chronological per Roberts et al. (2017): year ≤ 2021 → train, 2022 →
validation (for early stopping + meta-CV), 2023 → held-out test. This
matches v2 for back-compat and is what the rest of the UI assumes.

## Stacking

The meta-learner sees out-of-fold (OOF) predictions of the four base
models on the train+val partition (≤ 2022). OOF is produced by
`RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)` — the
Kuhn & Johnson (2013, §15.1) recommendation. The 2023 test set is held
out from base AND meta training; metrics on it are honest.

## Output

  models/yield_{family}_{crop}_v3.joblib    base + stacked artifacts
  data/processed/model_metrics_v3.json      train/val/test RMSE/MAE/R²/MAPE
                                            per (family, crop, split)

Run:
    uv run python scripts/train_yield_models_v3.py
    uv run python scripts/train_yield_models_v3.py --crop wheat --families xgboost,stack
    uv run python scripts/train_yield_models_v3.py --min-rows 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_yield_v5")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
METRICS_OUT = ROOT / "data" / "processed" / "model_metrics_v5.json"

# 23-feature tabular schema (v5 extension).
# v3 had 17 features; v5 adds 6 crop-specific weather-derivation features
# (overlap fraction, GDD proxy, crop-weighted precip/heat/drought, growing
# season length). This lets the model learn that "Apr-Jul heat hits wheat
# harder than corn" without needing fresh Open-Meteo daily data.
FEATURE_NAMES: tuple[str, ...] = (
    # v3 base — 17 features
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
    # v5 crop-specific extensions — 6 new features
    "crop_season_overlap_aprjul",
    "gdd_proxy",
    "precip_crop_weighted",
    "heat_stress_crop_weighted",
    "drought_crop_weighted",
    "growing_season_length_months",
)

ALL_FAMILIES: tuple[str, ...] = ("rf", "xgboost", "lightgbm", "stack")


@dataclass
class SplitMetrics:
    n: int
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    mape: float | None = None
    smape: float | None = None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    """RMSE / MAE / R² / MAPE / sMAPE."""
    if len(y_true) == 0:
        return SplitMetrics(n=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else None
    # MAPE — clip zeros to avoid div/0.
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    smape = float(
        np.mean(
            2 * np.abs(y_true - y_pred) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-6)
        ) * 100
    )
    return SplitMetrics(n=len(y_true), rmse=rmse, mae=mae, r2=r2, mape=mape, smape=smape)


def _split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2021]
    val = df[df["year"] == 2022]
    test = df[df["year"] == 2023]
    return train, val, test


# ─── Per-family training functions ─────────────────────────


def _train_rf(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def _train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray | None = None,
                   y_val: np.ndarray | None = None) -> dict[str, Any]:
    """Returns a dict with the point estimator + two quantile models
    for 5th / 95th percentile prediction intervals."""
    import xgboost as xgb

    common: dict[str, Any] = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, tree_method="hist",
    )

    point = xgb.XGBRegressor(objective="reg:squarederror", **common)
    if X_val is not None:
        point.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        point.fit(X_train, y_train)

    q_low = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.05, **common)
    q_low.fit(X_train, y_train)
    q_high = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.95, **common)
    q_high.fit(X_train, y_train)

    return {"point": point, "q_low": q_low, "q_high": q_high}


def _train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray | None = None,
                    y_val: np.ndarray | None = None) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=-1, num_leaves=31,
        learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    if X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def _train_stack(X_trainval: np.ndarray, y_trainval: np.ndarray) -> Any:
    """StackingRegressor: RF + XGB + LGBM → Ridge meta-learner.

    Stacking requires each row to receive *exactly one* OOF prediction
    (a partition), so we use plain `KFold(5, shuffle=True)` here, NOT
    `RepeatedKFold` — sklearn's `cross_val_predict` enforces the
    partition constraint and rejects repeated folds. Variance estimation
    via `RepeatedKFold(5×3)` is still applied later in `evaluate_models.py`
    on the final fitted stack — separate concern.
    """
    import lightgbm as lgb
    import xgboost as xgb

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base = [
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )),
        ("xgb", xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=-1, tree_method="hist",
            objective="reg:squarederror", verbosity=0,
        )),
        ("lgbm", lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ]
    stack = StackingRegressor(
        estimators=base,
        final_estimator=Ridge(alpha=1.0),
        cv=cv,
        n_jobs=1,  # base learners already parallelise — outer jobs=1 avoids fork bombs
        passthrough=False,
    )
    stack.fit(X_trainval, y_trainval)
    return stack


# ─── Save / metric writer ──────────────────────────────────


def _model_filename(family: str, crop: str) -> str:
    """Stable on-disk naming. `xgb` short form mirrors v1/v2 history."""
    short = {"xgboost": "xgb", "lightgbm": "lgbm", "rf": "rf", "stack": "stack"}[family]
    return f"yield_{short}_{crop}_v5.joblib"


def _save_payload(payload: dict, family: str, crop: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / _model_filename(family, crop)
    joblib.dump(payload, path)
    return path


def _save_metrics(metrics: dict) -> None:
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", METRICS_OUT)


def _eval_helper(model: Any, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute the metric set for one split."""
    preds = np.asarray(model.predict(X)) if len(y) else np.array([])
    return asdict(_compute_metrics(np.asarray(y), preds))


# ─── Main loop ─────────────────────────────────────────────


def train_one_crop(crop: str, df: pd.DataFrame, families: tuple[str, ...],
                   min_rows: int) -> dict[str, Any]:
    sub = df[df["crop"] == crop].copy()
    if len(sub) < min_rows:
        log.warning("Skipping %s — only %d rows (< %d).", crop, len(sub), min_rows)
        return {"skipped": True, "reason": "insufficient_rows", "n": int(len(sub))}

    train, val, test = _split_chronological(sub)
    X_train = train[list(FEATURE_NAMES)].to_numpy()
    y_train = train["yield_tha"].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy() if len(val) else np.empty((0, len(FEATURE_NAMES)))
    y_val = val["yield_tha"].to_numpy()
    X_test = test[list(FEATURE_NAMES)].to_numpy() if len(test) else np.empty((0, len(FEATURE_NAMES)))
    y_test = test["yield_tha"].to_numpy()

    X_trainval = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) else y_train

    out: dict[str, Any] = {"skipped": False, "n_train": len(train),
                          "n_val": len(val), "n_test": len(test),
                          "families": {}}

    for family in families:
        log.info("[%s] training %s…", crop, family)
        try:
            if family == "rf":
                model = _train_rf(X_train, y_train)
                meta_payload = {"model": model, "features": list(FEATURE_NAMES),
                                "family": "rf", "version": "v3"}
                _save_payload(meta_payload, family, crop)
                metrics = {
                    "train": _eval_helper(model, X_train, y_train),
                    "val": _eval_helper(model, X_val, y_val),
                    "test": _eval_helper(model, X_test, y_test),
                }

            elif family == "xgboost":
                bundle = _train_xgboost(X_train, y_train, X_val, y_val)
                meta_payload = {"point": bundle["point"], "q_low": bundle["q_low"],
                                "q_high": bundle["q_high"],
                                "features": list(FEATURE_NAMES),
                                "family": "xgboost", "version": "v3"}
                _save_payload(meta_payload, family, crop)
                point = bundle["point"]
                metrics = {
                    "train": _eval_helper(point, X_train, y_train),
                    "val": _eval_helper(point, X_val, y_val),
                    "test": _eval_helper(point, X_test, y_test),
                }

            elif family == "lightgbm":
                model = _train_lightgbm(X_train, y_train, X_val, y_val)
                meta_payload = {"model": model, "features": list(FEATURE_NAMES),
                                "family": "lightgbm", "version": "v3"}
                _save_payload(meta_payload, family, crop)
                metrics = {
                    "train": _eval_helper(model, X_train, y_train),
                    "val": _eval_helper(model, X_val, y_val),
                    "test": _eval_helper(model, X_test, y_test),
                }

            elif family == "stack":
                stack = _train_stack(X_trainval, y_trainval)
                meta_payload = {"model": stack, "features": list(FEATURE_NAMES),
                                "family": "stack", "version": "v3"}
                _save_payload(meta_payload, family, crop)
                metrics = {
                    "train": _eval_helper(stack, X_train, y_train),
                    "val": _eval_helper(stack, X_val, y_val),
                    "test": _eval_helper(stack, X_test, y_test),
                }
            else:
                log.warning("Unknown family %s — skipping.", family)
                continue

            out["families"][family] = metrics
            log.info("[%s/%s] test RMSE=%.3f R²=%s MAPE=%s",
                     crop, family,
                     metrics["test"].get("rmse") or float("nan"),
                     f"{metrics['test'].get('r2'):.3f}" if metrics["test"].get("r2") is not None else "n/a",
                     f"{metrics['test'].get('mape'):.1f}%" if metrics["test"].get("mape") is not None else "n/a")
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to train %s/%s: %s", crop, family, exc)
            out["families"][family] = {"error": str(exc)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", action="append", default=None,
                        help="restrict to one or more crops (repeatable)")
    parser.add_argument("--families", default=",".join(ALL_FAMILIES),
                        help=f"comma list of families to train (default {','.join(ALL_FAMILIES)})")
    parser.add_argument("--min-rows", type=int, default=50,
                        help="skip any crop with fewer rows (default 50)")
    parser.add_argument("--training-set", type=Path, default=TRAINING_SET,
                        help=f"override path to training set parquet (default {TRAINING_SET})")
    args = parser.parse_args()

    if not args.training_set.exists():
        log.error("Missing %s — run scripts/build_features_v3.py "
                  "or scripts/build_training_set_v3_synthetic.py first.", args.training_set)
        return 1

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    crops = args.crop or list(ALL_CROPS)

    log.info("Reading %s", args.training_set)
    df = pd.read_parquet(args.training_set)
    origin = df.get("features_origin", pd.Series(["unknown"] * len(df))).iloc[0]
    log.info("Loaded %d rows (features_origin=%s)", len(df), origin)
    log.info("Training crops: %s", crops)
    log.info("Families: %s", families)

    summary: dict[str, Any] = {
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(),
            "training_set": str(args.training_set),
            "features_origin": str(origin),
            "feature_names": list(FEATURE_NAMES),
            "splits": {"train": "<=2021", "val": "2022", "test": "2023"},
            "families": list(families),
        },
        "crops": {},
    }
    for crop in crops:
        summary["crops"][crop] = train_one_crop(crop, df, families, args.min_rows)

    _save_metrics(summary)
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
