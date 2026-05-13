"""Honest "real-to-real" v5 evaluation (Phase-4 of v5 plan).

## Why a separate evaluator?

The default `evaluate_models.py --version v5` measures R² on the
chronological test split (2023) — which is **synthetic** in our v5
parquet, because Держстат XLSX files are only available for 2018-2021.
Training on 4 real years + 3 synthetic years and testing on a
synthetic year gives a misleading number: the model is pulled toward
two different distributions.

This script measures the **honest** real-to-real generalisation:
  - Filter parquet to `is_real_yield=True` rows only (1066 rows for
    2018-2021).
  - Chronological split: train on 2018-2019 (real), val on 2020 (real),
    test on 2021 (real).
  - Train 4 model families (RF, XGBoost, LightGBM, Stack).
  - Report test R²/MAE/RMSE plus the standard scientific metric panel.

The pure-real numbers are what we cite in the thesis defence; the
mixed v5 numbers stay reported for back-compat (separate JSON file).

## Output

Writes `data/processed/evaluation_v5_real_only.json` with the same
structure as `evaluation_v3.json` / `evaluation_v4.json` — leaderboard
endpoint auto-picks v5 → v5-real-only → v4 → v3 chain when present.

## Run

    uv run python scripts/evaluate_v5_real_only.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_v5_real_only")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
EVAL_OUT = ROOT / "data" / "processed" / "evaluation_v5_real_only.json"

# Same 23-feature schema as v4/v5 trainers.
FEATURE_NAMES: tuple[str, ...] = (
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
    "crop_season_overlap_aprjul",
    "gdd_proxy",
    "precip_crop_weighted",
    "heat_stress_crop_weighted",
    "drought_crop_weighted",
    "growing_season_length_months",
)

TRAIN_YEARS = (2018, 2019)
VAL_YEAR = 2020
TEST_YEAR = 2021


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else None
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"n": len(y_true), "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def _build_families():
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import StackingRegressor
    from sklearn.model_selection import KFold

    base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=42)),
        ("xgb", xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 random_state=42, n_jobs=-1, tree_method="hist",
                                 objective="reg:squarederror", verbosity=0)),
        ("lgbm", lgb.LGBMRegressor(n_estimators=400, num_leaves=31,
                                   learning_rate=0.05, random_state=42,
                                   n_jobs=-1, verbose=-1)),
    ]
    return {
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                    n_jobs=-1, random_state=42),
        "xgboost": xgb.XGBRegressor(n_estimators=400, max_depth=5,
                                    learning_rate=0.05, subsample=0.85,
                                    colsample_bytree=0.85, random_state=42,
                                    n_jobs=-1, tree_method="hist",
                                    objective="reg:squarederror", verbosity=0),
        "lightgbm": lgb.LGBMRegressor(n_estimators=400, num_leaves=31,
                                      learning_rate=0.05, random_state=42,
                                      n_jobs=-1, verbose=-1),
        "stack": StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=1,
        ),
    }


def _per_oblast_residuals(df: pd.DataFrame, y_pred, y_true) -> list[dict]:
    res = pd.DataFrame({
        "iso_3166_2": df["iso_3166_2"].values,
        "oblast": df["oblast"].values,
        "abs_residual": np.abs(y_true - y_pred),
    })
    return (res.groupby(["iso_3166_2", "oblast"])
              .agg(mean_abs_residual=("abs_residual", "mean"),
                   max_abs_residual=("abs_residual", "max"),
                   n=("abs_residual", "size"))
              .reset_index()
              .to_dict("records"))


def evaluate_crop(df: pd.DataFrame, crop: str) -> dict:
    sub = df[df["crop"] == crop]
    train = sub[sub["year"].isin(TRAIN_YEARS)]
    val = sub[sub["year"] == VAL_YEAR]
    test = sub[sub["year"] == TEST_YEAR]
    if len(train) < 10 or len(test) < 5:
        return {"skipped": True, "reason": f"insufficient_real_rows (train={len(train)}, test={len(test)})"}

    X_train = train[list(FEATURE_NAMES)].to_numpy()
    y_train = train["yield_tha"].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy()
    y_val = val["yield_tha"].to_numpy()
    X_test = test[list(FEATURE_NAMES)].to_numpy()
    y_test = test["yield_tha"].to_numpy()

    # Stack uses train+val pool.
    X_trainval = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) else y_train

    families = _build_families()
    out: dict = {
        "skipped": False,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "split": f"train={'+'.join(map(str, TRAIN_YEARS))} val={VAL_YEAR} test={TEST_YEAR}",
        "families": {},
    }

    for name, model in families.items():
        try:
            if name == "stack":
                model.fit(X_trainval, y_trainval)
            else:
                model.fit(X_train, y_train)
            pred_test = model.predict(X_test)
            test_metrics = _metrics(y_test, pred_test)
            entry = {
                "test": test_metrics,
                "per_oblast_residuals": _per_oblast_residuals(test, pred_test, y_test),
                "pred_vs_actual": [
                    {"actual": float(a), "predicted": float(p),
                     "oblast": str(o), "iso_3166_2": str(iso)}
                    for a, p, o, iso in zip(
                        y_test.tolist(), pred_test.tolist(),
                        test["oblast"].tolist(), test["iso_3166_2"].tolist(),
                        strict=False,
                    )
                ],
            }
            out["families"][name] = entry
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s/%s] failed: %s", crop, name, exc)
            out["families"][name] = {"error": str(exc)}

    return out


def main() -> int:
    if not TRAINING_SET.exists():
        log.error("Missing %s", TRAINING_SET)
        return 1

    df = pd.read_parquet(TRAINING_SET)
    real = df[df.get("is_real_yield", False).astype(bool)].copy()
    log.info("Filtered to real-yield rows: %d / %d", len(real), len(df))
    log.info("Years available: %s", sorted(real["year"].unique().tolist()))

    summary: dict = {
        "metadata": {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "training_set": str(TRAINING_SET),
            "evaluation_kind": "pure_real_chronological",
            "split": f"train={'+'.join(map(str, TRAIN_YEARS))} val={VAL_YEAR} test={TEST_YEAR}",
            "feature_names": list(FEATURE_NAMES),
            "n_features": len(FEATURE_NAMES),
            "families": ["rf", "xgboost", "lightgbm", "stack"],
            "n_real_rows": int(len(real)),
        },
        "crops": {},
    }
    for crop in sorted(ALL_CROPS):
        log.info("Evaluating %s …", crop)
        summary["crops"][crop] = evaluate_crop(real, crop)

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s (%.1f KB)", EVAL_OUT, EVAL_OUT.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
