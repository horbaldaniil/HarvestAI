"""Hierarchical yield model — predicts deviation from oblast mean, not absolute yield.

## Why hierarchical?

For narrow-variance crops (oats, rye, buckwheat, sugar_beet) the v6 R²
collapses (oats: -0.95) because:
  - yield is dominated by a stable oblast mean (~2.7 t/ha for oats)
  - year-to-year variance within an oblast is ~0.3 t/ha (10% of mean)
  - features (NDVI, weather) explain a small fraction of the small variance
  - regressors waste capacity learning the big-picture mean instead of
    focusing on the actual signal

The hierarchical reformulation:
    y_target = yield - oblast_mean(crop)
The model predicts the residual — typically zero-centred [-0.5, +0.5]
in oats — much easier to fit. At inference time:
    y_pred = oblast_mean(crop, iso) + model.predict(features)

Reference: This is a special case of "mixed-effects" / "panel data"
modelling where the oblast-mean acts as a fixed effect. Standard in
agricultural econometrics (Lobell et al. 2011, Schauberger & Gornott 2017).

## What this script does

1. Loads `training_set_v3.parquet`, filters to `is_real_yield=True`.
2. Computes per-(crop, iso_3166_2) historic mean yield using train rows only.
3. Subtracts the oblast-mean → target column `yield_deviation`.
4. Trains 5 model families on the deviation (same chronological split as v6:
   train 2018-19, val 2020, test 2021).
5. At evaluation time, ADDS the oblast-mean back to predicted deviations
   and computes R² against the original `yield_tha`.
6. Saves models as `yield_*_v7h_{crop}.joblib` + metrics → `model_metrics_v7h.json`.

The registry treats v7h as a specialised variant: used by default for the
narrow-variance crops (oats, rye, buckwheat, sugar_beet) where it beats v6.

## Run

    uv run python scripts/train_yield_models_v7_hierarchical.py
    uv run python scripts/train_yield_models_v7_hierarchical.py --crop oats
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
log = logging.getLogger("train_yield_v7h")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
METRICS_OUT = ROOT / "data" / "processed" / "model_metrics_v7h.json"
OBLAST_MEAN_OUT = ROOT / "data" / "processed" / "oblast_mean_yields_v7h.json"

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

V7_TRAIN_YEARS = (2018, 2019)
V7_VAL_YEAR = 2020
V7_TEST_YEAR = 2021

ALL_FAMILIES = ("rf", "xgboost", "lightgbm", "stack")


@dataclass
class Metrics:
    n: int
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    mape: float | None = None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    if len(y_true) == 0:
        return Metrics(n=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else None
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return Metrics(n=len(y_true), rmse=rmse, mae=mae, r2=r2, mape=mape)


def _compute_oblast_means(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """`{crop: {iso: mean_yield}}` — computed ONLY on train years to avoid
    test-set leakage. Crops absent for an oblast use the crop's global mean."""
    out: dict[str, dict[str, float]] = {}
    for crop, sub in train_df.groupby("crop"):
        out[crop] = {}
        global_mean = float(sub["yield_tha"].mean())
        for iso, sub2 in sub.groupby("iso_3166_2"):
            out[crop][iso] = float(sub2["yield_tha"].mean())
        out[crop]["_GLOBAL"] = global_mean
    return out


def _apply_oblast_mean_offset(
    df: pd.DataFrame,
    oblast_means: dict[str, dict[str, float]],
    direction: str = "subtract",
) -> np.ndarray:
    """Subtract or add the oblast-mean. Returns array same length as df."""
    crops = df["crop"].values
    isos = df["iso_3166_2"].values
    means = np.array([
        oblast_means.get(c, {}).get(i, oblast_means.get(c, {}).get("_GLOBAL", 0.0))
        for c, i in zip(crops, isos, strict=False)
    ])
    if direction == "subtract":
        return df["yield_tha"].values - means
    return df["yield_tha"].values + means


def _make_model(family: str):
    """Same hyper-parameters as v6 trainer.

    CatBoost was previously a fourth family and stack base learner;
    removed entirely — the three remaining boosters cover the same
    model diversity for hierarchical residual prediction.
    """
    import lightgbm as lgb
    import xgboost as xgb

    if family == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=42)
    if family == "xgboost":
        return xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                subsample=0.85, colsample_bytree=0.85,
                                random_state=42, n_jobs=-1, tree_method="hist",
                                objective="reg:squarederror", verbosity=0)
    if family == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                                 random_state=42, n_jobs=-1, verbose=-1)
    if family == "stack":
        base = [
            ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                         n_jobs=-1, random_state=42)),
            ("xgb", xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                     learning_rate=0.05, subsample=0.85,
                                     colsample_bytree=0.85, random_state=42,
                                     n_jobs=-1, tree_method="hist",
                                     objective="reg:squarederror", verbosity=0)),
            ("lgbm", lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                                        learning_rate=0.05, random_state=42,
                                        n_jobs=-1, verbose=-1)),
        ]
        return StackingRegressor(
            estimators=base,
            final_estimator=Ridge(alpha=1.0),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=1,
        )
    raise ValueError(f"Unknown family: {family}")


def _save_model(family: str, crop: str, payload: dict) -> None:
    short = {"rf": "rf", "xgboost": "xgb", "lightgbm": "lgbm",
             "stack": "stack"}[family]
    path = MODELS_DIR / f"yield_{short}_{crop}_v7h.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def train_crop(crop: str, df_real: pd.DataFrame,
               oblast_means: dict[str, dict[str, float]],
               families: tuple[str, ...]) -> dict[str, Any]:
    sub = df_real[df_real["crop"] == crop]
    train = sub[sub["year"].isin(V7_TRAIN_YEARS)]
    val = sub[sub["year"] == V7_VAL_YEAR]
    test = sub[sub["year"] == V7_TEST_YEAR]
    if len(train) < 20 or len(test) < 5:
        return {"skipped": True,
                "reason": f"insufficient (train={len(train)}, test={len(test)})"}

    X_train = train[list(FEATURE_NAMES)].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy() if len(val) else np.empty((0, len(FEATURE_NAMES)))
    X_test = test[list(FEATURE_NAMES)].to_numpy()
    # Targets are DEVIATIONS from oblast mean — easier to fit on small data.
    y_train_dev = _apply_oblast_mean_offset(train, oblast_means)
    y_val_dev = _apply_oblast_mean_offset(val, oblast_means) if len(val) else np.empty(0)
    # We keep the ORIGINAL yield for R² evaluation.
    y_test_actual = test["yield_tha"].to_numpy()
    test_means = np.array([
        oblast_means.get(crop, {}).get(i, oblast_means.get(crop, {}).get("_GLOBAL", 0.0))
        for i in test["iso_3166_2"].values
    ])

    X_trainval = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y_trainval_dev = np.concatenate([y_train_dev, y_val_dev]) if len(y_val_dev) else y_train_dev

    out: dict[str, Any] = {
        "skipped": False,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "families": {},
    }
    for fam in families:
        try:
            model = _make_model(fam)
            if fam == "stack":
                model.fit(X_trainval, y_trainval_dev)
            else:
                model.fit(X_train, y_train_dev)
            pred_deviation = model.predict(X_test)
            pred_yield = test_means + pred_deviation
            m = _metrics(y_test_actual, pred_yield)
            out["families"][fam] = {"test": asdict(m)}
            _save_model(fam, crop, {
                "model": model,
                "features": list(FEATURE_NAMES),
                "family": fam, "version": "v7h",
                "oblast_means": oblast_means.get(crop, {}),
                "kind": "hierarchical_oblast_offset",
            })
            log.info("[%s/%s] test R²=%.3f MAE=%.3f",
                     crop, fam,
                     m.r2 if m.r2 is not None else float("nan"),
                     m.mae if m.mae is not None else float("nan"))
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s/%s] failed: %s", crop, fam, exc)
            out["families"][fam] = {"error": str(exc)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", action="append", default=None)
    parser.add_argument("--families", default=",".join(ALL_FAMILIES))
    parser.add_argument("--training-set", type=Path, default=TRAINING_SET)
    args = parser.parse_args()

    if not args.training_set.exists():
        log.error("Missing %s", args.training_set)
        return 1

    df = pd.read_parquet(args.training_set)
    if "is_real_yield" not in df.columns:
        log.error("Parquet missing is_real_yield column.")
        return 1
    df = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Real-only rows: %d", len(df))

    # Compute oblast means on train years ONLY (avoid leakage).
    train_only = df[df["year"].isin(V7_TRAIN_YEARS)]
    oblast_means = _compute_oblast_means(train_only)
    OBLAST_MEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    OBLAST_MEAN_OUT.write_text(
        json.dumps(oblast_means, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved oblast means to %s", OBLAST_MEAN_OUT)

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    crops = args.crop or list(ALL_CROPS)

    summary: dict[str, Any] = {
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(),
            "kind": "hierarchical_oblast_offset",
            "split": f"train={'+'.join(map(str, V7_TRAIN_YEARS))} val={V7_VAL_YEAR} test={V7_TEST_YEAR}",
            "features": list(FEATURE_NAMES),
            "families": list(families),
            "reference": "Lobell et al. 2011, Schauberger & Gornott 2017 — panel-data approach",
        },
        "crops": {},
    }
    for crop in crops:
        log.info("Training hierarchical %s …", crop)
        summary["crops"][crop] = train_crop(crop, df, oblast_means, families)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", METRICS_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
