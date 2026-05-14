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
import functools
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

# Optuna-tuned hyperparameters, when present, override the hand-tuned
# constants embedded in each `_train_*` helper below. See
# `scripts/optuna_tune_v7.py` for the sweep that generates this JSON
# and the rationale for restricting to 5 weak crops.
OPTUNA_BEST_PATH = ROOT / "data" / "processed" / "optuna_best_params_v7.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_yield_v7")

# v7 = real-only-trained production models. Same parquet input as v5
# (training_set_v3.parquet with v4 features + v5 yields), but the
# trainer FILTERS to is_real_yield=True rows before fitting. This
# produces models that match the honest pure-real evaluation:
#   train 2018-2019 (real), val 2020 (real), test 2021 (real)
# No synthetic test confounding.
TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
METRICS_OUT = ROOT / "data" / "processed" / "model_metrics_v7.json"

# Real-only data has years 2018-2021 (with partial 2025). For chronological
# split we use 2018-2019 → train, 2020 → val, 2021 → test. 2025 partial
# data is excluded (not within the chronological window).
V7_TRAIN_YEARS = (2018, 2019)
V7_VAL_YEAR = 2020
V7_TEST_YEAR = 2021

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
    # v7 SoilGrids extension — 7 soil features per oblast (from ISRIC
    # SoilGrids 250m 0-5cm topsoil). Drives potato / sugar_beet / cereals
    # yield prediction by encoding soil texture, fertility, pH.
    "bdod",    # bulk density (kg/dm³)
    "cec",     # cation exchange capacity (cmol(c)/kg)
    "clay",    # % mass
    "phh2o",   # soil pH
    "sand",    # % mass
    "silt",    # % mass
    "soc",     # soil organic carbon (g/kg)
    # Agroclimatic zone one-hot encoding — 5 categories matching the
    # `OblastRef.zone` literal. Lets tree models learn that Lviv-zone
    # barley typically yields ~4.7 t/ha vs Steppe-zone barley ~2.5 t/ha;
    # something the continuous centroid_lat/lon couldn't represent.
    "zone_polissia",
    "zone_forest_steppe",
    "zone_steppe_north",
    "zone_steppe_south",
    "zone_transcarpathia",
    # Lagged regional yield baselines (4 features). Multi-horizon
    # design — tree models pick whichever lag carries the most signal
    # via SHAP-importance. Stable cereals (rye, oats) tend to rely on
    # `lag_mean3` (the 3-year arithmetic mean smooths year-to-year
    # noise); volatile crops like sugar_beet pick `lag1`. Anti-leakage
    # is enforced at parquet-patch time (year-N never includes the
    # current year). See `app/ml/features.py:OBLAST_LAG_FEATURES`.
    "oblast_yield_lag1",
    "oblast_yield_lag2",
    "oblast_yield_lag3",
    "oblast_yield_lag_mean3",
    # Phase-D Option B tried adding `neighbor_yield_lag1_mean` here —
    # reverted because wheat regressed 0.08 R² without a fresh
    # Optuna sweep on the wider schema. Patcher + helper remain in
    # place; re-add this line when re-running the Optuna sweep.
    # Phase-A round-3 (Oct 2025): winter NDVI features. October /
    # November capture autumn establishment of winter crops; March
    # captures spring regrowth post-dormancy. The delta is a winter-
    # survival proxy. See `app/ml/features.py:WINTER_NDVI_FEATURES`.
    "ndvi_mean_october",
    "ndvi_mean_november",
    "ndvi_mean_march",
    "spring_regrowth_ndvi_delta",
    # Phase-A round-3: soil-moisture + winter-kill features. Address
    # the missing tuber-bulking (potato / sugar_beet) and winter-kill
    # (wheat / barley / rye / rapeseed) signals.
    "sm_jun_jul_mean",
    "sm_drydown_days",
    "winter_kill_days",
    # Phase-A round-3: BBCH-aligned weather windows. Replace the
    # one-size-fits-all Apr-Jul aggregate with per-phase splits.
    # Tree models pick whichever phase × variable combination carries
    # the highest signal for each crop via SHAP-importance ranking.
    "precip_early_veg",
    "precip_flowering",
    "precip_grain_fill",
    "temp_mean_flowering",
    "temp_mean_grain_fill",
    "heat_days_flowering",
    "heat_days_grain_fill",
    "drought_days_flowering",
    "drought_days_grain_fill",
    # v5 crop-specific extensions — 6 weather-derived features
    "crop_season_overlap_aprjul",
    "gdd_proxy",
    "precip_crop_weighted",
    "heat_stress_crop_weighted",
    "drought_crop_weighted",
    "growing_season_length_months",
    # Post-phenology-audit extension — 2 crop-aligned NDVI features.
    # `ndvi_at_crop_peak_month` reads the right monthly NDVI slot for
    # each crop's natural peak (wheat May, corn July, sugar_beet
    # August — previously all crops got the same Apr-Jul NDVI mean).
    # `ndvi_peak_timing_offset_weeks` is a signed week-difference
    # vs the expected peak — early peak = drought stress signal,
    # late peak = cold spring signal.
    "ndvi_at_crop_peak_month",
    "ndvi_peak_timing_offset_weeks",
)

ALL_FAMILIES: tuple[str, ...] = ("rf", "xgboost", "lightgbm", "stack")


# Crops where the target yield has a wide value range (>~15× ratio
# max:min) and is right-skewed enough that MSE-loss is dominated by
# the high-value tail. Applying `np.log1p(yield)` before fit and
# `np.expm1(preds)` after predict produces a more uniform error
# distribution and tightens the conformal CI for these crops.
#
# Empirically, the three included here have:
#   sugar_beet  raw range 26-67 т/га   (~2.6× ratio in real Держстат)
#   potato      raw range  8-22 т/га   (~2.8×)
#   corn_silage raw range  7-50 т/га   (~7×, biggest payoff)
#
# Wheat-class crops (1-7 т/га range, ratio ~5-7×) gain less from
# log-target because the absolute differences at the high end are
# already small in т/га, and the post-`expm1` predictions introduce
# their own asymmetric bias that's worse than the raw MSE issue.
LOG_TARGET_CROPS: frozenset[str] = frozenset({
    "sugar_beet", "potato", "corn_silage",
})


@functools.lru_cache(maxsize=1)
def _load_optuna_best() -> dict[str, dict[str, dict[str, Any]]]:
    """`{crop: {family: {param_name: value, ..., cv_r2: float}}}`.

    Returns empty dict if the Optuna sweep hasn't run yet (or the JSON
    is malformed). Trainers then fall back to the hand-tuned hyperparam
    constants embedded in `_train_*` helpers — backward compatible.
    """
    if not OPTUNA_BEST_PATH.exists():
        return {}
    try:
        blob = json.loads(OPTUNA_BEST_PATH.read_text(encoding="utf-8"))
        return blob.get("by_crop", {})
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load Optuna best params: %s", exc)
        return {}


def _optuna_overrides(crop: str, family: str) -> dict[str, Any]:
    """Optuna-tuned hyperparams for `(crop, family)` minus the `cv_r2`
    score key. Empty dict means no overrides — `_train_*` falls back
    to the hand-tuned constants below.
    """
    blob = _load_optuna_best().get(crop, {}).get(family, {})
    return {k: v for k, v in blob.items() if k != "cv_r2"}


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
    """v7 chronological split — adapted to real-data availability.

    v5 used train ≤2021 / val 2022 / test 2023, but 2022 + 2023 are
    synthetic (no Держстат XLSX for those years yet). v7 trains and
    tests entirely on real years:
      train: 2018-2019 (real)
      val:   2020 (real)
      test:  2021 (real)
    """
    train = df[df["year"].isin(V7_TRAIN_YEARS)]
    val = df[df["year"] == V7_VAL_YEAR]
    test = df[df["year"] == V7_TEST_YEAR]
    return train, val, test


# ─── Per-family training functions ─────────────────────────


def _train_rf(X_train: np.ndarray, y_train: np.ndarray,
              crop: str | None = None) -> Any:
    """Hand-tuned defaults; if Optuna has a `(crop, 'rf')` best-params
    entry, those keys override the defaults. `random_state` and
    `n_jobs` are never overridden by Optuna — they're plumbing, not
    hyperparameters."""
    params: dict[str, Any] = dict(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
    )
    if crop:
        params.update(_optuna_overrides(crop, "rf"))
    rf = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
    rf.fit(X_train, y_train)
    return rf


def _train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray | None = None,
                   y_val: np.ndarray | None = None,
                   crop: str | None = None) -> dict[str, Any]:
    """Returns a dict with the point estimator + two quantile models
    for 5th / 95th percentile prediction intervals.

    When `crop` is in the Optuna-swept set, the per-(crop, xgboost)
    best params override the hand-tuned defaults for ALL three sibling
    models (point + q_low + q_high). They share the same architecture
    by design — different objectives but identical capacity.
    """
    import xgboost as xgb

    common: dict[str, Any] = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
    )
    if crop:
        common.update(_optuna_overrides(crop, "xgboost"))
    common.update(random_state=42, n_jobs=-1, tree_method="hist")

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
                    y_val: np.ndarray | None = None,
                    crop: str | None = None) -> Any:
    import lightgbm as lgb

    params: dict[str, Any] = dict(
        n_estimators=400, num_leaves=31,
        learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
    )
    if crop:
        params.update(_optuna_overrides(crop, "lightgbm"))
    model = lgb.LGBMRegressor(
        max_depth=-1, random_state=42, n_jobs=-1, verbose=-1, **params,
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


def _train_stack(X_trainval: np.ndarray, y_trainval: np.ndarray,
                 crop: str | None = None) -> Any:
    """StackingRegressor: RF + XGB + LGBM → Ridge meta-learner.

    Stacking requires each row to receive *exactly one* OOF prediction
    (a partition), so we use plain `KFold(5, shuffle=True)` here, NOT
    `RepeatedKFold` — sklearn's `cross_val_predict` enforces the
    partition constraint and rejects repeated folds. Variance estimation
    via `RepeatedKFold(5×3)` is still applied later in `evaluate_models.py`
    on the final fitted stack — separate concern.

    Note: CatBoost was previously the fourth base learner. It's been
    removed entirely — the package was hard to maintain across Python
    versions and the three remaining boosters cover the same model
    diversity (gradient boosting on histograms via xgb+lgbm, plus
    randomised tree averaging via RF).
    """
    import lightgbm as lgb
    import xgboost as xgb

    # When Optuna best params exist for this crop, the stack base
    # learners inherit them from the standalone (crop, family) sweeps;
    # the meta Ridge alpha comes from the stack-specific sub-study.
    # The hand-tuned defaults below kick in for crops outside the
    # Optuna-sweep set (the 8 strong crops).
    rf_params: dict[str, Any] = dict(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
    )
    xgb_params: dict[str, Any] = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
    )
    lgbm_params: dict[str, Any] = dict(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
    )
    ridge_alpha = 1.0
    if crop:
        rf_params.update(_optuna_overrides(crop, "rf"))
        xgb_params.update(_optuna_overrides(crop, "xgboost"))
        lgbm_params.update(_optuna_overrides(crop, "lightgbm"))
        stack_overrides = _optuna_overrides(crop, "stack")
        if "ridge_alpha" in stack_overrides:
            ridge_alpha = float(stack_overrides["ridge_alpha"])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base = [
        ("rf", RandomForestRegressor(
            n_jobs=-1, random_state=42, **rf_params,
        )),
        ("xgb", xgb.XGBRegressor(
            random_state=42, n_jobs=-1, tree_method="hist",
            objective="reg:squarederror", verbosity=0, **xgb_params,
        )),
        ("lgbm", lgb.LGBMRegressor(
            random_state=42, n_jobs=-1, verbose=-1, **lgbm_params,
        )),
    ]
    stack = StackingRegressor(
        estimators=base,
        final_estimator=Ridge(alpha=ridge_alpha),
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
    return f"yield_{short}_{crop}_v7.joblib"


def _save_payload(payload: dict, family: str, crop: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / _model_filename(family, crop)
    joblib.dump(payload, path)
    return path


def _save_metrics(metrics: dict) -> None:
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", METRICS_OUT)


def _eval_helper(model: Any, X: np.ndarray, y_orig: np.ndarray,
                 *, log_target: bool = False) -> dict:
    """Compute the metric set for one split, always on the **original**
    yield scale.

    When `log_target=True` the model was trained on `np.log1p(y)`, so
    its `.predict()` output lives in log-space; we invert via
    `np.expm1` before computing metrics. The `y_orig` argument is
    therefore always the original-scale yield (т/га) regardless of
    whether the model is log-trained.
    """
    preds = np.asarray(model.predict(X)) if len(y_orig) else np.array([])
    if log_target and len(preds):
        preds = np.expm1(preds)
    return asdict(_compute_metrics(np.asarray(y_orig), preds))


# ─── Main loop ─────────────────────────────────────────────


def train_one_crop(crop: str, df: pd.DataFrame, families: tuple[str, ...],
                   min_rows: int) -> dict[str, Any]:
    sub = df[df["crop"] == crop].copy()
    if len(sub) < min_rows:
        log.warning("Skipping %s — only %d rows (< %d).", crop, len(sub), min_rows)
        return {"skipped": True, "reason": "insufficient_rows", "n": int(len(sub))}

    train, val, test = _split_chronological(sub)
    X_train = train[list(FEATURE_NAMES)].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy() if len(val) else np.empty((0, len(FEATURE_NAMES)))
    X_test = test[list(FEATURE_NAMES)].to_numpy() if len(test) else np.empty((0, len(FEATURE_NAMES)))

    # Original-scale targets — used for metrics + payload flag-driven
    # log inversion. The `y_*_fit` variants are what the models actually
    # learn from; identical to `y_*_orig` for non-log-target crops.
    y_train_orig = train["yield_tha"].to_numpy()
    y_val_orig = val["yield_tha"].to_numpy()
    y_test_orig = test["yield_tha"].to_numpy()
    log_target = crop in LOG_TARGET_CROPS
    if log_target:
        log.info("[%s] applying log1p target transform (LOG_TARGET_CROPS)", crop)
        y_train = np.log1p(y_train_orig)
        y_val = np.log1p(y_val_orig)
        y_test = np.log1p(y_test_orig)
    else:
        y_train = y_train_orig
        y_val = y_val_orig
        y_test = y_test_orig

    X_trainval = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) else y_train

    out: dict[str, Any] = {"skipped": False, "n_train": len(train),
                          "n_val": len(val), "n_test": len(test),
                          "log_target": log_target,
                          "families": {}}

    # Helper to build the payload header consistently across families
    # — every payload now carries `log_target` so the predict path
    # (`app/ml/yield_model.py`) and conformal calibrator know when to
    # apply `np.expm1` to the raw model output.
    def _eval_set(model, *, point: bool = False) -> dict:
        return {
            "train": _eval_helper(model, X_train, y_train_orig, log_target=log_target),
            "val": _eval_helper(model, X_val, y_val_orig, log_target=log_target),
            "test": _eval_helper(model, X_test, y_test_orig, log_target=log_target),
        }

    for family in families:
        log.info("[%s] training %s…", crop, family)
        try:
            if family == "rf":
                model = _train_rf(X_train, y_train, crop=crop)
                meta_payload = {"model": model, "features": list(FEATURE_NAMES),
                                "family": "rf", "version": "v3",
                                "log_target": log_target}
                _save_payload(meta_payload, family, crop)
                metrics = _eval_set(model)

            elif family == "xgboost":
                bundle = _train_xgboost(X_train, y_train, X_val, y_val, crop=crop)
                meta_payload = {"point": bundle["point"], "q_low": bundle["q_low"],
                                "q_high": bundle["q_high"],
                                "features": list(FEATURE_NAMES),
                                "family": "xgboost", "version": "v3",
                                "log_target": log_target}
                _save_payload(meta_payload, family, crop)
                metrics = _eval_set(bundle["point"])

            elif family == "lightgbm":
                model = _train_lightgbm(X_train, y_train, X_val, y_val, crop=crop)
                meta_payload = {"model": model, "features": list(FEATURE_NAMES),
                                "family": "lightgbm", "version": "v3",
                                "log_target": log_target}
                _save_payload(meta_payload, family, crop)
                metrics = _eval_set(model)

            elif family == "stack":
                stack = _train_stack(X_trainval, y_trainval, crop=crop)
                meta_payload = {"model": stack, "features": list(FEATURE_NAMES),
                                "family": "stack", "version": "v3",
                                "log_target": log_target}
                _save_payload(meta_payload, family, crop)
                metrics = _eval_set(stack)
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
    parser.add_argument("--min-rows", type=int, default=30,
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

    # v7: filter to is_real_yield=True only — no synthetic confound.
    if "is_real_yield" not in df.columns:
        log.error("v7 requires 'is_real_yield' column in parquet — run "
                  "build_yield_csv.py + parquet patch first.")
        return 1
    n_before = len(df)
    df = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Filtered to is_real_yield=True: %d / %d rows (%.1f%%)",
             len(df), n_before, 100 * len(df) / n_before)
    if len(df) == 0:
        log.error("No real-yield rows found — ingest Держстат XLSX first.")
        return 1

    log.info("Training crops: %s", crops)
    log.info("Families: %s", families)
    log.info("v7 chronological split: train=%s val=%d test=%d",
             list(V7_TRAIN_YEARS), V7_VAL_YEAR, V7_TEST_YEAR)

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
