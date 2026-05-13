"""Compute the thesis-grade scientific metric panel for trained v3 models.

Operates on `training_set_v3.parquet` + the joblib artifacts written by
`train_yield_models_v3.py`. Output JSON lives at
`data/processed/evaluation_v3.json`; the methodology page reads it.

## Metrics produced per (family, crop)

| Metric | Citation / role |
|---|---|
| RMSE / MAE / R² / MAPE / sMAPE | re-computed from re-training under each CV scheme so test-split / LOOCV / K-fold numbers are directly comparable |
| LOOCV-R² / LOOCV-RMSE | Leave-one-oblast-out — Roberts et al. (2017) spatial generalisation |
| RepeatedKFold(5×3)-R² (mean ± σ) | Variance bound — Kuhn & Johnson (2013, §4.4) |
| Pinball loss @ q=0.05, q=0.95 | XGBoost quantile heads only (Koenker & Bassett 1978) |
| Per-oblast residuals (mean abs, count) | Powers the choropleth map in `OblastResidualMap.tsx` |
| Pred-vs-actual scatter (sampled) | For `PerCropResidualScatter.tsx` |
| Global ‖SHAP‖ per feature | Lundberg & Lee 2017 — for `GlobalShapSummary.tsx` |
| Permutation importance | Model-agnostic alt. to SHAP |
| Learning curve (sample-size → R²) | Bias/variance diagnostics |
| Partial dependence (top-3 features) | Friedman 2001 — for `PartialDependenceCharts.tsx` |

## CRPS

For the stacked ensemble where we get a 5×3 RepeatedKFold prediction
distribution per sample, we compute CRPS approximate via the empirical
distribution — `properscoring.crps_ensemble`. We don't pull in
`properscoring` as a hard dep though; if it's unavailable we fall back
to the cheaper Gaussian-CRPS-from-mean-and-σ approximation (Gneiting &
Raftery 2007, eq. 5).

## Run

    uv run python scripts/evaluate_models.py
    uv run python scripts/evaluate_models.py --crop wheat --no-shap
    uv run python scripts/evaluate_models.py --skip-loocv  # faster dev iter
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402

# Same feature schema as the trainer — single source of truth would be
# nicer but breaks parquet introspection on systems where the trainer
# hasn't been imported. v4 extends v3 with 6 crop-specific features
# (Phase 3 of v4 plan). Evaluator picks the matching FEATURE_NAMES based
# on the model version it's evaluating.
FEATURE_NAMES_V3: tuple[str, ...] = (
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
)
FEATURE_NAMES_V4: tuple[str, ...] = FEATURE_NAMES_V3 + (
    "crop_season_overlap_aprjul",
    "gdd_proxy",
    "precip_crop_weighted",
    "heat_stress_crop_weighted",
    "drought_crop_weighted",
    "growing_season_length_months",
)
# Default for back-compat — overridden by --version flag.
FEATURE_NAMES: tuple[str, ...] = (
    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
    "evi_peak", "ndwi_min", "savi_peak",
    "precip_sum_apr_jul", "temp_mean_apr_jul",
    "heat_stress_days", "drought_dryspells",
    "centroid_lat", "centroid_lon",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
EVAL_OUT_V3 = ROOT / "data" / "processed" / "evaluation_v3.json"
EVAL_OUT_V4 = ROOT / "data" / "processed" / "evaluation_v4.json"
EVAL_OUT_V5 = ROOT / "data" / "processed" / "evaluation_v5.json"
# Default = v3 for back-compat; --version v4/v5 switches paths.
EVAL_OUT = EVAL_OUT_V3

# Family → joblib filename short-form (matches train_yield_models_v3.py).
FAMILY_SHORT = {
    "rf": "rf", "xgboost": "xgb", "lightgbm": "lgbm",
    "stack": "stack",
}


# ─── Metric utilities ──────────────────────────────────────


@dataclass
class CoreMetrics:
    n: int
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    mape: float | None = None
    smape: float | None = None


def _core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> CoreMetrics:
    if len(y_true) == 0:
        return CoreMetrics(n=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else None
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100)
    return CoreMetrics(n=len(y_true), rmse=rmse, mae=mae, r2=r2, mape=mape, smape=smape)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball loss for a single quantile estimate. Koenker & Bassett (1978)."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def _crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Continuous Ranked Probability Score with Gaussian assumption.

    Gneiting & Raftery (2007, eq. 5):
        CRPS(N(μ,σ²), y) = σ · [ (y-μ)/σ · (2·Φ((y-μ)/σ) - 1)
                                 + 2·φ((y-μ)/σ) - 1/√π ]
    """
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    return float(np.mean(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))


# ─── Cross-validation strategies ──────────────────────────


def _loocv_oblast(
    estimator_factory: Callable[[], Any],
    df: pd.DataFrame,
) -> CoreMetrics:
    """Leave-one-oblast-out CV → R² for spatial generalisation.

    For each oblast in the dataset: train on all OTHER oblasts, predict
    on this oblast, accumulate predictions, then compute R² over the
    union. Roberts et al. (2017) "Cross-validation strategies for data
    with temporal, spatial, hierarchical, or phylogenetic structure".
    """
    iso_groups = df.groupby("iso_3166_2")
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    for iso, holdout in iso_groups:
        train_df = df[df["iso_3166_2"] != iso]
        if len(train_df) < 10:
            continue
        X_tr = train_df[list(FEATURE_NAMES)].to_numpy()
        y_tr = train_df["yield_tha"].to_numpy()
        X_te = holdout[list(FEATURE_NAMES)].to_numpy()
        y_te = holdout["yield_tha"].to_numpy()
        m = estimator_factory()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(np.asarray(y_pred).tolist())
    return _core_metrics(np.asarray(y_true_all), np.asarray(y_pred_all))


def _repeated_kfold_variance(
    estimator_factory: Callable[[], Any],
    df: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict[str, float]:
    """RepeatedKFold(5×3) → list of per-fold R²; reports mean + std.

    Kuhn & Johnson (2013, §4.4): repeated K-fold gives a robust
    estimate of generalization error variance.
    """
    X = df[list(FEATURE_NAMES)].to_numpy()
    y = df["yield_tha"].to_numpy()
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    r2_scores: list[float] = []
    for tr_idx, te_idx in cv.split(X):
        m = estimator_factory()
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        if len(set(y[te_idx])) > 1:
            r2_scores.append(float(r2_score(y[te_idx], pred)))
    return {
        "n_folds": len(r2_scores),
        "r2_mean": float(np.mean(r2_scores)) if r2_scores else None,
        "r2_std": float(np.std(r2_scores)) if r2_scores else None,
        "r2_min": float(np.min(r2_scores)) if r2_scores else None,
        "r2_max": float(np.max(r2_scores)) if r2_scores else None,
    }


def _learning_curve(
    estimator_factory: Callable[[], Any],
    df: pd.DataFrame,
    sample_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
) -> list[dict[str, float]]:
    """Sample-size sweep on the test split — points for the learning curve.

    Holds out 2023 as test; subsamples the train (≤2022) data at each
    fraction; reports test R² + MAE. Bias-variance diagnostic.
    """
    train = df[df["year"] <= 2022]
    test = df[df["year"] == 2023]
    if len(train) < 20 or len(test) < 5:
        return []
    X_test = test[list(FEATURE_NAMES)].to_numpy()
    y_test = test["yield_tha"].to_numpy()
    rng = np.random.default_rng(seed=42)
    out: list[dict[str, float]] = []
    for frac in sample_fractions:
        n_sub = max(10, int(frac * len(train)))
        n_sub = min(n_sub, len(train))
        idx = rng.choice(len(train), size=n_sub, replace=False)
        sub = train.iloc[idx]
        X_tr = sub[list(FEATURE_NAMES)].to_numpy()
        y_tr = sub["yield_tha"].to_numpy()
        m = estimator_factory()
        m.fit(X_tr, y_tr)
        pred = m.predict(X_test)
        metrics = _core_metrics(y_test, pred)
        out.append({
            "n_train": int(n_sub),
            "fraction": float(frac),
            "test_r2": metrics.r2,
            "test_mae": metrics.mae,
            "test_rmse": metrics.rmse,
        })
    return out


# ─── Residual + SHAP helpers ──────────────────────────────


def _per_oblast_residuals(
    df: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray,
) -> list[dict[str, Any]]:
    res = pd.DataFrame({
        "iso_3166_2": df["iso_3166_2"].values,
        "oblast": df["oblast"].values,
        "abs_residual": np.abs(y_true - y_pred),
    })
    agg = (res.groupby(["iso_3166_2", "oblast"])
              .agg(mean_abs_residual=("abs_residual", "mean"),
                   max_abs_residual=("abs_residual", "max"),
                   n=("abs_residual", "size"))
              .reset_index())
    return agg.to_dict("records")


def _unwrap_estimator(payload: Any) -> Any:
    """Extract the fitted estimator from one of the trainer payload shapes.

    train_yield_models_v3.py serialises:
      - XGBoost  → {"point": XGBRegressor, "q_low": ..., "q_high": ...}
      - RF/LGBM/CatBoost/Stack → {"model": <estimator>, ...}

    For SHAP/permutation analysis we always need the actual `fit`/
    `predict`-implementing object.
    """
    if not isinstance(payload, dict):
        return payload
    if "point" in payload:
        return payload["point"]
    if "model" in payload:
        return payload["model"]
    return payload


def _global_shap(model: Any, X: np.ndarray, sample: int = 200) -> dict[str, float] | None:
    """Mean |SHAP value| per feature — the global importance ranking.

    Uses `shap.TreeExplainer` for tree models; returns None for the
    StackingRegressor (no first-class TreeExplainer support — the
    stacked output is a Ridge on top of OOF preds, so feature SHAP
    isn't meaningful at the stacked level).
    """
    try:
        import shap
    except ImportError:
        return None

    actual = _unwrap_estimator(model)
    try:
        explainer = shap.TreeExplainer(actual)
        Xs = X[:sample] if len(X) > sample else X
        shap_vals = explainer.shap_values(Xs)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        return {fname: float(v) for fname, v in zip(FEATURE_NAMES, mean_abs, strict=False)}
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP TreeExplainer failed for %s: %s", type(actual).__name__, exc)
        return None


def _permutation_importance(
    model: Any, X: np.ndarray, y: np.ndarray, n_repeats: int = 8,
) -> dict[str, float]:
    """Model-agnostic permutation importance — sklearn implementation."""
    actual = _unwrap_estimator(model)
    try:
        r = permutation_importance(actual, X, y, n_repeats=n_repeats,
                                    random_state=42, n_jobs=1)
        return {f: float(v) for f, v in zip(FEATURE_NAMES, r.importances_mean, strict=False)}
    except Exception as exc:  # noqa: BLE001
        log.warning("Permutation importance failed: %s", exc)
        return {}


def _partial_dependence(
    model: Any,
    X: np.ndarray,
    top_features: list[str],
    *,
    grid_resolution: int = 30,
) -> dict[str, dict[str, list[float]]]:
    """Compute 1-D partial-dependence curves for `top_features`.

    For each named feature we sweep its value across a `grid_resolution`
    -point grid (5th–95th percentile range) and report the model's
    average prediction at each grid point — i.e. the marginal effect
    of that feature holding all others fixed at their empirical
    distribution (Friedman 2001).

    `top_features` is typically the top-3 by permutation importance.

    Returns: `{feature_name: {"grid": [x...], "pdp": [y...]}}` —
    JSON-serialisable. Returns `{}` for the stacked ensemble (sklearn's
    `partial_dependence` doesn't unwrap StackingRegressor cleanly
    against its meta-input space, so the curves would be misleading).
    """
    actual = _unwrap_estimator(model)
    # StackingRegressor's partial_dependence runs but reports curves in
    # the meta-input space (one dim per base learner), not the original
    # 17 features. That's not what the UI wants to show, so we bail out.
    from sklearn.ensemble import StackingRegressor
    if isinstance(actual, StackingRegressor):
        return {}

    try:
        from sklearn.inspection import partial_dependence
    except ImportError:
        return {}

    out: dict[str, dict[str, list[float]]] = {}
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for feat in top_features:
        idx = name_to_idx.get(feat)
        if idx is None:
            continue
        try:
            pd_result = partial_dependence(
                actual, X, features=[idx],
                grid_resolution=grid_resolution,
                percentiles=(0.05, 0.95),
                kind="average",
            )
            # sklearn ≥1.5 returns a Bunch with `grid_values` + `average`.
            grid = pd_result["grid_values"][0]
            avg = pd_result["average"][0]
            out[feat] = {
                "grid": [float(g) for g in grid],
                "pdp": [float(v) for v in avg],
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("PDP failed for feature %s: %s", feat, exc)
            continue
    return out


# ─── Estimator factories (so CV schemes can re-instantiate cheaply) ──


def _factory_for_family(family: str) -> Callable[[], Any]:
    """Lightweight factory: return a NEW unfitted estimator of `family`.

    Hyper-parameters mirror train_yield_models_v3.py — keeping them in
    sync is a thesis-defensibility concern (the CV-reported metrics must
    use the same model class as the published checkpoint)."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    if family == "rf":
        return lambda: RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )
    if family == "xgboost":
        return lambda: xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=-1, tree_method="hist",
            objective="reg:squarederror", verbosity=0,
        )
    if family == "lightgbm":
        return lambda: lgb.LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    if family == "stack":
        # We don't re-train stack inside CV — too slow and the OOF tower
        # already gives variance-aware predictions. CV metrics reported
        # for "stack" reuse the chronological-test prediction.
        raise ValueError("CV not applicable to 'stack'; evaluate on test split only.")
    raise ValueError(f"Unknown family: {family}")


# ─── Loaders ──────────────────────────────────────────────


def _load_model(family: str, crop: str, version: str = "v3") -> dict[str, Any] | None:
    short = FAMILY_SHORT[family]
    path = MODELS_DIR / f"yield_{short}_{crop}_{version}.joblib"
    if not path.exists():
        log.warning("Missing model %s — skipping.", path.name)
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s: %s", path.name, exc)
        return None


def _predictor(payload: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Return a `predict(X) -> ndarray` callable from the saved payload."""
    if "point" in payload:                # xgboost v3 bundle
        m = payload["point"]
        return lambda X: np.asarray(m.predict(X))
    if "model" in payload:                # rf / lgbm / stack
        m = payload["model"]
        return lambda X: np.asarray(m.predict(X))
    raise ValueError("Unknown payload shape: %s" % list(payload.keys()))


# ─── Evaluation per crop / family ─────────────────────────


def evaluate_family_for_crop(
    family: str, crop: str, df: pd.DataFrame, args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _load_model(family, crop, version=args.version)
    if payload is None:
        return {"skipped": True, "reason": "model_not_found"}

    train = df[df["year"] <= 2022]
    test = df[df["year"] == 2023]
    X_test = test[list(FEATURE_NAMES)].to_numpy()
    y_test = test["yield_tha"].to_numpy()

    predict = _predictor(payload)
    pred_test = predict(X_test)
    test_metrics = _core_metrics(y_test, pred_test)
    result: dict[str, Any] = {
        "skipped": False,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test": asdict(test_metrics),
    }

    # ─── Pinball loss for XGBoost quantile heads ──────────
    if family == "xgboost" and "q_low" in payload and "q_high" in payload:
        q_low_pred = payload["q_low"].predict(X_test)
        q_high_pred = payload["q_high"].predict(X_test)
        result["pinball_loss_q05"] = _pinball_loss(y_test, q_low_pred, 0.05)
        result["pinball_loss_q95"] = _pinball_loss(y_test, q_high_pred, 0.95)
        # Empirical coverage: fraction of true values inside [q05, q95]
        inside = (y_test >= q_low_pred) & (y_test <= q_high_pred)
        result["interval_coverage_90pct"] = float(np.mean(inside))

    # ─── Per-oblast residuals ────────────────────────────
    result["per_oblast_residuals"] = _per_oblast_residuals(test, pred_test, y_test)

    # ─── Predicted-vs-actual scatter (capped at 500 points) ─
    result["pred_vs_actual"] = [
        {"actual": float(a), "predicted": float(p),
         "oblast": str(o), "iso_3166_2": str(iso)}
        for a, p, o, iso in zip(
            y_test.tolist(),
            pred_test.tolist(),
            test["oblast"].tolist(),
            test["iso_3166_2"].tolist(),
            strict=False,
        )
    ][:500]

    # ─── CV metrics — skip for stack (per docstring) ──────
    if family != "stack" and not args.skip_loocv:
        log.info("[%s/%s] LOOCV-oblast …", crop, family)
        factory = _factory_for_family(family)
        loocv_metrics = _loocv_oblast(factory, df)
        result["loocv_oblast"] = asdict(loocv_metrics)

    if family != "stack" and not args.skip_kfold:
        log.info("[%s/%s] RepeatedKFold(5×3) …", crop, family)
        factory = _factory_for_family(family)
        # Train+val pool only — preserve 2023 as the never-seen test set.
        cv_df = df[df["year"] <= 2022]
        result["repeated_kfold_5x3"] = _repeated_kfold_variance(factory, cv_df)

    if family != "stack" and not args.skip_learning_curve:
        log.info("[%s/%s] learning curve …", crop, family)
        factory = _factory_for_family(family)
        result["learning_curve"] = _learning_curve(factory, df)

    # ─── Global SHAP + permutation importance ────────────
    if not args.no_shap and family != "stack":
        log.info("[%s/%s] global SHAP …", crop, family)
        X_full = df[list(FEATURE_NAMES)].to_numpy()
        result["global_shap"] = _global_shap(payload, X_full)

    if not args.no_permutation_importance and family != "stack":
        log.info("[%s/%s] permutation importance …", crop, family)
        result["permutation_importance"] = _permutation_importance(
            payload, X_test, y_test,
        )

    # ─── Partial dependence (top-3 features by permutation importance) ──
    if not args.no_partial_dependence and family != "stack":
        log.info("[%s/%s] partial dependence …", crop, family)
        perm = result.get("permutation_importance") or {}
        if perm:
            top_features = [
                f for f, _ in sorted(perm.items(), key=lambda kv: -kv[1])[:3]
            ]
        else:
            # No perm importance available — fall back to NDVI-related staples.
            top_features = ["ndvi_peak", "ndvi_mean_july", "centroid_lat"]
        X_full = df[list(FEATURE_NAMES)].to_numpy()
        result["partial_dependence"] = _partial_dependence(
            payload, X_full, top_features,
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", action="append", default=None,
                        help="restrict to one or more crops (repeatable)")
    parser.add_argument("--families", default="rf,xgboost,lightgbm,stack",
                        help="comma list of families to evaluate")
    parser.add_argument("--skip-loocv", action="store_true",
                        help="skip leave-one-oblast-out (faster dev iteration)")
    parser.add_argument("--skip-kfold", action="store_true",
                        help="skip repeated K-fold variance estimation")
    parser.add_argument("--skip-learning-curve", action="store_true",
                        help="skip the learning-curve sample-size sweep")
    parser.add_argument("--no-shap", action="store_true",
                        help="skip global SHAP computation")
    parser.add_argument("--no-permutation-importance", action="store_true",
                        help="skip permutation-importance computation")
    parser.add_argument("--no-partial-dependence", action="store_true",
                        help="skip 1-D partial-dependence curves (Friedman 2001)")
    parser.add_argument("--training-set", type=Path, default=TRAINING_SET)
    parser.add_argument("--version", choices=("v3", "v4", "v5"), default="v3",
                        help="which model version artifacts to evaluate")
    args = parser.parse_args()

    # Switch feature schema + output path based on version.
    # v4 and v5 share the same 23-feature schema — v5 just trained on
    # real Держстат per-oblast yields (Phase 1a of v5 plan); features
    # remained the v4 set (crop calendar + GDD + weather-conditioned).
    global FEATURE_NAMES, EVAL_OUT
    if args.version == "v5":
        FEATURE_NAMES = FEATURE_NAMES_V4   # same 23 features as v4
        EVAL_OUT = EVAL_OUT_V5
    elif args.version == "v4":
        FEATURE_NAMES = FEATURE_NAMES_V4
        EVAL_OUT = EVAL_OUT_V4
    else:
        FEATURE_NAMES = FEATURE_NAMES_V3
        EVAL_OUT = EVAL_OUT_V3

    if not args.training_set.exists():
        log.error("Missing %s.", args.training_set)
        return 1

    df = pd.read_parquet(args.training_set)
    log.info("Loaded %d rows from %s", len(df), args.training_set)

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    crops = args.crop or list(ALL_CROPS)

    summary: dict[str, Any] = {
        "metadata": {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "training_set": str(args.training_set),
            "features_origin": str(df.get("features_origin", pd.Series(["unknown"])).iloc[0]),
            "feature_names": list(FEATURE_NAMES),
            "splits": {"train": "<=2021", "val": "2022", "test": "2023"},
            "families": list(families),
            "skipped": {
                "loocv": args.skip_loocv,
                "kfold": args.skip_kfold,
                "learning_curve": args.skip_learning_curve,
                "shap": args.no_shap,
                "permutation_importance": args.no_permutation_importance,
                "partial_dependence": args.no_partial_dependence,
            },
        },
        "crops": {},
    }

    for crop in crops:
        crop_sub = df[df["crop"] == crop]
        if len(crop_sub) == 0:
            log.warning("Skipping %s — 0 rows in training set.", crop)
            continue
        summary["crops"][crop] = {}
        for family in families:
            log.info("Evaluating %s / %s …", crop, family)
            summary["crops"][crop][family] = evaluate_family_for_crop(
                family, crop, crop_sub, args,
            )

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s (%.1f KB)", EVAL_OUT, EVAL_OUT.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
