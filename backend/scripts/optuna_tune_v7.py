"""Per-(crop, family) hyperparameter sweep on the 5 weakest crops.

Targets crops that consistently underperform on the test 2021 split
despite full feature coverage:

    oats         R²  = -0.65  (96 train rows — overfit)
    buckwheat    R²  =  0.08  (91 train rows — bug-fix surfaced
                               wider Steppe-South coverage)
    rye          R²  =  0.17  (91 train rows — same)
    sugar_beet   R²  =  0.35  (64 train rows + extreme range 26-67 т/га)
    peas         R²  =  0.33  (93 train rows — currently stack/v7
                               fallback)

For each `(crop, family) ∈ {weak_crops} × {rf, xgboost, lightgbm}`,
runs **50 Optuna trials** maximising 5-fold CV R² on the train-years
(2018-2019) subset of `training_set_v3.parquet[is_real_yield=True]`.
For `stack`, base learners are frozen to the per-family best params
and only the Ridge meta `alpha` is searched (30 trials, 1-D).

Strong crops (wheat, corn, sunflower, corn_silage, barley, potato,
soybean, rapeseed) are deliberately excluded — they sit at R² ≥ 0.4
on n=24 test rows, and hyperparameter perturbations there fall
inside test-set noise. We don't want a worse-by-noise regression in
exchange for the 1.5h compute.

## Output

`backend/data/processed/optuna_best_params_v7.json`:

```json
{
  "oats": {
    "rf":       {"n_estimators": 250, "max_depth": 5, ..., "cv_r2": 0.18},
    "xgboost":  {"n_estimators": 180, "max_depth": 4, ..., "cv_r2": 0.22},
    "lightgbm": {"n_estimators": 200, "num_leaves": 23, ..., "cv_r2": 0.20},
    "stack":    {"ridge_alpha": 0.4, "cv_r2": 0.24}
  },
  ...
}
```

The trainers (`scripts/train_yield_models_v7.py` +
`_v7_hierarchical.py`) read this JSON at fit time via an
`@lru_cache` helper and, when an entry exists for `(crop, family)`,
override the hand-tuned constants in the per-family `_train_*`
helpers. Backward compatible: missing entry → existing hand-tuned
constants remain.

## Log-target compatibility

`sugar_beet` is in both this weak-crop set AND the log-target set
(see `train_yield_models_v7.py:LOG_TARGET_CROPS`). The objective
function below applies `np.log1p` before fit and `np.expm1` after
predict when the crop is log-targeted, then computes CV R² in
**original yield units (т/га)** — same metric the hybrid evaluator
reports. Without this, Optuna would optimise for log-scale fit, which
isn't the metric users see.

## Run

    uv run python scripts/optuna_tune_v7.py
    uv run python scripts/optuna_tune_v7.py --crop oats   # single-crop debug
    uv run python scripts/optuna_tune_v7.py --trials 100  # bigger budget
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the trainer's canonical FEATURE_NAMES + LOG_TARGET_CROPS so a
# rename or feature-count change there propagates here automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yield_models_v7 import (  # noqa: E402
    FEATURE_NAMES,
    LOG_TARGET_CROPS,
    V7_TRAIN_YEARS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("optuna_tune_v7")

# Silence optuna's per-trial INFO chatter — we'll log summary lines
# ourselves. Per-trial dots still useful in DEBUG.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"
OUT = ROOT / "data" / "processed" / "optuna_best_params_v7.json"

# The 5-crop weak-set per the post-v7-lift round-2 plan.
WEAK_CROPS: tuple[str, ...] = (
    "oats", "buckwheat", "rye", "sugar_beet", "peas",
)

# Number of CV folds — KFold(5) is the standard practice for n~50
# train rows per crop (10 rows per fold; small but each model still
# sees ~40 train + 10 val).
N_FOLDS = 5

# Sweep budget. 50 trials gives Optuna's TPE sampler enough head-room
# to converge in 5-7 dimensional search; we found diminishing returns
# past 75 in pilot runs.
DEFAULT_TRIALS_BASE = 50
DEFAULT_TRIALS_STACK = 30


# ─── Objective functions ───────────────────────────────────


def _cv_score(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    *,
    log_target: bool,
) -> float:
    """5-fold CV R² in original yield units.

    When `log_target=True`, applies `np.log1p`/`np.expm1` around fit
    and predict so the model trains on log-yield but the R² is still
    measured in т/га — matching the metric the hybrid evaluator
    reports.

    Returns the mean across folds. Folds with `r2_score` returning
    NaN (constant y_val) are skipped from the mean.
    """
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        y_tr_fit = np.log1p(y_tr) if log_target else y_tr
        model = model_factory()
        model.fit(X_tr, y_tr_fit)
        pred = model.predict(X_va)
        if log_target:
            pred = np.expm1(pred)
        if len(np.unique(y_va)) < 2:
            continue
        scores.append(float(r2_score(y_va, pred)))
    return float(np.mean(scores)) if scores else float("-inf")


def _objective_rf(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray, log_target: bool,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
    }
    return _cv_score(
        lambda: RandomForestRegressor(
            n_jobs=-1, random_state=42, **params,
        ),
        X, y, log_target=log_target,
    )


def _objective_xgboost(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray, log_target: bool,
) -> float:
    import xgboost as xgb

    # Phase-A round-3: widened reg_alpha / reg_lambda ranges (0→5)
    # because the feature count grew 41→57 → d/n now 1.17, much
    # higher overfit risk. The TPE sampler will pick high
    # regularization for crops where the wider feature set adds
    # mostly noise rather than signal.
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    return _cv_score(
        lambda: xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42,
            n_jobs=-1, tree_method="hist", verbosity=0, **params,
        ),
        X, y, log_target=log_target,
    )


def _objective_lightgbm(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray, log_target: bool,
) -> float:
    import lightgbm as lgb

    # Phase-A round-3: widened reg_alpha / reg_lambda ranges (see XGBoost
    # objective above for rationale).
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    return _cv_score(
        lambda: lgb.LGBMRegressor(
            random_state=42, n_jobs=-1, verbose=-1, **params,
        ),
        X, y, log_target=log_target,
    )


def _objective_stack(
    trial: optuna.Trial,
    X: np.ndarray, y: np.ndarray,
    rf_params: dict, xgb_params: dict, lgbm_params: dict,
    log_target: bool,
) -> float:
    """1-D search over Ridge meta-learner alpha. Base learners are
    frozen to the per-family standalone-best params from earlier
    studies — avoids a 15-D search that would need ~500 trials to
    converge."""
    import lightgbm as lgb
    import xgboost as xgb

    alpha = trial.suggest_float("ridge_alpha", 0.01, 10.0, log=True)
    base = [
        ("rf", RandomForestRegressor(n_jobs=-1, random_state=42, **rf_params)),
        ("xgb", xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42,
            n_jobs=-1, tree_method="hist", verbosity=0, **xgb_params,
        )),
        ("lgbm", lgb.LGBMRegressor(
            random_state=42, n_jobs=-1, verbose=-1, **lgbm_params,
        )),
    ]
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    return _cv_score(
        lambda: StackingRegressor(
            estimators=base, final_estimator=Ridge(alpha=alpha),
            cv=inner_cv, n_jobs=1, passthrough=False,
        ),
        X, y, log_target=log_target,
    )


# ─── Main ──────────────────────────────────────────────────


def _tune_one_crop(
    crop: str,
    X: np.ndarray, y: np.ndarray,
    n_trials_base: int,
    n_trials_stack: int,
) -> dict[str, dict[str, Any]]:
    """Run the four sub-studies for one crop and return the merged
    `{family: {param, ..., cv_r2}}` dict."""
    log_target = crop in LOG_TARGET_CROPS
    log.info(
        "[%s] starting sweep (n=%d, features=%d, log_target=%s) …",
        crop, len(y), X.shape[1], log_target,
    )

    results: dict[str, dict[str, Any]] = {}

    for family, objective in (
        ("rf", _objective_rf),
        ("xgboost", _objective_xgboost),
        ("lightgbm", _objective_lightgbm),
    ):
        # Independent study per (crop, family); seed for reproducibility.
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda t, fam=family, obj=objective:  # noqa: B023
                obj(t, X, y, log_target),
            n_trials=n_trials_base,
            show_progress_bar=False,
        )
        results[family] = {**study.best_params, "cv_r2": float(study.best_value)}
        log.info(
            "[%s/%s] best CV R²=%+.4f after %d trials  params=%s",
            crop, family, study.best_value, len(study.trials), study.best_params,
        )

    # Stack — 1-D ridge_alpha search using frozen base learners.
    rf_only = {k: v for k, v in results["rf"].items() if k != "cv_r2"}
    xgb_only = {k: v for k, v in results["xgboost"].items() if k != "cv_r2"}
    lgbm_only = {k: v for k, v in results["lightgbm"].items() if k != "cv_r2"}

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: _objective_stack(t, X, y, rf_only, xgb_only, lgbm_only, log_target),
        n_trials=n_trials_stack,
        show_progress_bar=False,
    )
    results["stack"] = {**study.best_params, "cv_r2": float(study.best_value)}
    log.info(
        "[%s/stack] best CV R²=%+.4f  ridge_alpha=%.4f",
        crop, study.best_value, study.best_params["ridge_alpha"],
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", action="append", default=None,
                        help="restrict to one or more crops (repeatable). Default: 5 weak crops.")
    parser.add_argument("--all-crops", action="store_true",
                        help="sweep all 13 crops instead of the 5 weak-crop default. "
                             "Combine with --trials 200 for the wide Phase-D Option C sweep.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS_BASE,
                        help=f"trials per (crop, base-family) (default {DEFAULT_TRIALS_BASE})")
    parser.add_argument("--trials-stack", type=int, default=DEFAULT_TRIALS_STACK,
                        help=f"trials for stack ridge_alpha (default {DEFAULT_TRIALS_STACK})")
    parser.add_argument("--output", type=Path, default=OUT,
                        help=f"output JSON path (default {OUT})")
    args = parser.parse_args()

    if not PARQUET.exists():
        log.error("Missing parquet %s", PARQUET)
        return 1
    df = pd.read_parquet(PARQUET)
    if "is_real_yield" not in df.columns:
        log.error("Parquet missing is_real_yield column")
        return 1
    real = df[df["is_real_yield"].astype(bool)].copy()
    train_mask = real["year"].isin(V7_TRAIN_YEARS)
    train = real[train_mask]
    log.info(
        "Loaded %d real rows; %d in train years %s",
        len(real), len(train), list(V7_TRAIN_YEARS),
    )

    missing = [f for f in FEATURE_NAMES if f not in train.columns]
    if missing:
        log.error(
            "Parquet missing feature cols: %s. Run patch_parquet_oblast_yield_lag.py first.",
            missing[:5],
        )
        return 1

    # Crop-selection precedence: explicit --crop > --all-crops > WEAK_CROPS default.
    # Importing ALL_CROPS lazily so script header stays portable.
    if args.crop:
        crops = list(args.crop)
    elif args.all_crops:
        from app.data_reference.crop_zones import ALL_CROPS as _ALL_CROPS
        crops = list(_ALL_CROPS)
    else:
        crops = list(WEAK_CROPS)
    log.info("Crops to sweep: %s (%d total)", crops, len(crops))
    log.info("Trial budget per (crop, base-family): %d", args.trials)
    log.info("Trial budget per (crop, stack):       %d", args.trials_stack)

    all_results: dict[str, Any] = {}
    for crop in crops:
        sub = train[train["crop"] == crop]
        if len(sub) < 20:
            log.warning("[%s] only %d train rows — skipping (need ≥ 20)", crop, len(sub))
            continue
        X = sub[list(FEATURE_NAMES)].to_numpy(dtype=float)
        y = sub["yield_tha"].to_numpy(dtype=float)
        all_results[crop] = _tune_one_crop(
            crop, X, y, args.trials, args.trials_stack,
        )

    out = {
        "metadata": {
            "tuned_at": datetime.now(UTC).isoformat(),
            "train_years": list(V7_TRAIN_YEARS),
            "feature_count": len(FEATURE_NAMES),
            "trials_per_base_family": args.trials,
            "trials_for_stack": args.trials_stack,
            "n_folds": N_FOLDS,
            "weak_crops_swept": list(crops),
            "cv_metric": "r2_original_scale",
        },
        "by_crop": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
