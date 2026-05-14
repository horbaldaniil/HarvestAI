"""Multi-task yield regressor — ONE model per family across ALL 13 crops.

v7 trains 13 separate (crop, family) tabular models (52 .joblib files
total: 13 × {rf, xgb, lgbm, stack}). Each per-crop model sees only
~48 train rows on the chronological split — d/n = 57 features / 48
rows = 1.19 in deep overfit territory. Phase-A added agronomically
sound features (winter NDVI + soil moisture + BBCH-aligned weather)
but wheat regressed 0.638 → 0.526 because the extra features amplified
overfit on the small per-crop train set.

v8 takes the opposite approach: keep the same 57 features, add 13
`crop_{slug}` one-hot identity features (70 total), and train ONE
model per family on ALL crops jointly. Effective n_train grows from
48 (per-crop) to 624 (all crops × 2 train years × 24 oblasts), and
d/n drops to 70/624 = 0.11 — safely below the rule-of-thumb 0.3
overfit boundary.

Cross-crop knowledge transfer: tree splits learned on wheat-NDVI
generalize to barley, rye, oats (related cereals); soybean's
August-NDVI signal transfers to corn_silage; etc. The one-hots let
the model recover per-crop biases on top of the shared signal. Wang
et al. 2018 ("Deep Transfer Learning for Crop Yield Prediction")
and the broader multi-task-regression literature show +5-15 % R²
lift in low-n agricultural regimes — exactly our regime.

## Output

- **4 model files** (not 52): `yield_{rf,xgb,lgbm,stack}_v8.joblib`.
  No crop slot in the filename — one model serves all 13 crops.
- **One metrics file**: `data/processed/model_metrics_v8.json` in the
  same per-crop shape as `model_metrics_v7.json` (so
  `evaluate_v7_hybrid.py` can consume v8 as a 4th best-of-N candidate
  alongside v6/v7/v7h). Per-crop test metrics computed by splitting
  the global model's test predictions by `crop`.

## Rollback safety

Adding v8 to `evaluate_v7_hybrid` extends the candidate set
(v6, v7, v7h) → (v6, v7, v7h, v8). The evaluator picks the per-crop
best by test R². If v8 hurts a crop, that crop stays on v7/v7h
automatically — no rollback ceremony, the per-crop selector handles
it. Hard rollback (if all crops regress): delete the four .joblib
files + re-run the evaluator; the v8 candidate disappears.

## Run

    uv run python scripts/train_yield_models_v8.py
    uv run python scripts/evaluate_v7_hybrid.py     # picks per-crop winner
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the per-family trainers + metric helpers from v7. They take
# `crop=None` to skip the per-crop Optuna lookup; this is exactly
# what we want here (v8 uses one global hyperparam set, not per-crop).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yield_models_v7 import (  # noqa: E402
    ALL_FAMILIES,
    V7_TEST_YEAR,
    V7_TRAIN_YEARS,
    V7_VAL_YEAR,
    _compute_metrics,
    _train_lightgbm,
    _train_rf,
    _train_stack,
    _train_xgboost,
)

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402
from app.ml.features import V8_FEATURE_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_yield_v8")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
METRICS_OUT = ROOT / "data" / "processed" / "model_metrics_v8.json"

FEATURE_NAMES: tuple[str, ...] = V8_FEATURE_NAMES


def _model_filename(family: str) -> str:
    """v8 filename pattern: no crop slot, one model per family."""
    short = {"xgboost": "xgb", "lightgbm": "lgbm", "rf": "rf", "stack": "stack"}[family]
    return f"yield_{short}_v8.joblib"


def _save_payload(payload: dict, family: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / _model_filename(family)
    joblib.dump(payload, path)
    return path


def _per_crop_metrics(
    model: Any,
    test: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> dict[str, dict[str, dict]]:
    """Run model.predict on full {train, val, test} sets ONCE, then
    split predictions by crop and compute per-crop metrics. Returns
    `{crop: {train: {...}, val: {...}, test: {...}}}` matching the
    v7 metrics JSON shape so `evaluate_v7_hybrid.py` can consume v8
    as a drop-in candidate.

    Single global predict per split is much cheaper than 13 per-crop
    predicts (~13× speedup), and the splits are zero-overhead with
    pandas boolean masks.
    """
    X_train = train[list(FEATURE_NAMES)].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy() if len(val) else np.empty((0, len(FEATURE_NAMES)))
    X_test = test[list(FEATURE_NAMES)].to_numpy() if len(test) else np.empty((0, len(FEATURE_NAMES)))

    preds_train = np.asarray(model.predict(X_train)) if len(X_train) else np.array([])
    preds_val = np.asarray(model.predict(X_val)) if len(X_val) else np.array([])
    preds_test = np.asarray(model.predict(X_test)) if len(X_test) else np.array([])

    out: dict[str, dict[str, dict]] = {}
    for crop in ALL_CROPS:
        crop_train_mask = (train["crop"] == crop).to_numpy()
        crop_val_mask = (val["crop"] == crop).to_numpy() if len(val) else np.array([], dtype=bool)
        crop_test_mask = (test["crop"] == crop).to_numpy() if len(test) else np.array([], dtype=bool)

        # _compute_metrics returns SplitMetrics dataclass with .n etc.
        out[crop] = {
            "train": asdict(_compute_metrics(
                train.loc[crop_train_mask, "yield_tha"].to_numpy(),
                preds_train[crop_train_mask],
            )),
            "val": asdict(_compute_metrics(
                val.loc[crop_val_mask, "yield_tha"].to_numpy() if len(val) else np.array([]),
                preds_val[crop_val_mask] if len(preds_val) else np.array([]),
            )),
            "test": asdict(_compute_metrics(
                test.loc[crop_test_mask, "yield_tha"].to_numpy() if len(test) else np.array([]),
                preds_test[crop_test_mask] if len(preds_test) else np.array([]),
            )),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", default=",".join(ALL_FAMILIES),
                        help=f"comma list of families to train (default {','.join(ALL_FAMILIES)})")
    parser.add_argument("--training-set", type=Path, default=TRAINING_SET)
    args = parser.parse_args()

    if not args.training_set.exists():
        log.error("Missing %s", args.training_set)
        return 1

    df = pd.read_parquet(args.training_set)
    if "is_real_yield" not in df.columns:
        log.error("Parquet missing is_real_yield column")
        return 1
    real = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Real-yield rows: %d", len(real))

    # Verify the 13 crop one-hots are populated. If they're missing
    # (rebuilt parquet not yet patched), inject them here from the
    # `crop` column.
    missing_onehots = [
        f"crop_{c}" for c in ALL_CROPS if f"crop_{c}" not in real.columns
    ]
    if missing_onehots:
        log.info("Injecting %d crop one-hot columns inline", len(missing_onehots))
        for c in ALL_CROPS:
            col = f"crop_{c}"
            if col not in real.columns:
                real[col] = (real["crop"] == c).astype(int)

    train = real[real["year"].isin(V7_TRAIN_YEARS)].copy()
    val = real[real["year"] == V7_VAL_YEAR].copy()
    test = real[real["year"] == V7_TEST_YEAR].copy()
    log.info(
        "Split: train=%d (years %s) val=%d (year %d) test=%d (year %d)",
        len(train), list(V7_TRAIN_YEARS), len(val), V7_VAL_YEAR, len(test), V7_TEST_YEAR,
    )
    trainval = pd.concat([train, val], ignore_index=True)

    X_train = train[list(FEATURE_NAMES)].to_numpy()
    y_train = train["yield_tha"].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy() if len(val) else np.empty((0, len(FEATURE_NAMES)))
    y_val = val["yield_tha"].to_numpy()
    X_trainval = trainval[list(FEATURE_NAMES)].to_numpy()
    y_trainval = trainval["yield_tha"].to_numpy()

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    log.info("Families to train: %s", families)
    log.info("Features (multi-task v8): %d cols", len(FEATURE_NAMES))

    summary: dict[str, Any] = {
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(),
            "kind": "multi_task_v8",
            "training_set": str(args.training_set),
            "feature_names": list(FEATURE_NAMES),
            "splits": {"train": list(V7_TRAIN_YEARS), "val": V7_VAL_YEAR, "test": V7_TEST_YEAR},
            "families": list(families),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "rationale": (
                "Single model per family trained on ALL 13 crops jointly. "
                "Cross-crop signal sharing via tree splits lowers d/n to 0.11."
            ),
        },
        "crops": {},
    }

    for family in families:
        log.info("[v8/%s] training on %d rows × %d features…",
                 family, len(train), len(FEATURE_NAMES))
        try:
            # crop=None disables per-(crop, family) Optuna lookups in
            # the v7 helpers — we want one global hyperparam set for
            # the multi-task model.
            if family == "rf":
                model = _train_rf(X_train, y_train, crop=None)
            elif family == "xgboost":
                bundle = _train_xgboost(X_train, y_train, X_val, y_val, crop=None)
                # For multi-task we keep ONLY the point estimator on disk
                # — q05/q95 quantile siblings aren't meaningful when one
                # model spans 13 crops with wildly different y-scales
                # (e.g. corn 6-10 t/ha vs sugar_beet 26-67 t/ha share
                # the same residual distribution). Conformal CI takes
                # over for v8 at calibration time.
                model = bundle["point"]
            elif family == "lightgbm":
                model = _train_lightgbm(X_train, y_train, X_val, y_val, crop=None)
            elif family == "stack":
                model = _train_stack(X_trainval, y_trainval, crop=None)
            else:
                log.warning("Unknown family %s — skipping.", family)
                continue

            payload = {
                "model": model,
                "features": list(FEATURE_NAMES),
                "family": family,
                "version": "v8",
                "log_target": False,  # multi-task stays in linear yield space
                "kind": "multi_task",
            }
            path = _save_payload(payload, family)
            log.info("[v8/%s] saved → %s", family, path.name)

            # Per-crop metrics via single global predict + boolean splits.
            per_crop = _per_crop_metrics(model, test, train, val)
            for crop, metrics in per_crop.items():
                summary["crops"].setdefault(crop, {"families": {}})
                summary["crops"][crop]["families"][family] = metrics
                test_r2 = metrics["test"].get("r2")
                log.info(
                    "[v8/%s/%s] test n=%d R²=%s MAE=%s",
                    family, crop, metrics["test"].get("n", 0),
                    f"{test_r2:+.3f}" if test_r2 is not None else "n/a",
                    f"{metrics['test'].get('mae'):.3f}" if metrics["test"].get("mae") is not None else "n/a",
                )

        except Exception as exc:  # noqa: BLE001
            log.exception("[v8/%s] training failed: %s", family, exc)
            for crop in ALL_CROPS:
                summary["crops"].setdefault(crop, {"families": {}})
                summary["crops"][crop]["families"][family] = {"error": str(exc)}

    # Attach n_train / n_val / n_test per crop so the hybrid evaluator
    # has the same metadata shape as v7. Counts are post-split per
    # crop, not the global multi-task count.
    for crop in ALL_CROPS:
        if crop in summary["crops"]:
            summary["crops"][crop]["n_train"] = int((train["crop"] == crop).sum())
            summary["crops"][crop]["n_val"] = int((val["crop"] == crop).sum())
            summary["crops"][crop]["n_test"] = int((test["crop"] == crop).sum())
            summary["crops"][crop]["skipped"] = False

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote %s", METRICS_OUT)
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
