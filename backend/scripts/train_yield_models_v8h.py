"""Hierarchical multi-task yield regressor — v8 + v7h reformulation.

Combines two ideas:

1. **Multi-task** (from v8): ONE model per family trained on ALL 13
   crops jointly with 13 `crop_*` one-hot features. Effective n_train
   ≈ 624 (or higher when historical years 2016-2017 are present).
2. **Hierarchical** (from v7h): subtract the per-(crop, oblast)
   historical mean from the yield target. The model learns
   *deviations* from the regional baseline, not absolute yields.

For narrow-variance crops (oats, buckwheat, rye) where yield is
dominated by stable per-oblast baselines, the hierarchical
reformulation removes the noise floor that pure multi-task can't
escape. For wide-variance crops (sugar_beet, corn_silage) the per-
year deviation IS the agronomic signal and the same reformulation
applies cleanly.

## Inference

`yield_model.predict_yield` looks up `payload["oblast_means"]` for the
field's (crop, iso) and adds the mean back to the model's deviation
prediction. v7h already implements this for per-crop hierarchical
models; v8h reuses the exact same code path.

## Output

- **4 model files**: `yield_{rf,xgb,lgbm,stack}_v8h.joblib` — one per
  family, payload `kind="multi_task_hierarchical"`, version `"v8h"`.
- **`data/processed/model_metrics_v8h.json`** in the v7/v8 per-crop
  shape (consumable by `evaluate_v7_hybrid.py` as a 5th best-of-N
  candidate alongside v6/v7/v7h/v8).

## Rollback

Same as v8 — the hybrid evaluator picks per-crop best across
{v6, v7, v7h, v8, v8h}. If v8h hurts a crop, that crop reverts to
whichever older version was best. Delete the four .joblibs +
`model_metrics_v8h.json` to roll out entirely.

## Run

    uv run python scripts/train_yield_models_v8h.py
    uv run python scripts/evaluate_v7_hybrid.py     # adds v8h to candidates
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

# Reuse the per-family trainers + metric helpers + chronological-split
# constants from the v7 trainer. crop=None disables Optuna lookups
# (multi-task uses one global hyperparam set).
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
log = logging.getLogger("train_yield_v8h")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
METRICS_OUT = ROOT / "data" / "processed" / "model_metrics_v8h.json"

FEATURE_NAMES: tuple[str, ...] = V8_FEATURE_NAMES


def _model_filename(family: str) -> str:
    """v8h filename pattern mirrors v8: no crop slot, single global
    model per family. The `v8h` version tag distinguishes from `v8`."""
    short = {"xgboost": "xgb", "lightgbm": "lgbm", "rf": "rf", "stack": "stack"}[family]
    return f"yield_{short}_v8h.joblib"


def _save_payload(payload: dict, family: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / _model_filename(family)
    joblib.dump(payload, path)
    return path


def _compute_oblast_means(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """`{crop: {iso: mean_yield}}` computed from TRAIN YEARS ONLY.

    Mirror of `train_yield_models_v7_hierarchical._compute_oblast_means`.
    Crops with no row for an oblast use the crop's global mean as the
    `_GLOBAL` fallback so inference doesn't crash on a (crop, iso)
    combination the training years didn't cover.
    """
    out: dict[str, dict[str, float]] = {}
    for crop, sub in train_df.groupby("crop"):
        out[crop] = {}
        global_mean = float(sub["yield_tha"].mean())
        for iso, sub2 in sub.groupby("iso_3166_2"):
            out[crop][iso] = float(sub2["yield_tha"].mean())
        out[crop]["_GLOBAL"] = global_mean
    return out


def _apply_offset(
    df: pd.DataFrame,
    oblast_means: dict[str, dict[str, float]],
    direction: str = "subtract",
) -> np.ndarray:
    """Vectorised subtract or add of per-(crop, iso) mean. Returns an
    array of the same length as `df` containing either the deviation
    target (`subtract`) or the reconstructed absolute yield (`add`).
    """
    crops = df["crop"].to_numpy()
    isos = df["iso_3166_2"].to_numpy()
    means = np.array([
        oblast_means.get(c, {}).get(i, oblast_means.get(c, {}).get("_GLOBAL", 0.0))
        for c, i in zip(crops, isos, strict=False)
    ])
    if direction == "subtract":
        return df["yield_tha"].to_numpy() - means
    return df["yield_tha"].to_numpy() + means


def _per_crop_metrics(
    model: Any,
    test: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
    oblast_means: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict]]:
    """Same shape as v8's per-crop split, but add the oblast mean
    back to recover absolute yield before computing metrics."""

    def _split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if len(df) == 0:
            return np.array([]), np.array([])
        X = df[list(FEATURE_NAMES)].to_numpy()
        preds_dev = np.asarray(model.predict(X))
        # Add per-(crop, iso) mean back → absolute yield prediction.
        means = np.array([
            oblast_means.get(c, {}).get(i,
                oblast_means.get(c, {}).get("_GLOBAL", 0.0))
            for c, i in zip(df["crop"], df["iso_3166_2"], strict=False)
        ])
        return df["yield_tha"].to_numpy(), preds_dev + means

    y_train, preds_train = _split(train)
    y_val, preds_val = _split(val)
    y_test, preds_test = _split(test)

    out: dict[str, dict[str, dict]] = {}
    for crop in ALL_CROPS:
        ct = (train["crop"] == crop).to_numpy() if len(train) else np.array([], dtype=bool)
        cv = (val["crop"] == crop).to_numpy() if len(val) else np.array([], dtype=bool)
        cs = (test["crop"] == crop).to_numpy() if len(test) else np.array([], dtype=bool)
        out[crop] = {
            "train": asdict(_compute_metrics(y_train[ct], preds_train[ct])),
            "val": asdict(_compute_metrics(y_val[cv], preds_val[cv])),
            "test": asdict(_compute_metrics(y_test[cs], preds_test[cs])),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", default=",".join(ALL_FAMILIES))
    parser.add_argument("--training-set", type=Path, default=TRAINING_SET)
    args = parser.parse_args()

    if not args.training_set.exists():
        log.error("Missing %s", args.training_set)
        return 1

    df = pd.read_parquet(args.training_set)
    real = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Real-yield rows: %d", len(real))

    # Inject 13 crop one-hot columns inline (same as v8 trainer). The
    # parquet stores `crop` as a single text column; one-hots are a
    # train-time view, not a persisted column.
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

    # Compute per-(crop, iso) means from TRAIN YEARS only — strict
    # anti-leakage rule mirroring v7h. Means are persisted in the
    # payload so inference can add them back without re-reading the
    # parquet.
    oblast_means = _compute_oblast_means(train)
    log.info("Computed oblast_means for %d crops (avg %d isos each)",
             len(oblast_means),
             sum(len(v) - 1 for v in oblast_means.values()) // max(1, len(oblast_means)))

    # Deviation target. The same subtract operation maps each row to
    # its crop-and-oblast-specific zero-baseline. The model learns
    # "this row beat its oblast historical mean by N т/га", not the
    # absolute yield.
    train["y_target"] = _apply_offset(train, oblast_means, direction="subtract")
    val["y_target"] = _apply_offset(val, oblast_means, direction="subtract")
    test["y_target"] = _apply_offset(test, oblast_means, direction="subtract")

    X_train = train[list(FEATURE_NAMES)].to_numpy()
    y_train_dev = train["y_target"].to_numpy()
    X_val = val[list(FEATURE_NAMES)].to_numpy() if len(val) else np.empty((0, len(FEATURE_NAMES)))
    y_val_dev = val["y_target"].to_numpy()
    X_trainval = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y_trainval_dev = np.concatenate([y_train_dev, y_val_dev]) if len(y_val_dev) else y_train_dev

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    log.info("Families to train: %s", families)
    log.info("Features (v8h multi-task hierarchical): %d cols", len(FEATURE_NAMES))

    summary: dict[str, Any] = {
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(),
            "kind": "multi_task_hierarchical_v8h",
            "training_set": str(args.training_set),
            "feature_names": list(FEATURE_NAMES),
            "splits": {"train": list(V7_TRAIN_YEARS), "val": V7_VAL_YEAR, "test": V7_TEST_YEAR},
            "families": list(families),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "reference": (
                "Lobell et al. 2011, Schauberger & Gornott 2017 — panel-data "
                "approach. Multi-task variant — single model spans 13 crops."
            ),
        },
        "crops": {},
    }

    for family in families:
        log.info("[v8h/%s] training on %d rows (deviation target)…",
                 family, len(train))
        try:
            if family == "rf":
                model = _train_rf(X_train, y_train_dev, crop=None)
            elif family == "xgboost":
                bundle = _train_xgboost(X_train, y_train_dev, X_val, y_val_dev, crop=None)
                # Quantile siblings don't carry meaning across 13 crops
                # with mixed scales — keep point only (same as v8).
                model = bundle["point"]
            elif family == "lightgbm":
                model = _train_lightgbm(X_train, y_train_dev, X_val, y_val_dev, crop=None)
            elif family == "stack":
                model = _train_stack(X_trainval, y_trainval_dev, crop=None)
            else:
                log.warning("Unknown family %s — skipping.", family)
                continue

            payload = {
                "model": model,
                "features": list(FEATURE_NAMES),
                "family": family,
                "version": "v8h",
                "log_target": False,
                "kind": "multi_task_hierarchical",
                # `oblast_means` keyed as {crop: {iso: mean}} — the
                # predict-path reads `oblast_means[crop][iso]` to add
                # back the offset. Same shape v7h uses.
                "oblast_means": oblast_means,
            }
            path = _save_payload(payload, family)
            log.info("[v8h/%s] saved → %s", family, path.name)

            # Per-crop metrics on absolute yield (deviation + mean).
            per_crop = _per_crop_metrics(model, test, train, val, oblast_means)
            for crop, metrics in per_crop.items():
                summary["crops"].setdefault(crop, {"families": {}})
                summary["crops"][crop]["families"][family] = metrics
                test_r2 = metrics["test"].get("r2")
                log.info(
                    "[v8h/%s/%s] test n=%d R²=%s MAE=%s",
                    family, crop, metrics["test"].get("n", 0),
                    f"{test_r2:+.3f}" if test_r2 is not None else "n/a",
                    f"{metrics['test'].get('mae'):.3f}"
                        if metrics["test"].get("mae") is not None else "n/a",
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("[v8h/%s] training failed: %s", family, exc)
            for crop in ALL_CROPS:
                summary["crops"].setdefault(crop, {"families": {}})
                summary["crops"][crop]["families"][family] = {"error": str(exc)}

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
