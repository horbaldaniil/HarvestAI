"""v7-pruned: per-crop feature subset via SHAP-importance audit.

Phase D Option A. For each crop we read its top-K features from
`feature_importance_v7.json` (produced by
`scripts/feature_importance_audit_v7.py`) and train a v7 model on
that narrower subset. Goal: undo Phase A's d/n inflation on the
8-9 crops where the wider 57-feature schema added noise that
hurt test R².

## Selection policy

The audit script emits both fixed-K cuts (`top_{20,25,30,35,40}_features`)
and a per-crop adaptive cut (`auto_top_features`, the smallest set
that together explains 95 % of total |SHAP|). We use the **adaptive
cut** by default — every crop gets exactly as many features as its
data justifies. Narrow-variance crops (oats, buckwheat) get a
slightly larger set because their importance is more diffuse;
high-signal crops (corn, wheat) get a tighter set.

Override via `--policy top_25` etc. for ablation work — surfaces
the trade-off between coverage and overfit at fixed K.

## Output

- **One model per (family, crop)**: `yield_{family}_{crop}_v7p.joblib`.
- Payload carries the **per-crop** feature list in `payload["features"]`
  so the registry + inference path slice correctly.
- `data/processed/model_metrics_v7p.json` — same shape as `model_metrics_v7.json`
  so `evaluate_v7_hybrid.py` can consume it as a 6th best-of-N
  candidate alongside v6 / v7 / v7h / v8 / v8h.

## Rollback

Delete the .joblibs + `model_metrics_v7p.json`. The hybrid evaluator
gracefully falls back to its prior candidate set; no inference path
changes.

## Run

    uv run python scripts/feature_importance_audit_v7.py   # produces JSON
    uv run python scripts/train_yield_models_v7_pruned.py
    uv run python scripts/calibrate_conformal.py
    uv run python scripts/evaluate_v7_hybrid.py            # picks v7p where it wins
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yield_models_v7 import (  # noqa: E402
    ALL_FAMILIES,
    LOG_TARGET_CROPS,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_yield_v7p")

TRAINING_SET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
METRICS_OUT = ROOT / "data" / "processed" / "model_metrics_v7p.json"
SHAP_AUDIT = ROOT / "data" / "processed" / "feature_importance_v7.json"


def _model_filename(family: str, crop: str) -> str:
    short = {"xgboost": "xgb", "lightgbm": "lgbm", "rf": "rf", "stack": "stack"}[family]
    return f"yield_{short}_{crop}_v7p.joblib"


def _save_payload(payload: dict, family: str, crop: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / _model_filename(family, crop)
    joblib.dump(payload, path)
    return path


def _eval_helper(model: Any, X: np.ndarray, y_orig: np.ndarray,
                 *, log_target: bool = False) -> dict:
    preds = np.asarray(model.predict(X)) if len(y_orig) else np.array([])
    if log_target and len(preds):
        preds = np.expm1(preds)
    return asdict(_compute_metrics(np.asarray(y_orig), preds))


def _load_per_crop_features(policy: str) -> dict[str, list[str]]:
    """Read `feature_importance_v7.json` and return `{crop: [features]}`.

    `policy ∈ {auto, top_20, top_25, top_30, top_35, top_40}`. The
    audit script emits all five fixed-K lists + the auto (95 % cum
    share) list per crop. We just pick the right key per the policy.
    """
    if not SHAP_AUDIT.exists():
        log.error(
            "Missing %s — run scripts/feature_importance_audit_v7.py first",
            SHAP_AUDIT,
        )
        sys.exit(1)
    blob = json.loads(SHAP_AUDIT.read_text(encoding="utf-8"))
    by_crop = blob.get("by_crop") or {}
    key = f"{policy}_features"
    out: dict[str, list[str]] = {}
    for crop, body in by_crop.items():
        if policy == "auto":
            feats = body.get("auto_top_features")
        else:
            feats = body.get(key)
        if not feats:
            log.warning(
                "[%s] no '%s' entry in SHAP audit — skipping (will fall back "
                "to per-crop default)", crop, policy,
            )
            continue
        out[crop] = list(feats)
    return out


def _split_chronological(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = sub[sub["year"].isin(V7_TRAIN_YEARS)].copy()
    val = sub[sub["year"] == V7_VAL_YEAR].copy()
    test = sub[sub["year"] == V7_TEST_YEAR].copy()
    return train, val, test


def train_one_crop(
    crop: str, df: pd.DataFrame, features: list[str],
    families: tuple[str, ...], min_rows: int,
) -> dict[str, Any]:
    """Per-crop training loop. Differs from v7's `train_one_crop` in
    two places: features come from the SHAP-audit subset (passed
    in), and the version tag baked into payload/metadata is `v7p`."""
    sub = df[df["crop"] == crop].copy()
    if len(sub) < min_rows:
        log.warning("[%s] skip — only %d rows (< %d)", crop, len(sub), min_rows)
        return {"skipped": True, "reason": "insufficient_rows", "n": int(len(sub))}

    # Verify all requested features actually exist in the parquet.
    # Defensive — a feature could have been removed from the parquet
    # since the audit was run (rebuild scenarios).
    missing = [f for f in features if f not in sub.columns]
    if missing:
        log.warning(
            "[%s] %d audit feature(s) absent from parquet — dropping: %s",
            crop, len(missing), missing[:5],
        )
        features = [f for f in features if f in sub.columns]
    if len(features) < 5:
        log.warning("[%s] only %d valid features — skipping", crop, len(features))
        return {"skipped": True, "reason": "too_few_features"}

    train, val, test = _split_chronological(sub)
    X_train = train[features].to_numpy()
    X_val = val[features].to_numpy() if len(val) else np.empty((0, len(features)))
    X_test = test[features].to_numpy() if len(test) else np.empty((0, len(features)))

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
        y_train, y_val, y_test = y_train_orig, y_val_orig, y_test_orig

    X_trainval = np.vstack([X_train, X_val]) if len(X_val) else X_train
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) else y_train

    out: dict[str, Any] = {
        "skipped": False,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "log_target": log_target,
        "features_used": features,
        "n_features_used": len(features),
        "families": {},
    }

    def _eval_set(model: Any) -> dict:
        return {
            "train": _eval_helper(model, X_train, y_train_orig, log_target=log_target),
            "val": _eval_helper(model, X_val, y_val_orig, log_target=log_target),
            "test": _eval_helper(model, X_test, y_test_orig, log_target=log_target),
        }

    for family in families:
        log.info("[%s/%s] training on %d features…", crop, family, len(features))
        try:
            if family == "rf":
                model = _train_rf(X_train, y_train, crop=crop)
                payload = {
                    "model": model, "features": features,
                    "family": "rf", "version": "v7p",
                    "log_target": log_target,
                }
                _save_payload(payload, family, crop)
                metrics = _eval_set(model)
            elif family == "xgboost":
                bundle = _train_xgboost(X_train, y_train, X_val, y_val, crop=crop)
                payload = {
                    "point": bundle["point"], "q_low": bundle["q_low"],
                    "q_high": bundle["q_high"], "features": features,
                    "family": "xgboost", "version": "v7p",
                    "log_target": log_target,
                }
                _save_payload(payload, family, crop)
                metrics = _eval_set(bundle["point"])
            elif family == "lightgbm":
                model = _train_lightgbm(X_train, y_train, X_val, y_val, crop=crop)
                payload = {
                    "model": model, "features": features,
                    "family": "lightgbm", "version": "v7p",
                    "log_target": log_target,
                }
                _save_payload(payload, family, crop)
                metrics = _eval_set(model)
            elif family == "stack":
                stack = _train_stack(X_trainval, y_trainval, crop=crop)
                payload = {
                    "model": stack, "features": features,
                    "family": "stack", "version": "v7p",
                    "log_target": log_target,
                }
                _save_payload(payload, family, crop)
                metrics = _eval_set(stack)
            else:
                log.warning("Unknown family %s — skipping.", family)
                continue

            out["families"][family] = metrics
            test_r2 = metrics["test"].get("r2")
            log.info(
                "[%s/%s/v7p] test R²=%s MAE=%s n_feat=%d",
                crop, family,
                f"{test_r2:+.3f}" if test_r2 is not None else "n/a",
                f"{metrics['test'].get('mae'):.3f}"
                    if metrics["test"].get("mae") is not None else "n/a",
                len(features),
            )
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
                        help="comma list of families to train")
    parser.add_argument("--policy", default="auto",
                        choices=["auto", "top_20", "top_25", "top_30",
                                 "top_35", "top_40"],
                        help="per-crop feature-subset policy (default: auto)")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--training-set", type=Path, default=TRAINING_SET)
    args = parser.parse_args()

    if not args.training_set.exists():
        log.error("Missing %s", args.training_set)
        return 1

    df = pd.read_parquet(args.training_set)
    real = df[df["is_real_yield"].astype(bool)].copy()
    log.info("Loaded %d real-yield rows", len(real))

    per_crop_features = _load_per_crop_features(args.policy)
    log.info(
        "Feature-subset policy=%s; %d crops with audit subsets",
        args.policy, len(per_crop_features),
    )

    crops_to_run = args.crop or list(ALL_CROPS)
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())

    summary: dict[str, Any] = {
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(),
            "kind": "per_crop_pruned_features_v7p",
            "policy": args.policy,
            "training_set": str(args.training_set),
            "splits": {"train": list(V7_TRAIN_YEARS),
                       "val": V7_VAL_YEAR, "test": V7_TEST_YEAR},
            "families": list(families),
            "audit_source": str(SHAP_AUDIT),
            "reference": (
                "Lundberg & Lee 2017, NeurIPS — SHAP attribution used to "
                "select per-crop feature subset that explains 95 %% "
                "of total |SHAP| importance."
            ),
        },
        "crops": {},
    }

    for crop in crops_to_run:
        features = per_crop_features.get(crop)
        if not features:
            log.warning("[%s] no audit features — skipping", crop)
            summary["crops"][crop] = {"skipped": True, "reason": "no_audit_features"}
            continue
        result = train_one_crop(crop, real, features, families, args.min_rows)
        summary["crops"][crop] = result

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
