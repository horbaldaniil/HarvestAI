"""SHAP-importance audit for v7 per-crop models.

For each crop:
  1. Loads the v7 tree models (RF / XGBoost / LightGBM — Stack is
     skipped because its SHAP explainer reads the meta-layer's input
     space, not the original 57 features).
  2. Computes mean(|SHAP value|) per feature on the **training set**
     of real-yield rows. Average across the three tree families to
     get a robust ranking — a feature that only matters to one family
     usually isn't structural signal.
  3. Sorts features by descending importance and writes per-crop:
     - `ranked_features`: list of (feature_name, importance, cum_share)
     - `top_K_features` for `K ∈ {20, 25, 30, 35, 40}` — Phase D
       trainer reads whichever K is configured per crop.
     - `auto_top_features`: cumulative-95%-share cutoff — the smallest
       number of features that together explain 95 % of total |SHAP|.

## Why average across families (not max, not min)

Tree-specific quirks: XGBoost emphasises split-friendly thresholds,
LightGBM gradient boosting trees emphasise leaf-residuals, RF
emphasises bagging variance. Each over- or under-weights certain
features. The **mean** is the natural ensemble — the same logic as
why our prediction stack averages tree families. Any feature that
only one family considers important is likely a quirk; one with
consistent high importance across all three is structural.

## Output schema

```json
{
  "metadata": {...},
  "by_crop": {
    "wheat": {
      "n_train_rows": 56,
      "ranked_features": [
        {"name": "ndvi_peak", "importance": 0.0341, "cum_share": 0.082},
        ...
      ],
      "top_20_features": [...],
      "top_25_features": [...],
      "top_30_features": [...],
      "top_35_features": [...],
      "top_40_features": [...],
      "auto_top_features": [...]   // 95% cumulative cutoff
    },
    ...
  }
}
```

## Run

    uv run python scripts/feature_importance_audit_v7.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feature_importance_audit_v7")

PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"
MODELS_DIR = ROOT / "models"
OUT = ROOT / "data" / "processed" / "feature_importance_v7.json"

# v7 train years from the canonical v7 chronological split. We compute
# SHAP on the TRAIN set only — feature importance attribution should
# reflect what the model used to learn, not held-out evaluation rows.
V7_TRAIN_YEARS: tuple[int, ...] = (2018, 2019)
V7_VAL_YEAR: int = 2020

TREE_FAMILIES: tuple[str, ...] = ("rf", "xgboost", "lightgbm")
FAMILY_PREFIX = {"rf": "rf", "xgboost": "xgb", "lightgbm": "lgbm"}


def _per_family_shap(
    payload: dict, X: np.ndarray, feature_names: list[str],
) -> dict[str, float]:
    """`{feature_name: mean(|SHAP value|)}` on X for the loaded model."""
    model = payload.get("model") or payload.get("point")
    if model is None:
        return {}
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP failed: %s", exc)
        return {}
    # shap_values shape: (n_samples, n_features) for regression.
    # For tree models a list[(n,k)] sometimes shows up — flatten.
    if isinstance(vals, list):
        vals = vals[0]
    mean_abs = np.abs(np.asarray(vals)).mean(axis=0)
    return dict(zip(feature_names, mean_abs.tolist(), strict=True))


def _aggregate(per_family: list[dict[str, float]]) -> dict[str, float]:
    """Mean of (positive) per-family importances. Missing keys default
    to 0 — a family that didn't load contributes neutral weight, not
    NaN propagation. Returns the same dict shape with averaged values."""
    if not per_family:
        return {}
    all_keys = set()
    for d in per_family:
        all_keys.update(d.keys())
    out: dict[str, float] = {}
    for k in all_keys:
        vs = [d.get(k, 0.0) for d in per_family]
        out[k] = float(sum(vs) / len(per_family))
    return out


def _rank(importances: dict[str, float]) -> list[dict]:
    """Sort features by descending importance; emit cumulative-share
    alongside so a single pass can answer the K-cutoff question for
    any K. Filters out zero-importance entries (they're noise — the
    feature didn't appear in any split)."""
    items = [(k, v) for k, v in importances.items() if v > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in items) or 1.0
    out: list[dict] = []
    cum = 0.0
    for name, val in items:
        cum += val
        out.append({
            "name": name,
            "importance": round(val, 6),
            "cum_share": round(cum / total, 4),
        })
    return out


def main() -> int:
    if not PARQUET.exists():
        log.error("Missing parquet %s", PARQUET)
        return 1
    df = pd.read_parquet(PARQUET)
    real = df[df["is_real_yield"].astype(bool)].copy()
    # SHAP attribution uses TRAIN years only — held-out years would
    # measure something different (generalisation residual, not what
    # the model used to learn). Same rule as importance attribution
    # in the Lundberg-Lee 2017 paper.
    train_mask = real["year"].isin(list(V7_TRAIN_YEARS) + [V7_VAL_YEAR])
    train_df = real[train_mask].copy()
    log.info("Train+val rows for SHAP attribution: %d", len(train_df))

    by_crop: dict[str, dict] = {}

    for crop in ALL_CROPS:
        sub = train_df[train_df["crop"] == crop]
        if len(sub) < 5:
            log.info("[%s] skip — only %d rows", crop, len(sub))
            continue

        # Load each tree family's v7 payload. Skip silently when one
        # isn't on disk (e.g. trainer crashed for that crop+family
        # earlier; we still get a useful audit from the others).
        per_family_imports: list[dict[str, float]] = []
        ref_features: list[str] | None = None
        for family in TREE_FAMILIES:
            short = FAMILY_PREFIX[family]
            path = MODELS_DIR / f"yield_{short}_{crop}_v7.joblib"
            if not path.exists():
                continue
            try:
                payload = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load %s: %s", path.name, exc)
                continue
            feat_list = payload.get("features") or []
            if not feat_list:
                continue
            missing = [f for f in feat_list if f not in sub.columns]
            if missing:
                log.info("[%s/%s] parquet missing %d features — skipping",
                         crop, family, len(missing))
                continue
            X = sub[list(feat_list)].to_numpy()
            shap_imp = _per_family_shap(payload, X, list(feat_list))
            if shap_imp:
                per_family_imports.append(shap_imp)
                ref_features = list(feat_list)

        if not per_family_imports:
            log.warning("[%s] no tree-family SHAP — skipping", crop)
            continue

        aggregated = _aggregate(per_family_imports)
        ranked = _rank(aggregated)

        # Auto cutoff at 95 % cumulative |SHAP|. For most crops this
        # lands around 20-35 features — the natural Pareto knee.
        auto_top_n = next(
            (i + 1 for i, entry in enumerate(ranked) if entry["cum_share"] >= 0.95),
            len(ranked),
        )
        auto_top_features = [entry["name"] for entry in ranked[:auto_top_n]]

        # Fixed-K cuts. The trainer picks one of these per crop based
        # on a per-crop K-policy ("narrow crops → K=25, wide → K=35").
        topks: dict[str, list[str]] = {}
        for k in (20, 25, 30, 35, 40):
            topks[f"top_{k}_features"] = [e["name"] for e in ranked[:k]]

        by_crop[crop] = {
            "n_train_rows": int(len(sub)),
            "n_families_used": len(per_family_imports),
            "n_features_seen": len(ref_features or []),
            "ranked_features": ranked,
            "auto_top_n": auto_top_n,
            "auto_top_features": auto_top_features,
            **topks,
        }
        log.info(
            "[%s] n=%d, 95%% cutoff at %d features (top-3: %s)",
            crop, len(sub), auto_top_n,
            [e["name"] for e in ranked[:3]],
        )

    out = {
        "metadata": {
            "audited_at": datetime.now(UTC).isoformat(),
            "method": "mean_abs_shap_across_tree_families",
            "tree_families": list(TREE_FAMILIES),
            "train_years": list(V7_TRAIN_YEARS) + [V7_VAL_YEAR],
            "reference": "Lundberg & Lee 2017, NeurIPS — A Unified Approach to Interpreting Model Predictions",
        },
        "by_crop": by_crop,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
