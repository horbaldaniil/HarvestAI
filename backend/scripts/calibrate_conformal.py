"""Split-conformal prediction-interval calibration.

For each `(crop, family, version)` ∈ all-13 × {rf, xgb, lgbm, stack}
× {v7, v7h}, computes the 90 % conformal half-width from absolute
residuals on a held-out calibration set, and writes a flat JSON
lookup that `app/ml/yield_model.py` consumes at inference time.

## Background

Standard split conformal (Lei et al. 2018, §3):

  1. Hold out a calibration set the model never trained on.
  2. Predict on it, compute absolute residuals `r_i = |y_i − ŷ_i|`.
  3. Pick quantile `q = ⌈(n+1)(1-α)⌉ / n` of the residuals — guarantees
     coverage ≥ `1-α` in expectation under exchangeability.
  4. At test time, the `(1-α)` prediction interval for a new ŷ is
     `[ŷ − q, ŷ + q]`.

Where this beats the legacy heuristic `confidence = 0.3 + 0.1×(len(shap)-5)`:
the radius is **data-driven** from each model's actual error distribution.
A weak model (oats v7h R² ≈ -0.6) honestly reports a large band; a
strong model (wheat v7 R² 0.63) reports a tight one. Same number means
the same thing across crops — currently it doesn't, because the heuristic
ignores the model's actual accuracy entirely.

## Calibration set choice

We use **year 2021** (the test set in the v7 chronological split). Key
reason: the **Stack regressor** trains on `train + val` together (this
is the standard `StackingRegressor` workflow — meta-learner OOF
predictions are produced over the union), which means **2020 is NOT
truly held-out for stack** even though it is for individual tree
families. Calibrating stack against 2020 produces unrealistically tight
radii (wheat-stack on 2020 → 0.45 т/га while its tree siblings, which
DID hold 2020 out, produce 1.6 т/га). Using 2021 — held out from
every family by every trainer — gives a fair apples-to-apples radius
across the board.

Caveat: the reported `test_r2` in `evaluation_v7_hybrid.json` is
*also* computed from 2021 residuals. Conformal radius and R² therefore
come from the same residual set — that's fine because they measure
different things (R² = variance-explained; conformal radius = upper
quantile of absolute residuals). For coverage on a **fresh** future
season (e.g. 2026) you still rely on the exchangeability assumption
between 2021 and the future year — standard split-conformal limitation.

A separate held-out 2022 set would be cleaner but we don't yet have
real-yield 2022 from Держстат — calibrating against synthetic-yield
rows would inflate coverage by an unknown factor.

## Output schema

```json
{
  "metadata": {...},
  "alpha": 0.10,
  "by_crop": {
    "wheat": {
      "v7":  {"rf": 0.62, "xgboost": 0.58, "lightgbm": 0.55, "stack": 0.51},
      "v7h": {"rf": 0.49, ...},
    },
    ...
  }
}
```

Half-width values are in t/ha. `yield_model.py` reads this lazily on
first inference (lru_cached) and uses it to populate `value_tha_q05` /
`value_tha_q95` for all model families, replacing the XGBoost-only
quantile path.

## Run

    uv run python scripts/calibrate_conformal.py
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_reference.crop_zones import ALL_CROPS  # noqa: E402
from scripts.train_yield_models_v7 import FEATURE_NAMES as FEATURE_NAMES_V7  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calibrate_conformal")

PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"
OUT = ROOT / "data" / "processed" / "conformal_intervals_v7.json"
MODELS_DIR = ROOT / "models"

# 2021 — the chronological-split test year, held out from every family
# by every trainer (including stack, whose OOF folding spans train+val
# only). See the module docstring for the leakage argument that drove
# this choice over the more obvious "use the val year" default.
CALIB_YEAR = 2021
ALPHA = 0.10  # 90 % prediction interval


def _conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected quantile per Lei et al. 2018, Theorem 1.

    For `n` calibration points, the right quantile level is
    `⌈(n+1)(1-α)⌉ / n`. Clamp to 1.0 because for very small n this
    can exceed 1.0 (then we use the max residual — conservative)."""
    n = len(abs_residuals)
    if n == 0:
        return float("nan")
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(abs_residuals, q_level, method="higher"))


_NO_CROP_SLOT = frozenset({"v8", "v8h"})


def _load_payload(crop: str, family: str, version: str) -> dict | None:
    short = {"rf": "rf", "xgboost": "xgb", "lightgbm": "lgbm",
             "stack": "stack"}[family]
    # Phase-B / Phase-C: v8 + v8h multi-task models have no crop slot
    # in the filename. One file per family covers all 13 crops; for
    # v8h the per-(crop, iso) means are stored inside the payload.
    if version in _NO_CROP_SLOT:
        path = MODELS_DIR / f"yield_{short}_{version}.joblib"
    else:
        path = MODELS_DIR / f"yield_{short}_{crop}_{version}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s: %s", path.name, exc)
        return None


def _predict(payload: dict, X: np.ndarray, calib_iso: np.ndarray,
             version: str, crop: str) -> np.ndarray:
    """Run inference on the calibration set in **original yield units
    (т/га)** — the same scale residuals are reported in.

    Two inversions may apply:

      1. `payload["log_target"]=True` (v7 sugar_beet / potato /
         corn_silage): the regressor predicts in log-space; we
         `np.expm1` BEFORE adding the oblast offset (v7h/v8h) or
         returning directly (v7/v7p/v8). Without this, the conformal
         residuals would be computed in log-space and the persisted
         radius would mean nothing to the UI.
      2. **v7h** (per-crop hierarchical) and **v8h** (multi-task
         hierarchical) predict a deviation from per-(crop, oblast)
         mean. We add `oblast_means[iso]` (v7h, dict keyed by iso)
         or `oblast_means[crop][iso]` (v8h, nested dict) back to
         recover absolute yield. The oblast means themselves were
         stored in original yield units by both trainers (neither
         uses log-target), so the addition is always linear here.
    """
    model = payload.get("model")
    if model is None:
        model = payload.get("point")  # xgboost-triple
    if model is None:
        return np.full(len(X), np.nan)
    raw = np.asarray(model.predict(X), dtype=float)
    if payload.get("log_target"):
        raw = np.expm1(raw)
    if version in ("v7h", "v8h"):
        means_root = payload.get("oblast_means", {}) or {}
        # v8h: nested {crop: {iso: mean}}. v7h: flat {iso: mean}.
        means_by_iso = means_root.get(crop, {}) if version == "v8h" else means_root
        global_ = means_by_iso.get("_GLOBAL", 0.0)
        oblast_means_arr = np.array(
            [means_by_iso.get(i, global_) for i in calib_iso], dtype=float,
        )
        return raw + oblast_means_arr
    return raw


def main() -> int:
    if not PARQUET.exists():
        log.error("Missing parquet %s", PARQUET)
        return 1
    df = pd.read_parquet(PARQUET)
    if "is_real_yield" not in df.columns:
        log.error("Parquet missing is_real_yield column")
        return 1
    real = df[df["is_real_yield"].astype(bool)].copy()
    # Phase-B: v8 multi-task models expect 13 crop_* one-hot columns
    # but the parquet stores `crop` as a single text column. Inject the
    # one-hots inline (mirror of `scripts/train_yield_models_v8.py`).
    # No-op for crops already present.
    from app.data_reference.crop_zones import ALL_CROPS as _ALL_CROPS
    for c in _ALL_CROPS:
        col = f"crop_{c}"
        if col not in real.columns:
            real[col] = (real["crop"] == c).astype(int)

    calib = real[real["year"] == CALIB_YEAR]
    log.info("Calibration year=%d: %d real-yield rows", CALIB_YEAR, len(calib))

    feature_names_v7 = list(FEATURE_NAMES_V7)
    missing = [f for f in feature_names_v7 if f not in calib.columns]
    if missing:
        log.error("Parquet missing feature cols: %s", missing)
        return 1

    by_crop: dict[str, dict[str, dict[str, float]]] = {}

    for crop in ALL_CROPS:
        sub = calib[calib["crop"] == crop]
        if len(sub) < 5:
            log.info("[%s] skipping — only %d calibration rows", crop, len(sub))
            continue
        y = sub["yield_tha"].to_numpy()
        isos = sub["iso_3166_2"].to_numpy()

        crop_block: dict[str, dict[str, float]] = {}
        # Phase-B/C/D: include all calibration-eligible versions:
        #   - v7: per-crop classic
        #   - v7h: per-crop hierarchical (oblast_means offset)
        #   - v7p: per-crop with SHAP-pruned features (Phase D)
        #   - v8: multi-task plain
        #   - v8h: multi-task hierarchical
        # `_load_payload` + `_predict` branch on version where needed.
        for version in ("v7", "v7h", "v7p", "v8", "v8h"):
            ver_block: dict[str, float] = {}
            for family in ("rf", "xgboost", "lightgbm", "stack"):
                payload = _load_payload(crop, family, version)
                if payload is None:
                    continue
                # Each artifact records its own feature list (v7 = 38
                # incl. SoilGrids, v7h = 31 without soil). Reading
                # `payload["features"]` keeps us in lockstep with whatever
                # the trainer last wrote, instead of hard-coding the v7
                # schema and choking on v7h's narrower vector.
                version_features = payload.get("features") or feature_names_v7
                missing_for_version = [
                    f for f in version_features if f not in sub.columns
                ]
                if missing_for_version:
                    log.info(
                        "[%s/%s/%s] parquet missing feature(s) %s — skipping",
                        crop, family, version, missing_for_version[:3],
                    )
                    continue
                X = sub[list(version_features)].to_numpy()
                try:
                    preds = _predict(payload, X, isos, version, crop)
                    residuals = np.abs(y - preds)
                    if np.isnan(residuals).any():
                        log.info(
                            "[%s/%s/%s] residuals contain NaN — skipping",
                            crop, family, version,
                        )
                        continue
                    radius = _conformal_quantile(residuals, ALPHA)
                    ver_block[family] = round(radius, 3)
                    log.info(
                        "[%s/%s/%s] n=%d 90%%-CI radius=%.3f t/ha (MAE=%.3f)",
                        crop, family, version, len(residuals),
                        radius, residuals.mean(),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[%s/%s/%s] calibration failed: %s",
                        crop, family, version, exc,
                    )
            if ver_block:
                crop_block[version] = ver_block
        if crop_block:
            by_crop[crop] = crop_block

    out = {
        "metadata": {
            "calibrated_at": datetime.now(UTC).isoformat(),
            "calibration_year": CALIB_YEAR,
            "alpha": ALPHA,
            "coverage": 1 - ALPHA,
            "method": "split_conformal_absolute_residual",
            "reference": "Lei et al. 2018, JASA — Distribution-Free Predictive Inference for Regression",
        },
        "by_crop": by_crop,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
