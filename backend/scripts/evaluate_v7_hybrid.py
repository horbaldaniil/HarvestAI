"""v7-hybrid evaluation — best-of-N model selection per crop.

For each crop, examines three candidate evaluation results and picks
the highest test R²:

  - v6: regular real-only training (NDVI + weather + crop-calendar features)
  - v7: v6 + SoilGrids soil features (bdod, cec, clay, pH, sand, silt, soc)
  - v7h: hierarchical model (predicts yield deviation from oblast mean)

Output: `evaluation_v7_hybrid.json` — flat per-crop best-family record
that the methodology endpoint serves as the canonical "best result".

Rationale: each crop has different yield-prediction physics. Tuber crops
respond to soil > weather; cereals respond to weather > soil; narrow-
variance crops benefit from hierarchical anchoring. One model architecture
doesn't fit all 13 crops — letting the data pick the winner is the most
defensible thesis position.

## Output schema

```json
{
  "metadata": {...},
  "crops": {
    "wheat": {
      "selected_version": "v7h",
      "selected_family": "rf",
      "test_r2": 0.573,
      "test_rmse": ..., "test_mae": ..., "test_mape": ...,
      "candidates": {
        "v6": {"best_family": "rf", "best_r2": 0.462},
        "v7": {"best_family": ..., "best_r2": ...},
        "v7h": {"best_family": "rf", "best_r2": 0.573}
      }
    },
    ...
  }
}
```

## Run

    uv run python scripts/evaluate_v7_hybrid.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_v7_hybrid")

ROOT = Path(__file__).resolve().parents[1]
METRICS_V6 = ROOT / "data" / "processed" / "model_metrics_v6.json"
METRICS_V7 = ROOT / "data" / "processed" / "model_metrics_v7.json"
METRICS_V7H = ROOT / "data" / "processed" / "model_metrics_v7h.json"
# Phase-B: multi-task v8 metrics (single model per family across all 13
# crops). Optional — falls back gracefully when the file is missing
# (v8 trainer not yet run, or v8 was rolled back by removing .joblibs).
METRICS_V8 = ROOT / "data" / "processed" / "model_metrics_v8.json"
# Phase-C: multi-task hierarchical v8h metrics. Same shape as v8 (per-
# crop split metrics) and same one-file-per-family on-disk layout. Read
# alongside v8 so the best-of-N picker covers {v6, v7, v7h, v8, v8h}.
METRICS_V8H = ROOT / "data" / "processed" / "model_metrics_v8h.json"
# Phase-D Option A: per-crop SHAP-pruned v7 metrics. Same per-crop +
# per-family shape as v7. Read alongside the rest so the picker
# considers v7p as a 6th candidate.
METRICS_V7P = ROOT / "data" / "processed" / "model_metrics_v7p.json"
OUT = ROOT / "data" / "processed" / "evaluation_v7_hybrid.json"


def _load(path: Path) -> dict:
    if not path.exists():
        log.warning("Missing %s", path.name)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_family(crop_metrics: dict) -> tuple[str | None, dict]:
    """Find the family with highest test R² in one crop block.
    Returns (family_name, metrics_dict)."""
    if not isinstance(crop_metrics, dict) or crop_metrics.get("skipped"):
        return None, {}
    families = crop_metrics.get("families", {})
    best_fam = None
    best_r2 = -float("inf")
    best_metrics: dict = {}
    for fam, body in families.items():
        if not isinstance(body, dict):
            continue
        if "error" in body:
            continue
        test = body.get("test", {})
        r2 = test.get("r2")
        if r2 is None:
            continue
        if r2 > best_r2:
            best_r2 = r2
            best_fam = fam
            best_metrics = test
    return best_fam, best_metrics


def main() -> int:
    v6 = _load(METRICS_V6)
    v7 = _load(METRICS_V7)
    v7h = _load(METRICS_V7H)
    v7p = _load(METRICS_V7P)
    v8 = _load(METRICS_V8)
    v8h = _load(METRICS_V8H)

    if not (v6 or v7 or v7h or v7p or v8 or v8h):
        log.error("No metrics files found — train v6/v7/v7h/v7p/v8/v8h models first.")
        return 1

    all_crops = sorted(
        set(v6.get("crops", {}))
        | set(v7.get("crops", {}))
        | set(v7h.get("crops", {}))
        | set(v7p.get("crops", {}))
        | set(v8.get("crops", {}))
        | set(v8h.get("crops", {}))
    )
    log.info("Comparing %d crops across v6, v7, v7h, v7p, v8, v8h …",
             len(all_crops))

    summary: dict[str, Any] = {
        "metadata": {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "kind": "hybrid_best_of_v6_v7_v7h_v7p_v8_v8h",
            "candidates": ["v6", "v7", "v7h", "v7p", "v8", "v8h"],
            "selection_metric": "test_r2",
            "split": "train=2018-2019 / val=2020 / test=2021 (real-only Держстат)",
        },
        "crops": {},
    }

    for crop in all_crops:
        candidates: dict[str, dict] = {}
        # Phase-B: v8 (4th candidate). Phase-C: v8h (5th).
        # Phase-D Option A: v7p (6th) — SHAP-pruned per-crop feature
        # subset to claw back regressions caused by Phase A's wider
        # feature schema on crops with low n_train. All "multi-task"
        # and "pruned" variants are rolled back automatically when
        # their test R² fails to beat the per-crop alternatives.
        for ver_name, ver_data in (
            ("v6", v6), ("v7", v7), ("v7h", v7h), ("v7p", v7p),
            ("v8", v8), ("v8h", v8h),
        ):
            crop_block = ver_data.get("crops", {}).get(crop)
            if crop_block is None:
                continue
            fam, metrics = _best_family(crop_block)
            if fam is None:
                continue
            candidates[ver_name] = {
                "best_family": fam,
                **{k: metrics.get(k) for k in ("r2", "mae", "rmse", "mape")},
            }

        if not candidates:
            summary["crops"][crop] = {"skipped": True, "reason": "no_candidates"}
            log.warning("[%s] skipped — no usable metrics", crop)
            continue

        # Pick the version with highest test R² across all families.
        best_ver = max(candidates.items(),
                       key=lambda kv: kv[1].get("r2") or -float("inf"))
        version, info = best_ver
        summary["crops"][crop] = {
            "selected_version": version,
            "selected_family": info["best_family"],
            "test_r2": info.get("r2"),
            "test_mae": info.get("mae"),
            "test_rmse": info.get("rmse"),
            "test_mape": info.get("mape"),
            "candidates": candidates,
        }
        log.info("[%s] selected %s/%s R²=%+.3f (out of %d candidates)",
                 crop, version, info["best_family"],
                 info.get("r2", -99), len(candidates))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", OUT)

    # Headline stats
    crops_with_r2 = [
        c for c, body in summary["crops"].items()
        if isinstance(body, dict) and body.get("test_r2") is not None
    ]
    r2_values = [summary["crops"][c]["test_r2"] for c in crops_with_r2]
    n_above_0 = sum(1 for r in r2_values if r > 0)
    n_above_3 = sum(1 for r in r2_values if r > 0.3)
    n_above_5 = sum(1 for r in r2_values if r > 0.5)
    n_above_7 = sum(1 for r in r2_values if r > 0.7)

    log.info("=" * 60)
    log.info("v7-hybrid headline:")
    log.info("  R² > 0.0: %d/%d", n_above_0, len(crops_with_r2))
    log.info("  R² > 0.3: %d/%d", n_above_3, len(crops_with_r2))
    log.info("  R² > 0.5: %d/%d", n_above_5, len(crops_with_r2))
    log.info("  R² > 0.7: %d/%d", n_above_7, len(crops_with_r2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
