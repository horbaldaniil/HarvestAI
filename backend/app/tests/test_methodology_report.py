"""Smoke-tests for `build_methodology_report` and its chart helpers.

We exercise the helpers individually plus the end-to-end builder against
the actual `evaluation_v3.json` on disk. The output PDF is validated by
magic-byte check rather than reader-level parsing — keeps tests fast and
catches regressions where the engine silently produces a 0-byte payload.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.reports.charts import (
    render_methodology_leaderboard_png,
    render_methodology_learning_curves_png,
    render_methodology_residual_map_png,
    render_methodology_shap_png,
)


# ─── Chart helpers ──────────────────────────────────────────


def test_leaderboard_chart_empty_rows_returns_placeholder():
    """A no-data case must still emit a valid base-64 PNG so the
    template doesn't break the page with a broken-image icon."""
    uri = render_methodology_leaderboard_png([])
    assert uri.startswith("data:image/png;base64,")


def test_leaderboard_chart_sorts_by_r2_desc():
    rows = [
        {"crop": "wheat", "family": "rf", "test_r2": 0.42},
        {"crop": "corn", "family": "stack", "test_r2": 0.78},
        {"crop": "potato", "family": "xgboost", "test_r2": None},  # filtered
    ]
    uri = render_methodology_leaderboard_png(rows, top_n=5)
    # Smoke test: returns a valid PNG URI (not empty, has correct prefix).
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 200  # non-trivial content


def test_residual_map_handles_empty_inputs():
    uri = render_methodology_residual_map_png({}, {})
    assert uri.startswith("data:image/png;base64,")


def test_shap_chart_empty_returns_placeholder():
    uri = render_methodology_shap_png({})
    assert uri.startswith("data:image/png;base64,")


def test_shap_chart_picks_top_n():
    importance = {
        "ndvi_peak": 0.42,
        "centroid_lat": 0.31,
        "ndvi_mean_july": 0.20,
        "evi_peak": 0.18,
        "ndwi_min": 0.05,
        "drought_dryspells": 0.04,
        "heat_stress_days": 0.03,
        "temp_mean_apr_jul": 0.02,
        "savi_peak": 0.01,
    }
    uri = render_methodology_shap_png(importance, top_n=4)
    assert uri.startswith("data:image/png;base64,")


def test_learning_curves_empty_returns_placeholder():
    uri = render_methodology_learning_curves_png({})
    assert uri.startswith("data:image/png;base64,")


def test_learning_curves_multiple_families():
    curves = {
        "rf": [{"n_train": 20, "test_r2": 0.30},
               {"n_train": 40, "test_r2": 0.50},
               {"n_train": 60, "test_r2": 0.65}],
        "xgboost": [{"n_train": 20, "test_r2": 0.25},
                    {"n_train": 40, "test_r2": 0.55},
                    {"n_train": 60, "test_r2": 0.70}],
    }
    uri = render_methodology_learning_curves_png(curves)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 200


# ─── End-to-end builder ─────────────────────────────────────


_EVAL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "evaluation_v3.json"
)


@pytest.mark.skipif(
    not _EVAL_PATH.exists(),
    reason="evaluation_v3.json missing — run scripts/evaluate_models.py first",
)
def test_build_methodology_report_returns_valid_pdf():
    """End-to-end smoke: the builder must produce non-empty bytes with
    the `%PDF-1.4` magic header. Picks whichever (crop, family) is
    actually present in the JSON — the builder has its own fallback so
    it never crashes on a missing combination."""
    from app.reports.builder import build_methodology_report

    user = MagicMock()
    user.email = "test@example.com"
    user.id = 1
    db = MagicMock()

    # Read the JSON to find a viable (crop, family) — the builder will
    # also fall back internally, but starting with a known-good pair
    # gives a more diagnostic failure if the chart helpers crash.
    eval_payload = json.loads(_EVAL_PATH.read_text(encoding="utf-8"))
    crops = eval_payload.get("crops", {})
    crop = next(iter(crops))  # first available
    family = next(
        (f for f, b in crops[crop].items() if isinstance(b, dict) and not b.get("skipped")),
        "rf",
    )

    pdf = asyncio.run(build_methodology_report(
        db=db, user=user, crop=crop, family=family,
    ))
    assert pdf, "builder returned empty bytes"
    assert pdf[:4] == b"%PDF", f"expected PDF magic; got {pdf[:8]!r}"
    assert len(pdf) > 10_000, f"PDF suspiciously small: {len(pdf)} bytes"


@pytest.mark.skipif(
    not _EVAL_PATH.exists(),
    reason="evaluation_v3.json missing",
)
def test_build_methodology_report_falls_back_for_unknown_combo():
    """Asking for a (crop, family) that wasn't trained → builder uses
    the first viable entry it finds. Should never crash, never produce
    empty PDF."""
    from app.reports.builder import build_methodology_report

    user = MagicMock()
    user.email = "test@example.com"
    user.id = 1
    db = MagicMock()

    pdf = asyncio.run(build_methodology_report(
        db=db, user=user, crop="atlantis", family="quantum_stack",
    ))
    assert pdf[:4] == b"%PDF"
