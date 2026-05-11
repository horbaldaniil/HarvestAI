"""Tests for chart and QR rendering used by PDF reports."""
from __future__ import annotations

import base64

import pytest

from app.reports.charts import render_index_chart_png, render_shap_chart_png
from app.reports.qr import render_qr_png


def _decode_data_uri(uri: str) -> bytes:
    assert uri.startswith("data:image/png;base64,")
    return base64.b64decode(uri.removeprefix("data:image/png;base64,"))


def test_index_chart_returns_png_data_uri():
    points = [
        ("2025-05-01", 0.5),
        ("2025-05-15", 0.6),
        ("2025-06-01", 0.72),
        ("2025-06-15", None),  # cloudy gap
        ("2025-07-01", 0.81),
    ]
    uri = render_index_chart_png(points, index="ndvi")
    blob = _decode_data_uri(uri)
    # PNG magic bytes
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    # Non-trivial size
    assert len(blob) > 1000


def test_shap_chart_handles_empty_contributions():
    uri = render_shap_chart_png([])
    blob = _decode_data_uri(uri)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"


def test_shap_chart_renders_positive_and_negative():
    contribs = [
        {"name": "NDVI peak", "value": 0.78, "contribution": 1.2},
        {"name": "Опади Jun-Jul", "value": 120, "contribution": 0.6},
        {"name": "Heat stress days", "value": 8, "contribution": -0.4},
    ]
    uri = render_shap_chart_png(contribs)
    blob = _decode_data_uri(uri)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(blob) > 1500


def test_qr_returns_scannable_png():
    uri = render_qr_png("https://example.com/fields?selected=42")
    blob = _decode_data_uri(uri)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
