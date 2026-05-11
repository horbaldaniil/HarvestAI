"""Tests for the Statistical API response parser.

We don't hit Sentinel Hub; we feed canned JSON responses and verify parsing.
"""
from __future__ import annotations

from datetime import date

from app.integrations.sentinel_hub.statistical_api import (
    IndexAggregates,
    parse_statistical_response,
)


def _make_entry(date_iso: str, *, ndvi: dict | None, evi: dict | None = None):
    outputs: dict = {}
    if ndvi is not None:
        outputs["ndvi"] = {"bands": {"B0": {"stats": ndvi}}}
    if evi is not None:
        outputs["evi"] = {"bands": {"B0": {"stats": evi}}}
    return {
        "interval": {"from": f"{date_iso}T00:00:00Z", "to": f"{date_iso}T23:59:59Z"},
        "outputs": outputs,
    }


def test_parse_normal_response():
    payload = {
        "data": [
            _make_entry(
                "2024-07-01",
                ndvi={"mean": 0.73, "min": 0.45, "max": 0.91, "stDev": 0.05},
                evi={"mean": 0.6},
            )
        ]
    }
    rows = parse_statistical_response(payload)
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, IndexAggregates)
    assert r.observed_on == date(2024, 7, 1)
    assert r.ndvi_mean == 0.73
    assert r.evi_mean == 0.6


def test_parse_missing_outputs_yields_empty_row():
    payload = {
        "data": [
            {"interval": {"from": "2024-08-01T00:00:00Z"}, "outputs": {}}
        ]
    }
    rows = parse_statistical_response(payload)
    assert len(rows) == 1
    assert rows[0].ndvi_mean is None
    assert rows[0].evi_mean is None


def test_parse_nan_string_treated_as_none():
    payload = {
        "data": [
            _make_entry(
                "2024-06-15",
                ndvi={"mean": "NaN", "min": "NaN", "max": "NaN", "stDev": "NaN"},
            )
        ]
    }
    rows = parse_statistical_response(payload)
    assert rows[0].ndvi_mean is None


def test_parse_skips_entry_without_interval():
    payload = {"data": [{"outputs": {"ndvi": {}}}]}
    rows = parse_statistical_response(payload)
    assert rows == []
