"""Tests for the Open-Meteo response parser (no real HTTP)."""
from __future__ import annotations

from datetime import date

from app.integrations.openmeteo.client import _parse_daily


def test_parse_typical_response():
    payload = {
        "daily": {
            "time": ["2024-07-01", "2024-07-02", "2024-07-03"],
            "temperature_2m_max": [28.5, 31.2, 33.1],
            "temperature_2m_min": [16.0, 18.5, 19.0],
            "temperature_2m_mean": [22.5, 25.0, 26.5],
            "precipitation_sum": [0.0, 1.4, 12.3],
            "relative_humidity_2m_mean": [62, 58, 65],
            "shortwave_radiation_sum": [22.4, 24.1, 26.0],
        }
    }
    rows = _parse_daily(payload)
    assert len(rows) == 3
    assert rows[0].observed_on == date(2024, 7, 1)
    assert rows[0].temp_max_c == 28.5
    assert rows[2].precip_mm == 12.3


def test_parse_handles_missing_values():
    payload = {
        "daily": {
            "time": ["2024-07-01", "2024-07-02"],
            "temperature_2m_max": [28.5, None],
            "temperature_2m_min": [16.0, 18.0],
            "temperature_2m_mean": [22.5, None],
            "precipitation_sum": [0.0, None],
            "relative_humidity_2m_mean": [62, None],
            "shortwave_radiation_sum": [22.4, None],
        }
    }
    rows = _parse_daily(payload)
    assert len(rows) == 2
    assert rows[1].temp_max_c is None
    assert rows[1].precip_mm is None


def test_parse_empty_response():
    assert _parse_daily({}) == []
    assert _parse_daily({"daily": {}}) == []


def test_parse_skips_invalid_dates():
    payload = {
        "daily": {
            "time": ["not-a-date", "2024-07-01"],
            "temperature_2m_max": [25.0, 28.0],
            "temperature_2m_min": [15.0, 18.0],
            "temperature_2m_mean": [20.0, 23.0],
            "precipitation_sum": [0.0, 1.0],
            "relative_humidity_2m_mean": [60, 65],
            "shortwave_radiation_sum": [20.0, 22.0],
        }
    }
    rows = _parse_daily(payload)
    assert len(rows) == 1
    assert rows[0].observed_on == date(2024, 7, 1)
