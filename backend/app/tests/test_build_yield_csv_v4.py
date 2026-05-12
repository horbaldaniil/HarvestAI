"""Tests for the v4 weather-conditioned yield synthesis.

The Phase-1a fix replaces v3's ±3 % hash-noise with weather-derived
variation. These tests guard the conversion logic + the YAML/parquet
contract — without them, regressions are silent and only visible after
a full retrain.
"""
from __future__ import annotations

import pytest

from scripts.build_yield_csv import (
    CONFLICT_FACTOR,
    WEATHER_SENSITIVITY,
    _compute_oblast_baselines,
    _compute_weather_factor,
    _deterministic_noise,
    _lookup_real_yield,
)


# ─── Weather factor logic ──────────────────────────────────


def _mock_weather(precip_sd: float, heat_sd: float, drought_sd: float,
                  temp_sd: float = 0.0) -> dict:
    """Mock weather row at the given σ-deviations from oblast mean."""
    # Build a 7-year history where baseline mean=100, std=10 for each var.
    # The query year sits at mean + sd × 10.
    return {
        "precip": 100 + precip_sd * 10,
        "temp": 20 + temp_sd * 1,
        "heat": 10 + heat_sd * 5,
        "drought": 8 + drought_sd * 4,
    }


def _mock_baseline() -> dict:
    return {
        "precip": {"mean": 100.0, "std": 10.0},
        "temp": {"mean": 20.0, "std": 1.0},
        "heat": {"mean": 10.0, "std": 5.0},
        "drought": {"mean": 8.0, "std": 4.0},
    }


def test_weather_factor_returns_one_when_no_data():
    """If we have no weather features available, factor falls back to
    1.0 with `weather_used=False`."""
    factor, used = _compute_weather_factor("wheat", "UA-46", 2020, None, None)
    assert factor == 1.0
    assert used is False


def test_weather_factor_returns_one_when_zero_filler():
    """The 9 % of (iso, year) rows that hit Open-Meteo's 429 limit have
    precip=temp=0. We must NOT treat those as "no rain" → drought
    penalty; we treat them as missing."""
    weather = {("UA-46", 2020): {"precip": 0.0, "temp": 0.0, "heat": 0.0, "drought": 0.0}}
    baselines = {"UA-46": _mock_baseline()}
    factor, used = _compute_weather_factor("wheat", "UA-46", 2020, weather, baselines)
    assert factor == 1.0
    assert used is False


def test_weather_factor_drought_reduces_yield():
    """A year +2σ drier than baseline (≈+8 dry-spell days above oblast mean)
    should reduce the yield factor below 1.0."""
    weather = {("UA-46", 2020): _mock_weather(precip_sd=-2, heat_sd=0, drought_sd=+2)}
    baselines = {"UA-46": _mock_baseline()}
    factor, used = _compute_weather_factor("corn", "UA-46", 2020, weather, baselines)
    assert used is True
    assert factor < 1.0


def test_weather_factor_good_year_boosts_yield():
    """A wet, cool year (above-mean precip, low drought, low heat)
    should produce factor > 1.0."""
    weather = {("UA-46", 2020): _mock_weather(precip_sd=+1, heat_sd=-1, drought_sd=-1)}
    baselines = {"UA-46": _mock_baseline()}
    factor, used = _compute_weather_factor("wheat", "UA-46", 2020, weather, baselines)
    assert used is True
    assert factor > 1.0


def test_weather_factor_clipped_to_30_percent():
    """Extreme weather (+4σ everything) must clip to [0.7, 1.3] — real
    annual yield envelope (per Mazur & Roik 2014)."""
    bad = _mock_weather(precip_sd=-4, heat_sd=+4, drought_sd=+4)
    weather = {("UA-46", 2020): bad}
    baselines = {"UA-46": _mock_baseline()}
    factor, _ = _compute_weather_factor("corn", "UA-46", 2020, weather, baselines)
    assert 0.70 <= factor <= 1.30


def test_drought_sensitive_crops_react_more():
    """Corn is more drought-sensitive than rye per WEATHER_SENSITIVITY
    table → same drought should hurt corn more."""
    drought = _mock_weather(precip_sd=0, heat_sd=0, drought_sd=+2)
    weather = {("UA-46", 2020): drought}
    baselines = {"UA-46": _mock_baseline()}
    corn_factor, _ = _compute_weather_factor("corn", "UA-46", 2020, weather, baselines)
    rye_factor, _ = _compute_weather_factor("rye", "UA-46", 2020, weather, baselines)
    assert corn_factor < rye_factor  # corn penalised more


# ─── Oblast baseline computation ──────────────────────────


def test_baselines_use_only_nonzero_weather_data():
    """Zero-filled rows (rate-limit failures) must not contaminate the
    oblast's historical mean+std."""
    weather = {
        ("UA-46", 2017): {"precip": 200, "temp": 18, "heat": 5, "drought": 8},
        ("UA-46", 2018): {"precip": 250, "temp": 19, "heat": 7, "drought": 10},
        ("UA-46", 2019): {"precip": 220, "temp": 17, "heat": 4, "drought": 9},
        # 2020 was rate-limited → zero
        ("UA-46", 2020): {"precip": 0.0, "temp": 0.0, "heat": 0.0, "drought": 0.0},
    }
    bl = _compute_oblast_baselines(weather)
    assert bl is not None
    # The mean of [200, 250, 220] = 223.33; if zeros were included it'd be 167.5
    assert 200 < bl["UA-46"]["precip"]["mean"] < 240


def test_baselines_returns_none_when_weather_missing():
    assert _compute_oblast_baselines(None) is None


def test_baselines_handles_single_year():
    """With only 1 data point, std defaults to 1.0 to avoid div/0."""
    weather = {("UA-46", 2017): {"precip": 200, "temp": 18, "heat": 5, "drought": 8}}
    bl = _compute_oblast_baselines(weather)
    assert bl is not None
    assert bl["UA-46"]["precip"]["std"] == 1.0


# ─── Sensitivity table integrity ───────────────────────────


def test_all_crops_have_sensitivities():
    """Every crop in `ALL_CROPS` must have a sensitivity entry.
    A missing crop silently falls back to default sensitivities — okay
    for robustness but bad for thesis defensibility."""
    from app.data_reference.crop_zones import ALL_CROPS
    for crop in ALL_CROPS:
        assert crop in WEATHER_SENSITIVITY, f"missing sensitivity entry for {crop}"


def test_sensitivities_in_plausible_range():
    """Each sensitivity coefficient must be in (0, 0.20) to keep the
    total factor swing within the documented ±30 % envelope."""
    for crop, sens in WEATHER_SENSITIVITY.items():
        for var, value in sens.items():
            assert 0 < value < 0.20, f"{crop}.{var} = {value} out of range"


# ─── Conflict factor ───────────────────────────────────────


def test_conflict_factor_realistic():
    """Documented 40-50 % production loss in conflict zones (NaUKMA
    Crisis Atlas + UN OCHA estimates 2014-2023). 0.55 hits the middle."""
    assert 0.45 <= CONFLICT_FACTOR <= 0.65


# ─── Real-yield lookup ─────────────────────────────────────


def test_lookup_real_yield_returns_none_when_yaml_section_missing():
    assert _lookup_real_yield(None, "wheat", 2020, "UA-46") is None
    assert _lookup_real_yield({}, "wheat", 2020, "UA-46") is None


def test_lookup_real_yield_returns_match():
    yaml_block = {"wheat": {2020: {"UA-46": 4.85, "UA-63": 4.20}}}
    assert _lookup_real_yield(yaml_block, "wheat", 2020, "UA-46") == 4.85
    assert _lookup_real_yield(yaml_block, "wheat", 2020, "UA-63") == 4.20


def test_lookup_real_yield_partial_coverage():
    """If the YAML only has wheat (no corn), corn lookups return None,
    so build_yield_csv falls back to synthesis. This is the realistic
    incremental-population pattern."""
    yaml_block = {"wheat": {2020: {"UA-46": 4.85}}}
    assert _lookup_real_yield(yaml_block, "corn", 2020, "UA-46") is None
    assert _lookup_real_yield(yaml_block, "wheat", 2020, "UA-12") is None  # different oblast
    assert _lookup_real_yield(yaml_block, "wheat", 2018, "UA-46") is None  # different year


# ─── Deterministic noise hash ──────────────────────────────


def test_deterministic_noise_bounds():
    """±amplitude envelope."""
    for amp in (0.01, 0.015, 0.03):
        for i in range(50):
            noise = _deterministic_noise("seed", "wheat", 2020, f"o{i}", amp)
            assert -amp <= noise <= amp


def test_deterministic_noise_reproducible():
    """Same inputs → same noise (bit-for-bit). Guards reproducibility."""
    a = _deterministic_noise("v4", "wheat", 2020, "lviv", 0.015)
    b = _deterministic_noise("v4", "wheat", 2020, "lviv", 0.015)
    assert a == b


def test_deterministic_noise_distinct_per_crop():
    """Different crops at the same (year, oblast) get different noise —
    otherwise all crops in one oblast would be perfectly correlated."""
    n1 = _deterministic_noise("v4", "wheat", 2020, "lviv", 0.015)
    n2 = _deterministic_noise("v4", "corn", 2020, "lviv", 0.015)
    assert n1 != n2
