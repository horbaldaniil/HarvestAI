"""Tests for the crop × agro-climatic-zone feasibility matrix."""
from __future__ import annotations

import pytest

from app.data_reference.crop_zones import (
    ALL_CROPS,
    ALL_ZONES,
    feasible_crops,
    feasible_zones,
    is_grown,
    yield_multiplier,
)
from app.db.models.enums import CropType


# ─── Enum / matrix consistency ──────────────────────────────


def test_all_crops_matches_enum():
    """`ALL_CROPS` MUST mirror the `CropType` enum — otherwise a new
    crop added to one but not the other silently disappears from the
    training pipeline. This is the canary for that mistake."""
    enum_values = {c.value for c in CropType}
    assert set(ALL_CROPS) == enum_values
    assert len(ALL_CROPS) == 13


def test_all_zones_count():
    assert len(ALL_ZONES) == 5


# ─── Coverage rules ─────────────────────────────────────────


@pytest.mark.parametrize("crop", ALL_CROPS)
def test_every_crop_grows_in_at_least_one_zone(crop: str):
    """A crop with zero feasible zones is a bug — either the matrix
    is wrong or the crop shouldn't be in the enum."""
    zones = feasible_zones(crop)
    assert len(zones) >= 1, f"{crop} has no feasible zone"


@pytest.mark.parametrize("zone", ALL_ZONES)
def test_every_zone_grows_at_least_one_crop(zone: str):
    crops = feasible_crops(zone)
    assert len(crops) >= 1, f"{zone} grows no crop"


def test_forest_steppe_is_most_diverse_zone():
    """Sanity: forest-steppe is Ukraine's most productive and
    biome-diverse zone — it should support every cultivated crop."""
    crops = feasible_crops("forest_steppe")
    assert set(crops) == set(ALL_CROPS)


# ─── Negative cases reflecting agronomic reality ────────────


def test_sunflower_not_in_polissia():
    """Sunflower needs >130 frost-free days — northern Polissia is too
    cool. Multiple peer-reviewed sources confirm this; the matrix encodes it."""
    assert is_grown("sunflower", "polissia") is False
    assert yield_multiplier("sunflower", "polissia") is None


def test_sunflower_not_in_transcarpathia():
    assert is_grown("sunflower", "transcarpathia") is False


def test_sugar_beet_not_in_southern_steppe():
    """Sugar beet needs >450 mm rain; southern steppe is too dry for
    economic production."""
    assert is_grown("sugar_beet", "steppe_south") is False


def test_rye_not_in_southern_steppe():
    """Rye historically marginalised in hot/dry southern oblasts."""
    assert is_grown("rye", "steppe_south") is False


def test_buckwheat_not_in_southern_steppe():
    assert is_grown("buckwheat", "steppe_south") is False


# ─── Positive cases ────────────────────────────────────────


def test_wheat_grows_everywhere():
    """Wheat — winter and spring varieties combined — is cultivated in
    every Ukrainian agro-climatic zone."""
    for zone in ALL_ZONES:
        assert is_grown("wheat", zone), f"wheat missing in {zone}"


def test_corn_grows_everywhere():
    for zone in ALL_ZONES:
        assert is_grown("corn", zone), f"corn missing in {zone}"


def test_potato_strongest_in_polissia():
    """Polissia is Ukraine's potato heartland — its multiplier should be
    the highest across zones."""
    mults = {
        z: yield_multiplier("potato", z)
        for z in ALL_ZONES
        if yield_multiplier("potato", z) is not None
    }
    assert mults["polissia"] == max(mults.values())


def test_sunflower_strongest_in_northern_steppe():
    """Northern steppe is the historical sunflower belt."""
    mults = {
        z: yield_multiplier("sunflower", z)
        for z in ALL_ZONES
        if yield_multiplier("sunflower", z) is not None
    }
    assert mults["steppe_north"] == max(mults.values())


# ─── Multipliers in plausible range ─────────────────────────


def test_all_multipliers_in_plausible_range():
    """No multiplier should be wildly outside [0.5, 1.3] — that signals
    a typo (e.g. extra digit). Even the strongest regional advantage
    (potato in Polissia) sits at ≤1.20; the weakest at ≥0.70."""
    for crop in ALL_CROPS:
        for zone in ALL_ZONES:
            m = yield_multiplier(crop, zone)
            if m is None:
                continue
            assert 0.50 <= m <= 1.30, f"{crop}/{zone} multiplier {m} out of range"


# ─── Unknown crop / zone ────────────────────────────────────


def test_unknown_crop_returns_false_or_none():
    assert is_grown("watermelon", "steppe_south") is False
    assert yield_multiplier("watermelon", "steppe_south") is None


def test_unknown_zone_returns_false_or_none():
    assert is_grown("wheat", "tundra") is False  # type: ignore[arg-type]
    assert yield_multiplier("wheat", "tundra") is None  # type: ignore[arg-type]
