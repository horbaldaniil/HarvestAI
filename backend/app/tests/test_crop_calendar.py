"""Tests for the per-crop phenological calendar (Phase 3).

The calendar drives v4's crop-specific weather features. A bug here
(wrong sowing month, wrong GDD base) silently corrupts every per-crop
feature derivation. These tests catch the classic typo-class regressions.
"""
from __future__ import annotations

import pytest

from app.data_reference.crop_calendar import (
    CROP_CALENDAR,
    REFERENCE_WINDOW_MONTHS,
    flowering_in_reference,
    gdd_proxy,
    overlap_with_reference,
)
from app.data_reference.crop_zones import ALL_CROPS


# ─── Table integrity ────────────────────────────────────────


def test_all_crops_have_calendar_entries():
    """If we add a crop to the enum but forget the calendar, v4 features
    silently default to 0 — silent data corruption."""
    for crop in ALL_CROPS:
        assert crop in CROP_CALENDAR, f"missing calendar entry for {crop}"


def test_13_crops_total():
    assert len(CROP_CALENDAR) == 13


def test_all_gdd_bases_in_plausible_range():
    """GDD base temp typically 0-15 °C. >15 means we treat the whole
    growing season as below-base → zero accumulation (likely typo)."""
    for crop, cal in CROP_CALENDAR.items():
        assert 0.0 <= cal.gdd_base_c <= 15.0, f"{crop}.gdd_base = {cal.gdd_base_c}"


def test_all_months_in_1_12():
    for crop, cal in CROP_CALENDAR.items():
        assert 1 <= cal.sow_month <= 12
        assert 1 <= cal.peak_month <= 12
        assert 1 <= cal.harvest_month <= 12


# ─── Winter vs spring crops ─────────────────────────────────


@pytest.mark.parametrize("crop", ["wheat", "rye", "rapeseed"])
def test_winter_crops_sow_in_autumn(crop: str):
    """Winter wheat / winter rye / winter rapeseed sow in Sept-Oct."""
    cal = CROP_CALENDAR[crop]
    assert cal.sow_month in (9, 10, 11), f"{crop} sown in {cal.sow_month}"


@pytest.mark.parametrize("crop", ["wheat", "rye", "rapeseed"])
def test_winter_crops_wrap_growing_season(crop: str):
    """Winter crops' growing season wraps around year-end."""
    cal = CROP_CALENDAR[crop]
    gs = cal.growing_season_months
    # Should include both end-of-year months AND beginning-of-year months
    has_winter = any(m in (10, 11, 12) for m in gs)
    has_spring = any(m in (1, 2, 3, 4) for m in gs)
    assert has_winter and has_spring, f"{crop} growing season {gs}"


@pytest.mark.parametrize("crop", ["corn", "sunflower", "soybean", "potato",
                                   "sugar_beet", "barley", "oats"])
def test_spring_crops_sow_in_spring(crop: str):
    """Spring crops sow in April-May."""
    cal = CROP_CALENDAR[crop]
    assert cal.sow_month in (4, 5), f"{crop} sown in {cal.sow_month}"


# ─── Growing season order ───────────────────────────────────


def test_growing_season_includes_both_endpoints():
    """`growing_season_months` must include sow_month and harvest_month."""
    for crop, cal in CROP_CALENDAR.items():
        gs = cal.growing_season_months
        assert cal.sow_month in gs, f"{crop} missing sow month"
        assert cal.harvest_month in gs, f"{crop} missing harvest month"
        assert cal.peak_month in gs, f"{crop} missing peak month"


def test_growing_season_length_reasonable():
    """Crop seasons range 3 months (buckwheat) to 12 months (winter wheat
    + rye if we count the dormancy period)."""
    for crop, cal in CROP_CALENDAR.items():
        n = cal.growing_season_length_months
        assert 3 <= n <= 12, f"{crop} season {n} months"


# ─── Window overlap calculations ───────────────────────────


def test_spring_cereals_full_overlap_with_aprjul():
    """Barley/oats/peas — pure spring crops, growing season Apr-Jul → 100% overlap."""
    for crop in ("barley", "oats", "peas"):
        assert overlap_with_reference(crop) == 1.0, f"{crop} overlap"


def test_winter_wheat_partial_overlap():
    """Winter wheat October-July → 4 of 10 months overlap (Apr-Jul)."""
    overlap = overlap_with_reference("wheat")
    assert 0.35 <= overlap <= 0.45, f"wheat overlap = {overlap}"


def test_long_season_crops_lower_overlap():
    """Sugar beet (April-October, 7 months) has lower overlap than peas
    (April-July, 4 months) because more of its season is OUTSIDE Apr-Jul."""
    assert overlap_with_reference("sugar_beet") < overlap_with_reference("peas")
    assert overlap_with_reference("corn") < overlap_with_reference("oats")


def test_unknown_crop_overlap_zero():
    assert overlap_with_reference("atlantis") == 0.0


# ─── Flowering-in-window logic ─────────────────────────────


def test_flowering_in_window_for_spring_cereals():
    """Barley peak June, oats peak June, peas peak June — all in Apr-Jul."""
    for crop in ("barley", "oats", "peas", "wheat", "rapeseed"):
        assert flowering_in_reference(crop), f"{crop}"


def test_flowering_outside_window_for_late_crops():
    """Sunflower peaks in July (still in window), sugar beet peaks in
    August (outside Apr-Jul). Corn peaks in July (inside)."""
    assert not flowering_in_reference("sugar_beet")  # Aug peak — outside


def test_flowering_unknown_crop_false():
    assert not flowering_in_reference("atlantis")


# ─── GDD proxy ─────────────────────────────────────────────


def test_gdd_proxy_zero_when_temp_below_base():
    """If mean temp is below the crop's base (e.g. corn base 10 with
    temp 8) → no accumulation."""
    assert gdd_proxy("corn", 8.0) == 0.0


def test_gdd_proxy_scales_with_temp_excess():
    """Higher temp → higher GDD."""
    low = gdd_proxy("wheat", 12.0)  # base 5 → excess 7
    high = gdd_proxy("wheat", 18.0)  # base 5 → excess 13
    assert high > low


def test_gdd_proxy_none_when_temp_missing():
    assert gdd_proxy("wheat", None) is None
    assert gdd_proxy("wheat", 0.0) is None  # 0 = rate-limit filler


def test_gdd_proxy_uses_crop_specific_base():
    """Corn base 10 °C, rye base 4 °C → same mean temp gives less GDD for corn."""
    corn_gdd = gdd_proxy("corn", 15.0)        # excess 5
    rye_gdd = gdd_proxy("rye", 15.0)          # excess 11
    assert rye_gdd > corn_gdd


def test_gdd_proxy_unknown_crop_none():
    assert gdd_proxy("atlantis", 15.0) is None


# ─── Reference window constant ─────────────────────────────


def test_reference_window_is_apr_jul():
    """The constant matches the v3 Sentinel-Hub collection window."""
    assert REFERENCE_WINDOW_MONTHS == (4, 5, 6, 7)
