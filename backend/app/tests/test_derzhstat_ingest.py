"""Tests for the Phase-1 refactor of `derzhstat_ingest.py` — keyword-
based sheet matching + latest-month-per-year selection + partial-year
flag propagation. These tests guard against silent regression of the
ingest contract when Держстат changes bulletin numbering across years.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.derzhstat_ingest import (
    LATE_HARVEST_CROPS,
    _match_crop,
    parse_month_year,
    pick_latest_per_year,
)


# ─── _match_crop keyword regex ──────────────────────────────


@pytest.mark.parametrize("sheet_name,expected", [
    # Standard November / December bulletin (full set)
    ("6 пшен", "wheat"),
    ("12 кукур", "corn"),
    ("14 ячм", "barley"),
    ("20 житоОЗ", "rye"),
    ("22 овес", "oats"),
    ("24 гречка", "buckwheat"),
    ("31 горох", "peas"),
    ("33 соя", "soybean"),
    ("35 ріпак", "rapeseed"),
    ("41 соняш", "sunflower"),
    ("43 буряк цукр", "sugar_beet"),
    ("45 карт", "potato"),
    ("50 кукур корм", "corn_silage"),
])
def test_match_crop_november_bulletin(sheet_name: str, expected: str):
    assert _match_crop(sheet_name) == expected


@pytest.mark.parametrize("sheet_name,expected", [
    # July bulletin — fewer sheets, lower numeric prefix
    ("3 пшен", "wheat"),
    ("4 ячм", "barley"),
    ("5 житоОЗ", "rye"),
])
def test_match_crop_july_bulletin_positions_differ(sheet_name: str, expected: str):
    """Sheet POSITION changes between months; keyword regex must still match."""
    assert _match_crop(sheet_name) == expected


@pytest.mark.parametrize("sheet_name", [
    "8 пшенОЗ",        # winter-wheat-only — skip because "пшен" combined wins elsewhere
    "10 пшенЯР",       # spring wheat — same
    "16 ячмОЗ",
    "37 ріпакОЗ",
    "39 ріпакЯР",
])
def test_match_crop_skips_variety_only_sheets(sheet_name: str):
    """Pure-winter / pure-spring variants are skipped to avoid double-
    counting. The combined `пшен` / `ячм` / `ріпак` sheets aggregate both."""
    assert _match_crop(sheet_name) is None


@pytest.mark.parametrize("sheet_name", [
    "1",
    "Продовження / Continuation",
    "Заголовок",
    "",
])
def test_match_crop_non_crop_sheets_return_none(sheet_name: str):
    assert _match_crop(sheet_name) is None


def test_match_crop_silage_wins_before_grain():
    """`кукур корм` (silage) regex MUST match before plain `кукур`,
    otherwise sheet \"50 кукур корм\" would be classified as `corn`
    (grain), losing the silage data."""
    # Specific-first ordering is verified by both expectations.
    assert _match_crop("50 кукур корм") == "corn_silage"
    assert _match_crop("12 кукур") == "corn"


def test_match_crop_sugar_beet_qualifier():
    """`буряк` alone could mean fodder beet; we require `цукр` (sugar)."""
    assert _match_crop("43 буряк цукр") == "sugar_beet"


# ─── parse_month_year ──────────────────────────────────────


@pytest.mark.parametrize("stem,expected", [
    ("ovuzpsg_1118", (11, 2018)),
    ("ovuzpsg_1221", (12, 2021)),
    ("ovuzpsg_0725", (7, 2025)),
    ("ovuzpsg_0720", (7, 2020)),
    ("OVUZPSG_1019", (10, 2019)),   # case-insensitive
])
def test_parse_month_year(stem: str, expected: tuple[int, int]):
    assert parse_month_year(stem) == expected


def test_parse_month_year_unknown_format():
    assert parse_month_year("random_file") is None
    assert parse_month_year("yield_2021.xls") is None


# ─── pick_latest_per_year ──────────────────────────────────


def test_pick_latest_per_year_groups_by_year(tmp_path: Path):
    """Multiple months of same year → keep only the rightmost.
    Different years are independent."""
    files = [
        tmp_path / "ovuzpsg_0718.xls",
        tmp_path / "ovuzpsg_0918.xls",
        tmp_path / "ovuzpsg_1118.xls",       # latest 2018
        tmp_path / "ovuzpsg_1019.xls",
        tmp_path / "ovuzpsg_1119.xls",       # latest 2019
        tmp_path / "ovuzpsg_1221.xls",       # latest 2021
    ]
    for f in files:
        f.touch()
    out = pick_latest_per_year(files)
    assert sorted(out.keys()) == [2018, 2019, 2021]
    assert out[2018].name == "ovuzpsg_1118.xls"
    assert out[2019].name == "ovuzpsg_1119.xls"
    assert out[2021].name == "ovuzpsg_1221.xls"


def test_pick_latest_per_year_handles_single_month(tmp_path: Path):
    """A year with only one month present → that month is the latest."""
    files = [tmp_path / "ovuzpsg_0725.xls"]
    files[0].touch()
    out = pick_latest_per_year(files)
    assert out == {2025: files[0]}


def test_pick_latest_per_year_skips_unparseable(tmp_path: Path):
    """Files that don't match the naming pattern are ignored gracefully."""
    files = [
        tmp_path / "ovuzpsg_1121.xls",
        tmp_path / "random_extra.xls",
    ]
    for f in files:
        f.touch()
    out = pick_latest_per_year(files)
    assert list(out.keys()) == [2021]


# ─── Late-harvest crop annotations ─────────────────────────


def test_late_harvest_crops_set():
    """Sugar beet, potato, corn — these have substantial post-October
    harvest. Pre-Nov bulletins → flag as is_partial_year."""
    assert "sugar_beet" in LATE_HARVEST_CROPS
    assert "potato" in LATE_HARVEST_CROPS
    assert "corn" in LATE_HARVEST_CROPS
    # Early-harvest crops MUST NOT be in the late-harvest set
    # (otherwise they'd be flagged partial unnecessarily).
    assert "wheat" not in LATE_HARVEST_CROPS
    assert "barley" not in LATE_HARVEST_CROPS
    assert "rye" not in LATE_HARVEST_CROPS
