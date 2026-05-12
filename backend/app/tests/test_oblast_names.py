"""Tests for the canonical oblast naming table.

The whole point of `oblast_names.py` is to eliminate the "silent join
drop" bug class — a free-text name from one source not matching the
free-text name from another. These tests prove the round-trip works
for every alternative spelling we know about.
"""
from __future__ import annotations

import pytest

from app.data_reference.oblast_names import (
    ALL_OBLASTS,
    OBLAST_BY_ISO,
    agricultural_oblasts,
    iso_from_any_name,
    oblast_by_iso,
)


# ─── Table integrity ────────────────────────────────────────


def test_25_entries_total():
    """24 oblasts + Kyiv City = 25 administrative units."""
    assert len(ALL_OBLASTS) == 25


def test_24_oblasts_are_agricultural():
    agri = agricultural_oblasts()
    assert len(agri) == 24
    # Only Kyiv City should be excluded.
    excluded = {o.slug for o in ALL_OBLASTS} - {o.slug for o in agri}
    assert excluded == {"kyiv_city"}


def test_iso_codes_unique():
    isos = [o.iso_3166_2 for o in ALL_OBLASTS]
    assert len(set(isos)) == len(isos)


def test_slugs_unique_and_ascii():
    slugs = [o.slug for o in ALL_OBLASTS]
    assert len(set(slugs)) == len(slugs)
    for s in slugs:
        assert s.isascii(), f"slug {s!r} contains non-ASCII"
        assert s.islower(), f"slug {s!r} must be lowercase"
        assert " " not in s


def test_oblast_by_iso_lookup():
    assert oblast_by_iso("UA-46").slug == "lviv"
    assert oblast_by_iso("UA-32").slug == "kyiv"
    assert oblast_by_iso("UA-30").slug == "kyiv_city"
    assert oblast_by_iso("UA-99") is None
    assert oblast_by_iso(None) is None
    assert oblast_by_iso("") is None
    # Case-insensitive on input.
    assert oblast_by_iso("ua-46").slug == "lviv"


def test_oblast_by_iso_constant_dict_matches_tuple():
    """The pre-built `OBLAST_BY_ISO` dict must cover the same set."""
    assert {o.iso_3166_2 for o in ALL_OBLASTS} == set(OBLAST_BY_ISO.keys())


# ─── iso_from_any_name() — the main bug-class guard ─────────


@pytest.mark.parametrize(
    "spelling,expected_iso",
    [
        # Natural Earth romanisation (with apostrophes)
        ("L'viv",            "UA-46"),
        ("Khmel'nyts'kyy",   "UA-68"),
        ("Dnipropetrovs'k",  "UA-12"),
        ("Donets'k",         "UA-14"),
        ("Luhans'k",         "UA-09"),
        ("Ternopil'",        "UA-61"),
        # Canonical English (no apostrophes)
        ("Lviv",            "UA-46"),
        ("Khmelnytskyi",    "UA-68"),
        ("Dnipropetrovsk",  "UA-12"),
        ("Donetsk",         "UA-14"),
        ("Luhansk",         "UA-09"),
        ("Ternopil",        "UA-61"),
        # Legacy build_yield_csv.py spellings
        ("Dnipro",          "UA-12"),
        ("Odessa",          "UA-51"),
        ("Kyiv",            "UA-32"),
        ("Kiev",            "UA-32"),
        # Slugs
        ("lviv",            "UA-46"),
        ("kyiv_city",       "UA-30"),
        ("ivano_frankivsk", "UA-26"),
        # Ukrainian
        ("Львівська",        "UA-46"),
        ("Львівська область", "UA-46"),
        ("Київська",         "UA-32"),
        ("м. Київ",          "UA-30"),
        # Mixed case / whitespace
        ("  lviv ",         "UA-46"),
        ("LVIV",            "UA-46"),
    ],
)
def test_iso_from_any_name_round_trip(spelling: str, expected_iso: str):
    assert iso_from_any_name(spelling) == expected_iso


def test_iso_from_any_name_unknown_returns_none():
    assert iso_from_any_name("Atlantis") is None
    assert iso_from_any_name("Berlin") is None


def test_iso_from_any_name_none_or_empty():
    assert iso_from_any_name(None) is None
    assert iso_from_any_name("") is None
    assert iso_from_any_name("   ") is None


# ─── Geographic / agronomic sanity ──────────────────────────


def test_all_centroids_inside_ukraine_bbox():
    """Sanity: every centroid is within Ukraine's bounding box.

    Approximate WGS84 bbox: (44.0, 22.0) to (53.0, 41.0). Kyiv City
    obviously fits; the most peripheral oblasts (Zakarpattia, Luhansk)
    define the bounds.
    """
    for o in ALL_OBLASTS:
        assert 44.0 <= o.centroid_lat <= 53.0, f"{o.slug} lat out of bbox"
        assert 22.0 <= o.centroid_lon <= 41.0, f"{o.slug} lon out of bbox"


def test_conflict_oblasts_marked():
    """Donetsk / Luhansk / Kherson / Zaporizhzhia / Kharkiv must declare
    a `conflict_zone_since` year so the evaluator can exclude them."""
    conflict_isos = {
        o.iso_3166_2 for o in ALL_OBLASTS if o.conflict_zone_since is not None
    }
    expected = {"UA-14", "UA-09", "UA-65", "UA-23", "UA-63"}
    assert conflict_isos == expected
