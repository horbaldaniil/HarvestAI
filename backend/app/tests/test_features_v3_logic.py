"""Unit tests for the pieces of `build_features_v3.py` that we can
exercise without real Sentinel-2 / Open-Meteo data: the year-spec parser
of `collect_oblast_s2.py` and the ISO-attachment helper.

The full end-to-end run requires:
  - `oblast_samples_v2.geojson` (Phase 2)
  - `oblast_s2_observations_v2.parquet` (Phase 3 — Sentinel Hub PU spend)
  - `usda_ukraine_yield_2017_2023.csv` v3 (Phase 1 — already in repo)

That's verified manually on a wall-clock-bound run, not in CI.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_features_v3 import _attach_iso
from scripts.collect_oblast_s2 import _parse_years


# ─── --years parser ────────────────────────────────────────


def test_parse_years_single():
    assert _parse_years("2023") == (2023,)


def test_parse_years_range():
    assert _parse_years("2017-2023") == (2017, 2018, 2019, 2020, 2021, 2022, 2023)


def test_parse_years_list():
    assert _parse_years("2018,2021,2023") == (2018, 2021, 2023)


def test_parse_years_none_or_empty():
    assert _parse_years(None) is None
    assert _parse_years("") is None


def test_parse_years_whitespace_tolerant():
    assert _parse_years("  2017-2019 ") == (2017, 2018, 2019)


def test_parse_years_inclusive_range_endpoints():
    """Range MUST include both endpoints — half-open is a documented
    footgun in collect scripts (we keep saying 'years 2017-2023' meaning
    seven years, not six)."""
    years = _parse_years("2020-2022")
    assert 2020 in years and 2022 in years
    assert len(years) == 3


# ─── _attach_iso helper ─────────────────────────────────────


def test_attach_iso_resolves_natural_earth_spellings():
    df = pd.DataFrame({
        "oblast": ["L'viv", "Dnipropetrovs'k", "Khmel'nyts'kyy"],
        "year": [2023, 2023, 2023],
    })
    out = _attach_iso(df, "oblast")
    assert list(out["iso_3166_2"]) == ["UA-46", "UA-12", "UA-68"]


def test_attach_iso_resolves_legacy_short_names():
    """The legacy `build_yield_csv.py` (pre-v3) emitted short forms like
    `"Lviv"`, `"Dnipro"`, `"Odessa"`. These must still resolve so the
    feature builder works against the existing v2 parquet on disk."""
    df = pd.DataFrame({
        "oblast": ["Lviv", "Dnipro", "Odessa", "Kyiv"],
        "year": [2023, 2023, 2023, 2023],
    })
    out = _attach_iso(df, "oblast")
    assert set(out["iso_3166_2"]) == {"UA-46", "UA-12", "UA-51", "UA-32"}


def test_attach_iso_drops_unresolvable_rows():
    """An unknown name must NOT poison the dataframe with a None ISO —
    the row is dropped (with a warning) so downstream merges don't see
    a phantom oblast."""
    df = pd.DataFrame({
        "oblast": ["L'viv", "Atlantis", "Kharkiv"],
        "year": [2023, 2023, 2023],
    })
    out = _attach_iso(df, "oblast")
    assert "Atlantis" not in out["oblast"].values
    assert len(out) == 2


def test_attach_iso_preserves_other_columns():
    df = pd.DataFrame({
        "oblast": ["L'viv", "Kharkiv"],
        "year": [2023, 2022],
        "ndvi_peak": [0.82, 0.64],
    })
    out = _attach_iso(df, "oblast")
    # Original columns intact + new iso column appended.
    for col in ("oblast", "year", "ndvi_peak", "iso_3166_2"):
        assert col in out.columns
    assert out.loc[out["oblast"] == "L'viv", "ndvi_peak"].iloc[0] == pytest.approx(0.82)


def test_attach_iso_does_not_mutate_input():
    df = pd.DataFrame({"oblast": ["L'viv"], "year": [2023]})
    snapshot = df.copy()
    _ = _attach_iso(df, "oblast")
    pd.testing.assert_frame_equal(df, snapshot)


# ─── Yield CSV v3 integrity ─────────────────────────────────


def test_yield_csv_v3_has_iso_column():
    """The v3-built yield CSV must include `iso_3166_2` so v3 feature
    builder can join on the canonical key (not free-text)."""
    from pathlib import Path

    csv_path = (
        Path(__file__).resolve().parents[2] / "data" / "raw"
        / "usda_ukraine_yield_2017_2023.csv"
    )
    df = pd.read_csv(csv_path)
    assert "iso_3166_2" in df.columns, \
        "build_yield_csv.py v3 must emit iso_3166_2 column"
    assert df["iso_3166_2"].notna().all()


def test_yield_csv_v3_has_all_13_crops():
    from pathlib import Path

    csv_path = (
        Path(__file__).resolve().parents[2] / "data" / "raw"
        / "usda_ukraine_yield_2017_2023.csv"
    )
    df = pd.read_csv(csv_path)
    crops_in_csv = set(df["crop"].unique())
    from app.data_reference.crop_zones import ALL_CROPS
    # Every crop should appear at least once (in its feasible zones).
    missing = set(ALL_CROPS) - crops_in_csv
    assert missing == set(), f"Crops missing from CSV: {missing}"


def test_yield_csv_v3_covers_all_24_oblasts():
    from pathlib import Path

    csv_path = (
        Path(__file__).resolve().parents[2] / "data" / "raw"
        / "usda_ukraine_yield_2017_2023.csv"
    )
    df = pd.read_csv(csv_path)
    isos = set(df["iso_3166_2"].unique())
    # 24 oblasts; Kyiv City (UA-30) is non-agricultural and excluded.
    from app.data_reference.oblast_names import agricultural_oblasts
    expected = {o.iso_3166_2 for o in agricultural_oblasts()}
    assert isos == expected, f"Missing oblasts: {expected - isos}"


def test_yield_csv_v3_has_conflict_flag():
    from pathlib import Path

    csv_path = (
        Path(__file__).resolve().parents[2] / "data" / "raw"
        / "usda_ukraine_yield_2017_2023.csv"
    )
    df = pd.read_csv(csv_path)
    assert "conflict_zone" in df.columns
    # Conflict rows: Donetsk/Luhansk all 7 years + Kherson/Zaporizhzhia/Kharkiv
    # from 2022. Boolean column should have True for those.
    n_conflict = int(df["conflict_zone"].sum())
    assert n_conflict > 0, "Expected some conflict_zone=True rows"
