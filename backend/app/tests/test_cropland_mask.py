"""Unit tests for the WorldCover cropland-mask sampling filter.

We don't download real WorldCover tiles in CI — they're 80 MB each and
the AWS S3 bucket isn't always reachable from headless runners. Instead
we build a tiny synthetic GeoTIFF mimicking the same coordinate system
+ class encoding (class 40 = cropland) and exercise the filter against
it. The synthetic tile is a 100×100 pixel raster covering exactly the
3°×3° tile `N48E024` (so it occupies the same Lviv/Ternopil region as
the real product) — left half cropland, right half forest.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import box

from scripts.generate_oblast_samples import (
    CROPLAND_CLASS,
    MIN_CROPLAND_FRACTION,
    CroplandMask,
)


def _build_synthetic_tile(dest_dir: Path) -> Path:
    """Write a tiny WorldCover-shaped GeoTIFF covering N48E024..N51E027.

    Half the raster (longitude < 25.5°) is cropland (class 40), the
    other half (longitude ≥ 25.5°) is forest (class 10). Square samples
    placed near the centre divide land on a mix; samples deep in either
    half are unambiguous.

    Resolution: 3000 × 3000 pixels over 3°×3° = 0.001° per pixel
    (~100 m at 50°N). Coarser than real WorldCover (10 m), but lets a
    0.01° sample square cover ~100 raster pixels — enough granularity
    for partial-coverage threshold tests.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    minx, miny, maxx, maxy = 24.0, 48.0, 27.0, 51.0
    width = height = 3000  # 0.001° per pixel
    arr = np.full((height, width), 10, dtype="uint8")  # default = forest
    arr[:, : width // 2] = CROPLAND_CLASS                # left half = cropland

    dest = dest_dir / "ESA_WorldCover_10m_2021_v200_N48E024_Map.tif"
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    with rasterio.open(
        dest, "w",
        driver="GTiff", width=width, height=height,
        count=1, dtype="uint8", crs="EPSG:4326", transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)
    return dest


@pytest.fixture
def synthetic_worldcover_dir(tmp_path: Path) -> Path:
    _build_synthetic_tile(tmp_path)
    return tmp_path


# ─── Tile picking ───────────────────────────────────────────


def test_tile_for_point_within_n48e024():
    assert CroplandMask._tile_for_point(49.0, 25.0) == \
        "ESA_WorldCover_10m_2021_v200_N48E024_Map.tif"


def test_tile_for_point_within_n51e030():
    assert CroplandMask._tile_for_point(52.5, 31.5) == \
        "ESA_WorldCover_10m_2021_v200_N51E030_Map.tif"


def test_tile_for_point_at_boundary():
    # Boundaries fall to the lower-left tile (matches the (lat // 3) * 3 logic).
    assert CroplandMask._tile_for_point(48.0, 24.0) == \
        "ESA_WorldCover_10m_2021_v200_N48E024_Map.tif"


# ─── is_cropland() classification ───────────────────────────


def test_square_in_cropland_half_accepted(synthetic_worldcover_dir: Path):
    """A square centred at (49°N, 24.5°E) is entirely in the cropland half."""
    mask = CroplandMask(synthetic_worldcover_dir)
    sq = box(24.495, 48.995, 24.505, 49.005)  # ~1 km × 1 km
    assert mask.is_cropland(sq) is True
    mask.close()


def test_square_in_forest_half_rejected(synthetic_worldcover_dir: Path):
    mask = CroplandMask(synthetic_worldcover_dir)
    sq = box(26.495, 48.995, 26.505, 49.005)
    assert mask.is_cropland(sq) is False
    mask.close()


def test_square_outside_any_tile_rejected(synthetic_worldcover_dir: Path):
    """A square in a region without any downloaded tile must be rejected
    rather than crash. (The downstream sampling loop would otherwise
    silently accept everything when tiles are missing.)"""
    mask = CroplandMask(synthetic_worldcover_dir)
    # (40°N, 50°E) — nowhere near Ukraine, no tile available.
    sq = box(49.995, 39.995, 50.005, 40.005)
    assert mask.is_cropland(sq) is False
    mask.close()


def test_min_cropland_fraction_threshold(synthetic_worldcover_dir: Path):
    """A square straddling the 25.5° divide is ~50 % cropland — below
    the 0.6 default → rejected; above the relaxed 0.4 threshold → accepted."""
    strict = CroplandMask(synthetic_worldcover_dir, min_fraction=0.6)
    relaxed = CroplandMask(synthetic_worldcover_dir, min_fraction=0.4)
    sq = box(25.495, 48.995, 25.505, 49.005)
    assert strict.is_cropland(sq) is False
    assert relaxed.is_cropland(sq) is True
    strict.close()
    relaxed.close()


def test_default_min_fraction_in_plausible_range():
    """Guard against accidentally relaxing the threshold during tuning."""
    assert 0.4 <= MIN_CROPLAND_FRACTION <= 0.8


# ─── Class code sanity ──────────────────────────────────────


def test_cropland_class_is_40():
    """The CROPLAND_CLASS constant MUST be 40 — ESA WorldCover convention.
    A regression here would silently invalidate every masked sample."""
    assert CROPLAND_CLASS == 40
