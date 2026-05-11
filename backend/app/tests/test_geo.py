"""Unit tests for geometry validation and conversion helpers."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.schemas.field import PolygonGeometry
from app.utils.geo import (
    MAX_AREA_HA,
    MAX_VERTICES,
    MIN_AREA_HA,
    compute_area_ha,
    geojson_polygon_to_shape,
    polygon_to_wkt,
    validate_polygon,
)


def _square_around(lat: float, lon: float, side_m: float) -> PolygonGeometry:
    """Build a roughly-square polygon centered at (lat, lon) with given side in meters.

    Useful for size-bound tests. Uses a flat-Earth approximation — good enough
    for fields well under a few km on a side.
    """
    # ~111_320 m per degree latitude; longitude scales by cos(lat).
    import math

    half_lat = (side_m / 2.0) / 111_320.0
    half_lon = (side_m / 2.0) / (111_320.0 * math.cos(math.radians(lat)))

    return PolygonGeometry(
        type="Polygon",
        coordinates=[
            [
                [lon - half_lon, lat - half_lat],
                [lon + half_lon, lat - half_lat],
                [lon + half_lon, lat + half_lat],
                [lon - half_lon, lat + half_lat],
                [lon - half_lon, lat - half_lat],
            ]
        ],
    )


# ───── Conversions ───────────────────────────────────────────────


def test_geojson_polygon_to_shape_roundtrip():
    poly = _square_around(49.84, 24.03, 100.0)  # ~1 ha near Lviv
    shape_obj = geojson_polygon_to_shape(poly)
    assert shape_obj.is_valid
    # WKT contains "POLYGON" and the expected coordinate prefix.
    wkt = polygon_to_wkt(poly)
    assert wkt.startswith("POLYGON ((")


def test_compute_area_ha_matches_expected_within_1_percent():
    # 200 m x 200 m square near equator → ~4 ha
    poly = _square_around(0.0, 0.0, 200.0)
    area = compute_area_ha(poly)
    assert abs(area - 4.0) / 4.0 < 0.01


def test_compute_area_ha_at_high_latitude():
    # 100 m x 100 m square at lat 49.84 → ~1 ha despite lat distortion
    poly = _square_around(49.84, 24.03, 100.0)
    area = compute_area_ha(poly)
    assert abs(area - 1.0) / 1.0 < 0.02


# ───── Validation: valid cases ───────────────────────────────────


def test_validate_polygon_accepts_normal_field():
    poly = _square_around(49.84, 24.03, 200.0)  # ~4 ha near Lviv
    result = validate_polygon(poly)
    assert result.is_valid


def test_validate_polygon_accepts_at_min_boundary():
    # Slightly above 0.5 ha
    poly = _square_around(49.84, 24.03, 75.0)  # ~0.56 ha
    validate_polygon(poly)  # must not raise


def test_validate_polygon_accepts_at_max_boundary():
    # Around 100 ha (well below 1000 cap)
    poly = _square_around(49.84, 24.03, 1000.0)
    validate_polygon(poly)


# ───── Validation: failure cases ─────────────────────────────────


def test_validate_polygon_rejects_too_small():
    poly = _square_around(49.84, 24.03, 30.0)  # ~0.09 ha
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400
    assert "мал" in ei.value.detail.lower()


def test_validate_polygon_rejects_too_large():
    poly = _square_around(49.84, 24.03, 5000.0)  # ~2500 ha
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400
    assert "велик" in ei.value.detail.lower()


def test_validate_polygon_rejects_too_few_vertices():
    poly = PolygonGeometry(
        type="Polygon",
        coordinates=[[[0.0, 0.0], [0.001, 0.0], [0.0, 0.0]]],  # only 2 unique
    )
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400


def test_validate_polygon_rejects_holes():
    poly = PolygonGeometry(
        type="Polygon",
        coordinates=[
            # outer
            [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]],
            # hole (would be valid GeoJSON, but we forbid)
            [
                [0.002, 0.002],
                [0.008, 0.002],
                [0.008, 0.008],
                [0.002, 0.008],
                [0.002, 0.002],
            ],
        ],
    )
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400
    assert "отвор" in ei.value.detail.lower() or "контур" in ei.value.detail.lower()


def test_validate_polygon_rejects_self_intersection():
    # bowtie polygon — sides cross
    poly = PolygonGeometry(
        type="Polygon",
        coordinates=[
            [
                [0.0, 0.0],
                [0.01, 0.01],
                [0.01, 0.0],
                [0.0, 0.01],
                [0.0, 0.0],
            ]
        ],
    )
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400


def test_validate_polygon_rejects_out_of_range_lat():
    poly = PolygonGeometry(
        type="Polygon",
        coordinates=[
            [
                [0.0, 91.0],  # invalid latitude
                [0.001, 91.0],
                [0.001, 91.001],
                [0.0, 91.001],
                [0.0, 91.0],
            ]
        ],
    )
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400
    assert "широт" in ei.value.detail.lower()


def test_validate_polygon_rejects_too_many_vertices():
    # Generate MAX_VERTICES + 2 unique vertices on a tiny circle
    import math

    n = MAX_VERTICES + 2
    cx, cy = 24.03, 49.84
    r = 0.01
    ring = [
        [cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)]
        for i in range(n)
    ]
    ring.append(ring[0])  # close
    poly = PolygonGeometry(type="Polygon", coordinates=[ring])
    with pytest.raises(HTTPException) as ei:
        validate_polygon(poly)
    assert ei.value.status_code == 400
    assert "вершин" in ei.value.detail.lower()


# ───── Limits sanity ─────────────────────────────────────────────


def test_constants_are_sensible():
    assert MIN_AREA_HA > 0
    assert MAX_AREA_HA > MIN_AREA_HA
    assert MAX_VERTICES >= 100
