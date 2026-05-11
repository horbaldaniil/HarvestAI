"""Geometry helpers: GeoJSON <-> Shapely, validation, area calculation.

Why we validate on the application side even though PostGIS could:
- Errors are caught before the DB roundtrip (faster, cheaper).
- We can return precise Ukrainian-language error messages.
- Area is bounded business logic (0.5–1000 ha) — that's not a DB concern.
"""
from __future__ import annotations

from typing import Final

from fastapi import HTTPException, status
from shapely.geometry import Point, Polygon, mapping, shape

from app.schemas.field import PointGeometry, PolygonGeometry

# Business rules
MIN_AREA_HA: Final[float] = 0.5
MAX_AREA_HA: Final[float] = 1000.0
MAX_VERTICES: Final[int] = 500
MIN_VERTICES: Final[int] = 3  # exclusive of closing repeated vertex

# Earth equatorial radius for spherical-area approximation (meters).
# We could use pyproj for true geodesic area, but for validation a
# spherical approximation is more than precise enough; PostGIS will
# store the authoritative `area_ha`.
_EARTH_RADIUS_M: Final[float] = 6_371_008.8


def _spherical_area_m2(polygon: Polygon) -> float:
    """Compute polygon area in m² on a unit sphere, scaled by Earth radius².

    Uses the standard spherical excess formula for a single ring. Accurate to
    ~0.3% for fields under a few hundred km on a side, which is far below our
    1000 ha cap (~3.16 km × 3.16 km).
    """
    import math

    ring = list(polygon.exterior.coords)
    if len(ring) < 4:  # closed ring needs at least 4 points (3 unique + closure)
        return 0.0

    total = 0.0
    for i in range(len(ring) - 1):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[i + 1]
        lon1_r = math.radians(lon1)
        lon2_r = math.radians(lon2)
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        total += (lon2_r - lon1_r) * (math.sin(lat1_r) + math.sin(lat2_r))
    return abs(total) * _EARTH_RADIUS_M * _EARTH_RADIUS_M / 2.0


def geojson_polygon_to_shape(geojson: PolygonGeometry) -> Polygon:
    """Convert a Pydantic PolygonGeometry to a Shapely Polygon."""
    return shape(geojson.model_dump())  # type: ignore[return-value]


def shape_to_geojson_polygon(polygon: Polygon) -> PolygonGeometry:
    return PolygonGeometry(**mapping(polygon))  # type: ignore[arg-type]


def shape_to_geojson_point(point: Point) -> PointGeometry:
    return PointGeometry(**mapping(point))  # type: ignore[arg-type]


def polygon_to_wkt(geojson: PolygonGeometry) -> str:
    """Convert GeoJSON polygon to WKT (what PostGIS ingests)."""
    return geojson_polygon_to_shape(geojson).wkt


def compute_area_ha(geojson: PolygonGeometry) -> float:
    """Application-side area (hectares) for validation pre-check."""
    return _spherical_area_m2(geojson_polygon_to_shape(geojson)) / 10_000.0


def validate_polygon(geojson: PolygonGeometry) -> Polygon:
    """Run all business validations on an incoming polygon.

    Raises HTTPException(400) with a Ukrainian message if invalid.
    Returns the parsed Shapely polygon on success.
    """
    # 1. Single outer ring only (no holes).
    if len(geojson.coordinates) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Полігон повинен мати лише один зовнішній контур (без отворів).",
        )

    ring = geojson.coordinates[0]

    # 2. Vertex count.
    # GeoJSON closes the ring (first == last), so unique vertex count is len-1.
    unique_vertices = len(ring) - 1 if len(ring) >= 1 else 0
    if unique_vertices < MIN_VERTICES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Полігон повинен мати щонайменше {MIN_VERTICES} вершини.",
        )
    if unique_vertices > MAX_VERTICES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Полігон занадто складний: {unique_vertices} вершин "
                f"(максимум {MAX_VERTICES})."
            ),
        )

    # 3. Coordinate range sanity check (catches lat/lon swaps).
    for lon, lat in ring:
        if not (-180.0 <= lon <= 180.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Довгота поза допустимим діапазоном: {lon}",
            )
        if not (-90.0 <= lat <= 90.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Широта поза допустимим діапазоном: {lat}",
            )

    # 4. Shapely validity (no self-intersection, etc.).
    try:
        polygon = geojson_polygon_to_shape(geojson)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Некоректна геометрія полігона: {exc}",
        ) from exc

    if not polygon.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Полігон самоперетинається або має іншу геометричну помилку. "
                "Перевірте, що сторони не перетинаються."
            ),
        )

    # 5. Area bounds.
    area_ha = _spherical_area_m2(polygon) / 10_000.0
    if area_ha < MIN_AREA_HA:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Поле занадто мале: {area_ha:.2f} га "
                f"(мінімум {MIN_AREA_HA} га)."
            ),
        )
    if area_ha > MAX_AREA_HA:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Поле занадто велике: {area_ha:.1f} га "
                f"(максимум {MAX_AREA_HA} га)."
            ),
        )

    return polygon


__all__ = [
    "MAX_AREA_HA",
    "MAX_VERTICES",
    "MIN_AREA_HA",
    "MIN_VERTICES",
    "compute_area_ha",
    "geojson_polygon_to_shape",
    "polygon_to_wkt",
    "shape_to_geojson_point",
    "shape_to_geojson_polygon",
    "validate_polygon",
]
