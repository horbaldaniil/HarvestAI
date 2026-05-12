"""Static reference data shared across scripts, services and routers.

These modules hold the *constant* knowledge of the domain — Ukrainian
administrative geography (oblast names, ISO codes, agro-climatic zones)
and crop agronomy (which crop grows where, phenology, etc.) — that the
rest of the codebase joins on.

The single most important rule here: ALL joins between data sources
(GeoJSON, parquet, Держстат YAML, frontend i18n) go through the
canonical `iso_3166_2` key defined in `oblast_names.py`. Free-text name
columns (`"L'viv"` vs `"Lviv"` vs `"Львівська область"`) cause silent
~50% join drops, which is the bug class this module exists to prevent.
"""
from __future__ import annotations

from app.data_reference.crop_zones import (
    ALL_ZONES,
    AgroClimaticZone,
    is_grown,
    yield_multiplier,
)
from app.data_reference.oblast_names import (
    ALL_OBLASTS,
    OBLAST_BY_ISO,
    OblastRef,
    iso_from_any_name,
    oblast_by_iso,
)

__all__ = [
    "ALL_OBLASTS",
    "ALL_ZONES",
    "AgroClimaticZone",
    "OBLAST_BY_ISO",
    "OblastRef",
    "is_grown",
    "iso_from_any_name",
    "oblast_by_iso",
    "yield_multiplier",
]
