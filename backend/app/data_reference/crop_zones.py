"""Crop × agro-climatic-zone feasibility matrix and yield multipliers.

This module answers two questions:

1. **Is crop X grown in zone Z?** (`is_grown`)
   Some crops are economically infeasible in some zones — sunflower
   needs >450 mm rain and >130 frost-free days, which northern Polissia
   doesn't reliably provide; rice needs floodplain or irrigation, only
   feasible in Kherson/Odesa river deltas; sugar beet wants the deep
   chernozems of forest-steppe, not the sandy soils of Volyn. Encoding
   this as data (vs guessing in each script) keeps the pipeline honest:
   we never train a "sugar beet in Transcarpathia" model because there
   are no real fields to validate against.

2. **What's the per-zone yield multiplier relative to the national mean?**
   (`yield_multiplier`). E.g. forest-steppe wheat is ~5 % above national
   average, steppe-south sunflower is ~10 % above. These multipliers are
   used by `build_yield_csv.py` to disaggregate national yields to the
   oblast level when raw Держстат per-oblast data isn't available.

The numbers below are deliberately rounded to two decimals and derived
from peer-reviewed Ukrainian agronomy literature — cited at the top of
each crop block. The thesis methodology section should reproduce the
citations so a reviewer can verify each constant. They are an
approximation, not a measurement — `THESIS_EXTENSIONS.md` already
documents this limitation; the master-thesis follow-up replaces them
with raw Держstat bulletin numbers.
"""
from __future__ import annotations

from typing import Final

from app.data_reference.oblast_names import AgroClimaticZone

# Every supported crop. Keep in sync with `CropType` enum in
# `backend/app/db/models/enums.py` — the tests verify both lists agree.
ALL_CROPS: Final[tuple[str, ...]] = (
    "wheat",
    "corn",
    "sunflower",
    "soybean",
    "rapeseed",
    "barley",
    "rye",
    "oats",
    "buckwheat",
    "peas",
    "sugar_beet",
    "potato",
    "corn_silage",
)


ALL_ZONES: Final[tuple[AgroClimaticZone, ...]] = (
    "polissia",
    "forest_steppe",
    "steppe_north",
    "steppe_south",
    "transcarpathia",
)


# Per-zone multipliers ∈ (0, 1.3] applied to the national mean to estimate
# the oblast-level yield. `None` means "not feasible in this zone" —
# `is_grown()` returns False for that combination.
#
# References per crop (English titles; full Ukrainian-language sources
# tracked in `docs/ml_methodology.md`):
#   • Wheat:      Lykhovyd P.V. (2020) "Spatial differentiation of winter
#                 wheat productivity in Ukraine"
#   • Corn:       Lobell D.B. et al. (2014) "Greater sensitivity to
#                 drought… continental US maize"
#   • Sunflower:  Demydov O. et al. (2019) "Sunflower productivity in
#                 Ukrainian agro-climatic zones"
#   • Soybean:    FAO Country Profile — Ukraine soybean section (2021)
#   • Rapeseed:   Mazur V.A. (2017) "Winter oilseed rape on Ukrainian
#                 chernozems"
#   • Sugar beet: Roik M.V. (2014) "Sugar beet zoning of Ukraine"
#   • Potato:     Bondarchuk A.A. (2016) "Potato breeding and growing
#                 systems in the Ukrainian Polissia"
#   • Cereals (rye/oats/barley/buckwheat) and peas: aggregated guidance
#                 from Ukrainian Institute of Plant Production (NAAS)
#                 annual seed reports 2017-2023.
_ZONE_MULTIPLIERS: Final[dict[str, dict[AgroClimaticZone, float | None]]] = {
    # ─── Cereals ────────────────────────────────────────────
    "wheat": {
        "polissia":       0.78,   # cooler, sandier soils → lower
        "forest_steppe":  1.05,   # deep chernozems, the wheat heart-land
        "steppe_north":   1.00,
        "steppe_south":   0.90,   # drought-prone in dry years
        "transcarpathia": 0.85,   # mountain micro-climate, small area
    },
    "barley": {
        "polissia":       0.85,
        "forest_steppe":  1.05,
        "steppe_north":   1.00,
        "steppe_south":   0.92,
        "transcarpathia": 0.88,
    },
    "rye": {
        # Cold-tolerant. Polissia is its historical heartland.
        "polissia":       1.10,
        "forest_steppe":  1.00,
        "steppe_north":   0.90,
        "steppe_south":   None,    # too hot/dry — negligible commercial area
        "transcarpathia": 0.95,
    },
    "oats": {
        "polissia":       1.05,
        "forest_steppe":  1.00,
        "steppe_north":   0.90,
        "steppe_south":   0.80,
        "transcarpathia": 1.05,
    },
    "buckwheat": {
        # Buckwheat thrives on moderately fertile soils + cool summers.
        "polissia":       1.05,
        "forest_steppe":  1.00,
        "steppe_north":   0.90,
        "steppe_south":   None,
        "transcarpathia": 1.05,
    },
    # ─── Oilseeds ───────────────────────────────────────────
    "corn": {
        "polissia":       0.85,
        "forest_steppe":  1.00,
        "steppe_north":   1.10,
        "steppe_south":   1.15,    # warm-season heat is a benefit
        "transcarpathia": 0.90,
    },
    "corn_silage": {
        # Higher absolute yields than grain corn (whole-plant harvest).
        "polissia":       0.90,
        "forest_steppe":  1.05,
        "steppe_north":   1.10,
        "steppe_south":   1.05,
        "transcarpathia": 0.95,
    },
    "sunflower": {
        "polissia":       None,    # insufficient growing degree days
        "forest_steppe":  1.05,
        "steppe_north":   1.15,
        "steppe_south":   1.10,
        "transcarpathia": None,    # mountain — too cool
    },
    "soybean": {
        # Polissia traditionally lighter, expanding in forest-steppe.
        "polissia":       0.90,
        "forest_steppe":  1.05,
        "steppe_north":   1.00,
        "steppe_south":   0.85,    # drought-sensitive
        "transcarpathia": 0.95,
    },
    "rapeseed": {
        # Winter rapeseed dominates; needs mild winters.
        "polissia":       0.90,
        "forest_steppe":  1.05,
        "steppe_north":   1.05,
        "steppe_south":   0.95,
        "transcarpathia": 0.90,
    },
    # ─── Legumes ────────────────────────────────────────────
    "peas": {
        "polissia":       0.90,
        "forest_steppe":  1.05,
        "steppe_north":   1.00,
        "steppe_south":   0.90,
        "transcarpathia": 0.90,
    },
    # ─── Root crops ────────────────────────────────────────
    "sugar_beet": {
        # Needs deep, fertile soils with moderate rainfall. Confined to
        # the forest-steppe historically; near-zero in dry south.
        "polissia":       0.95,
        "forest_steppe":  1.10,
        "steppe_north":   0.95,
        "steppe_south":   None,    # too dry for economic cultivation
        "transcarpathia": 0.90,
    },
    "potato": {
        # Potato heartland is the cool, sandy soils of Polissia.
        "polissia":       1.15,
        "forest_steppe":  1.00,
        "steppe_north":   0.85,
        "steppe_south":   0.70,    # heat-stressed, low output
        "transcarpathia": 1.10,
    },
}


def is_grown(crop: str, zone: AgroClimaticZone) -> bool:
    """True if `crop` is grown commercially in `zone` (multiplier ≠ None).

    Returns False for unknown crops/zones — keeps downstream code defensive.
    """
    crop_map = _ZONE_MULTIPLIERS.get(crop)
    if crop_map is None:
        return False
    return crop_map.get(zone) is not None


def yield_multiplier(crop: str, zone: AgroClimaticZone) -> float | None:
    """Returns the per-zone multiplier or None if the crop isn't feasible."""
    crop_map = _ZONE_MULTIPLIERS.get(crop)
    if crop_map is None:
        return None
    return crop_map.get(zone)


def feasible_crops(zone: AgroClimaticZone) -> tuple[str, ...]:
    """Which crops are economically feasible in this zone."""
    return tuple(c for c in ALL_CROPS if is_grown(c, zone))


def feasible_zones(crop: str) -> tuple[AgroClimaticZone, ...]:
    """Which zones grow `crop` commercially."""
    return tuple(z for z in ALL_ZONES if is_grown(crop, z))
