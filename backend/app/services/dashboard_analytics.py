"""Pure-function helpers feeding the dashboard endpoint.

Keeps the router thin: each function takes already-loaded data + plain
arguments and returns a structured payload. Lets us unit-test the logic
without spinning up a DB.

Four independent topics live here:

1. Risk score — composite 0-100 score per field used by the dashboard
   table's "Risk" column and by best/worst card selection.
2. Phenology — static lookup table mapping (crop, month) to a growth phase
   label. Driven by averaged Ukrainian agronomy norms.
3. Oblast NDVI lookup — read training_set_v2.parquet (Week 6 output) and
   return the cropland-mask-aggregated NDVI mean for a given oblast/year.
   Used by the "your field vs oblast average" comparison column.
4. Field → oblast resolver — point-in-polygon over ukraine_oblasts.geojson
   so the dashboard knows WHICH oblast row to compare a field against.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ─── Phenology ─────────────────────────────────────────────────


# (crop, start_month, end_month_inclusive, phase_key)
# Phase keys mirror the i18n keys the frontend uses.
PHENOLOGY: dict[str, list[tuple[int, int, str]]] = {
    "wheat": [
        (9, 10, "sowing"),
        (11, 3, "dormancy"),
        (4, 5, "growth"),
        (6, 6, "flowering"),
        (7, 7, "ripening"),
        (8, 8, "harvest"),
    ],
    "corn": [
        (4, 4, "sowing"),
        (5, 6, "growth"),
        (7, 7, "flowering"),
        (8, 8, "ripening"),
        (9, 10, "harvest"),
    ],
    "sunflower": [
        (4, 4, "sowing"),
        (5, 6, "growth"),
        (7, 7, "flowering"),
        (8, 8, "ripening"),
        (9, 9, "harvest"),
    ],
}


def phenology_phase(crop: str, today: date | None = None) -> str | None:
    """Return the current growth phase for `crop` at `today`, or None.

    Handles wraparound months (e.g. wheat dormancy spans Nov-Mar).
    """
    if today is None:
        today = date.today()
    spec = PHENOLOGY.get(crop)
    if not spec:
        return None
    month = today.month
    for start, end, label in spec:
        if start <= end:
            if start <= month <= end:
                return label
        else:
            # Wraparound (Nov-Mar): hits if month >= start OR month <= end.
            if month >= start or month <= end:
                return label
    return None


def phenology_calendar(crop: str) -> list[dict[str, Any]]:
    """Return the full phase list with normalised month ranges for the UI."""
    out = []
    for start, end, label in PHENOLOGY.get(crop, []):
        out.append({"phase": label, "start_month": start, "end_month": end})
    return out


# ─── Risk score ────────────────────────────────────────────────


# Open-Meteo `soil_moisture_0_10cm` is volumetric water content (m³/m³).
# Typical agronomic bands for a temperate loam-clay topsoil:
#   < 0.10 — wilting point, severe stress
#   0.10–0.18 — refill / dryness threshold
#   0.18–0.30 — adequate
#   > 0.30 — near field capacity
# We flag below 18 % which catches "soil is drying out" before the crop
# is wilting. This replaces the older `drought_days < 1mm/day` proxy that
# misfired after a recent rain shower because it only looked at counts of
# dry days, not at actual moisture present in the soil.
SOIL_MOISTURE_LOW_THRESHOLD: float = 0.18


def compute_risk_score(
    *,
    current_ndvi: float | None,
    oblast_avg_ndvi: float | None,
    alerts_by_severity: dict[str, int],
    recent_soil_moisture: float | None = None,
    heat_days_recent: int = 0,
) -> tuple[int, list[str]]:
    """Composite risk score 0-100 plus short factor tags for tooltip.

    Factors:
      - NDVI deficit vs oblast average: up to 30 points
      - Active alerts: 10 per warning, 30 per critical (capped at 50)
      - Soil moisture below `SOIL_MOISTURE_LOW_THRESHOLD`: 15 points
        (replaces the old <1 mm/day counter — see threshold constant
        for the agronomic rationale)
      - Recent heat stress (>5 days T_max > 30 °C): 15 points

    Score is clamped to [0, 100]. Higher = worse.
    """
    score = 0
    factors: list[str] = []

    # NDVI deficit relative to oblast. Only applied if oblast comparison
    # exists; otherwise no signal in either direction.
    if current_ndvi is not None and oblast_avg_ndvi is not None:
        deficit = oblast_avg_ndvi - current_ndvi
        if deficit > 0.05:
            penalty = min(30, int(deficit * 100))
            score += penalty
            factors.append(f"NDVI нижче області на {deficit:.2f}")
    elif current_ndvi is not None and current_ndvi < 0.3:
        # Standalone: very low NDVI even without comparison is a flag.
        score += 25
        factors.append(f"Низький NDVI {current_ndvi:.2f}")

    n_warn = alerts_by_severity.get("warning", 0)
    n_crit = alerts_by_severity.get("critical", 0)
    alert_score = min(50, n_warn * 10 + n_crit * 30)
    if alert_score:
        score += alert_score
        tag_parts = []
        if n_crit:
            tag_parts.append(f"{n_crit} критичних")
        if n_warn:
            tag_parts.append(f"{n_warn} попереджень")
        factors.append("Alerts: " + ", ".join(tag_parts))

    if (
        recent_soil_moisture is not None
        and recent_soil_moisture < SOIL_MOISTURE_LOW_THRESHOLD
    ):
        score += 15
        factors.append(
            f"Низька волога ґрунту ({recent_soil_moisture * 100:.0f}%)"
        )
    if heat_days_recent > 5:
        score += 15
        factors.append(f"Спека ({heat_days_recent} днів T_max>30°C)")

    return min(100, max(0, score)), factors


# ─── Oblast NDVI baseline (Week 6 connection) ─────────────────


@lru_cache(maxsize=1)
def _load_oblast_baseline() -> dict[tuple[str, int], float]:
    """Read training_set_v2.parquet into (oblast, year) → mean NDVI.

    File may be missing (Week 6 data collection is offline + month-long).
    Cached on first hit; tests that mutate the parquet should reset the cache.
    """
    path = Path("data/processed/training_set_v2.parquet")
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=["oblast", "year", "ndvi_peak", "ndvi_mean_july"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load oblast baseline: %s", exc)
        return {}

    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        # Use ndvi_peak as the comparable summary — it's the most stable
        # single number across (oblast, year) and the easiest to communicate
        # in the UI ("oblast peak NDVI was X, yours is Y").
        if row.get("ndvi_peak") is None:
            continue
        key = (str(row["oblast"]), int(row["year"]))
        # If multiple crops share the row, they collapse to the same average —
        # we already aggregate per (oblast, year) upstream. Last write wins.
        out[key] = float(row["ndvi_peak"])
    return out


@lru_cache(maxsize=1)
def _load_oblast_latest_year() -> dict[str, tuple[int, float]]:
    """Per oblast → (latest year present in parquet, ndvi_peak for that year).

    Used as the fallback when the current calendar year hasn't been
    collected yet (training_set_v2 holds 2019–2023 by default; we run in
    2026). Picking the most recent year keeps the comparison "fresh".
    """
    baseline = _load_oblast_baseline()
    latest: dict[str, tuple[int, float]] = {}
    for (ob, yr), val in baseline.items():
        cur = latest.get(ob)
        if cur is None or yr > cur[0]:
            latest[ob] = (yr, val)
    return latest


def oblast_avg_ndvi(oblast_name: str | None, year: int | None) -> float | None:
    """Lookup helper for the dashboard field-row enrichment.

    Resolution order:
      1. Exact (oblast, year) match — best when the requested year is in
         the dataset.
      2. Most-recent year for that oblast — covers the common case where
         the calendar year hasn't been collected yet.
      3. None — oblast isn't in the dataset at all.
    """
    if not oblast_name:
        return None
    name = str(oblast_name).strip()
    baseline = _load_oblast_baseline()
    if year is not None and (name, int(year)) in baseline:
        return baseline[(name, int(year))]
    latest = _load_oblast_latest_year()
    entry = latest.get(name)
    return entry[1] if entry else None


def oblast_baseline_year(oblast_name: str | None) -> int | None:
    """Which year the baseline NDVI for `oblast_name` actually came from.

    Lets the UI label the comparison correctly ("Львівська обл., 2023")
    instead of implying it's a current-season number.
    """
    if not oblast_name:
        return None
    latest = _load_oblast_latest_year()
    entry = latest.get(str(oblast_name).strip())
    return entry[0] if entry else None


def reset_oblast_baseline_cache() -> None:
    """Test helper — call after the parquet is rewritten."""
    _load_oblast_baseline.cache_clear()
    _load_oblast_latest_year.cache_clear()


# ─── Field → oblast resolver (Week 7 connection) ──────────────


@lru_cache(maxsize=1)
def _load_oblast_polygons() -> list[tuple[str, Any]]:
    """Read `ukraine_oblasts.geojson` once → list of (name, shapely.Polygon).

    Missing file → empty list. The resolver then returns None for every
    field, and the dashboard falls back to "data not collected" state.
    """
    path = Path("data/raw/ukraine_oblasts.geojson")
    if not path.exists():
        log.warning("ukraine_oblasts.geojson missing — field→oblast disabled.")
        return []
    try:
        from shapely.geometry import shape
    except ImportError:
        log.warning("shapely not available — field→oblast disabled.")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("Bad oblasts.geojson: %s", exc)
        return []

    out: list[tuple[str, Any]] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        # Natural Earth names live under "name"; Ukrainian under "name_uk".
        name = props.get("name") or props.get("name_uk")
        geom = feat.get("geometry")
        if not name or not geom:
            continue
        try:
            out.append((str(name), shape(geom)))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not parse oblast geometry for %s: %s", name, exc)
    return out


def find_oblast_for_centroid(
    lat: float | None, lon: float | None,
) -> str | None:
    """Point-in-polygon over Ukrainian oblasts. Returns the English name."""
    if lat is None or lon is None:
        return None
    try:
        from shapely.geometry import Point
    except ImportError:
        return None
    point = Point(lon, lat)  # GeoJSON convention: (x=lon, y=lat)
    for name, polygon in _load_oblast_polygons():
        try:
            if polygon.contains(point):
                return name
        except Exception:  # noqa: BLE001
            continue
    return None


def reset_oblast_polygons_cache() -> None:
    """Test helper — re-read the geojson after fixtures rewrite it."""
    _load_oblast_polygons.cache_clear()


# Human-readable Ukrainian names for the UI. Falls back to the English
# Natural Earth name if a translation isn't pinned here.
OBLAST_NAME_UK: dict[str, str] = {
    "Cherkasy": "Черкаська обл.",
    "Chernihiv": "Чернігівська обл.",
    "Chernivtsi": "Чернівецька обл.",
    "Dnipropetrovs'k": "Дніпропетровська обл.",
    "Donets'k": "Донецька обл.",
    "Ivano-Frankivs'k": "Івано-Франківська обл.",
    "Kharkiv": "Харківська обл.",
    "Kherson": "Херсонська обл.",
    "Khmel'nyts'kyy": "Хмельницька обл.",
    "Kirovohrad": "Кіровоградська обл.",
    "Kiev": "Київська обл.",
    "Kiev City": "м. Київ",
    "Luhans'k": "Луганська обл.",
    "L'viv": "Львівська обл.",
    "Mykolayiv": "Миколаївська обл.",
    "Odessa": "Одеська обл.",
    "Poltava": "Полтавська обл.",
    "Rivne": "Рівненська обл.",
    "Sumy": "Сумська обл.",
    "Ternopil'": "Тернопільська обл.",
    "Transcarpathia": "Закарпатська обл.",
    "Vinnytsya": "Вінницька обл.",
    "Volyn": "Волинська обл.",
    "Zaporizhzhya": "Запорізька обл.",
    "Zhytomyr": "Житомирська обл.",
}


def oblast_name_uk(english_name: str | None) -> str | None:
    if not english_name:
        return None
    return OBLAST_NAME_UK.get(english_name, english_name)


__all__ = [
    "OBLAST_NAME_UK",
    "PHENOLOGY",
    "compute_risk_score",
    "find_oblast_for_centroid",
    "oblast_avg_ndvi",
    "oblast_baseline_year",
    "oblast_name_uk",
    "phenology_calendar",
    "phenology_phase",
    "reset_oblast_baseline_cache",
    "reset_oblast_polygons_cache",
]
