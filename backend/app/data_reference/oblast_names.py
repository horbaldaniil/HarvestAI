"""Canonical Ukrainian oblast reference table.

ONE source of truth keyed by ISO 3166-2 code (`UA-46` for Lviv, `UA-32`
for Kyiv oblast, etc.). Every consumer in the codebase that needs to
identify an oblast — yield CSV builder, parquet feature builder, GeoJSON
loader, frontend i18n — must look up via `iso_3166_2`, not by free-text
name. This eliminates the entire class of bugs where `"L'viv"` (Natural
Earth) silently fails to join against `"Lviv"` (build_yield_csv.py) or
`"Львівська область"` (Держстат), each dropping ~50% of rows undetected.

The table also encodes which agro-climatic zone each oblast belongs to
(see `crop_zones.py`), and whether the territory had active agriculture
during the 2017-2023 study window — used by collectors to skip Kyiv City
(no agriculture) and by the evaluator to flag conflict-affected oblasts.

Zone classification follows the standard Ukrainian agricultural
geography (Polissia / Forest-Steppe / Northern Steppe / Southern Steppe
/ Transcarpathia) as published in:
  - Балюк С.А. et al. "Ґрунтово-кліматичне районування території
    України" (NAAS, 2014)
  - FAO Country Profile — Ukraine (2018), agro-ecological zoning section

References:
  - ISO 3166-2:UA — https://en.wikipedia.org/wiki/ISO_3166-2:UA
  - Natural Earth admin-1 — https://www.naturalearthdata.com/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

AgroClimaticZone = Literal[
    "polissia",          # forested north — cooler, more rainfall, sandy soils
    "forest_steppe",     # transitional centre — best wheat/corn yields
    "steppe_north",      # northern-steppe — corn/sunflower dominate
    "steppe_south",      # arid south — drought-prone, sunflower belt
    "transcarpathia",    # mountain west — distinct micro-climate
]


@dataclass(frozen=True)
class OblastRef:
    """Single row in the canonical oblast table.

    Attributes:
        iso_3166_2: ISO code, e.g. `"UA-46"` — the join key. Stable, unique.
        slug: ASCII slug for filenames / URLs (`"lviv"`, `"kyiv_city"`).
        name_en: Canonical English label used in metrics JSON, logs, etc.
        natural_earth_name: How the Natural Earth admin-1 layer spells it
            (`"L'viv"`, `"Khmel'nyts'kyy"`). This is what
            `training_set_v2.parquet` currently stores — kept as a join
            target for the v2→v3 migration only.
        name_uk_short: Short Ukrainian label for compact UI cells
            (`"Львівська"`, `"Київська"`).
        name_uk_full: Full Ukrainian label (`"Львівська область"`) — for
            page titles and the methodology table.
        derzhstat_name: How Держstat бюлетень spells the oblast in the
            "Сільське господарство України" annual report. Used by the
            YAML→features join.
        zone: Agro-climatic zone — drives yield multipliers and feasibility.
        centroid_lat / centroid_lon: Approximate centroid (for the per-
            oblast weather fetch in `build_features_v3.py`).
        is_agricultural: False for Kyiv City (mostly urban — excluded
            from sampling and training). True for all 24 oblasts.
        conflict_zone_since: Year from which the oblast is considered
            partially/fully under conflict and yields are unreliable;
            None for unaffected oblasts. Used by `evaluate_models.py`
            to optionally exclude these rows.
    """

    iso_3166_2: str
    slug: str
    name_en: str
    natural_earth_name: str
    name_uk_short: str
    name_uk_full: str
    derzhstat_name: str
    zone: AgroClimaticZone
    centroid_lat: float
    centroid_lon: float
    is_agricultural: bool = True
    conflict_zone_since: int | None = None


# 25 entries: 24 oblasts + Kyiv City (non-agricultural).
# Order matches Natural Earth GeoJSON for easy visual cross-check.
ALL_OBLASTS: Final[tuple[OblastRef, ...]] = (
    OblastRef(
        iso_3166_2="UA-74", slug="chernihiv",
        name_en="Chernihiv", natural_earth_name="Chernihiv",
        name_uk_short="Чернігівська", name_uk_full="Чернігівська область",
        derzhstat_name="Чернігівська",
        zone="polissia", centroid_lat=51.50, centroid_lon=31.30,
    ),
    OblastRef(
        iso_3166_2="UA-07", slug="volyn",
        name_en="Volyn", natural_earth_name="Volyn",
        name_uk_short="Волинська", name_uk_full="Волинська область",
        derzhstat_name="Волинська",
        zone="polissia", centroid_lat=51.20, centroid_lon=24.70,
    ),
    OblastRef(
        iso_3166_2="UA-56", slug="rivne",
        name_en="Rivne", natural_earth_name="Rivne",
        name_uk_short="Рівненська", name_uk_full="Рівненська область",
        derzhstat_name="Рівненська",
        zone="polissia", centroid_lat=50.78, centroid_lon=26.43,
    ),
    OblastRef(
        iso_3166_2="UA-18", slug="zhytomyr",
        name_en="Zhytomyr", natural_earth_name="Zhytomyr",
        name_uk_short="Житомирська", name_uk_full="Житомирська область",
        derzhstat_name="Житомирська",
        zone="polissia", centroid_lat=50.26, centroid_lon=28.66,
    ),
    OblastRef(
        iso_3166_2="UA-32", slug="kyiv",
        # Natural Earth labels Kyiv oblast as `"Kiev"` (the legacy
        # romanisation). We store the canonical Kyiv form for everything
        # else but keep `natural_earth_name` as-is for the GeoJSON join.
        name_en="Kyiv", natural_earth_name="Kiev",
        name_uk_short="Київська", name_uk_full="Київська область",
        derzhstat_name="Київська",
        zone="forest_steppe", centroid_lat=50.10, centroid_lon=30.50,
    ),
    OblastRef(
        iso_3166_2="UA-21", slug="zakarpattia",
        name_en="Zakarpattia", natural_earth_name="Transcarpathia",
        name_uk_short="Закарпатська", name_uk_full="Закарпатська область",
        derzhstat_name="Закарпатська",
        zone="transcarpathia", centroid_lat=48.62, centroid_lon=22.30,
    ),
    OblastRef(
        iso_3166_2="UA-77", slug="chernivtsi",
        name_en="Chernivtsi", natural_earth_name="Chernivtsi",
        name_uk_short="Чернівецька", name_uk_full="Чернівецька область",
        derzhstat_name="Чернівецька",
        zone="forest_steppe", centroid_lat=48.29, centroid_lon=25.94,
    ),
    OblastRef(
        iso_3166_2="UA-26", slug="ivano_frankivsk",
        name_en="Ivano-Frankivsk", natural_earth_name="Ivano-Frankivs'k",
        name_uk_short="Івано-Франківська", name_uk_full="Івано-Франківська область",
        derzhstat_name="Івано-Франківська",
        zone="forest_steppe", centroid_lat=48.92, centroid_lon=24.71,
    ),
    OblastRef(
        iso_3166_2="UA-51", slug="odesa",
        name_en="Odesa", natural_earth_name="Odessa",
        name_uk_short="Одеська", name_uk_full="Одеська область",
        derzhstat_name="Одеська",
        zone="steppe_south", centroid_lat=46.48, centroid_lon=30.73,
    ),
    OblastRef(
        iso_3166_2="UA-05", slug="vinnytsia",
        name_en="Vinnytsia", natural_earth_name="Vinnytsya",
        name_uk_short="Вінницька", name_uk_full="Вінницька область",
        derzhstat_name="Вінницька",
        zone="forest_steppe", centroid_lat=49.23, centroid_lon=28.47,
    ),
    OblastRef(
        iso_3166_2="UA-46", slug="lviv",
        name_en="Lviv", natural_earth_name="L'viv",
        name_uk_short="Львівська", name_uk_full="Львівська область",
        derzhstat_name="Львівська",
        zone="forest_steppe", centroid_lat=49.84, centroid_lon=24.03,
    ),
    OblastRef(
        iso_3166_2="UA-59", slug="sumy",
        name_en="Sumy", natural_earth_name="Sumy",
        name_uk_short="Сумська", name_uk_full="Сумська область",
        derzhstat_name="Сумська",
        # Officially classified as "Polissia / Forest-Steppe transition";
        # we tag forest_steppe to match its dominant southern-half agriculture.
        zone="forest_steppe", centroid_lat=50.92, centroid_lon=34.12,
    ),
    OblastRef(
        iso_3166_2="UA-63", slug="kharkiv",
        name_en="Kharkiv", natural_earth_name="Kharkiv",
        name_uk_short="Харківська", name_uk_full="Харківська область",
        derzhstat_name="Харківська",
        zone="forest_steppe", centroid_lat=49.99, centroid_lon=36.23,
        conflict_zone_since=2022,  # partially shelled, NE border
    ),
    OblastRef(
        iso_3166_2="UA-09", slug="luhansk",
        name_en="Luhansk", natural_earth_name="Luhans'k",
        name_uk_short="Луганська", name_uk_full="Луганська область",
        derzhstat_name="Луганська",
        zone="steppe_south", centroid_lat=48.57, centroid_lon=39.31,
        conflict_zone_since=2014,  # most of oblast occupied since 2014
    ),
    OblastRef(
        iso_3166_2="UA-14", slug="donetsk",
        name_en="Donetsk", natural_earth_name="Donets'k",
        name_uk_short="Донецька", name_uk_full="Донецька область",
        derzhstat_name="Донецька",
        zone="steppe_south", centroid_lat=48.02, centroid_lon=37.80,
        conflict_zone_since=2014,
    ),
    OblastRef(
        iso_3166_2="UA-65", slug="kherson",
        name_en="Kherson", natural_earth_name="Kherson",
        name_uk_short="Херсонська", name_uk_full="Херсонська область",
        derzhstat_name="Херсонська",
        zone="steppe_south", centroid_lat=46.65, centroid_lon=32.62,
        conflict_zone_since=2022,
    ),
    OblastRef(
        iso_3166_2="UA-23", slug="zaporizhzhia",
        name_en="Zaporizhzhia", natural_earth_name="Zaporizhzhya",
        name_uk_short="Запорізька", name_uk_full="Запорізька область",
        derzhstat_name="Запорізька",
        zone="steppe_south", centroid_lat=47.84, centroid_lon=35.14,
        conflict_zone_since=2022,
    ),
    OblastRef(
        iso_3166_2="UA-48", slug="mykolaiv",
        name_en="Mykolaiv", natural_earth_name="Mykolayiv",
        name_uk_short="Миколаївська", name_uk_full="Миколаївська область",
        derzhstat_name="Миколаївська",
        zone="steppe_south", centroid_lat=46.97, centroid_lon=31.99,
    ),
    OblastRef(
        iso_3166_2="UA-53", slug="poltava",
        name_en="Poltava", natural_earth_name="Poltava",
        name_uk_short="Полтавська", name_uk_full="Полтавська область",
        derzhstat_name="Полтавська",
        zone="forest_steppe", centroid_lat=49.59, centroid_lon=34.55,
    ),
    OblastRef(
        iso_3166_2="UA-68", slug="khmelnytskyi",
        name_en="Khmelnytskyi", natural_earth_name="Khmel'nyts'kyy",
        name_uk_short="Хмельницька", name_uk_full="Хмельницька область",
        derzhstat_name="Хмельницька",
        zone="forest_steppe", centroid_lat=49.42, centroid_lon=26.99,
    ),
    OblastRef(
        iso_3166_2="UA-61", slug="ternopil",
        name_en="Ternopil", natural_earth_name="Ternopil'",
        name_uk_short="Тернопільська", name_uk_full="Тернопільська область",
        derzhstat_name="Тернопільська",
        zone="forest_steppe", centroid_lat=49.55, centroid_lon=25.62,
    ),
    OblastRef(
        iso_3166_2="UA-12", slug="dnipropetrovsk",
        name_en="Dnipropetrovsk", natural_earth_name="Dnipropetrovs'k",
        name_uk_short="Дніпропетровська", name_uk_full="Дніпропетровська область",
        derzhstat_name="Дніпропетровська",
        zone="steppe_north", centroid_lat=48.46, centroid_lon=35.03,
    ),
    OblastRef(
        iso_3166_2="UA-71", slug="cherkasy",
        name_en="Cherkasy", natural_earth_name="Cherkasy",
        name_uk_short="Черкаська", name_uk_full="Черкаська область",
        derzhstat_name="Черкаська",
        zone="forest_steppe", centroid_lat=49.43, centroid_lon=32.06,
    ),
    OblastRef(
        iso_3166_2="UA-35", slug="kirovohrad",
        name_en="Kirovohrad", natural_earth_name="Kirovohrad",
        name_uk_short="Кіровоградська", name_uk_full="Кіровоградська область",
        derzhstat_name="Кіровоградська",
        zone="steppe_north", centroid_lat=48.51, centroid_lon=32.27,
    ),
    OblastRef(
        iso_3166_2="UA-30", slug="kyiv_city",
        name_en="Kyiv City", natural_earth_name="Kiev City",
        name_uk_short="м. Київ", name_uk_full="місто Київ",
        derzhstat_name="м. Київ",
        # The city is its own ISO entity (not part of Kyiv oblast). It
        # has effectively no commercial agriculture inside the ring road,
        # so we keep it in the lookup table for completeness but mark
        # it non-agricultural — downstream code skips it.
        zone="forest_steppe", centroid_lat=50.45, centroid_lon=30.52,
        is_agricultural=False,
    ),
)


# Pre-built lookup. Constant after import — safe to share globally.
OBLAST_BY_ISO: Final[dict[str, OblastRef]] = {o.iso_3166_2: o for o in ALL_OBLASTS}


# Reverse-lookup map: every alternative spelling → canonical OblastRef.
# Built lazily on first call; cheap enough to recompute that we don't
# bother memoising aggressively.
def _build_name_index() -> dict[str, OblastRef]:
    """Every known spelling → OblastRef. Case-insensitive keys.

    Handles:
      - Natural Earth English  (`"L'viv"`, `"Khmel'nyts'kyy"`)
      - Canonical English      (`"Lviv"`, `"Khmelnytskyi"`)
      - Legacy build_yield_csv (`"Lviv"`, `"Kyiv"`, `"Dnipro"`)
      - Slug                   (`"lviv"`, `"kyiv_city"`)
      - Ukrainian short        (`"Львівська"`)
      - Ukrainian full         (`"Львівська область"`)
      - Держстат name          (`"Львівська"`)
    """
    index: dict[str, OblastRef] = {}
    for o in ALL_OBLASTS:
        keys = {
            o.iso_3166_2,
            o.slug,
            o.name_en,
            o.natural_earth_name,
            o.name_uk_short,
            o.name_uk_full,
            o.derzhstat_name,
        }
        # A few historical aliases retained from `build_yield_csv.py`
        # (pre-canonicalisation). Listed explicitly so a grep finds them.
        if o.iso_3166_2 == "UA-12":
            keys.add("Dnipro")          # legacy short form
        if o.iso_3166_2 == "UA-51":
            keys.add("Odessa")          # alternative romanisation
        for k in keys:
            if not k:
                continue
            index[k.casefold()] = o
    return index


_NAME_INDEX: dict[str, OblastRef] | None = None


def iso_from_any_name(name: str | None) -> str | None:
    """Resolve any reasonable spelling of an oblast to its ISO 3166-2 code.

    Returns None if the name is None, empty, or unknown. Case-insensitive.
    Trailing/leading whitespace is stripped.

    >>> iso_from_any_name("L'viv")
    'UA-46'
    >>> iso_from_any_name("Lviv")
    'UA-46'
    >>> iso_from_any_name("Львівська")
    'UA-46'
    >>> iso_from_any_name("Львівська область")
    'UA-46'
    >>> iso_from_any_name("lviv")
    'UA-46'
    >>> iso_from_any_name("Atlantis") is None
    True
    """
    global _NAME_INDEX
    if not name:
        return None
    if _NAME_INDEX is None:
        _NAME_INDEX = _build_name_index()
    hit = _NAME_INDEX.get(name.strip().casefold())
    return hit.iso_3166_2 if hit else None


def oblast_by_iso(iso: str | None) -> OblastRef | None:
    """Lookup by ISO code; returns None if unknown or input is None."""
    if not iso:
        return None
    return OBLAST_BY_ISO.get(iso.strip().upper())


def agricultural_oblasts() -> tuple[OblastRef, ...]:
    """The 24 oblasts that participate in training (skips Kyiv City)."""
    return tuple(o for o in ALL_OBLASTS if o.is_agricultural)
