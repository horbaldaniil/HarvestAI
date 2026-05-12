"""Per-oblast SoilGrids 250m 0-5cm topsoil features.

Values come from ISRIC SoilGrids v2 (CC-BY 4.0) — sampled at each
oblast's 4-8 representative cropland polygons and aggregated as
oblast-level median. Originally produced by
`scripts/add_soilgrids_features.py` during the v7 model upgrade;
those exact values are now baked in here so the prediction path
doesn't need to read the 50 MB training parquet at inference time.

Used by `app/ml/features.py` to fill the 7 soil columns of the v7
feature schema (bdod / cec / clay / phh2o / sand / silt / soc).

Properties (0-5 cm depth):
    bdod  — bulk density (kg/dm³)
    cec   — cation exchange capacity (cmol(c)/kg)
    clay  — clay content (% mass)
    phh2o — soil pH (in water)
    sand  — sand content (% mass)
    silt  — silt content (% mass)
    soc   — soil organic carbon (g/kg)

Reference:
    Poggio, L. et al. (2021) "SoilGrids 2.0: producing soil information
    for the globe with quantified spatial uncertainty." *SOIL* 7,
    217–240. https://soilgrids.org/
"""
from __future__ import annotations

from typing import Final

# Order matters: the keys here are the column names in the v7 feature
# matrix. Any reorder must be mirrored in `FEATURE_NAMES` in
# `scripts/train_yield_models_v7.py` (and in any newly trained model
# payloads) — the model expects positional inputs.
SOIL_FEATURES: Final[tuple[str, ...]] = (
    "bdod", "cec", "clay", "phh2o", "sand", "silt", "soc",
)


# Per-oblast values keyed by ISO 3166-2 code. Use `iso_from_any_name()`
# from `app.data_reference.oblast_names` to resolve Natural Earth /
# Cyrillic / slug spellings before lookup.
SOILGRIDS_BY_ISO: Final[dict[str, dict[str, float]]] = {
    "UA-71": dict(bdod=1.190, cec=32.900, clay=26.450, phh2o=6.700, sand=20.400, silt=54.550, soc=52.200),   # Cherkasy
    "UA-74": dict(bdod=1.040, cec=36.100, clay=17.400, phh2o=6.600, sand=38.000, silt=46.950, soc=98.600),   # Chernihiv
    "UA-77": dict(bdod=1.270, cec=29.450, clay=29.850, phh2o=6.800, sand=20.500, silt=47.750, soc=47.200),   # Chernivtsi
    "UA-12": dict(bdod=1.185, cec=31.250, clay=30.700, phh2o=7.050, sand=16.000, silt=53.100, soc=62.100),   # Dnipropetrovsk
    "UA-14": dict(bdod=1.145, cec=31.950, clay=34.900, phh2o=7.000, sand=16.150, silt=48.650, soc=71.550),   # Donetsk
    "UA-26": dict(bdod=1.290, cec=27.700, clay=30.600, phh2o=6.300, sand=22.900, silt=47.300, soc=41.100),   # Ivano-Frankivsk
    "UA-63": dict(bdod=1.110, cec=32.100, clay=33.600, phh2o=6.900, sand=17.700, silt=48.200, soc=88.550),   # Kharkiv
    "UA-65": dict(bdod=1.260, cec=29.400, clay=39.700, phh2o=6.950, sand=8.300,  silt=52.900, soc=36.850),   # Kherson
    "UA-68": dict(bdod=1.190, cec=28.600, clay=26.100, phh2o=6.600, sand=19.100, silt=55.300, soc=58.300),   # Khmelnytskyi
    "UA-35": dict(bdod=1.220, cec=30.500, clay=31.900, phh2o=6.850, sand=21.200, silt=48.800, soc=51.800),   # Kirovohrad
    "UA-32": dict(bdod=1.150, cec=30.250, clay=25.400, phh2o=6.600, sand=26.900, silt=45.700, soc=64.750),   # Kyiv
    "UA-09": dict(bdod=1.115, cec=30.900, clay=29.400, phh2o=6.950, sand=21.650, silt=47.800, soc=79.250),   # Luhansk
    "UA-46": dict(bdod=1.220, cec=26.900, clay=22.050, phh2o=6.550, sand=26.800, silt=50.550, soc=55.650),   # Lviv
    "UA-48": dict(bdod=1.265, cec=28.600, clay=34.450, phh2o=7.150, sand=19.250, silt=46.050, soc=42.550),   # Mykolaiv
    "UA-51": dict(bdod=1.275, cec=30.400, clay=34.500, phh2o=7.250, sand=16.100, silt=48.400, soc=33.800),   # Odesa
    "UA-53": dict(bdod=1.165, cec=33.050, clay=25.350, phh2o=6.800, sand=22.500, silt=52.300, soc=72.350),   # Poltava
    "UA-56": dict(bdod=1.190, cec=29.300, clay=21.450, phh2o=6.900, sand=24.600, silt=50.950, soc=60.100),   # Rivne
    "UA-59": dict(bdod=1.080, cec=37.350, clay=21.350, phh2o=6.800, sand=33.450, silt=46.400, soc=87.050),   # Sumy
    "UA-61": dict(bdod=1.165, cec=30.150, clay=23.450, phh2o=6.650, sand=20.200, silt=54.750, soc=55.250),   # Ternopil
    "UA-05": dict(bdod=1.195, cec=29.450, clay=26.850, phh2o=6.750, sand=23.350, silt=49.800, soc=45.500),   # Vinnytsia
    "UA-07": dict(bdod=1.150, cec=29.000, clay=20.050, phh2o=6.850, sand=28.550, silt=47.300, soc=59.500),   # Volyn
    "UA-21": dict(bdod=1.310, cec=26.100, clay=31.300, phh2o=6.300, sand=24.200, silt=44.800, soc=45.450),   # Zakarpattia
    "UA-23": dict(bdod=1.270, cec=30.050, clay=38.450, phh2o=6.900, sand=9.800,  silt=52.050, soc=49.250),   # Zaporizhzhia
    "UA-18": dict(bdod=1.105, cec=27.600, clay=16.600, phh2o=6.600, sand=37.550, silt=44.000, soc=55.100),   # Zhytomyr
}


# National-median fallback for fields whose centroid falls outside the
# oblast polygons we have (typically Kyiv City, or a point in a river/
# lake near a boundary). Computed as the mean across all 24 oblasts so
# the model gets a realistic value instead of NaN — RandomForest does
# not handle NaN, and falling back to 0.0 would skew predictions hard.
def _national_median() -> dict[str, float]:
    keys = SOIL_FEATURES
    rows = list(SOILGRIDS_BY_ISO.values())
    return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}


NATIONAL_MEDIAN: Final[dict[str, float]] = _national_median()


def soilgrids_for_oblast(iso_3166_2: str | None) -> dict[str, float]:
    """Return the 7-feature soil dict for an oblast, or national-median
    fallback if the oblast isn't in the table (most commonly: input is
    None or the lookup hit Kyiv City, which is excluded from the
    training parquet because it's non-agricultural).
    """
    if iso_3166_2:
        hit = SOILGRIDS_BY_ISO.get(iso_3166_2)
        if hit is not None:
            return dict(hit)
    return dict(NATIONAL_MEDIAN)


__all__ = [
    "NATIONAL_MEDIAN",
    "SOIL_FEATURES",
    "SOILGRIDS_BY_ISO",
    "soilgrids_for_oblast",
]
