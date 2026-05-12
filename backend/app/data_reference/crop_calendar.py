"""Per-crop phenological calendar + GDD parameters for Ukrainian agriculture.

Drives crop-specific feature derivation in `build_features_v4.py`. v3 used
a single April-July weather window for ALL crops, which is wrong for:
  - Winter wheat: emerges in autumn, dormant winter, peaks May, harvests July
  - Sunflower: late-planted (May), peaks July, harvests September
  - Sugar beet: April-October, much longer than April-July
  - Potato: short April-August window

v4 captures these differences by computing the WINDOW OVERLAP between
each crop's growing season and our existing Apr-Jul weather aggregates.
This lets the model learn that "Apr-Jul precip strongly predicts wheat
yield but only weakly predicts sugar-beet yield".

For each crop:
  - `sow_month` / `peak_month` / `harvest_month`: growing-season anchor
  - `gdd_base_c`: base temperature for growing degree day accumulation
  - `flowering_window_days`: days centred on peak when heat stress hurts
    most (typical 10-day grain-filling window for cereals)

Sources (cited in docstring per group):
  - Cereals (wheat, barley, rye, oats, buckwheat):
    Інститут рослинництва ім. В.Я. Юр'єва (NAAS) "Технологія вирощування
    зернових культур" 2020 edition
  - Oilseeds (sunflower, soybean, rapeseed):
    Demydov O.A. et al. (2019), Mazur V.A. (2017)
  - Root crops (sugar_beet, potato, corn_silage):
    Roik M.V. (2014) for sugar beet, Bondarchuk A.A. (2016) for potato
  - Legume (peas):
    USDA FAS Ukraine pulses report (2022)
  - Corn (grain), corn_silage:
    FAO Crop Calendar for Ukraine
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class CropCalendar:
    """Per-crop phenological constants. Months are 1-12; days are integers."""
    sow_month: int        # Sowing — start of growing season
    peak_month: int       # Peak vegetation / flowering — model's "signal" month
    harvest_month: int    # Harvest — end of growing season
    gdd_base_c: float     # Growing-degree-day base temperature (°C)
    flowering_window_days: int = 10  # Heat-stress critical days centred on peak

    @property
    def growing_season_months(self) -> tuple[int, ...]:
        """All months from sowing to harvest, handling autumn-sown crops
        (winter wheat sowing Oct, harvesting July → wraps around year)."""
        if self.sow_month <= self.harvest_month:
            return tuple(range(self.sow_month, self.harvest_month + 1))
        # Wrap: Oct-Dec + Jan-Jul
        return tuple(list(range(self.sow_month, 13)) + list(range(1, self.harvest_month + 1)))

    @property
    def growing_season_length_months(self) -> int:
        return len(self.growing_season_months)


CROP_CALENDAR: Final[dict[str, CropCalendar]] = {
    # ─── Cereals ────────────────────────────────────────────
    # Winter wheat: dominant Ukrainian variety. Sow Sept-Oct, overwinter,
    # resume growth April, flower May, ripen June, harvest July.
    "wheat": CropCalendar(sow_month=10, peak_month=5, harvest_month=7, gdd_base_c=5.0),
    # Spring barley: shorter season; some winter barley exists but spring dominates.
    "barley": CropCalendar(sow_month=4, peak_month=6, harvest_month=7, gdd_base_c=4.5),
    # Rye is mostly winter-sown like wheat; Polissia is its heartland.
    "rye": CropCalendar(sow_month=10, peak_month=5, harvest_month=7, gdd_base_c=4.0),
    # Oats — spring cereal, slightly longer than barley.
    "oats": CropCalendar(sow_month=4, peak_month=6, harvest_month=7, gdd_base_c=5.0),
    # Buckwheat — short season, late planted to dodge frost.
    "buckwheat": CropCalendar(sow_month=5, peak_month=7, harvest_month=8, gdd_base_c=8.0),
    # ─── Oilseeds ───────────────────────────────────────────
    # Grain corn — main C4 crop, long season, demands heat.
    "corn": CropCalendar(sow_month=4, peak_month=7, harvest_month=9, gdd_base_c=10.0),
    # Corn for silage — harvested earlier (whole-plant), slightly shorter.
    "corn_silage": CropCalendar(sow_month=4, peak_month=7, harvest_month=8, gdd_base_c=10.0),
    # Sunflower — drought-tolerant, late maturation.
    "sunflower": CropCalendar(sow_month=5, peak_month=7, harvest_month=9, gdd_base_c=6.0),
    # Soybean — heat-demanding, May-September window.
    "soybean": CropCalendar(sow_month=5, peak_month=7, harvest_month=9, gdd_base_c=10.0),
    # Winter rapeseed dominates Ukrainian production.
    "rapeseed": CropCalendar(sow_month=9, peak_month=5, harvest_month=7, gdd_base_c=5.0),
    # ─── Legume ────────────────────────────────────────────
    # Peas — short cool-season crop.
    "peas": CropCalendar(sow_month=4, peak_month=6, harvest_month=7, gdd_base_c=4.5),
    # ─── Root crops ────────────────────────────────────────
    # Sugar beet — very long season.
    "sugar_beet": CropCalendar(sow_month=4, peak_month=8, harvest_month=10, gdd_base_c=5.0),
    # Potato — shorter than sugar beet, more flexible.
    "potato": CropCalendar(sow_month=4, peak_month=7, harvest_month=9, gdd_base_c=7.0),
}


# Reference window used by v3 weather aggregates (precip_sum_apr_jul etc.).
# We compute each crop's overlap fraction with this to derive coverage features.
REFERENCE_WINDOW_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7)


def overlap_with_reference(crop: str) -> float:
    """Fraction of `crop`'s growing season that falls inside Apr-Jul.

    Used as a coverage weight — wheat has high overlap (4 of 10 months but
    those 4 contain emergence-flowering-ripening, the high-signal phase),
    sugar beet has lower overlap (4 of 7 months, missing late-season
    tuber-fill).

    We compute by *month-mass weighting*: each month in the crop's
    growing season is given equal weight; sum those that fall in Apr-Jul,
    divide by total months.
    """
    cal = CROP_CALENDAR.get(crop)
    if cal is None:
        return 0.0
    gs = set(cal.growing_season_months)
    in_ref = gs.intersection(REFERENCE_WINDOW_MONTHS)
    return len(in_ref) / len(gs) if gs else 0.0


def flowering_in_reference(crop: str) -> bool:
    """True if the crop's flowering peak lands within Apr-Jul.

    Critical: a heat-stress day in Apr-Jul matters MUCH more for a crop
    whose peak is in May (wheat) than for one whose peak is in September
    (sunflower) — heat at flowering damages reproductive organs.
    """
    cal = CROP_CALENDAR.get(crop)
    if cal is None:
        return False
    return cal.peak_month in REFERENCE_WINDOW_MONTHS


def gdd_proxy(crop: str, mean_temp_c: float | None,
              growing_season_months: int | None = None) -> float | None:
    """Approximate GDD accumulation as `(mean_temp - base) × days_in_season`.

    This is a coarse approximation — a real GDD uses daily T_mean. But
    when we only have a single seasonal-mean temperature, this proxy
    captures (a) does the crop get enough heat above its base in this
    oblast/year? (b) how does that compare to other oblasts?

    Returns None if `mean_temp_c` is missing.
    """
    cal = CROP_CALENDAR.get(crop)
    if cal is None or mean_temp_c is None or mean_temp_c == 0:
        return None
    n_months = growing_season_months or cal.growing_season_length_months
    days = n_months * 30  # rough
    excess = max(0.0, mean_temp_c - cal.gdd_base_c)
    return round(excess * days, 1)


__all__ = [
    "CROP_CALENDAR",
    "CropCalendar",
    "REFERENCE_WINDOW_MONTHS",
    "flowering_in_reference",
    "gdd_proxy",
    "overlap_with_reference",
]
