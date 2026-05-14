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

Sources (per crop):
  - wheat (winter):    NAAS "Технологія вирощування зернових культур" 2020;
                       GDD base 5°C per Hatfield & Prueger (2015)
  - barley (spring):   NAAS 2020; GDD base 4.5°C standard for Hordeum vulgare.
                       NOTE: winter barley (~20 % of UA area) is treated under
                       the same `barley` slug — future work could split into
                       `barley_winter` with its own calendar.
  - rye (winter):      NAAS 2020; rye is the most cold-tolerant cereal, hence
                       GDD base 4°C (lowest in the cereal group).
  - oats (spring):     NAAS 2020.
  - buckwheat:         NAAS 2020; GDD base 8°C reflects buckwheat's heat-loving
                       behaviour (one of the highest among small-grain crops).
  - corn (grain):      FAO Crop Calendar for Ukraine + FAO ECOCROP; GDD base
                       10°C is the canonical C4 reference.
  - corn_silage:       FAO Crop Calendar; same physiology as grain corn but
                       harvested at milk-dough stage (August).
  - sunflower:         FAO ECOCROP (Helianthus annuus) GDD base 7.2°C;
                       Demydov et al. 2019 cite 7-8°C range. Previous 6.0°C
                       was at the low end of the literature.
  - soybean:           FAO ECOCROP GDD base 10°C. Peak_month=8 (R5 full-canopy
                       NDVI peak) rather than 7 (flowering R3) for more
                       accurate NDVI feature alignment.
  - rapeseed (winter): Mazur V.A. 2017; sowing window in Ukraine is late
                       August to first week of September — we store 9 (Sept)
                       as the integer-month approximation.
  - peas:              USDA FAS Ukraine pulses report 2022.
  - sugar_beet:        Roik M.V. 2014; peak_month=8 reflects mid-August root-
                       and-canopy biomass peak (NDVI signal).
  - potato:            Bondarchuk A.A. 2016 + FAO ECOCROP; GDD base 5.5°C
                       is the canonical value for Solanum tuberosum (was
                       7.0°C, which was at the upper end of the literature).
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

    # ─── BBCH-aligned phase splits (Phase-A round-3) ─────────────
    #
    # Instead of one generic Apr-Jul aggregate per crop, give the model
    # weather signals separated by phenological phase. Each phase has its
    # own dominant agronomic stress mode:
    #
    #   - early_veg: vegetative growth (BBCH 10-39). Water demand
    #     moderate; cool-season crops prefer adequate spring rain.
    #   - flowering: reproductive phase (BBCH 51-69). HEAT-SENSITIVE —
    #     a single 35°C day at flowering can sterilise wheat anthers.
    #     Drought also amplified here (no recovery possible later).
    #   - grain_fill: yield accumulation (BBCH 71-89). Water-driven;
    #     soil-moisture depletion limits grain weight.
    #
    # Boundaries derived mechanically from `sow_month` / `peak_month` /
    # `harvest_month` — `peak_month` is the agronomic flowering anchor
    # used everywhere else in the codebase (e.g. `_crop_features`).

    @property
    def early_veg_months(self) -> tuple[int, ...]:
        """Sowing → month before peak (vegetative growth phase).

        Handles wraparound for winter crops (sow=10, peak=5 → returns
        (10, 11, 12, 1, 2, 3, 4)). Excludes the peak month itself — that
        belongs to `flowering_months`.
        """
        gs = list(self.growing_season_months)
        try:
            peak_idx = gs.index(self.peak_month)
        except ValueError:
            # Peak not in growing season? Shouldn't happen, but fall back
            # to "everything before harvest" as a defensive default.
            return tuple(gs[:-1])
        return tuple(gs[:peak_idx])

    @property
    def flowering_months(self) -> tuple[int, ...]:
        """Peak month ± ceil(flowering_window_days / 30).

        With the default `flowering_window_days=10`, returns just the
        peak month itself (one-month window). For crops with longer
        flowering periods (sunflower 30+ days) the window expands. Wrap-
        around-aware: handles peak=5 + window_days=20 → (4, 5, 6) ok,
        or peak=12 + window_days=45 → (11, 12, 1).
        """
        import math
        half = max(1, math.ceil(self.flowering_window_days / 30))
        months: list[int] = [self.peak_month]
        for offset in range(1, half + 1):
            before = ((self.peak_month - offset - 1) % 12) + 1
            after = ((self.peak_month + offset - 1) % 12) + 1
            if before not in months:
                months.insert(0, before)
            if after not in months:
                months.append(after)
        # Restrict to months actually inside the growing season — the
        # ±window can spill into dormancy for winter crops with short
        # flowering windows; we don't want Feb on a wheat-flowering list.
        gs = set(self.growing_season_months)
        return tuple(m for m in months if m in gs)

    @property
    def grain_fill_months(self) -> tuple[int, ...]:
        """Month after peak → harvest month (grain-fill / maturation)."""
        gs = list(self.growing_season_months)
        try:
            peak_idx = gs.index(self.peak_month)
        except ValueError:
            return ()
        return tuple(gs[peak_idx + 1:])


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
    # Sunflower — drought-tolerant, late maturation. GDD base updated
    # 6.0 → 7.2 per FAO ECOCROP (Demydov 2019 cites 7-8°C range).
    "sunflower": CropCalendar(sow_month=5, peak_month=7, harvest_month=9, gdd_base_c=7.2),
    # Soybean — heat-demanding, May-September window. peak_month shifted
    # 7 → 8: NDVI canopy peak is the R5 full-canopy stage (August), not
    # R3 flowering (July) — this lets `ndvi_at_crop_peak_month` pull
    # the right monthly slot for soybean fields.
    "soybean": CropCalendar(sow_month=5, peak_month=8, harvest_month=9, gdd_base_c=10.0),
    # Winter rapeseed dominates Ukrainian production.
    "rapeseed": CropCalendar(sow_month=9, peak_month=5, harvest_month=7, gdd_base_c=5.0),
    # ─── Legume ────────────────────────────────────────────
    # Peas — short cool-season crop.
    "peas": CropCalendar(sow_month=4, peak_month=6, harvest_month=7, gdd_base_c=4.5),
    # ─── Root crops ────────────────────────────────────────
    # Sugar beet — very long season.
    "sugar_beet": CropCalendar(sow_month=4, peak_month=8, harvest_month=10, gdd_base_c=5.0),
    # Potato — shorter than sugar beet, more flexible. GDD base updated
    # 7.0 → 5.5 per FAO ECOCROP (Solanum tuberosum); previous value
    # overestimated GDD accumulation in cool springs and skewed the
    # gdd_proxy feature downward for northern oblasts.
    "potato": CropCalendar(sow_month=4, peak_month=7, harvest_month=9, gdd_base_c=5.5),
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
