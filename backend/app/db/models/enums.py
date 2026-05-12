"""Domain enums shared across ORM models and Pydantic schemas."""
from __future__ import annotations

from enum import Enum


class CropType(str, Enum):
    """Supported crop types for Ukrainian agriculture.

    Expanded from the original three (wheat / corn / sunflower — top by
    sown area) to 13 covering the bulk of Ukrainian arable production:
    main cereals (wheat/barley/rye/oats/buckwheat), oilseeds
    (sunflower/soybean/rapeseed), the corn dyad (grain corn + silage),
    legumes (peas) and the two major root crops (sugar beet, potato).

    Selection criteria:
      - Coverage: each crop occupies >0.5 % of the national arable area
        in 2017-2023, per Держстат annual statistics.
      - Distinct phenology: each crop has a recognisably different NDVI
        curve shape — winter wheat peaks in May/June, sunflower in July,
        sugar beet in August, etc. — so the time-series features actually
        carry discriminative signal.
      - Available yield data: per-oblast yields are published by Держстат
        for all 13 crops; without this, training is impossible.

    Crop labels are also used as join keys in `training_set_v3.parquet`,
    `derzhstat_yield_2017_2023.yaml`, and the frontend i18n bundle —
    DO NOT rename without coordinated migration of those files.
    """

    # — Original three (used by every model version v1 / v2)
    WHEAT = "wheat"
    CORN = "corn"
    SUNFLOWER = "sunflower"

    # — Major oilseeds added in v3 (alongside sunflower)
    SOYBEAN = "soybean"
    RAPESEED = "rapeseed"

    # — Spring/winter cereals
    BARLEY = "barley"
    RYE = "rye"
    OATS = "oats"
    BUCKWHEAT = "buckwheat"

    # — Legume
    PEAS = "peas"

    # — Root crops
    SUGAR_BEET = "sugar_beet"
    POTATO = "potato"

    # — Forage variant of corn (whole-plant harvest, distinct yield profile)
    CORN_SILAGE = "corn_silage"

    @property
    def display_uk(self) -> str:
        return {
            CropType.WHEAT: "Пшениця",
            CropType.CORN: "Кукурудза",
            CropType.SUNFLOWER: "Соняшник",
            CropType.SOYBEAN: "Соя",
            CropType.RAPESEED: "Ріпак",
            CropType.BARLEY: "Ячмінь",
            CropType.RYE: "Жито",
            CropType.OATS: "Овес",
            CropType.BUCKWHEAT: "Гречка",
            CropType.PEAS: "Горох",
            CropType.SUGAR_BEET: "Цукровий буряк",
            CropType.POTATO: "Картопля",
            CropType.CORN_SILAGE: "Кукурудза на силос",
        }[self]

    @property
    def default_color(self) -> str:
        """Hex color used when the user has not chosen a custom one.

        Palette chosen to be visually distinguishable on the dashboard
        crops-breakdown donut and on Recharts time-series. Hue picked
        roughly to match the crop's visual identity in agricultural
        photography (wheat=amber, sunflower=gold, peas=fresh green, etc.).
        """
        return {
            CropType.WHEAT: "#f0c419",         # amber — ripe wheat
            CropType.CORN: "#e67e22",          # orange — corn cob
            CropType.SUNFLOWER: "#f1c40f",     # gold — sunflower disk
            CropType.SOYBEAN: "#8bc34a",       # light green — soybean canopy
            CropType.RAPESEED: "#fdd835",      # bright yellow — rapeseed flowers
            CropType.BARLEY: "#d4a373",        # tan — mature barley
            CropType.RYE: "#9c8866",           # darker tan — rye
            CropType.OATS: "#c7b07a",          # warm grey — oat panicles
            CropType.BUCKWHEAT: "#b08fb8",     # pinkish-purple — buckwheat flowers
            CropType.PEAS: "#7cb342",          # fresh green — pea pods
            CropType.SUGAR_BEET: "#5d4037",    # brown — soil/beet
            CropType.POTATO: "#a1887f",        # earthy brown — potato skin
            CropType.CORN_SILAGE: "#558b2f",   # dark green — whole-plant corn
        }[self]
