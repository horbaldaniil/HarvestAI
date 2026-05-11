"""Domain enums shared across ORM models and Pydantic schemas."""
from __future__ import annotations

from enum import Enum


class CropType(str, Enum):
    """Supported crop types for Ukrainian agriculture.

    Wheat / corn / sunflower were chosen because:
    - They are the top-3 grain/oilseed crops by area in Ukraine
    - Derzhstat publishes yield data for all three (training set for ML)
    - Their NDVI seasonal curves are sufficiently distinct to demonstrate
      crop-classification in later weeks.
    """

    WHEAT = "wheat"
    CORN = "corn"
    SUNFLOWER = "sunflower"

    @property
    def display_uk(self) -> str:
        return {
            CropType.WHEAT: "Пшениця",
            CropType.CORN: "Кукурудза",
            CropType.SUNFLOWER: "Соняшник",
        }[self]

    @property
    def default_color(self) -> str:
        """Hex color used when the user has not chosen a custom one."""
        return {
            CropType.WHEAT: "#f0c419",       # amber
            CropType.CORN: "#e67e22",        # orange
            CropType.SUNFLOWER: "#f1c40f",   # gold
        }[self]
