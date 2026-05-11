"""ORM models. Importing this module registers all models with `Base.metadata`."""
from app.db.models.alert import Alert
from app.db.models.enums import CropType
from app.db.models.field import Field
from app.db.models.observation import PuUsageMonthly, SatelliteObservation
from app.db.models.prediction import Prediction
from app.db.models.refresh_token import RefreshToken
from app.db.models.user import User
from app.db.models.weather import WeatherObservation

__all__ = [
    "Alert",
    "CropType",
    "Field",
    "Prediction",
    "PuUsageMonthly",
    "RefreshToken",
    "SatelliteObservation",
    "User",
    "WeatherObservation",
]
