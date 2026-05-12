"""User-settings endpoints — currently powering the dashboard income card.

Backed by `users.settings_json` (JSONB, Alembic 0007). One place for any
small-scale per-user preference so we don't sprinkle 1:1 tables.

Schema we agree on for crop prices:

  settings_json["crop_prices"] = {
      "currency": "UAH",
      "wheat": 8500.0,        # ціна за тонну
      "corn": 7200.0,
      "sunflower": 17000.0,
  }
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.db.models import User
from app.deps import CurrentUser, DbSession

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Currencies the UI offers as a dropdown. Free-form would be too lax (we'd
# end up with "uah"/"UAH"/"грн" inconsistencies polluting reports).
Currency = Literal["UAH", "USD", "EUR"]

DEFAULT_CURRENCY: Currency = "UAH"


class CropPricesRead(BaseModel):
    """All prices keyed by crop_type. None = user hasn't set a price for it."""
    model_config = ConfigDict(extra="ignore")
    currency: Currency = DEFAULT_CURRENCY
    wheat: float | None = None
    corn: float | None = None
    sunflower: float | None = None


class CropPricesUpdate(BaseModel):
    """Same shape but accepts None to clear individual entries."""
    currency: Currency | None = None
    wheat: float | None = Field(default=None, ge=0)
    corn: float | None = Field(default=None, ge=0)
    sunflower: float | None = Field(default=None, ge=0)


@router.get("/crop-prices", response_model=CropPricesRead)
async def get_crop_prices(current_user: CurrentUser) -> CropPricesRead:
    raw = (current_user.settings_json or {}).get("crop_prices") or {}
    return CropPricesRead(**raw)


@router.put("/crop-prices", response_model=CropPricesRead)
async def update_crop_prices(
    payload: CropPricesUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> CropPricesRead:
    settings = dict(current_user.settings_json or {})
    current = dict(settings.get("crop_prices") or {})

    # Partial-update semantics: only fields present in the payload are
    # written. Setting `null` on a field clears that single crop's price
    # without affecting siblings.
    payload_dict = payload.model_dump(exclude_unset=True)
    for key, value in payload_dict.items():
        if value is None and key != "currency":
            current.pop(key, None)
        else:
            current[key] = value

    settings["crop_prices"] = current
    # SQLAlchemy needs us to reassign the dict so JSONB diff is detected.
    current_user.settings_json = settings
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return CropPricesRead(**current)
