"""User-settings endpoints — powers the Settings page (Profile +
CropPrices cards) and the dashboard income projection.

Backed by `users.settings_json` (JSONB, Alembic 0007). One place for any
small-scale per-user preference so we don't sprinkle 1:1 tables.

Schema we agree on:

  settings_json["crop_prices"] = {
      "currency": "UAH",
      "wheat": 8500.0,        # ціна за тонну
      "corn": 7200.0,
      # … all 13 crops supported; missing keys = "not set"
  }
  settings_json["profile"] = {
      "farm_name": "АгроСвіт",
      "phone_number": "+380501234567",
  }
  # `full_name` stays on the `users` table (auth-adjacent column), but
  # the /profile endpoint surfaces it alongside the JSONB fields so
  # the frontend has a single form.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.db.models import User
from app.deps import CurrentUser, DbSession

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Currencies the UI offers as a dropdown. Free-form would be too lax (we'd
# end up with "uah"/"UAH"/"грн" inconsistencies polluting reports).
Currency = Literal["UAH", "USD", "EUR"]

DEFAULT_CURRENCY: Currency = "UAH"


class CropPricesRead(BaseModel):
    """All prices keyed by crop slug. None = user hasn't set a price for it.

    Mirrors the 13-member `CropType` enum in
    `app/db/models/enums.py` — when a new crop is added there, add the
    same field here (and to `CropPricesUpdate` below).
    """
    model_config = ConfigDict(extra="ignore")
    currency: Currency = DEFAULT_CURRENCY
    wheat: float | None = None
    corn: float | None = None
    sunflower: float | None = None
    soybean: float | None = None
    rapeseed: float | None = None
    barley: float | None = None
    rye: float | None = None
    oats: float | None = None
    buckwheat: float | None = None
    peas: float | None = None
    sugar_beet: float | None = None
    potato: float | None = None
    corn_silage: float | None = None


class CropPricesUpdate(BaseModel):
    """Same shape but every field is independently clearable.

    Partial-update semantics: only fields *present* in the payload are
    written; sending `null` clears that single crop's price without
    touching siblings. See the loop in `update_crop_prices` below.
    """
    currency: Currency | None = None
    wheat: float | None = Field(default=None, ge=0)
    corn: float | None = Field(default=None, ge=0)
    sunflower: float | None = Field(default=None, ge=0)
    soybean: float | None = Field(default=None, ge=0)
    rapeseed: float | None = Field(default=None, ge=0)
    barley: float | None = Field(default=None, ge=0)
    rye: float | None = Field(default=None, ge=0)
    oats: float | None = Field(default=None, ge=0)
    buckwheat: float | None = Field(default=None, ge=0)
    peas: float | None = Field(default=None, ge=0)
    sugar_beet: float | None = Field(default=None, ge=0)
    potato: float | None = Field(default=None, ge=0)
    corn_silage: float | None = Field(default=None, ge=0)


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


# ─── Profile (full_name + farm metadata) ─────────────────────


class ProfileRead(BaseModel):
    """User-facing profile card payload.

    `full_name` is stored on the `users` table (auth-adjacent column),
    while `farm_name` and `phone_number` live in `settings_json["profile"]`
    to avoid alembic migrations for every new tweak. The frontend
    consumes a single flat shape — the split is internal.
    """
    model_config = ConfigDict(extra="ignore")
    full_name: str | None = None
    farm_name: str | None = None
    phone_number: str | None = None


class ProfileUpdate(BaseModel):
    """Partial update: only fields present in the payload are written.
    Empty string is treated the same as None — clears the field.
    """
    # Light validation: trim length so a runaway paste can't blow up
    # the JSONB column or break PDF report headers.
    full_name: str | None = Field(default=None, max_length=255)
    farm_name: str | None = Field(default=None, max_length=255)
    phone_number: str | None = Field(default=None, max_length=64)


def _empty_to_none(v: str | None) -> str | None:
    """Treat `""` as `None` so the UI's "clear field" gesture
    (delete all chars + Save) actually removes the value."""
    if v is None:
        return None
    stripped = v.strip()
    return stripped or None


@router.get("/profile", response_model=ProfileRead)
async def get_profile(current_user: CurrentUser) -> ProfileRead:
    profile = (current_user.settings_json or {}).get("profile") or {}
    return ProfileRead(
        full_name=current_user.full_name,
        farm_name=profile.get("farm_name"),
        phone_number=profile.get("phone_number"),
    )


@router.put("/profile", response_model=ProfileRead)
async def update_profile(
    payload: ProfileUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> ProfileRead:
    payload_dict = payload.model_dump(exclude_unset=True)

    # full_name lives on the users table — write directly to the column.
    if "full_name" in payload_dict:
        current_user.full_name = _empty_to_none(payload_dict["full_name"])

    # farm_name + phone_number live in settings_json["profile"].
    settings = dict(current_user.settings_json or {})
    profile = dict(settings.get("profile") or {})
    for key in ("farm_name", "phone_number"):
        if key in payload_dict:
            val = _empty_to_none(payload_dict[key])
            if val is None:
                profile.pop(key, None)
            else:
                profile[key] = val
    settings["profile"] = profile
    # Reassign so JSONB diff is detected by SQLAlchemy.
    current_user.settings_json = settings

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return ProfileRead(
        full_name=current_user.full_name,
        farm_name=profile.get("farm_name"),
        phone_number=profile.get("phone_number"),
    )


# ─── AI-suggested crop prices ──────────────────────────────────


class CropPricesSuggestRequest(BaseModel):
    """Optional context for the AI — currency to quote in. Defaults to
    UAH, which is the only currency we expect 95 % of users to want."""
    currency: Currency = DEFAULT_CURRENCY


@router.post("/crop-prices/suggest", response_model=CropPricesRead)
async def suggest_crop_prices(
    payload: CropPricesSuggestRequest,
    current_user: CurrentUser,
) -> CropPricesRead:
    """Ask the LLM for a plausible-current-season price quote for each
    of the 13 supported crops in `payload.currency`/тонна.

    Returns a `CropPricesRead`-shaped payload that the frontend treats
    as a *suggestion* — the user reviews the numbers in the form, can
    edit any cell, and clicks Save to persist via PUT /crop-prices.
    No write happens here; this endpoint is read-only by design.
    """
    _ = current_user  # auth-only — no per-user personalisation in the prompt
    from app.services.ai_crop_prices import suggest_prices_via_openai

    try:
        prices = await suggest_prices_via_openai(payload.currency)
    except Exception as exc:  # noqa: BLE001
        log.warning("AI crop-prices suggestion failed: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Не вдалося отримати пропозицію від AI. "
            "Перевірте, що OPENAI_API_KEY налаштовано, і спробуйте ще раз.",
        ) from exc

    return CropPricesRead(currency=payload.currency, **prices)
