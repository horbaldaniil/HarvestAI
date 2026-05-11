"""Function-calling tools exposed to the chat assistant.

Each tool is a *read-only* projection of the user's data: the bot may inspect
fields, observations, predictions, alerts, weather, but it cannot mutate
anything. Ownership is enforced inside every tool, not by the LLM —
even if the bot fabricates a field_id, the tool checks `user_id` server-side.

Result payloads are intentionally compact (truncated lists, rounded floats)
so they fit in OpenAI's tool-response token budget. We document the
truncation explicitly in the response (e.g. `{"truncated": true, ...}`) so
the assistant can warn the user.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Alert,
    Field,
    Prediction,
    SatelliteObservation,
    User,
    WeatherObservation,
)

log = logging.getLogger(__name__)

# How many observation rows we feed back to the LLM in one tool call.
# 50 weekly buckets ≈ ~1 year, which covers the typical question scope
# while keeping the response under ~3KB.
MAX_OBSERVATIONS = 50
# Weather: similarly capped to one growing season's worth of days.
MAX_WEATHER_DAYS = 180
INDEX_NAMES = {"ndvi", "evi", "ndwi", "savi"}


class ToolError(Exception):
    """Returned to the bot as `{"error": "..."}` instead of crashing."""


# ─── Tool schemas (OpenAI function-calling format) ─────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_field_info",
            "description": (
                "Returns basic metadata about the user's field: name, crop "
                "type, season year, area in hectares, and the centroid "
                "latitude/longitude. Use this whenever you need to talk about "
                "a specific field's identity or location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id": {
                        "type": "integer",
                        "description": "Database id of the field.",
                    },
                },
                "required": ["field_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_observations",
            "description": (
                "Returns the satellite-derived vegetation index time series "
                "for a field. One row per Sentinel-2 acquisition. NULL values "
                "mean the bucket was too cloudy. Use this for any question "
                "involving NDVI/EVI/NDWI/SAVI trends, peaks, or drops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id": {"type": "integer"},
                    "index": {
                        "type": "string",
                        "enum": list(INDEX_NAMES),
                        "description": "Which vegetation index to return.",
                    },
                    "since": {
                        "type": "string",
                        "description": "Optional ISO date lower bound (YYYY-MM-DD).",
                    },
                    "until": {
                        "type": "string",
                        "description": "Optional ISO date upper bound (YYYY-MM-DD).",
                    },
                },
                "required": ["field_id", "index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prediction",
            "description": (
                "Returns the most recent yield prediction for a field, "
                "including the predicted t/ha value, confidence interval, "
                "model name/version, and SHAP top-5 feature contributions. "
                "Returns null if no prediction exists yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {"field_id": {"type": "integer"}},
                "required": ["field_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alerts",
            "description": (
                "Returns active (unacknowledged) anomaly alerts. Pass "
                "field_id to scope to one field, or omit to get all alerts "
                "for the user. Each alert has a severity (info/warning/"
                "critical), type (ndvi_drop/drought_stress/heat_stress), "
                "and a Ukrainian message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id": {
                        "type": "integer",
                        "description": "Optional: filter to one field.",
                    },
                    "include_acknowledged": {
                        "type": "boolean",
                        "description": "Default false. Set true to include history.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Returns daily weather aggregates (temperature, precipitation, "
                "humidity, radiation) for a field's location. By default returns "
                "the last 30 days of history. Set `forecast=true` to instead "
                "return the upcoming 14-day forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id": {"type": "integer"},
                    "forecast": {
                        "type": "boolean",
                        "description": "True = 14-day forecast. False = recent history.",
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "How many days of history (1-180). Default 30.",
                    },
                },
                "required": ["field_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_fields",
            "description": (
                "Returns a compact list of all of the user's fields, with id, "
                "name, crop type, area, and current NDVI status. Use this "
                "before answering portfolio-level questions like 'what is "
                "my worst-performing field'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ─── Tool implementations ─────────────────────────────────────


async def _owned_field(db: AsyncSession, user: User, field_id: int) -> Field:
    field = await db.get(Field, field_id)
    if field is None or field.user_id != user.id:
        raise ToolError(f"field_not_found: id={field_id}")
    return field


def _parse_iso_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise ToolError(f"bad_date_format: {s}") from exc


async def get_field_info(
    *, db: AsyncSession, user: User, field_id: int
) -> dict[str, Any]:
    field = await _owned_field(db, user, field_id)
    # Fetch centroid as WKT through PostGIS — cheaper than loading the geom.
    from geoalchemy2 import functions as geofn  # local import: keep cold start light
    from sqlalchemy import select as sa_select

    centroid_wkt = await db.scalar(
        sa_select(geofn.ST_AsText(field.centroid))
    )
    lat: float | None = None
    lon: float | None = None
    if centroid_wkt:
        # WKT: "POINT(lon lat)"
        try:
            inner = centroid_wkt.replace("POINT(", "").rstrip(")")
            lon_s, lat_s = inner.split()
            lat = float(lat_s)
            lon = float(lon_s)
        except ValueError:
            pass

    return {
        "field_id": field.id,
        "name": field.name,
        "crop_type": field.crop_type.value if field.crop_type else None,
        "season_year": field.season_year,
        "area_ha": round(float(field.area_ha), 2) if field.area_ha is not None else None,
        "centroid_lat": round(lat, 5) if lat is not None else None,
        "centroid_lon": round(lon, 5) if lon is not None else None,
    }


async def get_observations(
    *,
    db: AsyncSession,
    user: User,
    field_id: int,
    index: str,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    await _owned_field(db, user, field_id)
    if index not in INDEX_NAMES:
        raise ToolError(f"bad_index: {index}")
    col = getattr(SatelliteObservation, f"{index}_mean")

    stmt = (
        select(
            SatelliteObservation.observed_on,
            col,
            SatelliteObservation.cloud_cover,
        )
        .where(SatelliteObservation.field_id == field_id)
        .order_by(SatelliteObservation.observed_on.desc())
        .limit(MAX_OBSERVATIONS + 1)
    )
    d_since = _parse_iso_date(since)
    d_until = _parse_iso_date(until)
    if d_since:
        stmt = stmt.where(SatelliteObservation.observed_on >= d_since)
    if d_until:
        stmt = stmt.where(SatelliteObservation.observed_on <= d_until)

    rows = (await db.execute(stmt)).all()
    truncated = len(rows) > MAX_OBSERVATIONS
    rows = rows[:MAX_OBSERVATIONS]
    # Send oldest-first so the model sees the natural trend direction.
    rows.reverse()

    return {
        "field_id": field_id,
        "index": index,
        "count": len(rows),
        "truncated": truncated,
        "observations": [
            {
                "date": d.isoformat(),
                "value": round(float(v), 3) if v is not None else None,
                "cloud_cover": round(float(cc), 1) if cc is not None else None,
            }
            for d, v, cc in rows
        ],
    }


async def get_prediction(
    *, db: AsyncSession, user: User, field_id: int
) -> dict[str, Any]:
    await _owned_field(db, user, field_id)
    result = await db.execute(
        select(Prediction)
        .where(Prediction.field_id == field_id)
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        return {"field_id": field_id, "prediction": None}

    return {
        "field_id": field_id,
        "prediction": {
            "value_tha": round(float(pred.value_tha), 2),
            "confidence_tha": (
                round(float(pred.confidence), 2) if pred.confidence is not None else None
            ),
            "model_name": pred.model_name,
            "model_version": pred.model_version,
            "predicted_at": pred.predicted_at.isoformat(),
            "shap_top": [
                {
                    "name": s.get("name"),
                    "value": s.get("value"),
                    "contribution": round(float(s.get("contribution", 0)), 3),
                }
                for s in (pred.shap_top_json or [])
            ],
        },
    }


async def get_alerts(
    *,
    db: AsyncSession,
    user: User,
    field_id: int | None = None,
    include_acknowledged: bool = False,
) -> dict[str, Any]:
    stmt = (
        select(Alert, Field.name)
        .join(Field, Field.id == Alert.field_id, isouter=True)
        .where(Alert.user_id == user.id)
        .order_by(desc(Alert.created_at))
        .limit(50)
    )
    if not include_acknowledged:
        stmt = stmt.where(Alert.acknowledged.is_(False))
    if field_id is not None:
        await _owned_field(db, user, field_id)
        stmt = stmt.where(Alert.field_id == field_id)

    rows = (await db.execute(stmt)).all()
    return {
        "count": len(rows),
        "alerts": [
            {
                "id": a.id,
                "field_id": a.field_id,
                "field_name": fname,
                "severity": a.severity,
                "type": a.type,
                "message_uk": a.message_uk,
                "metric_value": (
                    round(float(a.metric_value), 3) if a.metric_value is not None else None
                ),
                "threshold": (
                    round(float(a.threshold), 3) if a.threshold is not None else None
                ),
                "acknowledged": a.acknowledged,
                "created_at": a.created_at.isoformat(),
            }
            for a, fname in rows
        ],
    }


async def get_weather(
    *,
    db: AsyncSession,
    user: User,
    field_id: int,
    forecast: bool = False,
    days_back: int = 30,
) -> dict[str, Any]:
    await _owned_field(db, user, field_id)
    days_back = max(1, min(days_back, MAX_WEATHER_DAYS))
    today = datetime.now(tz=timezone.utc).date()

    stmt = (
        select(WeatherObservation)
        .where(WeatherObservation.field_id == field_id)
        .order_by(WeatherObservation.observed_on.asc())
    )
    if forecast:
        stmt = stmt.where(WeatherObservation.is_forecast.is_(True))
    else:
        stmt = stmt.where(WeatherObservation.is_forecast.is_(False))
        stmt = stmt.where(WeatherObservation.observed_on >= today - timedelta(days=days_back))

    rows = list((await db.execute(stmt)).scalars().all())

    return {
        "field_id": field_id,
        "kind": "forecast" if forecast else "history",
        "count": len(rows),
        "days": [
            {
                "date": r.observed_on.isoformat(),
                "temp_min_c": _round(r.temp_min_c, 1),
                "temp_max_c": _round(r.temp_max_c, 1),
                "temp_mean_c": _round(r.temp_mean_c, 1),
                "precip_mm": _round(r.precip_mm, 1),
                "humidity_pct": _round(r.humidity_pct, 1),
                "radiation_mj": _round(r.radiation_mj, 2),
            }
            for r in rows
        ],
    }


async def list_user_fields(
    *, db: AsyncSession, user: User
) -> dict[str, Any]:
    rows = (
        await db.execute(
            select(Field)
            .where(Field.user_id == user.id)
            .order_by(Field.name.asc())
        )
    ).scalars().all()

    fields_payload = []
    for f in rows:
        # Latest NDVI from satellite_observations (cheap correlated subquery).
        latest_ndvi = await db.scalar(
            select(SatelliteObservation.ndvi_mean)
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.ndvi_mean.is_not(None))
            .order_by(SatelliteObservation.observed_on.desc())
            .limit(1)
        )
        fields_payload.append(
            {
                "field_id": f.id,
                "name": f.name,
                "crop_type": f.crop_type.value if f.crop_type else None,
                "season_year": f.season_year,
                "area_ha": _round(f.area_ha, 2),
                "latest_ndvi": _round(latest_ndvi, 3),
            }
        )

    return {"count": len(fields_payload), "fields": fields_payload}


# ─── Dispatcher ───────────────────────────────────────────────


_DISPATCH = {
    "get_field_info": get_field_info,
    "get_observations": get_observations,
    "get_prediction": get_prediction,
    "get_alerts": get_alerts,
    "get_weather": get_weather,
    "list_user_fields": list_user_fields,
}


async def dispatch_tool(
    name: str, args: dict[str, Any], *, db: AsyncSession, user: User
) -> dict[str, Any]:
    """Run a single tool call, returning a JSON-serializable result.

    Tool errors are caught and returned as `{"error": "..."}` so the bot
    can recover instead of the whole assistant turn aborting.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown_tool: {name}"}
    try:
        return await fn(db=db, user=user, **(args or {}))
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        # Wrong arg name from the model — give it a hint instead of crashing.
        log.warning("Tool %s called with bad args %s: %s", name, args, exc)
        return {"error": f"bad_arguments: {exc}"}
    except Exception as exc:  # noqa: BLE001
        log.exception("Tool %s raised: %s", name, exc)
        return {"error": "internal_error"}


def _round(v: Any, ndigits: int) -> float | None:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


__all__ = [
    "TOOL_SCHEMAS",
    "ToolError",
    "dispatch_tool",
    "get_alerts",
    "get_field_info",
    "get_observations",
    "get_prediction",
    "get_weather",
    "list_user_fields",
]
