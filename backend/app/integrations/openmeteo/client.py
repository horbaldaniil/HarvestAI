"""Open-Meteo HTTP client.

Two endpoints we care about:
- archive-api.open-meteo.com (historical, 1940+) — used both at training
  time (notebook 02) and at runtime for filling weather_observations
- api.open-meteo.com/v1/forecast (today + 14 days) — used for Dashboard
  forecast card

We keep this thin: no caching here, no DB writes — callers (jobs +
notebooks) decide what to persist. Single async wrapper that surfaces a
clean dataclass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Daily-aggregated variables we ask Open-Meteo to return. The first six
# feed the ML pipeline + dashboard KPIs; the last three (wind, cloud,
# soil moisture) drive the dedicated /weather page (Week 8).
DAILY_VARS: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "cloud_cover_mean",
    "soil_moisture_0_to_10cm_mean",
)


@dataclass(frozen=True, slots=True)
class DailyWeather:
    observed_on: date
    temp_max_c: float | None
    temp_min_c: float | None
    temp_mean_c: float | None
    precip_mm: float | None
    humidity_pct: float | None
    radiation_mj: float | None
    # Week 8 fields — all optional so older cached rows that predate the
    # columns still parse cleanly.
    wind_speed_max_ms: float | None = None
    cloud_cover_pct: float | None = None
    soil_moisture_0_10cm: float | None = None


class OpenMeteoError(RuntimeError):
    pass


class OpenMeteoClient:
    """Async client for Open-Meteo daily endpoints (historical + forecast)."""

    def __init__(self, http: httpx.AsyncClient | None = None):
        self._http = http or httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_historical(
        self, lat: float, lon: float, start: date, end: date
    ) -> list[DailyWeather]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto",
        }
        return await self._fetch(HISTORICAL_URL, params)

    async def fetch_forecast(
        self, lat: float, lon: float, days: int = 14
    ) -> list[DailyWeather]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": min(max(days, 1), 16),
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto",
        }
        return await self._fetch(FORECAST_URL, params)

    async def _fetch(self, url: str, params: dict) -> list[DailyWeather]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TransportError, _TransientHTTP)),
            reraise=True,
        ):
            with attempt:
                resp = await self._http.get(url, params=params)
                if resp.status_code in (429, 502, 503, 504):
                    raise _TransientHTTP(f"{resp.status_code}")
                if resp.status_code >= 400:
                    raise OpenMeteoError(
                        f"Open-Meteo {resp.status_code}: {resp.text[:200]}"
                    )
                return _parse_daily(resp.json())
        raise OpenMeteoError("Exhausted retries")


def _parse_daily(payload: dict) -> list[DailyWeather]:
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    result: list[DailyWeather] = []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    tmean = daily.get("temperature_2m_mean") or []
    precip = daily.get("precipitation_sum") or []
    humid = daily.get("relative_humidity_2m_mean") or []
    rad = daily.get("shortwave_radiation_sum") or []
    wind = daily.get("wind_speed_10m_max") or []
    cloud = daily.get("cloud_cover_mean") or []
    soil = daily.get("soil_moisture_0_to_10cm_mean") or []

    for i, day in enumerate(dates):
        try:
            observed_on = date.fromisoformat(day)
        except ValueError:
            continue
        result.append(
            DailyWeather(
                observed_on=observed_on,
                temp_max_c=_safe_float(tmax, i),
                temp_min_c=_safe_float(tmin, i),
                temp_mean_c=_safe_float(tmean, i),
                precip_mm=_safe_float(precip, i),
                humidity_pct=_safe_float(humid, i),
                radiation_mj=_safe_float(rad, i),
                wind_speed_max_ms=_safe_float(wind, i),
                cloud_cover_pct=_safe_float(cloud, i),
                soil_moisture_0_10cm=_safe_float(soil, i),
            )
        )
    return result


def _safe_float(arr: list, i: int) -> float | None:
    try:
        v = arr[i]
    except IndexError:
        return None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class _TransientHTTP(Exception):
    pass


__all__ = [
    "DAILY_VARS",
    "DailyWeather",
    "OpenMeteoClient",
    "OpenMeteoError",
]
