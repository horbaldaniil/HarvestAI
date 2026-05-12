"""Rule-based agronomic recommendations for the /weather page.

Pure functions — no DB, no LLM. Each rule reads the cached daily forecast
rows and emits at most one advisory. Rules are intentionally conservative
("спека критична для пшениці" rather than "polyivayte zaraz") because we
don't want a course-project demo to issue real agronomic prescriptions.

The frontend renders the advisories as a stacked card.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls
from typing import Literal

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True, slots=True)
class WeatherAdvice:
    severity: Severity
    title: str
    detail: str


def _has_value(v) -> bool:
    return v is not None


def build_advices(
    days: list[dict],
    *,
    crop: str | None = None,
) -> list[WeatherAdvice]:
    """Inspect the next 7 days and emit applicable advisories.

    Each `days` entry is a dict-shaped weather row with at least:
    `observed_on, temp_min_c, temp_max_c, precip_mm, wind_speed_max_ms,
    cloud_cover_pct, soil_moisture_0_10cm`.
    """
    if not days:
        return []

    out: list[WeatherAdvice] = []
    out.extend(_drought_rule(days))
    out.extend(_heat_rule(days, crop=crop))
    out.extend(_frost_rule(days, crop=crop))
    out.extend(_wind_rule(days))
    out.extend(_rain_rule(days))
    out.extend(_workday_rule(days))
    return out


# ─── Individual rules ──────────────────────────────────────────


def _drought_rule(days: list[dict]) -> list[WeatherAdvice]:
    """≤1 mm precipitation for the full window → flag drought stress."""
    total_precip = sum(_to_f(d.get("precip_mm")) or 0 for d in days)
    if total_precip < 1.0:
        return [WeatherAdvice(
            severity="warning",
            title="Сухий період попереду",
            detail=f"Сумарні опади за 7 днів < 1 мм. Розгляньте полив, "
                   f"якщо фаза розвитку культури того потребує.",
        )]
    if total_precip < 5.0:
        return [WeatherAdvice(
            severity="info",
            title="Малі опади",
            detail=f"Очікується лише {total_precip:.1f} мм за 7 днів — це нижче "
                   f"норми для більшості культур у вегетаційний період.",
        )]
    return []


def _heat_rule(days: list[dict], *, crop: str | None) -> list[WeatherAdvice]:
    """≥3 days with T_max > 30°C → heat stress flag. Wheat is most sensitive."""
    hot = [d for d in days if (_to_f(d.get("temp_max_c")) or 0) > 30]
    if len(hot) >= 5:
        sev: Severity = "critical"
    elif len(hot) >= 3:
        sev = "warning"
    else:
        return []
    crop_note = ""
    if crop == "wheat":
        crop_note = " Для пшениці у фазі цвітіння/наливу зерна — критично."
    elif crop == "corn":
        crop_note = " Кукурудза витримує краще, але цвітіння (липень) чутливе."
    return [WeatherAdvice(
        severity=sev,
        title=f"Очікується {len(hot)} спекотних днів",
        detail=f"T_max > 30°C — рослини можуть закрити продихи й уповільнити "
               f"фотосинтез.{crop_note}",
    )]


def _frost_rule(days: list[dict], *, crop: str | None) -> list[WeatherAdvice]:
    """Negative night-time temperatures in spring/autumn → frost risk."""
    frost = [d for d in days if (_to_f(d.get("temp_min_c")) or 99) < 0]
    if not frost:
        return []
    today = days[0].get("observed_on")
    month = _parse_month(today)
    relevant = month in {3, 4, 5, 9, 10, 11} if month else True
    if not relevant:
        return []
    sev: Severity = "critical" if len(frost) >= 2 else "warning"
    crop_note = " Сходи й суцвіття дуже вразливі." if crop in {"corn", "sunflower"} else ""
    return [WeatherAdvice(
        severity=sev,
        title=f"Ризик заморозків ({len(frost)} ноч.)",
        detail=f"Прогнозується мінімум нижче 0°C.{crop_note}",
    )]


def _wind_rule(days: list[dict]) -> list[WeatherAdvice]:
    """Wind > 6 m/s → advise against ground spraying."""
    windy = [
        (d.get("observed_on"), _to_f(d.get("wind_speed_max_ms")))
        for d in days
        if (_to_f(d.get("wind_speed_max_ms")) or 0) > 6.0
    ]
    if not windy:
        return []
    return [WeatherAdvice(
        severity="info",
        title=f"Вітряні дні: {len(windy)}",
        detail="Уникайте наземної обробки пестицидами при вітрі > 6 м/с — "
               "знос капель далеко від цільового поля.",
    )]


def _rain_rule(days: list[dict]) -> list[WeatherAdvice]:
    """≥1 day with precip > 10 mm → flag — too wet for fieldwork."""
    heavy = [d for d in days if (_to_f(d.get("precip_mm")) or 0) > 10]
    if not heavy:
        return []
    return [WeatherAdvice(
        severity="info",
        title=f"Сильні опади: {len(heavy)} дн.",
        detail="У ці дні поле буде грузьке — техніку краще тримати у дворі.",
    )]


def _workday_rule(days: list[dict]) -> list[WeatherAdvice]:
    """Find the day with the best work window — no rain, mild temperature,
    low wind."""
    candidates = []
    for d in days:
        precip = _to_f(d.get("precip_mm")) or 0
        wind = _to_f(d.get("wind_speed_max_ms")) or 0
        tmax = _to_f(d.get("temp_max_c")) or 99
        if precip < 1.0 and wind < 6.0 and 10 <= tmax <= 28:
            candidates.append((d.get("observed_on"), tmax, wind))
    if not candidates:
        return []
    candidates.sort(key=lambda x: (x[1], x[2]))
    best = candidates[0]
    return [WeatherAdvice(
        severity="info",
        title="Найкращий день для роботи",
        detail=f"{best[0]} — без дощу, температура ~{best[1]:.0f}°C, "
               f"вітер {best[2]:.1f} м/с.",
    )]


# ─── Helpers ──────────────────────────────────────────────────


def _to_f(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_month(d) -> int | None:
    if d is None:
        return None
    if isinstance(d, date_cls):
        return d.month
    try:
        return date_cls.fromisoformat(str(d)).month
    except ValueError:
        return None


__all__ = ["WeatherAdvice", "build_advices"]
