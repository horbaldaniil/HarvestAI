"""Rule-based anomaly detectors for NDVI drop, drought, and heat stress.

Why rule-based (not Isolation Forest etc.):
- Explainable. Every alert maps to a concrete threshold the user can
  understand (e.g. "NDVI on 2024-07-15 was 2.3 std-devs below the wheat
  norm for week 28").
- Stable across small training sets. ML anomaly detectors need a lot of
  history we don't have for Ukrainian fields.
- Easy to tune from the project README without retraining.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from app.db.models import SatelliteObservation, WeatherObservation
from app.db.models.enums import CropType

log = logging.getLogger(__name__)

# Z-score thresholds for NDVI drop relative to per-crop seasonal norms.
NDVI_WARNING_Z = -1.5
NDVI_CRITICAL_Z = -2.5

# Drought threshold — sum of precip over the last N days as fraction of
# expected (which we approximate as 15 mm / week × 2 weeks = 30 mm).
DROUGHT_WINDOW_DAYS = 14
DROUGHT_EXPECTED_PRECIP_MM = 30.0
DROUGHT_WARNING_RATIO = 0.30
DROUGHT_CRITICAL_RATIO = 0.10

# Heat stress — count of T_max > 30°C days in the last N days.
HEAT_WINDOW_DAYS = 14
HEAT_WARNING_DAYS = 5
HEAT_CRITICAL_DAYS = 8


@dataclass(frozen=True, slots=True)
class AnomalySignal:
    type: str          # "ndvi_drop" | "drought_stress" | "heat_stress"
    severity: str      # "info" | "warning" | "critical"
    message_uk: str
    metric_value: float | None
    threshold: float | None


def detect_ndvi_anomalies(
    observations: list[SatelliteObservation],
    crop: CropType,
    seasonal_norms: dict,
) -> list[AnomalySignal]:
    """Flag observations whose NDVI is more than N std-devs below the
    per-crop, per-ISO-week seasonal norm.
    """
    crop_norms = (seasonal_norms or {}).get(crop.value)
    if not crop_norms:
        return []

    signals: list[AnomalySignal] = []
    for obs in observations:
        if obs.ndvi_mean is None:
            continue
        week = obs.observed_on.isocalendar()[1]
        norm = crop_norms.get(str(week))
        if not norm:
            continue
        mu = float(norm["mean"])
        sigma = float(norm["std"]) or 0.05
        z = (float(obs.ndvi_mean) - mu) / sigma

        if z <= NDVI_CRITICAL_Z:
            signals.append(AnomalySignal(
                type="ndvi_drop",
                severity="critical",
                message_uk=(
                    f"Критичне падіння NDVI: {float(obs.ndvi_mean):.2f} "
                    f"(норма для тижня {week}: {mu:.2f} ± {sigma:.2f})"
                ),
                metric_value=round(float(obs.ndvi_mean), 3),
                threshold=round(mu + NDVI_CRITICAL_Z * sigma, 3),
            ))
        elif z <= NDVI_WARNING_Z:
            signals.append(AnomalySignal(
                type="ndvi_drop",
                severity="warning",
                message_uk=(
                    f"NDVI нижче норми: {float(obs.ndvi_mean):.2f} "
                    f"(норма для тижня {week}: {mu:.2f} ± {sigma:.2f})"
                ),
                metric_value=round(float(obs.ndvi_mean), 3),
                threshold=round(mu + NDVI_WARNING_Z * sigma, 3),
            ))

    # Deduplicate: keep only the most recent NDVI drop per (week, severity).
    seen = set()
    deduped = []
    for s in sorted(signals, key=lambda x: x.metric_value or 0):
        key = (s.type, s.severity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def detect_drought(
    weather: list[WeatherObservation], today: date
) -> AnomalySignal | None:
    """Insufficient precipitation in the last DROUGHT_WINDOW_DAYS days."""
    if not weather:
        return None
    recent = [
        w for w in weather
        if w.is_forecast is False
        and (today - w.observed_on).days <= DROUGHT_WINDOW_DAYS
        and (today - w.observed_on).days >= 0
    ]
    if len(recent) < DROUGHT_WINDOW_DAYS // 2:
        return None
    total = sum(float(w.precip_mm or 0) for w in recent)
    ratio = total / DROUGHT_EXPECTED_PRECIP_MM
    if ratio < DROUGHT_CRITICAL_RATIO:
        return AnomalySignal(
            type="drought_stress",
            severity="critical",
            message_uk=(
                f"Критична посуха: лише {total:.1f} мм опадів за "
                f"{DROUGHT_WINDOW_DAYS} днів (норма ~{DROUGHT_EXPECTED_PRECIP_MM:.0f} мм)"
            ),
            metric_value=round(total, 1),
            threshold=round(DROUGHT_EXPECTED_PRECIP_MM * DROUGHT_CRITICAL_RATIO, 1),
        )
    if ratio < DROUGHT_WARNING_RATIO:
        return AnomalySignal(
            type="drought_stress",
            severity="warning",
            message_uk=(
                f"Дефіцит опадів: {total:.1f} мм за {DROUGHT_WINDOW_DAYS} днів "
                f"(норма ~{DROUGHT_EXPECTED_PRECIP_MM:.0f} мм)"
            ),
            metric_value=round(total, 1),
            threshold=round(DROUGHT_EXPECTED_PRECIP_MM * DROUGHT_WARNING_RATIO, 1),
        )
    return None


def detect_heat_stress(
    weather: list[WeatherObservation], today: date
) -> AnomalySignal | None:
    """Count of T_max > 30°C days in the last HEAT_WINDOW_DAYS days."""
    if not weather:
        return None
    recent = [
        w for w in weather
        if w.is_forecast is False
        and (today - w.observed_on).days <= HEAT_WINDOW_DAYS
        and (today - w.observed_on).days >= 0
    ]
    if not recent:
        return None
    hot_days = sum(
        1 for w in recent
        if w.temp_max_c is not None and float(w.temp_max_c) > 30
    )
    if hot_days >= HEAT_CRITICAL_DAYS:
        return AnomalySignal(
            type="heat_stress",
            severity="critical",
            message_uk=(
                f"Критична спека: {hot_days} днів з T>30°C за "
                f"останні {HEAT_WINDOW_DAYS} днів"
            ),
            metric_value=hot_days,
            threshold=HEAT_CRITICAL_DAYS,
        )
    if hot_days >= HEAT_WARNING_DAYS:
        return AnomalySignal(
            type="heat_stress",
            severity="warning",
            message_uk=(
                f"Тривала спека: {hot_days} днів з T>30°C за "
                f"останні {HEAT_WINDOW_DAYS} днів"
            ),
            metric_value=hot_days,
            threshold=HEAT_WARNING_DAYS,
        )
    return None
