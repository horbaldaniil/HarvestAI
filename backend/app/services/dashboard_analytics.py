"""Pure-function helpers feeding the dashboard endpoint.

Keeps the router thin: each function takes already-loaded data + plain
arguments and returns a structured payload. Lets us unit-test the logic
without spinning up a DB.

Three independent topics live here:

1. Risk score — composite 0-100 score per field used by the dashboard
   table's "Risk" column and by best/worst card selection.
2. Phenology — static lookup table mapping (crop, month) to a growth phase
   label. Driven by averaged Ukrainian agronomy norms.
3. Oblast NDVI lookup — read training_set_v2.parquet (Week 6 output) and
   return the cropland-mask-aggregated NDVI mean for a given oblast/year.
   Used by the "your field vs oblast average" comparison column.
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ─── Phenology ─────────────────────────────────────────────────


# (crop, start_month, end_month_inclusive, phase_key)
# Phase keys mirror the i18n keys the frontend uses.
PHENOLOGY: dict[str, list[tuple[int, int, str]]] = {
    "wheat": [
        (9, 10, "sowing"),
        (11, 3, "dormancy"),
        (4, 5, "growth"),
        (6, 6, "flowering"),
        (7, 7, "ripening"),
        (8, 8, "harvest"),
    ],
    "corn": [
        (4, 4, "sowing"),
        (5, 6, "growth"),
        (7, 7, "flowering"),
        (8, 8, "ripening"),
        (9, 10, "harvest"),
    ],
    "sunflower": [
        (4, 4, "sowing"),
        (5, 6, "growth"),
        (7, 7, "flowering"),
        (8, 8, "ripening"),
        (9, 9, "harvest"),
    ],
}


def phenology_phase(crop: str, today: date | None = None) -> str | None:
    """Return the current growth phase for `crop` at `today`, or None.

    Handles wraparound months (e.g. wheat dormancy spans Nov-Mar).
    """
    if today is None:
        today = date.today()
    spec = PHENOLOGY.get(crop)
    if not spec:
        return None
    month = today.month
    for start, end, label in spec:
        if start <= end:
            if start <= month <= end:
                return label
        else:
            # Wraparound (Nov-Mar): hits if month >= start OR month <= end.
            if month >= start or month <= end:
                return label
    return None


def phenology_calendar(crop: str) -> list[dict[str, Any]]:
    """Return the full phase list with normalised month ranges for the UI."""
    out = []
    for start, end, label in PHENOLOGY.get(crop, []):
        out.append({"phase": label, "start_month": start, "end_month": end})
    return out


# ─── Risk score ────────────────────────────────────────────────


def compute_risk_score(
    *,
    current_ndvi: float | None,
    oblast_avg_ndvi: float | None,
    alerts_by_severity: dict[str, int],
    drought_days_recent: int = 0,
    heat_days_recent: int = 0,
) -> tuple[int, list[str]]:
    """Composite risk score 0-100 plus short factor tags for tooltip.

    Factors:
      - NDVI deficit vs oblast average: up to 30 points
      - Active alerts: 10 per warning, 30 per critical (capped at 50)
      - Recent extreme weather: 15 for drought (>10 dry days), 15 for heat
        (>5 hot days)

    Score is clamped to [0, 100]. Higher = worse.
    """
    score = 0
    factors: list[str] = []

    # NDVI deficit relative to oblast. Only applied if oblast comparison
    # exists; otherwise no signal in either direction.
    if current_ndvi is not None and oblast_avg_ndvi is not None:
        deficit = oblast_avg_ndvi - current_ndvi
        if deficit > 0.05:
            penalty = min(30, int(deficit * 100))
            score += penalty
            factors.append(f"NDVI нижче області на {deficit:.2f}")
    elif current_ndvi is not None and current_ndvi < 0.3:
        # Standalone: very low NDVI even without comparison is a flag.
        score += 25
        factors.append(f"Низький NDVI {current_ndvi:.2f}")

    n_warn = alerts_by_severity.get("warning", 0)
    n_crit = alerts_by_severity.get("critical", 0)
    alert_score = min(50, n_warn * 10 + n_crit * 30)
    if alert_score:
        score += alert_score
        tag_parts = []
        if n_crit:
            tag_parts.append(f"{n_crit} критичних")
        if n_warn:
            tag_parts.append(f"{n_warn} попереджень")
        factors.append("Alerts: " + ", ".join(tag_parts))

    if drought_days_recent > 10:
        score += 15
        factors.append(f"Посуха ({drought_days_recent} днів)")
    if heat_days_recent > 5:
        score += 15
        factors.append(f"Спека ({heat_days_recent} днів T_max>30°C)")

    return min(100, max(0, score)), factors


def pick_best_worst(
    rows: list[dict[str, Any]],
) -> tuple[dict | None, dict | None]:
    """Pick the highest-performer (NDVI) and most-at-risk (risk_score) field.

    Both can be the same field in extreme cases — we still return both so
    the UI shows the duality. Returns None entries when no candidates.
    """
    if not rows:
        return None, None
    with_ndvi = [r for r in rows if r.get("current_ndvi") is not None]
    best = max(with_ndvi, key=lambda r: r["current_ndvi"], default=None)
    worst = max(rows, key=lambda r: r.get("risk_score", 0), default=None)

    def _shape(r: dict, reason: str) -> dict:
        return {
            "field_id": r["field_id"],
            "name": r["name"],
            "crop_type": r["crop_type"],
            "current_ndvi": r.get("current_ndvi"),
            "predicted_tha": r.get("predicted_tha"),
            "risk_score": r.get("risk_score", 0),
            "reason": reason,
        }

    best_payload = (
        _shape(best, f"Найвищий NDVI ({best['current_ndvi']:.2f})")
        if best is not None else None
    )
    worst_payload = None
    if worst is not None and worst.get("risk_score", 0) > 0:
        factors = worst.get("risk_factors") or []
        reason = factors[0] if factors else f"Risk score {worst.get('risk_score')}"
        worst_payload = _shape(worst, reason)
    return best_payload, worst_payload


# ─── Oblast NDVI baseline (Week 6 connection) ─────────────────


@lru_cache(maxsize=1)
def _load_oblast_baseline() -> dict[tuple[str, int], float]:
    """Read training_set_v2.parquet into (oblast, year) → mean NDVI.

    File may be missing (Week 6 data collection is offline + month-long).
    Cached on first hit; tests that mutate the parquet should reset the cache.
    """
    path = Path("data/processed/training_set_v2.parquet")
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=["oblast", "year", "ndvi_peak", "ndvi_mean_july"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load oblast baseline: %s", exc)
        return {}

    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        # Use ndvi_peak as the comparable summary — it's the most stable
        # single number across (oblast, year) and the easiest to communicate
        # in the UI ("oblast peak NDVI was X, yours is Y").
        if row.get("ndvi_peak") is None:
            continue
        key = (str(row["oblast"]), int(row["year"]))
        # If multiple crops share the row, they collapse to the same average —
        # we already aggregate per (oblast, year) upstream. Last write wins.
        out[key] = float(row["ndvi_peak"])
    return out


def oblast_avg_ndvi(oblast_name: str | None, year: int | None) -> float | None:
    """Lookup helper for the dashboard field-row enrichment."""
    if not oblast_name or not year:
        return None
    baseline = _load_oblast_baseline()
    return baseline.get((oblast_name, int(year)))


def reset_oblast_baseline_cache() -> None:
    """Test helper — call after the parquet is rewritten."""
    _load_oblast_baseline.cache_clear()


__all__ = [
    "PHENOLOGY",
    "compute_risk_score",
    "oblast_avg_ndvi",
    "phenology_calendar",
    "phenology_phase",
    "pick_best_worst",
    "reset_oblast_baseline_cache",
]
