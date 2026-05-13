"""Aggregated dashboard endpoint.

Returns everything the Dashboard page needs in one round-trip:
KPIs, per-field summaries (with risk_score + geometry), legacy YoY +
top-3 NDVI movers, best/worst highlight cards, crop breakdown, and
per-field 7-day weather summaries.

Query parameters:
- `crops`: comma-separated crop types to filter to. Defaults to all.

There used to be a `time_range_days` filter; removed because most of the
numbers it touched (last NDVI, last NDWI, drought/heat counters) were
either point-in-time or anchored to a fixed 14-day window matching the
anomaly detector. The filter had no visible effect for typical users.
"""
from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.db.models import (
    Alert,
    Field,
    Prediction,
    SatelliteObservation,
    WeatherObservation,
)
from app.deps import CurrentUser, DbSession
from app.schemas.prediction import (
    CropBreakdownItem,
    DashboardFieldRow,
    DashboardKpis,
    DashboardResponse,
    FieldWeather,
    FieldYoYDelta,
    WeatherDayRead,
    YearOverYear,
)
from app.services.dashboard_analytics import (
    compute_risk_score,
    find_oblast_for_centroid,
    oblast_avg_ndvi,
    oblast_avg_yield,
    oblast_baseline_year,
    oblast_name_uk,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Fixed weather window used everywhere — matches the anomaly detector
# (drought_stress / heat_stress operate on the same 14-day slice).
WEATHER_WINDOW_DAYS = 14


def _crop_value(crop_type) -> str:
    return crop_type.value if hasattr(crop_type, "value") else str(crop_type)


def _field_polygon_geojson(field: Field) -> dict | None:
    """Convert PostGIS POLYGON → GeoJSON dict for the mini-map."""
    try:
        from shapely import wkb

        poly = wkb.loads(bytes(field.geom.data))
        return poly.__geo_interface__
    except Exception:  # noqa: BLE001
        return None


def _field_centroid(field: Field) -> tuple[float, float] | None:
    try:
        from shapely import wkb

        pt = wkb.loads(bytes(field.centroid.data))
        return pt.y, pt.x  # lat, lon
    except Exception:  # noqa: BLE001
        return None


@router.get("", response_model=DashboardResponse)
async def get_dashboard(
    current_user: CurrentUser,
    db: DbSession,
    crops: str = Query(default="", description="Comma-separated crop names"),
) -> DashboardResponse:
    """Aggregated dashboard payload."""
    crop_filter = [c.strip() for c in crops.split(",") if c.strip()] if crops else []

    fields_all = list((await db.execute(
        select(Field).where(Field.user_id == current_user.id)
    )).scalars().all())
    fields = [
        f for f in fields_all
        if not crop_filter or _crop_value(f.crop_type) in crop_filter
    ]

    today = date.today()
    weather_window_start = today - timedelta(days=WEATHER_WINDOW_DAYS)
    current_year = today.year

    # YTD comparison window for the "Найбільші зміни NDVI" movers card —
    # Jan 1 → today this year vs Jan 1 → same date prior year. Honest
    # mid-season comparison; the old `EXTRACT(year FROM observed_on) = Y`
    # filter pitted partial-year current against full-year prior, which
    # made mid-season YoY look misleadingly negative.
    ytd_start_cy = date(current_year, 1, 1)
    ytd_end_cy = today
    ytd_start_py = date(current_year - 1, 1, 1)
    try:
        ytd_end_py = date(current_year - 1, today.month, today.day)
    except ValueError:
        # Feb 29 → clamp to Feb 28 when prior year isn't a leap year.
        ytd_end_py = date(current_year - 1, today.month, today.day - 1)

    # Crop breakdown computed on the full portfolio (not crop-filtered) so
    # the donut/calendar always reflects the full mix.
    breakdown_acc: dict[str, dict[str, float]] = {}
    for f in fields_all:
        crop_val = _crop_value(f.crop_type)
        entry = breakdown_acc.setdefault(crop_val, {"count": 0, "area": 0.0})
        entry["count"] += 1
        entry["area"] += float(f.area_ha)

    field_rows: list[DashboardFieldRow] = []
    movers_pool: list[dict] = []
    total_area = 0.0
    total_predicted_yield = 0.0
    have_any_pred = False
    cy_ndvi: list[float] = []
    py_ndvi: list[float] = []
    ndwi_pool: list[float] = []

    for f in fields:
        crop_val = _crop_value(f.crop_type)

        pred = (await db.execute(
            select(Prediction)
            .where(Prediction.field_id == f.id)
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        predicted_tha = float(pred.value_tha) if pred is not None else None

        # Latest NDVI + NDWI without any window — always the most recent we have.
        latest_ndvi = (await db.execute(
            select(SatelliteObservation.ndvi_mean)
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.ndvi_mean.is_not(None))
            .order_by(SatelliteObservation.observed_on.desc())
            .limit(1)
        )).scalar_one_or_none()
        current_ndvi = float(latest_ndvi) if latest_ndvi is not None else None

        latest_ndwi = (await db.execute(
            select(SatelliteObservation.ndwi_mean)
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.ndwi_mean.is_not(None))
            .order_by(SatelliteObservation.observed_on.desc())
            .limit(1)
        )).scalar_one_or_none()
        current_ndwi = float(latest_ndwi) if latest_ndwi is not None else None
        if current_ndwi is not None:
            ndwi_pool.append(current_ndwi)

        # YTD averages: Jan 1 → today this year vs same window prior
        # year. See `ytd_start_cy` block at the top of this function for
        # the leap-year edge case (Feb 29 clamps to Feb 28).
        cy_avg = (await db.execute(
            select(func.avg(SatelliteObservation.ndvi_mean))
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.observed_on >= ytd_start_cy)
            .where(SatelliteObservation.observed_on <= ytd_end_cy)
        )).scalar()
        py_avg = (await db.execute(
            select(func.avg(SatelliteObservation.ndvi_mean))
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.observed_on >= ytd_start_py)
            .where(SatelliteObservation.observed_on <= ytd_end_py)
        )).scalar()
        if cy_avg is not None:
            cy_ndvi.append(float(cy_avg))
        if py_avg is not None:
            py_ndvi.append(float(py_avg))
        # Movers eligible: have BOTH cy and py to compute a signed delta.
        if cy_avg is not None and py_avg is not None and float(py_avg) != 0:
            cy_f = float(cy_avg)
            py_f = float(py_avg)
            movers_pool.append({
                "field_id": f.id,
                "name": f.name,
                "crop_type": crop_val,
                "current_year_ndvi": round(cy_f, 3),
                "prev_year_ndvi": round(py_f, 3),
                "diff_pct": round((cy_f - py_f) / py_f * 100, 1),
            })

        # Alerts grouped by severity for risk weighting.
        alert_rows = (await db.execute(
            select(Alert.severity, func.count())
            .where(Alert.field_id == f.id)
            .where(Alert.acknowledged.is_(False))
            .group_by(Alert.severity)
        )).all()
        alerts_by_severity = {sev: int(n) for sev, n in alert_rows}
        has_alerts = sum(alerts_by_severity.values()) > 0

        # Weather counters anchored to a fixed 14-day window.
        weather_recent = list((await db.execute(
            select(WeatherObservation)
            .where(WeatherObservation.field_id == f.id)
            .where(WeatherObservation.observed_on >= weather_window_start)
            .where(WeatherObservation.is_forecast.is_(False))
        )).scalars().all())

        # Recent soil moisture — averaged over the last 3 days of actual
        # observations. Replaces the old "11 of 14 days had <1 mm precip
        # → drought" heuristic, which mis-fired after a fresh shower
        # because it ignored the moisture actually present in the soil.
        # Open-Meteo's `soil_moisture_0_10cm` is volumetric water content
        # (m³/m³); see `SOIL_MOISTURE_LOW_THRESHOLD` for the agronomic
        # band that triggers the risk flag.
        moisture_window_start = today - timedelta(days=3)
        recent_moisture_vals = [
            float(w.soil_moisture_0_10cm) for w in weather_recent
            if w.soil_moisture_0_10cm is not None
            and w.observed_on >= moisture_window_start
        ]
        recent_soil_moisture = (
            sum(recent_moisture_vals) / len(recent_moisture_vals)
            if recent_moisture_vals else None
        )

        heat_days = sum(
            1 for w in weather_recent
            if w.temp_max_c is not None and float(w.temp_max_c) > 30.0
        )

        # Oblast comparison: resolve the field's oblast from its centroid
        # via point-in-polygon over ukraine_oblasts.geojson, then look up
        # the Week 6 baseline NDVI for that oblast. Falls back to None when:
        #   - oblast geojson is missing,
        #   - centroid lands outside Ukraine,
        #   - training_set_v2.parquet doesn't have data for that oblast yet.
        centroid_latlon = _field_centroid(f)
        oblast_english = (
            find_oblast_for_centroid(*centroid_latlon)
            if centroid_latlon is not None else None
        )
        oblast_baseline = oblast_avg_ndvi(oblast_english, current_year)
        oblast_yr = oblast_baseline_year(oblast_english) if oblast_baseline is not None else None

        # User-facing "Поле vs середнє по області" baseline — predicted
        # yield vs published Держстат mean for the same (oblast, crop).
        # NDVI baseline above still feeds compute_risk_score; these are
        # separate concerns.
        oblast_yield_avg, oblast_yield_year = oblast_avg_yield(
            oblast_english, crop_val, current_year,
        )

        risk_score, risk_factors = compute_risk_score(
            current_ndvi=current_ndvi,
            oblast_avg_ndvi=oblast_baseline,
            alerts_by_severity=alerts_by_severity,
            recent_soil_moisture=recent_soil_moisture,
            heat_days_recent=heat_days,
        )

        total_area += float(f.area_ha)
        if predicted_tha is not None:
            total_predicted_yield += predicted_tha * float(f.area_ha)
            have_any_pred = True

        row_dict = {
            "field_id": f.id,
            "name": f.name,
            "crop_type": crop_val,
            "area_ha": float(f.area_ha),
            "current_ndvi": round(current_ndvi, 3) if current_ndvi is not None else None,
            "current_ndwi": round(current_ndwi, 3) if current_ndwi is not None else None,
            "predicted_tha": round(predicted_tha, 2) if predicted_tha is not None else None,
            "has_alerts": has_alerts,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "oblast_name": oblast_name_uk(oblast_english),
            "oblast_avg_ndvi": round(oblast_baseline, 3) if oblast_baseline is not None else None,
            "oblast_baseline_year": oblast_yr,
            "oblast_avg_yield_tha": oblast_yield_avg,
            "oblast_avg_yield_year": oblast_yield_year,
            "geometry": _field_polygon_geojson(f),
        }
        field_rows.append(DashboardFieldRow(**row_dict))

    unread_alerts = await db.scalar(
        select(func.count()).select_from(Alert)
        .where(Alert.user_id == current_user.id)
        .where(Alert.acknowledged.is_(False))
    )

    avg_ndvi_current = round(sum(cy_ndvi) / len(cy_ndvi), 3) if cy_ndvi else None
    avg_ndwi_current = round(sum(ndwi_pool) / len(ndwi_pool), 3) if ndwi_pool else None

    cy_mean = sum(cy_ndvi) / len(cy_ndvi) if cy_ndvi else None
    py_mean = sum(py_ndvi) / len(py_ndvi) if py_ndvi else None
    diff_pct = None
    if cy_mean is not None and py_mean is not None and py_mean != 0:
        diff_pct = round((cy_mean - py_mean) / py_mean * 100, 1)

    # Top-3 movers by |diff_pct|.
    top_movers = [
        FieldYoYDelta(**m)
        for m in sorted(movers_pool, key=lambda m: abs(m["diff_pct"]), reverse=True)[:3]
    ]

    weather_by_field = await _build_weather_by_field(db, fields)

    return DashboardResponse(
        kpis=DashboardKpis(
            total_fields=len(fields),
            total_area_ha=round(total_area, 2),
            predicted_total_yield_t=round(total_predicted_yield, 1) if have_any_pred else None,
            avg_ndvi_current=avg_ndvi_current,
            avg_ndwi_current=avg_ndwi_current,
            active_alerts_count=int(unread_alerts or 0),
        ),
        fields=field_rows,
        yoy=YearOverYear(
            current_year=current_year,
            current_year_avg_ndvi=round(cy_mean, 3) if cy_mean is not None else None,
            prev_year_avg_ndvi=round(py_mean, 3) if py_mean is not None else None,
            diff_pct=diff_pct,
        ),
        top_movers=top_movers,
        crops_breakdown=[
            CropBreakdownItem(
                crop_type=crop, field_count=int(v["count"]), area_ha=round(v["area"], 2),
            )
            for crop, v in sorted(breakdown_acc.items())
        ],
        weather_by_field=weather_by_field,
        applied_crops=crop_filter,
    )


async def _build_weather_by_field(db, fields: list[Field]) -> list[FieldWeather]:
    """One FieldWeather payload per field with a non-empty 7-day forecast.

    Fields without cached forecast simply don't appear (UI hides them) —
    that keeps the dropdown short and avoids dead picker entries.
    """
    if not fields:
        return []

    today = date.today()
    cutoff = today + timedelta(days=7)

    # One DB call for everything, then bucket in Python.
    rows = list((await db.execute(
        select(WeatherObservation)
        .where(WeatherObservation.field_id.in_([f.id for f in fields]))
        .where(WeatherObservation.is_forecast.is_(True))
        .where(WeatherObservation.observed_on >= today)
        .where(WeatherObservation.observed_on <= cutoff)
        .order_by(WeatherObservation.observed_on.asc())
    )).scalars().all())

    by_field: dict[int, list[WeatherObservation]] = {}
    for r in rows:
        by_field.setdefault(r.field_id, []).append(r)

    out: list[FieldWeather] = []
    for f in fields:
        bucket = by_field.get(f.id, [])
        if not bucket:
            continue

        days = [
            WeatherDayRead(
                observed_on=w.observed_on,
                temp_min_c=_as_float(w.temp_min_c),
                temp_max_c=_as_float(w.temp_max_c),
                temp_mean_c=_as_float(w.temp_mean_c),
                precip_mm=_as_float(w.precip_mm),
                humidity_pct=_as_float(w.humidity_pct),
                radiation_mj=_as_float(w.radiation_mj),
                is_forecast=True,
            )
            for w in bucket
        ]
        temp_max_7d = max(
            (d.temp_max_c for d in days if d.temp_max_c is not None), default=None,
        )
        temp_min_7d = min(
            (d.temp_min_c for d in days if d.temp_min_c is not None), default=None,
        )
        avg_vals = [d.temp_mean_c for d in days if d.temp_mean_c is not None]
        temp_avg_7d = (
            round(sum(avg_vals) / len(avg_vals), 1) if avg_vals else None
        )
        precip_sum_7d = sum(d.precip_mm or 0 for d in days)
        heat_days_count = sum(
            1 for d in days if d.temp_max_c is not None and d.temp_max_c > 30.0
        )
        centroid = _field_centroid(f)
        out.append(FieldWeather(
            field_id=f.id,
            field_name=f.name,
            crop_type=_crop_value(f.crop_type),
            centroid_lat=round(centroid[0], 5) if centroid else None,
            centroid_lon=round(centroid[1], 5) if centroid else None,
            days=days,
            temp_max_7d=round(temp_max_7d, 1) if temp_max_7d is not None else None,
            temp_min_7d=round(temp_min_7d, 1) if temp_min_7d is not None else None,
            temp_avg_7d=temp_avg_7d,
            precip_sum_7d=round(precip_sum_7d, 1) if days else None,
            heat_stress_days_7d=heat_days_count,
        ))
    return out


def _as_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
