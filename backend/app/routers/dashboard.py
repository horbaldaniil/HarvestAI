"""Aggregated dashboard endpoint.

Returns everything the Dashboard needs in one round-trip: KPIs, per-field
prediction summaries, and YoY NDVI comparison.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter
from sqlalchemy import and_, func, select

from app.db.models import (
    Alert,
    Field,
    Prediction,
    SatelliteObservation,
)
from app.deps import CurrentUser, DbSession
from app.schemas.prediction import (
    DashboardFieldRow,
    DashboardKpis,
    DashboardResponse,
    YearOverYear,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse)
async def get_dashboard(
    current_user: CurrentUser, db: DbSession,
) -> DashboardResponse:
    # User's fields.
    fields = list(
        (await db.execute(
            select(Field).where(Field.user_id == current_user.id)
        )).scalars().all()
    )

    field_rows: list[DashboardFieldRow] = []
    total_area = 0.0
    total_predicted_yield = 0.0
    have_any_pred = False
    current_year = date.today().year
    cy_ndvi: list[float] = []
    py_ndvi: list[float] = []

    for f in fields:
        # Latest prediction (if any)
        pred = (await db.execute(
            select(Prediction)
            .where(Prediction.field_id == f.id)
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        predicted_tha = float(pred.value_tha) if pred is not None else None

        # Latest NDVI mean (within current year)
        latest_ndvi_row = (await db.execute(
            select(SatelliteObservation.ndvi_mean)
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.ndvi_mean.isnot(None))
            .order_by(SatelliteObservation.observed_on.desc())
            .limit(1)
        )).scalar_one_or_none()
        current_ndvi = float(latest_ndvi_row) if latest_ndvi_row is not None else None

        # YoY accumulators (avg NDVI per year)
        cy_avg = (await db.execute(
            select(func.avg(SatelliteObservation.ndvi_mean))
            .where(SatelliteObservation.field_id == f.id)
            .where(func.extract("year", SatelliteObservation.observed_on) == current_year)
        )).scalar()
        py_avg = (await db.execute(
            select(func.avg(SatelliteObservation.ndvi_mean))
            .where(SatelliteObservation.field_id == f.id)
            .where(func.extract("year", SatelliteObservation.observed_on) == current_year - 1)
        )).scalar()
        if cy_avg is not None:
            cy_ndvi.append(float(cy_avg))
        if py_avg is not None:
            py_ndvi.append(float(py_avg))

        # Has unack'd alerts?
        has_alerts_row = (await db.execute(
            select(func.count())
            .select_from(Alert)
            .where(and_(
                Alert.field_id == f.id,
                Alert.acknowledged.is_(False),
            ))
        )).scalar()
        has_alerts = bool(has_alerts_row)

        total_area += float(f.area_ha)
        if predicted_tha is not None:
            total_predicted_yield += predicted_tha * float(f.area_ha)
            have_any_pred = True

        crop_val = f.crop_type if isinstance(f.crop_type, str) else f.crop_type.value
        field_rows.append(DashboardFieldRow(
            field_id=f.id,
            name=f.name,
            crop_type=crop_val,
            area_ha=float(f.area_ha),
            current_ndvi=round(current_ndvi, 3) if current_ndvi is not None else None,
            predicted_tha=round(predicted_tha, 2) if predicted_tha is not None else None,
            has_alerts=has_alerts,
        ))

    unread_alerts = await db.scalar(
        select(func.count()).select_from(Alert)
        .where(Alert.user_id == current_user.id)
        .where(Alert.acknowledged.is_(False))
    )

    avg_ndvi_current: float | None = None
    if cy_ndvi:
        avg_ndvi_current = round(sum(cy_ndvi) / len(cy_ndvi), 3)

    cy_mean = sum(cy_ndvi) / len(cy_ndvi) if cy_ndvi else None
    py_mean = sum(py_ndvi) / len(py_ndvi) if py_ndvi else None
    diff_pct = None
    if cy_mean is not None and py_mean is not None and py_mean != 0:
        diff_pct = round((cy_mean - py_mean) / py_mean * 100, 1)

    return DashboardResponse(
        kpis=DashboardKpis(
            total_fields=len(fields),
            total_area_ha=round(total_area, 2),
            predicted_total_yield_t=round(total_predicted_yield, 1) if have_any_pred else None,
            avg_ndvi_current=avg_ndvi_current,
            active_alerts_count=int(unread_alerts or 0),
        ),
        fields=field_rows,
        yoy=YearOverYear(
            current_year=current_year,
            current_year_avg_ndvi=round(cy_mean, 3) if cy_mean is not None else None,
            prev_year_avg_ndvi=round(py_mean, 3) if py_mean is not None else None,
            diff_pct=diff_pct,
        ),
    )
