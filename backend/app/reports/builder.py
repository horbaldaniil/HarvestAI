"""Build per-field and portfolio PDF reports.

Both builders follow the same shape:
  1. Pull data from DB (one query per "section" — fields, observations, etc.)
  2. Render server-side PNG charts via matplotlib (`charts.py`)
  3. Optionally call OpenAI for a one-paragraph summary (gracefully degrades
     to None if OPENAI_API_KEY isn't configured)
  4. Render the Jinja2 template
  5. Pipe through `engine.render_html_to_pdf` → bytes

All-around defensive: missing data → empty section, missing OpenAI → no
summary box, missing heatmap → skip the image. The PDF must build even
when the user has only just created a field with no observations yet.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import (
    Alert,
    Field,
    Prediction,
    SatelliteObservation,
    User,
    WeatherObservation,
)
from app.db.models.enums import CropType
from app.integrations.openai.client import OpenAIClient, OpenAIError
from app.reports.charts import (
    INDEX_LABELS_UK,
    render_index_chart_png,
    render_methodology_leaderboard_png,
    render_methodology_learning_curves_png,
    render_methodology_residual_map_png,
    render_methodology_shap_png,
    render_multi_field_chart_png,
    render_shap_chart_png,
)
from app.reports.engine import render_html_to_pdf
from app.reports.qr import render_qr_png

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
INDEX_NAMES = ("ndvi", "evi", "ndwi", "savi")
SEVERITY_LABELS_UK = {"info": "Інфо", "warning": "Увага", "critical": "Критично"}


def _crop_label_uk(value: str) -> str:
    """Resolve a Cyrillic crop label for PDF rendering.

    Delegates to `CropType.display_uk` so this stays in sync with the
    enum's display names for all 13 crops (wheat, corn, sunflower,
    soybean, rapeseed, barley, rye, oats, buckwheat, peas, sugar_beet,
    potato, corn_silage). Falls back to the raw slug if the value is
    unknown — defensive for stale rows or legacy data, so PDFs keep
    rendering even after a future schema rename.
    """
    try:
        return CropType(value).display_uk
    except (ValueError, KeyError):
        return value

# Field-report section keys. Builder lets the user pick a subset of these;
# unspecified = all enabled (legacy behaviour for the inline export button).
ALL_FIELD_SECTIONS = {
    "summary",       # AI-summary paragraph + heatmap thumbnail
    "indices",       # 4 vegetation-index time-series charts
    "prediction",    # yield prediction + SHAP top-5
    "alerts",        # active anomaly alerts
    "forecast",      # 14-day weather forecast table
}


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _frontend_url(path: str = "") -> str:
    return settings.frontend_origin.rstrip("/") + path


# ─── Per-field report ────────────────────────────────────────


async def build_field_report(
    field_id: int,
    *,
    db: AsyncSession,
    user: User,
    openai: OpenAIClient | None = None,
    sections: set[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> bytes:
    """Render a per-field PDF report.

    `sections` — a set of strings from ALL_FIELD_SECTIONS picking which
    blocks to include (None = all). `date_from`/`date_to` constrain the
    observation window for the index charts (default: last 2 years).
    """
    enabled = sections if sections is not None else set(ALL_FIELD_SECTIONS)
    field = await db.get(Field, field_id)
    if field is None or field.user_id != user.id:
        raise ValueError(f"field {field_id} not found for user {user.id}")

    # Pull centroid lat/lon for the metadata table.
    from geoalchemy2 import functions as geofn

    centroid_lat: float | None = None
    centroid_lon: float | None = None
    centroid_wkt = await db.scalar(select(geofn.ST_AsText(field.centroid)))
    if centroid_wkt:
        try:
            inner = centroid_wkt.replace("POINT(", "").rstrip(")")
            lon_s, lat_s = inner.split()
            centroid_lat = float(lat_s)
            centroid_lon = float(lon_s)
        except ValueError:
            pass

    # Observations: window from caller, or default 2 years back to today.
    today = date.today()
    obs_start = date_from or (today - timedelta(days=730))
    obs_end = date_to or today
    obs_rows = list(
        (await db.execute(
            select(SatelliteObservation)
            .where(SatelliteObservation.field_id == field_id)
            .where(SatelliteObservation.observed_on >= obs_start)
            .where(SatelliteObservation.observed_on <= obs_end)
            .order_by(SatelliteObservation.observed_on.asc())
        )).scalars().all()
    )

    # One chart per vegetation index, rendered as a base64 PNG data URI.
    index_charts = []
    for idx in INDEX_NAMES:
        col = f"{idx}_mean"
        points = [(o.observed_on.isoformat(), _to_float(getattr(o, col))) for o in obs_rows]
        index_charts.append(
            {
                "label": INDEX_LABELS_UK[idx],
                "uri": render_index_chart_png(points, index=idx),
            }
        )

    # Latest prediction (if any).
    pred = (await db.execute(
        select(Prediction)
        .where(Prediction.field_id == field_id)
        .order_by(Prediction.predicted_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    prediction_payload: dict | None = None
    shap_uri = None
    if pred is not None:
        prediction_payload = {
            "value_tha": float(pred.value_tha),
            "confidence_tha": float(pred.confidence) if pred.confidence is not None else None,
            "model_name": pred.model_name,
            "model_version": pred.model_version,
            "predicted_at": pred.predicted_at.strftime("%d.%m.%Y %H:%M"),
        }
        shap_uri = render_shap_chart_png(pred.shap_top_json or [])

    # Active alerts (5 most recent + unack'd) for this field.
    alerts_rows = list((await db.execute(
        select(Alert)
        .where(Alert.field_id == field_id)
        .where(Alert.acknowledged.is_(False))
        .order_by(Alert.created_at.desc())
        .limit(10)
    )).scalars().all())
    alerts_payload = [
        {
            "severity": a.severity,
            "severity_label": SEVERITY_LABELS_UK.get(a.severity, a.severity),
            "type": a.type,
            "message_uk": a.message_uk,
            "created_at": a.created_at.strftime("%d.%m.%Y %H:%M"),
        }
        for a in alerts_rows
    ]

    # 14-day forecast from the cache (populated by fetch_weather job).
    today = date.today()
    forecast_rows = list((await db.execute(
        select(WeatherObservation)
        .where(WeatherObservation.field_id == field_id)
        .where(WeatherObservation.is_forecast.is_(True))
        .where(WeatherObservation.observed_on >= today)
        .order_by(WeatherObservation.observed_on.asc())
        .limit(14)
    )).scalars().all())
    forecast_payload = [
        {
            "date": w.observed_on.strftime("%d.%m"),
            "temp_min_c": _round(w.temp_min_c, 1),
            "temp_max_c": _round(w.temp_max_c, 1),
            "precip_mm": _round(w.precip_mm, 1),
            "humidity_pct": _round(w.humidity_pct, 0),
        }
        for w in forecast_rows
    ]

    # Optional heatmap thumbnail: latest cached PNG on disk, if any.
    heatmap_uri = _load_latest_heatmap(field_id)
    heatmap_date = _latest_heatmap_date(field_id)

    # LLM summary (best effort).
    summary_text = await _summarise_field(
        openai,
        field=field,
        observations=obs_rows,
        prediction=pred,
        alerts=alerts_rows,
        forecast=forecast_rows,
    )

    # QR back to the field's web page.
    qr_uri = render_qr_png(_frontend_url(f"/fields?selected={field_id}"))

    # Section filtering: hand the template empty payloads for disabled
    # sections so the existing `{% if ... %}` blocks no-op. This keeps the
    # template logic-free.
    if "summary" not in enabled:
        summary_text = None
        heatmap_uri = None
        heatmap_date = None
    if "indices" not in enabled:
        index_charts = []
    if "prediction" not in enabled:
        prediction_payload = None
        shap_uri = None
    if "alerts" not in enabled:
        alerts_payload = []
    if "forecast" not in enabled:
        forecast_payload = []

    # Render.
    crop_value = field.crop_type if isinstance(field.crop_type, str) else field.crop_type.value
    context = {
        "title": f"Звіт по полю — {field.name}",
        "heading": field.name,
        "subheading": f"{_crop_label_uk(crop_value)} · {field.season_year} · "
                      f"{float(field.area_ha):.2f} га",
        "generated_at": _now_uk(),
        "field": {
            "name": field.name,
            "crop_label": _crop_label_uk(crop_value),
            "season_year": field.season_year,
            "area_ha": float(field.area_ha),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        },
        "summary_text": summary_text,
        "heatmap_uri": heatmap_uri,
        "heatmap_date": heatmap_date,
        "index_charts": index_charts,
        "prediction": prediction_payload,
        "shap_uri": shap_uri,
        "alerts": alerts_payload,
        "forecast": forecast_payload,
        "qr_uri": qr_uri,
    }
    html = _env().get_template("field_report.html").render(context)
    return render_html_to_pdf(html, base_url=str(TEMPLATES_DIR))


# ─── Portfolio report ────────────────────────────────────────


async def build_portfolio_report(
    *,
    db: AsyncSession,
    user: User,
    openai: OpenAIClient | None = None,
) -> bytes:
    fields = list((await db.execute(
        select(Field).where(Field.user_id == user.id)
    )).scalars().all())

    current_year = date.today().year
    cy_ndvi: list[float] = []
    py_ndvi: list[float] = []
    fields_payload = []
    total_area = 0.0
    total_predicted_yield = 0.0
    have_pred = False

    for f in fields:
        pred = (await db.execute(
            select(Prediction)
            .where(Prediction.field_id == f.id)
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        predicted_tha = float(pred.value_tha) if pred is not None else None

        latest_ndvi = await db.scalar(
            select(SatelliteObservation.ndvi_mean)
            .where(SatelliteObservation.field_id == f.id)
            .where(SatelliteObservation.ndvi_mean.is_not(None))
            .order_by(SatelliteObservation.observed_on.desc())
            .limit(1)
        )
        cy_avg = await db.scalar(
            select(func.avg(SatelliteObservation.ndvi_mean))
            .where(SatelliteObservation.field_id == f.id)
            .where(func.extract("year", SatelliteObservation.observed_on) == current_year)
        )
        py_avg = await db.scalar(
            select(func.avg(SatelliteObservation.ndvi_mean))
            .where(SatelliteObservation.field_id == f.id)
            .where(func.extract("year", SatelliteObservation.observed_on) == current_year - 1)
        )
        if cy_avg is not None:
            cy_ndvi.append(float(cy_avg))
        if py_avg is not None:
            py_ndvi.append(float(py_avg))

        has_alerts = bool(await db.scalar(
            select(func.count()).select_from(Alert)
            .where(Alert.field_id == f.id)
            .where(Alert.acknowledged.is_(False))
        ))

        total_area += float(f.area_ha)
        if predicted_tha is not None:
            total_predicted_yield += predicted_tha * float(f.area_ha)
            have_pred = True

        crop_value = f.crop_type if isinstance(f.crop_type, str) else f.crop_type.value
        fields_payload.append({
            "field_id": f.id,
            "name": f.name,
            "crop_label": _crop_label_uk(crop_value),
            "area_ha": float(f.area_ha),
            "current_ndvi": float(latest_ndvi) if latest_ndvi is not None else None,
            "predicted_tha": predicted_tha,
            "has_alerts": has_alerts,
        })

    active_alerts_count = int(await db.scalar(
        select(func.count()).select_from(Alert)
        .where(Alert.user_id == user.id)
        .where(Alert.acknowledged.is_(False))
    ) or 0)

    kpis = {
        "total_fields": len(fields),
        "total_area_ha": total_area,
        "predicted_total_yield_t": total_predicted_yield if have_pred else None,
        "active_alerts_count": active_alerts_count,
    }

    cy_mean = sum(cy_ndvi) / len(cy_ndvi) if cy_ndvi else None
    py_mean = sum(py_ndvi) / len(py_ndvi) if py_ndvi else None
    diff_pct = None
    if cy_mean is not None and py_mean is not None and py_mean != 0:
        diff_pct = (cy_mean - py_mean) / py_mean * 100
    yoy = {
        "current_year": current_year,
        "current_year_avg_ndvi": cy_mean,
        "prev_year_avg_ndvi": py_mean,
        "diff_pct": diff_pct,
    }

    # All active alerts across user's fields (joined with field names).
    alerts_rows = (await db.execute(
        select(Alert, Field.name)
        .join(Field, Field.id == Alert.field_id, isouter=True)
        .where(Alert.user_id == user.id)
        .where(Alert.acknowledged.is_(False))
        .order_by(Alert.created_at.desc())
        .limit(20)
    )).all()
    alerts_payload = [
        {
            "severity": a.severity,
            "severity_label": SEVERITY_LABELS_UK.get(a.severity, a.severity),
            "type": a.type,
            "message_uk": a.message_uk,
            "field_name": fname,
            "created_at": a.created_at.strftime("%d.%m.%Y %H:%M"),
        }
        for a, fname in alerts_rows
    ]

    summary_text = await _summarise_portfolio(
        openai, fields=fields_payload, kpis=kpis, yoy=yoy,
    )

    qr_uri = render_qr_png(_frontend_url("/dashboard"))

    context = {
        "title": "Звіт по портфелю",
        "heading": "Огляд портфеля",
        "subheading": f"Користувач: {user.email} · {_now_uk()}",
        "generated_at": _now_uk(),
        "kpis": kpis,
        "fields": fields_payload,
        "yoy": yoy,
        "alerts": alerts_payload,
        "summary_text": summary_text,
        "qr_uri": qr_uri,
    }
    html = _env().get_template("portfolio_report.html").render(context)
    return render_html_to_pdf(html, base_url=str(TEMPLATES_DIR))


# ─── Compare report (multi-field) ────────────────────────────


async def build_compare_report(
    field_ids: list[int],
    *,
    db: AsyncSession,
    user: User,
    openai: OpenAIClient | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    index: str = "ndvi",
) -> bytes:
    """Render a side-by-side comparison report for multiple fields.

    Cross-field view: aggregate metrics table + one multi-line NDVI chart.
    Useful for spotting under-performers within a portfolio without
    flipping through individual reports.
    """
    if not field_ids:
        raise ValueError("compare report requires at least 1 field_id")
    if index not in INDEX_NAMES:
        raise ValueError(f"unknown index: {index}")

    today = date.today()
    obs_start = date_from or (today - timedelta(days=730))
    obs_end = date_to or today

    fields_payload: list[dict] = []
    series: list[dict] = []
    total_area = 0.0
    total_predicted_t = 0.0
    have_pred = False

    obs_col = f"{index}_mean"

    for fid in field_ids:
        field = await db.get(Field, fid)
        if field is None or field.user_id != user.id:
            # Silently skip — caller validated, but defence in depth.
            continue

        # Aggregate stats for this field within the window.
        obs_rows = list(
            (await db.execute(
                select(SatelliteObservation)
                .where(SatelliteObservation.field_id == fid)
                .where(SatelliteObservation.observed_on >= obs_start)
                .where(SatelliteObservation.observed_on <= obs_end)
                .order_by(SatelliteObservation.observed_on.asc())
            )).scalars().all()
        )
        vals = [float(getattr(o, obs_col)) for o in obs_rows if getattr(o, obs_col) is not None]
        ndvi_mean = sum(vals) / len(vals) if vals else None
        ndvi_peak = max(vals) if vals else None

        pred = (await db.execute(
            select(Prediction)
            .where(Prediction.field_id == fid)
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        predicted_tha = float(pred.value_tha) if pred is not None else None

        alerts_count = int(await db.scalar(
            select(func.count()).select_from(Alert)
            .where(Alert.field_id == fid)
            .where(Alert.acknowledged.is_(False))
        ) or 0)

        total_area += float(field.area_ha)
        if predicted_tha is not None:
            total_predicted_t += predicted_tha * float(field.area_ha)
            have_pred = True

        crop_val = field.crop_type if isinstance(field.crop_type, str) else field.crop_type.value
        fields_payload.append({
            "field_id": fid,
            "name": field.name,
            "crop_label": _crop_label_uk(crop_val),
            "area_ha": float(field.area_ha),
            "ndvi_mean": ndvi_mean,
            "ndvi_peak": ndvi_peak,
            "predicted_tha": predicted_tha,
            "alerts_count": alerts_count,
        })
        series.append({
            "name": field.name,
            "points": [
                (o.observed_on.isoformat(), _to_float(getattr(o, obs_col)))
                for o in obs_rows
            ],
        })

    ndvi_chart_uri = render_multi_field_chart_png(series, index=index)

    summary_text = await _summarise_compare(
        openai, fields=fields_payload, date_from=obs_start, date_to=obs_end,
    )

    qr_uri = render_qr_png(_frontend_url("/dashboard"))

    context = {
        "title": "Порівняльний звіт",
        "heading": "Порівняння полів",
        "subheading": f"{len(fields_payload)} полів · {user.email}",
        "generated_at": _now_uk(),
        "fields": fields_payload,
        "total_area_ha": total_area,
        "total_predicted_t": total_predicted_t if have_pred else None,
        "date_from": obs_start.isoformat(),
        "date_to": obs_end.isoformat(),
        "ndvi_chart_uri": ndvi_chart_uri,
        "summary_text": summary_text,
        "qr_uri": qr_uri,
    }
    html = _env().get_template("compare_report.html").render(context)
    return render_html_to_pdf(html, base_url=str(TEMPLATES_DIR))


# ─── Helpers ──────────────────────────────────────────────────


def _round(v, ndigits):
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _now_uk() -> str:
    return datetime.now(tz=timezone.utc).astimezone().strftime("%d.%m.%Y %H:%M")


def _load_latest_heatmap(field_id: int) -> str | None:
    """Find the most recent cached heatmap PNG for this field, return as data URI."""
    base = settings.rasters_dir / str(field_id)
    if not base.exists():
        return None
    pngs = sorted(base.glob("*.png"), reverse=True)
    if not pngs:
        return None
    import base64
    data = pngs[0].read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _latest_heatmap_date(field_id: int) -> str | None:
    base = settings.rasters_dir / str(field_id)
    if not base.exists():
        return None
    pngs = sorted(base.glob("*.png"), reverse=True)
    if not pngs:
        return None
    # Filename format: YYYY-MM-DD_index.png
    stem = pngs[0].stem
    date_part = stem.split("_")[0] if "_" in stem else stem
    try:
        return datetime.strptime(date_part, "%Y-%m-%d").strftime("%d.%m.%Y")
    except ValueError:
        return None


async def _summarise_field(
    openai: OpenAIClient | None,
    *,
    field,
    observations,
    prediction,
    alerts,
    forecast,
) -> str | None:
    """One paragraph (≤120 words) summary. Returns None on failure / no key."""
    if openai is None or not settings.openai_api_key:
        return None

    latest_ndvi = next(
        (o.ndvi_mean for o in reversed(observations) if o.ndvi_mean is not None),
        None,
    )
    bullets = [
        f"Поле: {field.name}, культура: {field.crop_type.value if hasattr(field.crop_type, 'value') else field.crop_type}, "
        f"сезон: {field.season_year}, площа: {float(field.area_ha):.2f} га",
        f"Останній NDVI: {float(latest_ndvi):.2f}" if latest_ndvi is not None else "Спостережень NDVI ще немає",
        f"Активних аномалій: {len(alerts)}",
    ]
    if prediction is not None:
        bullets.append(
            f"Прогноз: {float(prediction.value_tha):.2f} т/га"
            + (f" ± {float(prediction.confidence):.2f}" if prediction.confidence is not None else "")
        )
    if forecast:
        avg_t = [float(w.temp_mean_c) for w in forecast if w.temp_mean_c is not None]
        total_p = sum(float(w.precip_mm) for w in forecast if w.precip_mm is not None)
        if avg_t:
            bullets.append(
                f"14-денний прогноз: середня темп. {sum(avg_t)/len(avg_t):.1f}°C, "
                f"опади {total_p:.0f} мм"
            )

    prompt = (
        "Користувач просить короткий звіт по своєму полю. Напиши 2-3 речення "
        "українською про поточний стан, базуючись лише на цих даних. Не вигадуй цифри.\n\n"
        + "\n".join(f"• {b}" for b in bullets)
    )
    try:
        resp = await openai.completion(
            messages=[
                {"role": "system", "content": "Ти короткий аналітик-агроном. Пиши стисло."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=220,
        )
        return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
    except OpenAIError as exc:
        log.warning("LLM summary failed: %s", exc)
        return None


async def _summarise_portfolio(
    openai: OpenAIClient | None,
    *,
    fields,
    kpis,
    yoy,
) -> str | None:
    if openai is None or not settings.openai_api_key:
        return None

    lines = [
        f"Полів: {kpis['total_fields']}, загальна площа: {kpis['total_area_ha']:.1f} га",
    ]
    if kpis["predicted_total_yield_t"] is not None:
        lines.append(f"Прогноз сумарного врожаю: {kpis['predicted_total_yield_t']:.1f} т")
    if kpis["active_alerts_count"]:
        lines.append(f"Активних сповіщень: {kpis['active_alerts_count']}")
    if yoy["diff_pct"] is not None:
        sign = "+" if yoy["diff_pct"] >= 0 else ""
        lines.append(f"NDVI рік-до-року: {sign}{yoy['diff_pct']:.1f}%")
    for f in fields[:5]:
        ndvi = f"NDVI={f['current_ndvi']:.2f}" if f["current_ndvi"] is not None else "немає даних"
        lines.append(f"• {f['name']} ({f['crop_label']}): {ndvi}")

    prompt = (
        "Напиши 2-3 речення українською — короткий стан портфеля користувача. "
        "Не вигадуй цифр.\n\n" + "\n".join(lines)
    )
    try:
        resp = await openai.completion(
            messages=[
                {"role": "system", "content": "Ти короткий аналітик-агроном. Пиши стисло."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=220,
        )
        return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
    except OpenAIError as exc:
        log.warning("LLM portfolio summary failed: %s", exc)
        return None


async def _summarise_compare(
    openai: OpenAIClient | None,
    *,
    fields: list[dict],
    date_from: date,
    date_to: date,
) -> str | None:
    if openai is None or not settings.openai_api_key:
        return None

    lines = [f"Період: {date_from.isoformat()} – {date_to.isoformat()}"]
    for f in fields:
        ndvi = f"NDVI ср.={f['ndvi_mean']:.2f}" if f["ndvi_mean"] is not None else "немає даних"
        pred = f", прогноз {f['predicted_tha']:.1f} т/га" if f["predicted_tha"] is not None else ""
        alerts = f", alerts: {f['alerts_count']}" if f["alerts_count"] else ""
        lines.append(f"• {f['name']} ({f['crop_label']}): {ndvi}{pred}{alerts}")

    prompt = (
        "Порівняй ці поля українською (2-3 речення). Вкажи яке найкраще, "
        "яке найгірше і чому. Не вигадуй цифр.\n\n" + "\n".join(lines)
    )
    try:
        resp = await openai.completion(
            messages=[
                {"role": "system", "content": "Ти короткий аналітик-агроном. Пиши стисло."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=240,
        )
        return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
    except OpenAIError as exc:
        log.warning("LLM compare summary failed: %s", exc)
        return None


# ─── Methodology report (Phase-5 finishing iteration) ────────


def _load_evaluation_v3() -> dict:
    """Read `data/processed/evaluation_v3.json`. Returns `{}` if missing."""
    import json
    p = Path("data/processed/evaluation_v3.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("evaluation_v3.json unparseable: %s", exc)
        return {}


def _load_oblast_geojson() -> dict:
    import json
    p = Path("data/raw/ukraine_oblasts.geojson")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("ukraine_oblasts.geojson unparseable: %s", exc)
        return {}


def _flatten_leaderboard(eval_payload: dict) -> list[dict]:
    """Same shape as /api/methodology/leaderboard — used for the PDF
    cross-crop chart. Keeps single source of truth."""
    rows: list[dict] = []
    for crop, by_family in eval_payload.get("crops", {}).items():
        for family, body in by_family.items():
            if not isinstance(body, dict) or body.get("skipped"):
                continue
            test = body.get("test") or {}
            rows.append({
                "crop": crop,
                "family": family,
                "test_r2": test.get("r2"),
                "test_rmse": test.get("rmse"),
                "test_mae": test.get("mae"),
                "test_mape": test.get("mape"),
            })
    return rows


def _fmt(v, digits: int = 3, suffix: str = "") -> str:
    """Format a number for display in the PDF KPI cards; `—` for None."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


async def build_methodology_report(
    *,
    db: AsyncSession,
    user: User,
    crop: str = "wheat",
    family: str = "stack",
) -> bytes:
    """Render the portfolio-wide ML methodology PDF.

    Reads `evaluation_v3.json` + `ukraine_oblasts.geojson`, builds four
    PNG figures (leaderboard, residual choropleth, global SHAP, learning
    curves), pipes through the same `engine.render_html_to_pdf` path as
    the existing field/portfolio reports.

    The `crop`/`family` selectors pick which (crop × family) entry drives
    the headline metrics + residual map + SHAP. The leaderboard chart
    spans all entries regardless of selectors.
    """
    _ = db  # methodology report doesn't read user data; auth is the gate

    eval_payload = _load_evaluation_v3()
    crops_block = eval_payload.get("crops", {})

    # Headline (crop, family) — fall back gracefully when the requested
    # combo is missing (e.g. user asks for stack on a crop that was
    # pruned at training time).
    body = (crops_block.get(crop) or {}).get(family) or {}
    if body.get("skipped") or not body:
        # Pick the first non-skipped entry as a safety net so the PDF
        # still has data — operator sees what's available rather than a
        # blank page.
        for c, families_block in crops_block.items():
            for f, b in families_block.items():
                if isinstance(b, dict) and not b.get("skipped") and b.get("test"):
                    crop, family, body = c, f, b
                    break
            if body:
                break

    test = body.get("test") or {}
    loocv = body.get("loocv_oblast") or {}
    headline_metrics = {
        "test_r2": _fmt(test.get("r2"), 3),
        "test_rmse": _fmt(test.get("rmse"), 3),
        "test_mape": _fmt(test.get("mape"), 1, suffix=" %"),
        "loocv_r2": _fmt(loocv.get("r2"), 3),
    }

    # Cross-crop leaderboard chart
    leaderboard_rows = _flatten_leaderboard(eval_payload)
    leaderboard_top_n = 15
    leaderboard_png = render_methodology_leaderboard_png(
        leaderboard_rows, top_n=leaderboard_top_n,
    )

    # Residual choropleth
    geojson = _load_oblast_geojson()
    residuals_by_iso = {
        r["iso_3166_2"]: float(r["mean_abs_residual"])
        for r in (body.get("per_oblast_residuals") or [])
        if r.get("iso_3166_2") and r.get("mean_abs_residual") is not None
    }
    residual_map_png = render_methodology_residual_map_png(
        geojson, residuals_by_iso,
    )

    # Global SHAP (or permutation importance fall-back).
    shap_data = body.get("global_shap") or body.get("permutation_importance") or {}
    shap_title = "Global SHAP — топ фічі" if body.get("global_shap") else "Permutation importance — топ фічі"
    shap_png = render_methodology_shap_png(
        shap_data, top_n=8, title=shap_title,
    )

    # Learning curves: aggregate across all families of the SELECTED
    # crop so the chart shows the full bias/variance picture per crop.
    curves_by_family: dict[str, list[dict]] = {}
    for f, b in (crops_block.get(crop) or {}).items():
        if isinstance(b, dict) and b.get("learning_curve"):
            curves_by_family[f] = b["learning_curve"]
    learning_curves_png = render_methodology_learning_curves_png(curves_by_family)

    # Run metadata
    md = eval_payload.get("metadata", {})
    run_meta = {
        "crops_count": len(crops_block),
        "oblasts_count": 24,
        "year_range": "2017–2023",
        "features_origin": md.get("features_origin", "—"),
    }

    qr_uri = render_qr_png(_frontend_url("/methodology"))

    context = {
        "title": "Methodology · HarvestAI",
        "heading": f"Methodology: {crop} / {family}",
        "subheading": f"Phase 4 scientific metric panel · {_now_uk()}",
        "generated_at": _now_uk(),
        "selected_crop": crop,
        "selected_family": family,
        "headline_metrics": headline_metrics,
        "leaderboard_png": leaderboard_png,
        "leaderboard_top_n": leaderboard_top_n,
        "residual_map_png": residual_map_png,
        "shap_png": shap_png,
        "learning_curves_png": learning_curves_png,
        "run_meta": run_meta,
        "qr_uri": qr_uri,
    }
    html = _env().get_template("methodology_report.html").render(context)
    return render_html_to_pdf(html, base_url=str(TEMPLATES_DIR))


__all__ = [
    "ALL_FIELD_SECTIONS",
    "build_compare_report",
    "build_field_report",
    "build_methodology_report",
    "build_portfolio_report",
]
