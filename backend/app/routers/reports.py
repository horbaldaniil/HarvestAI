"""PDF report endpoints with history + builder.

Per-user persistence:
- Each generated PDF is written to `settings.reports_dir/<user_id>/<id>.pdf`
  and a metadata row goes into `generated_reports`.
- The History tab on /reports lists them in reverse-chronological order.
- Auto-prune: when a user crosses 50 reports we delete the oldest (DB row +
  file). Cap configurable via `REPORTS_PER_USER_LIMIT` later if needed.

Endpoints:
- POST /api/fields/{id}/report       — single-field quick export (uses defaults)
- POST /api/portfolio/report         — portfolio quick export
- POST /api/reports/builder          — customisable: sections, date range,
                                       field selection, single/portfolio/compare
- GET  /api/reports                  — paginated history for current user
- GET  /api/reports/{id}/download    — serve cached PDF (no re-generation)
- DELETE /api/reports/{id}           — delete row + file
"""
from __future__ import annotations

import io
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field as PField, model_validator
from sqlalchemy import delete, func, select

from app.config import settings
from app.db.models import Field, GeneratedReport
from app.deps import CurrentUser, DbSession
from app.integrations.openai.client import OpenAIClient
from app.reports.builder import (
    ALL_FIELD_SECTIONS,
    build_compare_report,
    build_field_report,
    build_portfolio_report,
)

log = logging.getLogger(__name__)

REPORTS_PER_USER_LIMIT = 50

router = APIRouter(prefix="/api", tags=["reports"])


# ─── Pydantic schemas ─────────────────────────────────────────


class GeneratedReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: str
    title: str
    params_json: dict
    file_size_bytes: int
    generated_at: datetime


class BuilderRequest(BaseModel):
    """Body for POST /api/reports/builder.

    `kind` selects which builder runs. Other fields are interpreted per-kind:
    - field: requires `field_id`; honours `sections`, `date_from`, `date_to`
    - portfolio: ignores `field_id` / `field_ids` / `sections`
    - compare: requires `field_ids` (≥1); honours `date_from`, `date_to`
    """
    kind: Literal["field", "portfolio", "compare"]
    field_id: int | None = None
    field_ids: list[int] | None = None
    sections: list[str] | None = None
    date_from: date | None = None
    date_to: date | None = None
    title: str | None = PField(default=None, max_length=255)

    @model_validator(mode="after")
    def _check_consistency(self) -> BuilderRequest:
        if self.kind == "field" and self.field_id is None:
            raise ValueError("field kind requires field_id")
        if self.kind == "compare" and (not self.field_ids or len(self.field_ids) == 0):
            raise ValueError("compare kind requires field_ids (≥1)")
        if self.sections:
            unknown = set(self.sections) - ALL_FIELD_SECTIONS
            if unknown:
                raise ValueError(f"unknown sections: {sorted(unknown)}")
        return self


# ─── Helpers ──────────────────────────────────────────────────


def _safe_filename(name: str) -> str:
    """Drop characters that browsers/file systems don't like."""
    safe = re.sub(r"[^A-Za-z0-9_\-.Ѐ-ӿ]", "_", name)
    return safe[:80] or "report"


def _user_reports_dir(user_id: int) -> Path:
    d = settings.reports_dir / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _persist_report(
    *,
    db,
    user_id: int,
    kind: str,
    title: str,
    params: dict[str, Any],
    pdf_bytes: bytes,
) -> GeneratedReport:
    """Save bytes to disk + DB row, then prune to REPORTS_PER_USER_LIMIT."""
    row = GeneratedReport(
        user_id=user_id,
        kind=kind,
        title=title[:255],
        params_json=params,
        file_path="",  # set after we know id
        file_size_bytes=len(pdf_bytes),
    )
    db.add(row)
    await db.flush()  # gets row.id

    rel = f"{user_id}/{row.id}.pdf"
    abs_path = _user_reports_dir(user_id) / f"{row.id}.pdf"
    abs_path.write_bytes(pdf_bytes)
    row.file_path = rel
    await db.flush()

    await _prune_old_reports(db, user_id)
    await db.commit()
    await db.refresh(row)
    return row


async def _prune_old_reports(db, user_id: int) -> None:
    """If the user has > limit reports, delete oldest (row + disk file)."""
    total = await db.scalar(
        select(func.count()).select_from(GeneratedReport)
        .where(GeneratedReport.user_id == user_id)
    )
    excess = (total or 0) - REPORTS_PER_USER_LIMIT
    if excess <= 0:
        return
    old = (await db.execute(
        select(GeneratedReport)
        .where(GeneratedReport.user_id == user_id)
        .order_by(GeneratedReport.generated_at.asc())
        .limit(excess)
    )).scalars().all()
    for row in old:
        try:
            (settings.reports_dir / row.file_path).unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001
            log.warning("Could not unlink %s: %s", row.file_path, exc)
        await db.delete(row)


def _stream_pdf(pdf_bytes: bytes, filename: str) -> StreamingResponse:
    """Send a PDF as a streaming attachment with a Unicode-safe filename.

    HTTP/1.1 headers are restricted to latin-1 (RFC 7230 §3.2.4); Ukrainian
    Cyrillic in `filename` would otherwise raise UnicodeEncodeError when
    Starlette encodes the headers. RFC 5987 §3.2 specifies the
    `filename*=UTF-8''<percent-encoded>` extension, which all modern
    browsers (Chrome 9+, Firefox 8+, Safari 6+, Edge) understand. We
    additionally emit a fallback ASCII `filename=` for ancient HTTP
    clients — it's transliterated to ASCII-safe form (drop accents,
    replace non-Latin chars with `_`) so the header itself stays
    latin-1-encodable.
    """
    from urllib.parse import quote
    import unicodedata

    # Fallback for clients that don't read filename* (rare these days).
    # NFKD-normalise then strip everything beyond ASCII; replace gaps with _
    ascii_fallback = (
        unicodedata.normalize("NFKD", filename)
        .encode("ascii", "ignore")
        .decode("ascii")
        or "report.pdf"
    )
    # RFC 5987: percent-encoded UTF-8 for the canonical name (Cyrillic OK).
    utf8_filename = quote(filename, safe="")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_fallback}"; '
                f"filename*=UTF-8''{utf8_filename}"
            ),
        },
    )


# ─── Quick exports (existing buttons in BottomPanel + Dashboard) ──


@router.post("/fields/{field_id}/report")
async def field_report(
    field_id: int, current_user: CurrentUser, db: DbSession,
) -> StreamingResponse:
    field = await db.get(Field, field_id)
    if field is None or field.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Поле не знайдено.")

    openai = OpenAIClient()
    try:
        pdf_bytes = await build_field_report(
            field_id, db=db, user=current_user, openai=openai,
        )
    finally:
        await openai.aclose()

    await _persist_report(
        db=db, user_id=current_user.id, kind="field",
        title=f"{field.name} · {field.season_year}",
        params={"field_id": field_id, "sections": sorted(ALL_FIELD_SECTIONS)},
        pdf_bytes=pdf_bytes,
    )

    filename = f"HarvestAI_{_safe_filename(field.name)}_{field.season_year}.pdf"
    return _stream_pdf(pdf_bytes, filename)


@router.post("/portfolio/report")
async def portfolio_report(
    current_user: CurrentUser, db: DbSession,
) -> StreamingResponse:
    openai = OpenAIClient()
    try:
        pdf_bytes = await build_portfolio_report(
            db=db, user=current_user, openai=openai,
        )
    finally:
        await openai.aclose()

    await _persist_report(
        db=db, user_id=current_user.id, kind="portfolio",
        title=f"Портфель · {_now_compact()}",
        params={},
        pdf_bytes=pdf_bytes,
    )

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    return _stream_pdf(pdf_bytes, f"HarvestAI_portfolio_{stamp}.pdf")


# ─── Builder (customisable) ───────────────────────────────────


@router.post("/reports/builder")
async def builder(
    payload: BuilderRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    openai = OpenAIClient()
    try:
        if payload.kind == "field":
            field = await db.get(Field, payload.field_id)
            if field is None or field.user_id != current_user.id:
                raise HTTPException(404, "Поле не знайдено.")
            sections = set(payload.sections) if payload.sections else None
            pdf_bytes = await build_field_report(
                payload.field_id,  # type: ignore[arg-type]
                db=db, user=current_user, openai=openai,
                sections=sections,
                date_from=payload.date_from,
                date_to=payload.date_to,
            )
            title = payload.title or f"{field.name} (builder)"
            filename = f"HarvestAI_{_safe_filename(field.name)}_custom.pdf"
        elif payload.kind == "portfolio":
            pdf_bytes = await build_portfolio_report(
                db=db, user=current_user, openai=openai,
            )
            title = payload.title or f"Портфель · {_now_compact()}"
            filename = f"HarvestAI_portfolio_custom_{_now_compact()}.pdf"
        else:  # compare
            # Validate ownership of every field before kicking off generation.
            ids = list(payload.field_ids or [])
            owned = set((await db.execute(
                select(Field.id).where(Field.id.in_(ids))
                .where(Field.user_id == current_user.id)
            )).scalars().all())
            if owned != set(ids):
                raise HTTPException(404, "Одне або більше полів не знайдено.")
            pdf_bytes = await build_compare_report(
                ids, db=db, user=current_user, openai=openai,
                date_from=payload.date_from,
                date_to=payload.date_to,
            )
            title = payload.title or f"Порівняння {len(ids)} полів"
            filename = f"HarvestAI_compare_{_now_compact()}.pdf"
    finally:
        await openai.aclose()

    await _persist_report(
        db=db, user_id=current_user.id, kind=payload.kind,
        title=title,
        params=payload.model_dump(mode="json", exclude={"title"}),
        pdf_bytes=pdf_bytes,
    )

    return _stream_pdf(pdf_bytes, filename)


# ─── History ──────────────────────────────────────────────────


@router.get("/reports", response_model=list[GeneratedReportRead])
async def list_reports(
    current_user: CurrentUser, db: DbSession,
) -> list[GeneratedReportRead]:
    rows = (await db.execute(
        select(GeneratedReport)
        .where(GeneratedReport.user_id == current_user.id)
        .order_by(GeneratedReport.generated_at.desc())
        .limit(REPORTS_PER_USER_LIMIT)
    )).scalars().all()
    return [GeneratedReportRead.model_validate(r) for r in rows]


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: int, current_user: CurrentUser, db: DbSession,
):
    row = await db.get(GeneratedReport, report_id)
    if row is None or row.user_id != current_user.id:
        raise HTTPException(404, "Звіт не знайдено.")
    abs_path = settings.reports_dir / row.file_path
    if not abs_path.exists():
        # The DB row points at a missing file (e.g. someone wiped the volume).
        raise HTTPException(410, "Файл недоступний — повторіть генерацію.")
    return FileResponse(
        abs_path,
        media_type="application/pdf",
        filename=f"{_safe_filename(row.title)}.pdf",
    )


@router.delete("/reports/{report_id}", status_code=204)
async def delete_report(
    report_id: int, current_user: CurrentUser, db: DbSession,
) -> None:
    row = await db.get(GeneratedReport, report_id)
    if row is None or row.user_id != current_user.id:
        raise HTTPException(404, "Звіт не знайдено.")
    try:
        (settings.reports_dir / row.file_path).unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001
        log.warning("Failed to unlink %s: %s", row.file_path, exc)
    await db.execute(delete(GeneratedReport).where(GeneratedReport.id == report_id))
    await db.commit()


def _now_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
