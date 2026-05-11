"""PDF rendering engine — picks WeasyPrint when available, falls back to xhtml2pdf.

WeasyPrint produces noticeably better PDFs (real CSS3, web fonts, flexbox).
But on Windows it depends on GTK3 + cairo, which most students don't have
installed. xhtml2pdf is pure-Python and ships well, at the cost of a more
restricted CSS subset (HTML 4 / CSS 2.1-ish).

We choose at import time. If the user installs GTK3 later, just `pip
install weasyprint` again — we'll switch automatically on next process
start.

The interface is a single function: `render_html_to_pdf(html, base_url=None)
-> bytes`.

Cyrillic note: ReportLab (which xhtml2pdf wraps) ships with PDF-14 fonts
that have NO Cyrillic glyphs. To avoid the "tofu boxes" rendering bug we
register DejaVu Sans (Cyrillic-capable, already bundled by matplotlib) with
ReportLab on first use, and the HTML templates use it as the body font.
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

_USE_WEASYPRINT: bool
_FONTS_REGISTERED: bool = False


def _can_use_weasyprint() -> bool:
    try:
        from weasyprint import HTML  # noqa: F401  (probe-import)
    except (OSError, ImportError) as exc:
        log.info("WeasyPrint unavailable (%s) — using xhtml2pdf fallback.", exc)
        return False
    return True


_USE_WEASYPRINT = _can_use_weasyprint()


def _register_cyrillic_fonts() -> None:
    """Make DejaVu Sans available to xhtml2pdf+ReportLab for Cyrillic text.

    Three steps, each addressing a different layer:
      1. `pdfmetrics.registerFont` — tells ReportLab the actual TTF file.
      2. `addMapping` — declares the bold/italic relationship within the
         family so `<strong>` (which xhtml2pdf renders as bold) picks the
         bold TTF instead of falling back to Helvetica.
      3. `xhtml2pdf.default.DEFAULT_FONT` — primes xhtml2pdf's per-document
         font lookup table BEFORE pisaContext is instantiated. Without this
         step, xhtml2pdf would still resolve `font-family: "DejaVuSans"`
         through @font-face → loadFont, which on Windows hits a tempfile
         locking bug (NamedTemporaryFile keeps the file open exclusively
         and ReportLab's TTFOpenFile then fails to read it).

    Looks up the font matplotlib already ships, so the repo doesn't need a
    bundled TTF. Safe to call repeatedly — the flag short-circuits.
    """
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    try:
        from matplotlib import font_manager
        from reportlab.lib.fonts import addMapping
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from xhtml2pdf import default as xhtml2pdf_default

        regular = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"),
            fallback_to_default=False,
        )
        bold = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
            fallback_to_default=False,
        )
        # Step 1: register the TTF files with ReportLab.
        pdfmetrics.registerFont(TTFont("DejaVuSans", regular))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold))

        # Step 2: declare the family. addMapping(family, bold, italic, font).
        addMapping("DejaVuSans", 0, 0, "DejaVuSans")
        addMapping("DejaVuSans", 1, 0, "DejaVuSans-Bold")
        # No italic variant bundled — fall back to the regular roman for italic.
        addMapping("DejaVuSans", 0, 1, "DejaVuSans")
        addMapping("DejaVuSans", 1, 1, "DejaVuSans-Bold")

        # Step 3: prime xhtml2pdf's lookup table. Keys are lowercase per
        # xhtml2pdf's `getFontName` (it lowercases incoming font-family).
        # We accept both "dejavusans" (the no-space CSS name we use in
        # templates) and "dejavu sans" (the system / WeasyPrint convention).
        xhtml2pdf_default.DEFAULT_FONT["dejavusans"] = "DejaVuSans"
        xhtml2pdf_default.DEFAULT_FONT["dejavu sans"] = "DejaVuSans"

        _FONTS_REGISTERED = True
        log.info("Registered DejaVuSans for Cyrillic PDF rendering.")
    except Exception as exc:  # noqa: BLE001
        # Don't crash report generation if the font can't be located —
        # text will render with tofu boxes but the rest of the PDF is fine.
        log.warning("Could not register Cyrillic font: %s", exc)


def render_html_to_pdf(html: str, *, base_url: str | None = None) -> bytes:
    """Render an HTML string to PDF bytes.

    `base_url` controls how relative `<img src=...>` / `<link href=...>` get
    resolved. Pass a `file://` URL for the templates directory if your HTML
    references local assets.
    """
    if _USE_WEASYPRINT:
        return _render_weasyprint(html, base_url=base_url)
    return _render_xhtml2pdf(html)


def _render_weasyprint(html: str, *, base_url: str | None) -> bytes:
    from weasyprint import HTML

    return HTML(string=html, base_url=base_url).write_pdf()


def _render_xhtml2pdf(html: str) -> bytes:
    """xhtml2pdf path: takes HTML, writes to a BytesIO."""
    _register_cyrillic_fonts()
    from xhtml2pdf import pisa

    buf = io.BytesIO()
    # `raise_exception=False` so a single malformed CSS rule doesn't abort
    # the whole document; we still log errors so they show up in tests.
    result = pisa.CreatePDF(
        src=html,
        dest=buf,
        encoding="utf-8",
        raise_exception=False,
    )
    if result.err:
        log.warning("xhtml2pdf produced %d errors", result.err)
    return buf.getvalue()


def current_engine() -> str:
    return "weasyprint" if _USE_WEASYPRINT else "xhtml2pdf"


__all__ = ["current_engine", "render_html_to_pdf"]
