"""Smoke-tests for the PDF engine selector + xhtml2pdf renderer.

We don't exercise the full per-field/portfolio builders here (they need
PostgreSQL+PostGIS for the underlying tables). Builder tests live in the
integration suite. These ensure the engine layer itself produces valid PDF
output from raw HTML — the contract every builder relies on.
"""
from __future__ import annotations

from app.reports.engine import current_engine, render_html_to_pdf


HTML_SAMPLE = """
<!DOCTYPE html>
<html><head><title>Test</title>
<style>body { font-family: "DejaVuSans"; } h1 { color: #14532d; }</style>
</head>
<body>
    <h1>HarvestAI тестовий звіт</h1>
    <p>Текст українською для перевірки кириличного рендерингу.</p>
    <table border="1"><tr><th>А</th><th>Б</th></tr><tr><td>1</td><td>2</td></tr></table>
</body></html>
"""


def test_current_engine_is_one_of_known():
    assert current_engine() in {"weasyprint", "xhtml2pdf"}


def test_render_produces_pdf_bytes():
    pdf = render_html_to_pdf(HTML_SAMPLE)
    # PDF magic header
    assert pdf.startswith(b"%PDF-")
    assert b"%%EOF" in pdf[-1024:]
    # Reasonable size (cyrillic content, small CSS)
    assert len(pdf) > 500


def test_render_handles_empty_body():
    pdf = render_html_to_pdf("<html><body></body></html>")
    assert pdf.startswith(b"%PDF-")


def test_cyrillic_font_registered_for_xhtml2pdf():
    """Regression: PDFs were rendering Cyrillic as tofu boxes because
    ReportLab's default Helvetica has no Cyrillic glyphs.
    """
    from app.reports.engine import _register_cyrillic_fonts
    _register_cyrillic_fonts()
    from reportlab.pdfbase import pdfmetrics

    names = pdfmetrics.getRegisteredFontNames()
    assert "DejaVuSans" in names
    assert "DejaVuSans-Bold" in names


def test_cyrillic_text_uses_dejavu_in_pdf():
    """End-to-end: render a cyrillic page and assert DejaVuSans is referenced
    in the PDF font table (xhtml2pdf inlines the font name into the PDF).
    """
    pdf = render_html_to_pdf(HTML_SAMPLE)
    # PDFs encode font references both literally and inside FontDescriptor.
    # We only need to confirm the font got embedded.
    assert b"DejaVuSans" in pdf
