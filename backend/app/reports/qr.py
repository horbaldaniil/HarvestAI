"""QR-code generation as base-64 PNG data URI.

Used in the report footer: scanning the QR opens the field's web view in
the browser. We size it small (~150 px) so it's a quick tap target without
dominating the page.
"""
from __future__ import annotations

import base64
import io

import qrcode


def render_qr_png(url: str, *, box_size: int = 4, border: int = 1) -> str:
    """Return a data URI for a PNG-encoded QR code pointing at `url`."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


__all__ = ["render_qr_png"]
