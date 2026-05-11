"""PDF report generation package.

Two report types are exposed:
- `build_field_report(...)` — 3-5 page report for a single field
- `build_portfolio_report(...)` — 1-2 page summary for the whole user

Both produce bytes (the PDF body) and are mounted on `/api/.../report`
endpoints in `app.routers.reports`.
"""
from app.reports.builder import build_field_report, build_portfolio_report

__all__ = ["build_field_report", "build_portfolio_report"]
