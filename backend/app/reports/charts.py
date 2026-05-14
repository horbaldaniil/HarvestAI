"""Server-side chart rendering for the PDF report.

We use matplotlib with the Agg backend (no GUI) and emit PNG via base-64
data URIs. xhtml2pdf, our default PDF engine on Windows, doesn't accept
inline `<svg>` elements — only `<img src="data:image/png;base64,...">` —
so PNG is the lowest-common-denominator format.

Charts are kept intentionally simple (single-axis time series, horizontal
bar SHAP plot) because A4 page real estate is small and a busy chart in
print is unreadable.
"""
from __future__ import annotations

import base64
import io
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")  # noqa: E402 (must be before pyplot import)
import matplotlib.pyplot as plt  # noqa: E402

INDEX_COLOURS = {
    "ndvi": "#5e7d36",
    "evi": "#7d9c49",
    "ndwi": "#3b82f6",
    "savi": "#a16207",
}
INDEX_LABELS_UK = {
    "ndvi": "NDVI",
    "evi": "EVI",
    "ndwi": "NDWI",
    "savi": "SAVI",
}


def _to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_index_chart_png(
    points: Iterable[tuple[str, float | None]],
    *,
    index: str,
    width_in: float = 6.5,
    height_in: float = 2.4,
) -> str:
    """Render a vegetation index time-series chart as a base-64 PNG data URI.

    `points` is an iterable of (iso_date_string, value_or_None) tuples. None
    values are interpreted as cloudy buckets and shown as gaps in the line.
    """
    dates = []
    values = []
    for d, v in points:
        dates.append(d)
        values.append(v)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.18)

    colour = INDEX_COLOURS.get(index, "#444")
    # Convert None → nan so matplotlib draws gaps.
    ys = [v if v is not None else float("nan") for v in values]
    ax.plot(dates, ys, color=colour, linewidth=1.5, marker="o", markersize=3)

    ax.set_title(f"{INDEX_LABELS_UK.get(index, index.upper())}", fontsize=10)
    ax.set_ylabel("значення", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Trim x-axis to ≤ 12 ticks so labels don't overlap.
    if len(dates) > 12:
        step = len(dates) // 12
        ax.set_xticks(dates[::step])

    return _to_data_uri(fig)


def render_shap_chart_png(
    contributions: list[dict],
    *,
    width_in: float = 6.5,
    height_in: float = 2.6,
) -> str:
    """Render the SHAP top-N horizontal bar chart.

    `contributions` items must have at least `name` (str) and `contribution`
    (float). Positive = increases yield, negative = decreases.
    """
    if not contributions:
        # Empty chart with a hint, so the PDF doesn't have a dead hole.
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "Немає даних SHAP", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    names = [c.get("name", "?") for c in contributions]
    vals = [float(c.get("contribution", 0.0)) for c in contributions]
    colours = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.18)

    y_pos = range(len(names))
    ax.barh(list(y_pos), vals, color=colours)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("внесок у прогноз (т/га)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(0, color="#777", linewidth=0.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_title("Топ-фактори впливу", fontsize=10)

    return _to_data_uri(fig)


def render_multi_field_chart_png(
    series: list[dict],
    *,
    index: str = "ndvi",
    width_in: float = 7.2,
    height_in: float = 3.0,
) -> str:
    """Render a single chart with one line per field for a single index.

    `series` items: `{"name": "Field A", "points": [(iso_date, value), ...]}`.
    Used by the compare report to show side-by-side NDVI trajectories.
    """
    if not series:
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "Немає даних", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.20)

    # tab10 cycles through 10 colours — plenty for typical compare cases.
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(series):
        pts = s.get("points") or []
        if not pts:
            continue
        dates, vals = zip(*[(d, v if v is not None else float("nan")) for d, v in pts])
        ax.plot(
            dates, vals,
            color=cmap(i % 10),
            linewidth=1.5, marker="o", markersize=3,
            label=s.get("name", f"#{i+1}"),
        )

    ax.set_title(INDEX_LABELS_UK.get(index, index.upper()) + " — порівняння", fontsize=10)
    ax.set_ylabel("значення", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    # Cap x-axis ticks so labels don't overlap.
    all_dates: list[str] = []
    for s in series:
        for d, _ in s.get("points") or []:
            all_dates.append(d)
    uniq = sorted(set(all_dates))
    if len(uniq) > 12:
        step = len(uniq) // 12
        ax.set_xticks(uniq[::step])

    return _to_data_uri(fig)


__all__ = [
    "INDEX_COLOURS",
    "INDEX_LABELS_UK",
    "render_index_chart_png",
    "render_multi_field_chart_png",
    "render_shap_chart_png",
]
