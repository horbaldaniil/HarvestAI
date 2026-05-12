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


# ─── Methodology PDF charts (Phase 5 finishing iteration) ────────────


def render_methodology_leaderboard_png(
    rows: list[dict],
    *,
    top_n: int = 15,
    width_in: float = 7.2,
    height_in: float = 4.5,
) -> str:
    """Top-N (crop, family) entries sorted by test R² → horizontal bar.

    `rows` items must have at least: `crop`, `family`, `test_r2`. Sorted
    desc by R² (NaN/None last); cropped to top_n.
    """
    valid = [r for r in rows if r.get("test_r2") is not None]
    valid.sort(key=lambda r: r["test_r2"], reverse=True)
    valid = valid[:top_n]

    if not valid:
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "Метрики ML ще не обчислені", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    labels = [f"{r['crop']} / {r['family']}" for r in valid]
    r2_values = [r["test_r2"] for r in valid]
    # Colour scheme: stack rows in deep purple, others in muted green.
    colours = ["#9333ea" if r["family"] == "stack" else "#5e7d36" for r in valid]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.93, bottom=0.10)

    y_pos = range(len(labels))
    ax.barh(list(y_pos), r2_values, color=colours)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # Top-1 at the top.
    ax.set_xlabel("Test R² (out-of-time generalisation)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(0, color="#777", linewidth=0.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_title(f"Лідерборд моделей (top-{top_n} за test R²)", fontsize=10)

    return _to_data_uri(fig)


def render_methodology_residual_map_png(
    oblast_geojson: dict,
    residuals_by_iso: dict[str, float],
    *,
    width_in: float = 7.2,
    height_in: float = 4.5,
) -> str:
    """Ukraine oblast choropleth coloured by mean abs residual.

    `oblast_geojson` is the FeatureCollection from `ukraine_oblasts.geojson`
    (each feature has `iso_3166_2` in its properties). `residuals_by_iso`
    maps `UA-XX` → float (mean absolute residual). Unmapped oblasts are
    drawn in light grey.
    """
    if not oblast_geojson or not residuals_by_iso:
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "Дані залишків відсутні", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    try:
        import geopandas as gpd
        from matplotlib.colors import LinearSegmentedColormap
        from shapely.geometry import shape
    except ImportError:
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "geopandas not installed", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    geoms = []
    iso_codes = []
    names = []
    values = []
    vmin = min(residuals_by_iso.values())
    vmax = max(residuals_by_iso.values())

    for feature in oblast_geojson.get("features", []):
        props = feature.get("properties", {}) or {}
        iso = props.get("iso_3166_2")
        if not iso:
            continue
        try:
            geom = shape(feature["geometry"])
        except Exception:  # noqa: BLE001
            continue
        geoms.append(geom)
        iso_codes.append(iso)
        names.append(props.get("name_uk") or props.get("name") or iso)
        values.append(residuals_by_iso.get(iso))

    gdf = gpd.GeoDataFrame({
        "iso": iso_codes, "name": names, "residual": values,
    }, geometry=geoms, crs="EPSG:4326")

    # Yellow → red gradient (matches frontend OblastResidualMap.tsx).
    cmap = LinearSegmentedColormap.from_list("yellow_red", ["#fde047", "#dc2626"])

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Two-layer plot: greys for missing data, colour for mapped.
    gdf[gdf["residual"].isna()].plot(
        ax=ax, color="#e5e7eb", edgecolor="#9ca3af", linewidth=0.5,
    )
    mapped = gdf[gdf["residual"].notna()]
    if not mapped.empty:
        mapped.plot(
            ax=ax, column="residual", cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolor="#4b5563", linewidth=0.5,
        )

    ax.set_title(
        f"Mean absolute residual по областях (test split)\n"
        f"шкала: {vmin:.3f} → {vmax:.3f} т/га",
        fontsize=10,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")

    return _to_data_uri(fig)


def render_methodology_shap_png(
    feature_importance: dict[str, float],
    *,
    top_n: int = 8,
    width_in: float = 6.5,
    height_in: float = 3.5,
    title: str = "Global SHAP — топ фічі",
) -> str:
    """Horizontal bar chart of mean |SHAP value| per feature, top-N.

    `feature_importance` is `{feature_name: value}` (typically from
    evaluate_models.py global_shap or permutation_importance).
    """
    if not feature_importance:
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "SHAP не доступний для цієї моделі",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    items = sorted(feature_importance.items(), key=lambda kv: abs(kv[1]),
                   reverse=True)[:top_n]
    names = [n for n, _ in items]
    vals = [abs(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.12)

    y_pos = range(len(names))
    ax.barh(list(y_pos), vals, color="#5e7d36")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_title(title, fontsize=10)

    return _to_data_uri(fig)


def render_methodology_learning_curves_png(
    curves_by_family: dict[str, list[dict]],
    *,
    width_in: float = 6.8,
    height_in: float = 3.5,
) -> str:
    """One line per family: test R² as a function of training sample size.

    `curves_by_family` is `{family_name: [{n_train, test_r2}, ...]}`
    (typically the `learning_curve` entries from evaluation_v3.json).
    """
    valid = {f: c for f, c in curves_by_family.items() if c}
    if not valid:
        fig, ax = plt.subplots(figsize=(width_in, 1.0))
        ax.text(0.5, 0.5, "Learning curves не доступні", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _to_data_uri(fig)

    palette = {
        "rf": "#5e7d36",
        "xgboost": "#3b82f6",
        "lightgbm": "#a16207",
        "catboost": "#dc2626",
        "stack": "#9333ea",
        "lstm": "#0ea5e9",
    }

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.15)

    for family, curve in valid.items():
        # Sort by n_train so the line is monotonic.
        rows = sorted(curve, key=lambda p: p.get("n_train", 0))
        xs = [p.get("n_train") for p in rows]
        ys = [p.get("test_r2") if p.get("test_r2") is not None else float("nan") for p in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=family,
                color=palette.get(family, "#444"))

    ax.set_xlabel("n train", fontsize=8)
    ax.set_ylabel("test R²", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_title("Криві навчання — bias/variance діагностика", fontsize=10)

    return _to_data_uri(fig)


__all__ = [
    "INDEX_COLOURS",
    "INDEX_LABELS_UK",
    "render_index_chart_png",
    "render_methodology_leaderboard_png",
    "render_methodology_learning_curves_png",
    "render_methodology_residual_map_png",
    "render_methodology_shap_png",
    "render_multi_field_chart_png",
    "render_shap_chart_png",
]
