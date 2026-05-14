"""Add 2025 rows to `training_set_v3.parquet` without rebuilding all years.

Run AFTER:
  * `scripts/collect_oblast_s2.py --year 2025` (192 PU-paid calls)
  * The 2025 Держстат XLS (`ovuzpsg_1025.xls`) parsed into the YAML
  * The build_yield_csv.py bug fix landed (recovers real-data rows that
    were dropped by the zone-feasibility gate)

This is a **surgical add** — does NOT touch any of the existing 2017-
2021 rows. Reuses the helpers from `build_features_v2.py` (S2
aggregation) and `train_yield_models.py` (Open-Meteo fetch) so the
2025 features are computed with exactly the same arithmetic as every
other year.

## Output schema match

Every column the 2018-2021 rows have (vegetation, weather, soil,
zone one-hots, crop features, `is_real_yield`, `oblast_yield_lag1`,
etc.) is populated for the new 2025 rows too — so downstream
consumers (lag-1 lookup, hierarchical model evaluation, future
training when 2025 enters the split) don't need to special-case the
new year.

## Yield handling

  * **Real Держстат row in YAML** → use it, `is_real_yield=True`.
  * **No real row + crop feasible in zone** → weather-conditioned
    synthesis (same arithmetic as `build_yield_csv.py`), `is_real_yield=False`.
  * **No real row + crop infeasible in zone** → skip the row (don't
    fabricate a sunflower yield in Polissia).

2025 is partial-year (October bulletin), so corn/sugar_beet/potato may
have synthetic values where Держstat hasn't yet published.

## Idempotency

Skips any (iso, crop, 2025) triple already present in the parquet, so
rerunning the script is safe.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the aggregator + weather helpers — exact same arithmetic as
# build_features_v3.py so new 2025 rows are schema-identical to the
# existing years.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features_v2 import aggregate_oblast_year  # noqa: E402
from train_yield_models import fetch_weather_for_rows  # noqa: E402

from app.data_reference.crop_calendar import (  # noqa: E402
    CROP_CALENDAR,
    flowering_in_reference,
    gdd_proxy as _gdd_proxy_helper,
    overlap_with_reference,
)
from app.data_reference.crop_zones import ALL_CROPS, yield_multiplier  # noqa: E402
from app.data_reference.oblast_names import (  # noqa: E402
    iso_from_any_name,
    oblast_by_iso,
)
from app.db.models.enums import CropType  # noqa: E402
from app.ml.features import (  # noqa: E402
    BBCH_PHASE_FEATURES,
    _bbch_phase_weather,
)


def _bbch_features_for_2025(
    iso: str,
    crop_slug: str,
    daily_map: dict[tuple[float, float, int], list],
    centroids: dict[str, dict[str, float]],
) -> dict[str, float | int | None]:
    """Compute 9 BBCH-phase features for one (iso, crop) at YEAR=2025.

    Reuses the inference-path helper from `app.ml.features` so the
    train-time augment row matches what `build_feature_vector`
    produces at predict time — identical arithmetic guarantees
    SHAP-importance comparisons and conformal radii are coherent.
    """
    c = centroids.get(iso)
    if c is None:
        return {k: None for k in BBCH_PHASE_FEATURES}
    daily = daily_map.get(
        (float(c["lat"]), float(c["lon"]), YEAR), [],
    )
    try:
        crop_enum = CropType(crop_slug)
    except (KeyError, ValueError):
        return {k: None for k in BBCH_PHASE_FEATURES}
    return _bbch_phase_weather(daily, crop_enum)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("augment_parquet_year2025")

YEAR = 2025
SAMPLES = ROOT / "data" / "processed" / "oblast_samples_v2.geojson"
S2_PARQUET = ROOT / "data" / "processed" / "oblast_s2_observations_v2.parquet"
YIELD_YAML = ROOT / "data" / "raw" / "derzhstat_yield_2017_2023.yaml"
PARQUET = ROOT / "data" / "processed" / "training_set_v3.parquet"

# Synthesis params — mirror build_yield_csv.py constants. Kept inline so
# this script doesn't import the full pipeline (avoids circular Python
# import dance with `agricultural_oblasts` etc.).
NOISE_AMPLITUDE = 0.015
NOISE_SEED = "harvestai-v4"
CONFLICT_FACTOR = 0.55

# Heat weights from `app/ml/features.py` — same constants.
_HEAT_W_IN_REF = 1.5
_HEAT_W_OUT_REF = 0.5

# ISO week of mid-month — mirror of features.py constant.
_ISO_WEEK_OF_MID_MONTH = {
    1: 3, 2: 7, 3: 11, 4: 16, 5: 20, 6: 24,
    7: 29, 8: 33, 9: 37, 10: 42, 11: 46, 12: 50,
}


def _deterministic_noise(seed: str, crop: str, year: int, oblast_slug: str,
                         amplitude: float) -> float:
    """Per-(crop, year, oblast) noise — same arithmetic as build_yield_csv.py."""
    h = hashlib.sha256(f"{seed}|{crop}|{year}|{oblast_slug}".encode("utf-8")).hexdigest()
    raw = int(h[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * amplitude


def _compute_crop_features(crop: str, weather: dict, veg: dict) -> dict:
    """Inline mirror of `app/ml/features.py:_crop_features`."""
    cal = CROP_CALENDAR.get(crop)
    if cal is None:
        return {
            "crop_season_overlap_aprjul": None,
            "gdd_proxy": None,
            "precip_crop_weighted": None,
            "heat_stress_crop_weighted": None,
            "drought_crop_weighted": None,
            "growing_season_length_months": None,
            "ndvi_at_crop_peak_month": None,
            "ndvi_peak_timing_offset_weeks": None,
        }

    overlap = overlap_with_reference(crop)
    season_months = cal.growing_season_length_months
    temp_mean = weather.get("temp_mean_apr_jul")
    precip = weather.get("precip_sum_apr_jul")
    heat_days = weather.get("heat_stress_days")
    drought = weather.get("drought_dryspells")
    heat_w = _HEAT_W_IN_REF if flowering_in_reference(crop) else _HEAT_W_OUT_REF

    ndvi_month_keys = {
        5: "ndvi_mean_may", 6: "ndvi_mean_june",
        7: "ndvi_mean_july", 8: "ndvi_mean_august",
    }
    peak_key = ndvi_month_keys.get(cal.peak_month)
    ndvi_at_peak = veg.get(peak_key) if peak_key is not None else veg.get("ndvi_peak")
    expected_peak_week = _ISO_WEEK_OF_MID_MONTH.get(cal.peak_month, 20)
    observed_peak_week = veg.get("ndvi_peak_week")
    timing_offset = (
        int(observed_peak_week) - expected_peak_week
        if observed_peak_week is not None and not pd.isna(observed_peak_week)
        else None
    )

    return {
        "crop_season_overlap_aprjul": round(overlap, 3),
        "gdd_proxy": _gdd_proxy_helper(crop, temp_mean, season_months),
        "precip_crop_weighted": (
            round(float(precip) * overlap, 1)
            if precip is not None and not pd.isna(precip) else None
        ),
        "heat_stress_crop_weighted": (
            round(float(heat_days) * heat_w, 1)
            if heat_days is not None and not pd.isna(heat_days) else None
        ),
        "drought_crop_weighted": (
            round(float(drought) * overlap, 2)
            if drought is not None and not pd.isna(drought) else None
        ),
        "growing_season_length_months": season_months,
        "ndvi_at_crop_peak_month": (
            round(float(ndvi_at_peak), 3)
            if ndvi_at_peak is not None and not pd.isna(ndvi_at_peak) else None
        ),
        "ndvi_peak_timing_offset_weeks": timing_offset,
    }


def _weather_factor(crop: str, weather: dict, baselines: dict) -> float:
    """Convert (oblast, year) weather to a yield-multiplier ∈ [0.7, 1.3].

    Mirrors `build_yield_csv.py:_compute_weather_factor` with the same
    crop-specific sensitivities, z-score normalisation, and clip range."""
    import statistics

    if not weather or not baselines:
        return 1.0
    sens_default = {"drought": 0.06, "heat": 0.06, "precip": 0.04}
    # Inline copy of WEATHER_SENSITIVITY for the 13 crops (mirror of build_yield_csv.py).
    sens_map = {
        "wheat":       {"drought": 0.06, "heat": 0.08, "precip": 0.04},
        "corn":        {"drought": 0.10, "heat": 0.06, "precip": 0.05},
        "sunflower":   {"drought": 0.04, "heat": 0.03, "precip": 0.02},
        "soybean":     {"drought": 0.10, "heat": 0.07, "precip": 0.05},
        "rapeseed":    {"drought": 0.05, "heat": 0.10, "precip": 0.04},
        "barley":      {"drought": 0.04, "heat": 0.05, "precip": 0.03},
        "rye":         {"drought": 0.03, "heat": 0.04, "precip": 0.02},
        "oats":        {"drought": 0.05, "heat": 0.06, "precip": 0.03},
        "buckwheat":   {"drought": 0.05, "heat": 0.06, "precip": 0.04},
        "peas":        {"drought": 0.08, "heat": 0.07, "precip": 0.04},
        "sugar_beet":  {"drought": 0.12, "heat": 0.05, "precip": 0.06},
        "potato":      {"drought": 0.10, "heat": 0.08, "precip": 0.06},
        "corn_silage": {"drought": 0.08, "heat": 0.05, "precip": 0.05},
    }
    sens = sens_map.get(crop, sens_default)

    def _z(var: str) -> float:
        b = baselines.get(var) or {"mean": 0, "std": 1}
        v = weather.get(var, 0)
        return max(-2.0, min(2.0, (v - b["mean"]) / max(b["std"], 1e-3)))

    factor = (
        1.0
        + sens["precip"] * _z("precip")
        - sens["drought"] * _z("drought")
        - sens["heat"] * _z("heat")
    )
    return max(0.70, min(1.30, factor))


def _oblast_weather_baselines(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Per-iso mean+std for each weather variable across years already in parquet."""
    import statistics

    var_map = {
        "precip": "precip_sum_apr_jul",
        "temp": "temp_mean_apr_jul",
        "heat": "heat_stress_days",
        "drought": "drought_dryspells",
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for iso, grp in df.groupby("iso_3166_2"):
        # Dedupe to (year) since per-crop rows share weather cols.
        unique = grp.drop_duplicates(subset=["year"])
        out[iso] = {}
        for short, col in var_map.items():
            vals = unique[col].dropna().astype(float).tolist()
            # Filter out the rate-limit "0,0" sentinels.
            vals = [v for v in vals if not (short in ("precip", "temp") and v == 0)]
            if len(vals) >= 2:
                out[iso][short] = {
                    "mean": float(statistics.mean(vals)),
                    "std": float(max(statistics.stdev(vals), 1e-3)),
                }
            else:
                out[iso][short] = {"mean": vals[0] if vals else 0.0, "std": 1.0}
    return out


async def main() -> int:
    if not S2_PARQUET.exists():
        log.error("Missing S2 parquet — run scripts/collect_oblast_s2.py --year 2025")
        return 1
    if not PARQUET.exists():
        log.error("Missing training_set_v3.parquet — nothing to augment.")
        return 1

    # ─── Load existing parquet (the donor + skip-existing source) ────────
    df = pd.read_parquet(PARQUET)
    log.info("Existing parquet: %d rows; years %s",
             len(df), sorted(df["year"].astype(int).unique().tolist()))
    existing_2025 = set(
        zip(
            df.loc[df["year"] == YEAR, "iso_3166_2"],
            df.loc[df["year"] == YEAR, "crop"],
        )
    )
    if existing_2025:
        log.info(
            "Found %d existing %d rows — those (iso, crop) pairs will be skipped.",
            len(existing_2025), YEAR,
        )

    # ─── Load S2 + per-(iso, 2025) aggregates ───────────────────────────
    s2 = pd.read_parquet(S2_PARQUET)
    s2_2025 = s2[s2["year"] == YEAR].copy()
    if s2_2025.empty:
        log.error("No 2025 rows in S2 parquet — re-run collect_oblast_s2.py --year 2025")
        return 1
    log.info("S2 parquet has %d 2025 rows across %d (oblast, sample) buckets",
             len(s2_2025),
             s2_2025[["oblast", "sample_idx"]].drop_duplicates().shape[0])

    # Resolve oblast → ISO so the aggregation key matches everything else.
    s2_2025["iso_3166_2"] = s2_2025["oblast"].apply(iso_from_any_name)
    s2_2025 = s2_2025.dropna(subset=["iso_3166_2"])

    veg_by_iso: dict[str, dict] = {}
    for iso, grp in s2_2025.groupby("iso_3166_2"):
        agg = aggregate_oblast_year(grp)
        if agg:
            veg_by_iso[iso] = agg
    log.info("Computed S2 aggregates for %d oblasts (year=%d)", len(veg_by_iso), YEAR)

    # ─── Load oblast centroids from samples geojson ─────────────────────
    samples = gpd.read_file(SAMPLES)
    samples["iso_3166_2"] = samples["oblast"].apply(iso_from_any_name)
    samples = samples.dropna(subset=["iso_3166_2"])
    centroids = (
        samples.assign(lat=samples.geometry.centroid.y, lon=samples.geometry.centroid.x)
        .groupby("iso_3166_2")[["lat", "lon"]]
        .mean()
        .to_dict("index")
    )

    # ─── Fetch Open-Meteo for each (iso, 2025) ──────────────────────────
    weather_input = [
        {"centroid_lat": float(centroids[iso]["lat"]),
         "centroid_lon": float(centroids[iso]["lon"]),
         "year": YEAR}
        for iso in veg_by_iso
    ]
    log.info("Fetching Open-Meteo weather for %d (centroid, year=%d) keys...",
             len(weather_input), YEAR)
    weather_map, daily_map = await fetch_weather_for_rows(weather_input)
    log.info("Got weather for %d keys", len(weather_map))

    weather_by_iso: dict[str, dict] = {}
    for iso in veg_by_iso:
        c = centroids[iso]
        ws = weather_map.get(
            (float(c["lat"]), float(c["lon"]), YEAR),
        )
        if ws is None:
            log.warning("Open-Meteo returned no data for iso=%s — using zeros", iso)
            weather_by_iso[iso] = {
                "precip_sum_apr_jul": 0.0,
                "temp_mean_apr_jul": 0.0,
                "heat_stress_days": 0,
                "drought_dryspells": 0,
                # Phase-A round-3 zeros
                "sm_jun_jul_mean": 0.0,
                "sm_drydown_days": 0,
                "winter_kill_days": 0,
            }
        else:
            weather_by_iso[iso] = {
                "precip_sum_apr_jul": ws.precip_sum_apr_jul,
                "temp_mean_apr_jul": ws.temp_mean_apr_jul,
                "heat_stress_days": ws.heat_stress_days,
                "drought_dryspells": ws.drought_dryspells,
                # Phase-A round-3 additions
                "sm_jun_jul_mean": ws.sm_jun_jul_mean,
                "sm_drydown_days": ws.sm_drydown_days,
                "winter_kill_days": ws.winter_kill_days,
            }

    # ─── Donor rows for soil + zone + crop-static cols (per iso) ──────
    donor_cols = [
        "bdod", "cec", "clay", "phh2o", "sand", "silt", "soc",
        "zone_polissia", "zone_forest_steppe", "zone_steppe_north",
        "zone_steppe_south", "zone_transcarpathia",
        "oblast", "oblast_uk", "zone",
        "centroid_lat", "centroid_lon",
    ]
    donor_by_iso: dict[str, pd.Series] = {}
    for iso, grp in df.groupby("iso_3166_2"):
        # Prefer rows from real-yield years (have all soil cols populated).
        # Any year donor works — soil + zone + centroid are time-invariant.
        donor_by_iso[iso] = grp.iloc[0]

    # ─── Real Держстат yields for 2025 ──────────────────────────────────
    cfg = yaml.safe_load(YIELD_YAML.read_text(encoding="utf-8")) or {}
    real_yields_2025: dict[str, dict[str, float]] = {}
    for crop, by_year in (cfg.get("per_oblast_yield") or {}).items():
        yr_block = by_year.get(YEAR) or {}
        if yr_block:
            real_yields_2025[crop] = {iso: float(v) for iso, v in yr_block.items()}
    log.info(
        "YAML has %d 2025 real-yield triples across %d crops",
        sum(len(v) for v in real_yields_2025.values()), len(real_yields_2025),
    )

    # National yields (for synthesis fallback).
    national: dict[str, dict[int, float]] = cfg.get("national_yield", {})

    # Conflict-zone reductions (mirror build_yield_csv.py).
    conflict_since: dict[str, int] = {
        row["iso"]: row["since"]
        for row in (cfg.get("conflict_exclusions") or [])
        if isinstance(row.get("since"), int) and row.get("iso")
    }

    # ─── Compute weather baselines from existing parquet for synthesis ──
    weather_baselines = _oblast_weather_baselines(df)

    # ─── Compute lag1 lookup from real-yield rows ───────────────────────
    real = df[df["is_real_yield"].astype(bool)]
    lag_lookup = real.groupby(["iso_3166_2", "crop", "year"])["yield_tha"].mean().to_dict()
    # For 2025 lag1 we look up year-1 = 2024. We don't have 2024 in the
    # parquet, so for every 2025 row the fallback path applies — use
    # crop mean over train-years 2018-2019.
    train_only = real[real["year"].isin((2018, 2019))]
    crop_mean = train_only.groupby("crop")["yield_tha"].mean().to_dict()
    global_mean = float(real["yield_tha"].mean())

    def _lag_n(iso: str, crop: str, n: int) -> float:
        """Same fallback rule as `patch_parquet_oblast_yield_lag.py`
        — use (iso, crop, YEAR-n) when present, else crop-train-mean."""
        key = (iso, crop, YEAR - n)
        if key in lag_lookup:
            return float(lag_lookup[key])
        return float(crop_mean.get(crop, global_mean))

    def _lag_features(iso: str, crop: str) -> dict[str, float]:
        """4-feature lag block for the new 2025 rows. Since we have no
        2024 data in the parquet (Держстат hasn't published it yet),
        every 2025 row's lag1 falls back to the crop-train-mean. lag2
        pulls 2023 (also unavailable → fallback), lag3 pulls 2022 (also
        unavailable → fallback). Therefore all four features are the
        same constant per crop until 2024 data arrives — that's the
        documented fallback path, not a bug."""
        l1 = _lag_n(iso, crop, 1)
        l2 = _lag_n(iso, crop, 2)
        l3 = _lag_n(iso, crop, 3)
        return {
            "oblast_yield_lag1": l1,
            "oblast_yield_lag2": l2,
            "oblast_yield_lag3": l3,
            "oblast_yield_lag_mean3": (l1 + l2 + l3) / 3.0,
        }

    # ─── Compose new rows ───────────────────────────────────────────────
    new_rows: list[dict] = []
    n_real = n_syn = n_skipped_infeasible = 0
    for iso, veg in veg_by_iso.items():
        donor = donor_by_iso.get(iso)
        if donor is None:
            log.warning("No donor row for iso=%s — skipping entire oblast", iso)
            continue
        weather = weather_by_iso[iso]
        ref = oblast_by_iso(iso)
        zone_name = ref.zone if ref else None
        in_conflict = (
            iso in conflict_since and YEAR >= conflict_since[iso]
        )

        for crop in ALL_CROPS:
            if (iso, crop) in existing_2025:
                continue
            real_val = real_yields_2025.get(crop, {}).get(iso)
            mult = yield_multiplier(crop, zone_name) if zone_name else None

            if real_val is not None:
                yield_val = real_val
                is_real = True
                weather_used = False
                weather_factor = 1.0
                n_real += 1
            elif mult is None:
                # No real datum AND zone-infeasible — don't fabricate.
                n_skipped_infeasible += 1
                continue
            else:
                # Synthesise from national × zone × weather.
                nat = (national.get(crop) or {}).get(YEAR)
                if nat is None:
                    # 2025 absent in national yield table (likely — XLS partial).
                    # Fall back to 2021 national value as a stand-in.
                    nat = (national.get(crop) or {}).get(2021)
                    if nat is None:
                        n_skipped_infeasible += 1
                        continue
                bls = weather_baselines.get(iso, {})
                wf = _weather_factor(crop, weather, bls)
                # Use slug (lowercased name_en) — same as build_yield_csv.py.
                slug = (ref.name_en if ref else iso).lower().replace(" ", "_")
                noise = _deterministic_noise(NOISE_SEED, crop, YEAR, slug, NOISE_AMPLITUDE)
                conflict_adj = CONFLICT_FACTOR if in_conflict else 1.0
                yield_val = nat * mult * wf * conflict_adj * (1.0 + noise)
                is_real = False
                weather_used = True
                weather_factor = wf
                n_syn += 1

            # Compose the row.
            row: dict = {}
            # Shared cols from donor (soil, zone one-hots, oblast name, centroid).
            for col in donor_cols:
                row[col] = donor[col]
            # Vegetation features from S2.
            for k, v in veg.items():
                row[k] = v
            # Weather features.
            for k, v in weather.items():
                row[k] = v
            # Crop features.
            row.update(_compute_crop_features(crop, weather, veg))
            row["crop_flowering_in_aprjul"] = bool(flowering_in_reference(crop))
            # Lag features (4): lag1 + lag2 + lag3 + lag_mean3.
            row.update(_lag_features(iso, crop))
            # Phase-A round-3: BBCH-phase weather features per crop.
            # Pulls the daily weather list cached during the
            # Open-Meteo fetch above and runs the same arithmetic
            # as the inference-path helper.
            row.update(_bbch_features_for_2025(iso, crop, daily_map, centroids))
            # Identity + label.
            row["iso_3166_2"] = iso
            row["year"] = YEAR
            row["crop"] = crop
            row["yield_tha"] = round(yield_val, 2)
            row["is_real_yield"] = is_real
            row["weather_used"] = weather_used
            row["weather_factor"] = round(weather_factor, 4)
            row["conflict_zone"] = in_conflict
            row["features_origin"] = "sentinel_hub_v3"
            new_rows.append(row)

    log.info(
        "Composed: real=%d synthetic=%d skipped_infeasible=%d (total new=%d)",
        n_real, n_syn, n_skipped_infeasible, len(new_rows),
    )
    if not new_rows:
        log.info("Nothing new to add — exiting.")
        return 0

    new_df = pd.DataFrame(new_rows)
    # Align dtypes with existing parquet to avoid object-dtype contamination.
    for col in new_df.columns:
        if col in df.columns:
            try:
                new_df[col] = new_df[col].astype(df[col].dtype)
            except (TypeError, ValueError):
                pass

    out = pd.concat([df, new_df], ignore_index=True)
    out.to_parquet(PARQUET, index=False)

    by_crop = new_df.groupby("crop").size().to_dict()
    log.info("Per-crop %d rows added: %s", YEAR, by_crop)
    log.info(
        "Parquet now: %d total rows (was %d), real-yield rows: %d (was %d)",
        len(out), len(df),
        int(out["is_real_yield"].sum()),
        int(df["is_real_yield"].sum()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
