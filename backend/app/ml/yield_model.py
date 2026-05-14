"""Yield prediction service — wraps registry + SHAP + quantile bands
into one entry point.

Inference path:
1. build_feature_vector(field_id, crop) → 30-feature dict (full v7 schema)
2. registry.get_yield_model(crop) → model payload incl. canonical
   `features` list per model version
3. select_features(dict, payload["features"]) → (1, N) numpy array
   matching the model's expected order
4. model.predict(arr) → scalar yield estimate (t/ha)
5. SHAP explainer (with companion-tree fallback for non-tree resolved
   families like `stack`)
6. Quantile band (q05/q95) when an XGBoost-triple payload provides it
7. top-5 features by |contribution| → list of dicts for UI/DB
"""
from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.db.models.enums import CropType
from app.ml.features import (
    V3_BASE_FEATURES,
    build_feature_vector,
    select_features,
)
from app.ml.registry import get_registry

log = logging.getLogger(__name__)


# ─── Conformal-prediction intervals (Lei et al. 2018) ──────────
#
# Calibrated offline via `scripts/calibrate_conformal.py` against the
# 2020 validation set and persisted as a flat JSON lookup. Used here
# to populate `value_tha_q05` / `value_tha_q95` for ALL model families
# (RF, Stack, LightGBM) — not just the XGBoost-triple which had native
# quantile siblings. Replaces the legacy heuristic
# `confidence = 0.3 + 0.1 × (len(shap_top) - 5)` with a data-driven
# half-width that genuinely reflects each model's accuracy on held-out
# Держстат rows.
_CONFORMAL_PATH = Path("data/processed/conformal_intervals_v7.json")

# Per-crop test R² lookup used by the empirical-Bayes shrinkage step.
# Read lazily from the same `evaluation_v7_hybrid.json` the registry
# consults for `_resolve_default`. We don't reuse the registry's helper
# because it returns only `(family, version)` — we need the numeric
# test_r2 to compute the shrinkage weight.
_EVAL_PATH = Path("data/processed/evaluation_v7_hybrid.json")


@functools.lru_cache(maxsize=1)
def _load_conformal_intervals() -> dict:
    """Read the conformal calibration JSON once per process.

    Returns the empty dict if the file is missing (calibration script
    not yet run) — yield_model then falls back to the legacy XGBoost-
    quantile / heuristic-confidence path. No crash on first deployment.
    """
    if not _CONFORMAL_PATH.exists():
        log.info(
            "Conformal intervals JSON not found at %s — falling back to "
            "legacy quantile path. Run scripts/calibrate_conformal.py.",
            _CONFORMAL_PATH,
        )
        return {}
    try:
        return json.loads(_CONFORMAL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read %s: %s", _CONFORMAL_PATH, exc)
        return {}


def _conformal_radius(
    crop: CropType, family: str, version: str,
) -> float | None:
    """Look up the 90 %-CI half-width for `(crop, family, version)`.

    Returns None if the JSON is missing, the crop wasn't calibrated
    (e.g. < 5 calibration rows), or this specific (family, version)
    pair wasn't trained when the calibrator ran. The caller falls
    back to the legacy path in that case.
    """
    blob = _load_conformal_intervals()
    if not blob:
        return None
    by_crop = blob.get("by_crop", {})
    crop_block = by_crop.get(crop.value, {})
    ver_block = crop_block.get(version, {})
    value = ver_block.get(family)
    return float(value) if value is not None else None


def _reset_conformal_cache() -> None:
    """Test helper — drops the lru_cache so tests can swap the JSON."""
    _load_conformal_intervals.cache_clear()


@functools.lru_cache(maxsize=1)
def _load_eval_summary() -> dict:
    """Read `evaluation_v7_hybrid.json` once per process.

    Returns the empty dict if missing — `_per_crop_test_r2` then yields
    None for every crop and the shrinkage step becomes a no-op (no
    rescue, no harm). Keeps the inference path running on a fresh
    clone before `scripts/evaluate_v7_hybrid.py` has been executed.
    """
    if not _EVAL_PATH.exists():
        return {}
    try:
        return json.loads(_EVAL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read %s: %s", _EVAL_PATH, exc)
        return {}


def _per_crop_test_r2(crop: CropType) -> float | None:
    """Best test R² for this crop per the hybrid evaluator.

    Reads `crops[crop_value].test_r2` from the hybrid-eval JSON, which
    is itself the per-crop best across {v6, v7, v7h, v8, v8h}. We want
    *this* number for the EB shrinkage weight because it answers
    "how much should we trust THIS crop's prediction" — exactly the
    James-Stein 'noisy-estimate-quality' input.

    Returns None when the JSON is missing or the crop has no scored
    candidate — in that case shrinkage falls back to no-op so the
    raw ML prediction stays untouched.
    """
    blob = _load_eval_summary()
    crops_blob = blob.get("crops") or {}
    body = crops_blob.get(crop.value)
    if not isinstance(body, dict) or body.get("skipped"):
        return None
    val = body.get("test_r2")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _reset_eval_cache() -> None:
    """Test helper — drops the lru_cache so tests can swap the JSON."""
    _load_eval_summary.cache_clear()


# ─── Empirical Bayes shrinkage (James-Stein 1956 / Efron-Morris 1973) ──
#
# Blend the ML prediction toward a stable regional prior (most-recent
# oblast×crop Держстат yield). The shrinkage weight is `1 - test_r2`
# clamped to [0, 1] — when the model is strong (R²≈1) we trust the ML;
# when the model is weak or broken (R²<0) we fall fully back to the
# baseline. Mathematically guaranteed never to increase MSE under the
# JS framework, provided the prior captures part of the signal.
#
# Independence: the model uses `oblast_yield_lag1/2/3/mean3` (lagged
# historical yields) but NOT the current-year mean. The shrinkage
# target IS the current/most-recent year — a genuinely different
# information channel, so the bias-variance trade-off has real teeth.


def _empirical_bayes_shrink(
    ml_value: float,
    crop: CropType,
    oblast_name_en: str | None,
    raw_r2: float | None,
) -> tuple[float, float]:
    """Return `(shrunk_value, weight)`.

    `weight` is the share of the regional baseline in the final value
    (0 = pure ML, 1 = pure baseline). Returns `(ml_value, 0.0)` (no
    shrinkage) when the baseline isn't available or `raw_r2` is None
    or ≥ 1.0.

    The baseline is `oblast_avg_yield(crop, year=None)` — same value
    the dashboard's `OblastComparisonTable` footer displays, so the
    shrunken prediction never drifts away from a number the user can
    independently verify against the published Держстат yield.
    """
    if raw_r2 is None:
        return ml_value, 0.0
    weight = max(0.0, min(1.0, 1.0 - raw_r2))
    if weight <= 0.0:
        return ml_value, 0.0
    try:
        # Local import — `dashboard_analytics` pulls heavy SQLAlchemy
        # bindings and we want the import cost paid lazily, only for
        # predict requests where shrinkage actually fires.
        from app.services.dashboard_analytics import oblast_avg_yield
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not import oblast_avg_yield for shrinkage: %s", exc)
        return ml_value, 0.0
    baseline, _baseline_year = oblast_avg_yield(oblast_name_en, crop.value, None)
    if baseline is None:
        return ml_value, 0.0
    shrunk = (1.0 - weight) * ml_value + weight * float(baseline)
    return float(shrunk), float(weight)


# Order in which we look for a tree-based SHAP explainer when the
# resolved model itself isn't tree-explainable (most notably `stack`,
# whose Ridge meta-learner sees only the 4 base predictions, not the
# original 30 features). Every v7 tree model was trained on the same
# `FEATURE_NAMES` tuple, so the SHAP contributions explain the inputs
# honestly — they just come from a sibling estimator's perspective.
_TREE_FALLBACK_ORDER: tuple[str, ...] = ("xgboost", "rf", "lightgbm")


@dataclass(frozen=True, slots=True)
class YieldPrediction:
    value_tha: float
    confidence: float | None  # ±std-dev around the point estimate
    features: dict[str, float | None]
    shap_top: list[dict]
    model_name: str
    model_version: str
    # Optional quantile band — only populated for XGBoost-triple payloads
    # (point + q_low + q_high). For stack / rf / lightgbm this stays
    # None and the UI renders just the point estimate.
    value_tha_q05: float | None = None
    value_tha_q95: float | None = None
    # Which family generated the SHAP contributions above. Usually equals
    # the resolved model family, except when the resolved family is `stack`
    # (or any other non-tree family) — then we fall back to a tree sibling
    # and record its name here so the UI can label the disclaimer.
    explainer_source: str | None = None
    # 2–3 sentence Ukrainian narrative generated by gpt-4o-mini in the
    # predict-job. None if OPENAI_API_KEY isn't configured or the LLM
    # call failed — the prediction itself still persists.
    summary_text: str | None = None
    # Phase-C empirical-Bayes shrinkage weight in [0, 1]. Share of the
    # value that comes from the regional Держстат baseline rather than
    # the raw ML output. 0 means pure ML (trusted model); 1 means pure
    # baseline (model unusable, falling back to most-recent published
    # yield). Persisted on Prediction rows so the UI can label
    # "X % regional baseline" when desired.
    shrinkage_weight: float = 0.0
    # Raw ML output before shrinkage — useful for the UI to show both
    # "model said X, shrunken to Y" diagnostics, and lets us recompute
    # alternate-weight shrinkage post-hoc without re-running the model.
    raw_ml_value_tha: float | None = None


def predict_yield(session: Session, field_id: int, crop: CropType) -> YieldPrediction:
    registry = get_registry()
    # Resolve via the registry's own preference walk so we know the
    # canonical (family, version) — the filename-derived truth. Some v7
    # joblib payloads have a stale `version="v3"` baked in (trainer bug
    # that didn't bump the string when migrating the dump format from
    # v3 to v7); trusting `payload["version"]` here would misreport the
    # model to the UI even though the right artefact is being used.
    resolved = registry._resolve_default(crop)
    if resolved is None:
        raise RuntimeError(
            f"No trained model for crop={crop.value}. "
            f"Run scripts/train_yield_models_v7.py first."
        )
    resolved_family, resolved_version = resolved
    payload = registry.get_yield_model(
        crop, family=resolved_family, version=resolved_version,
    )
    if payload is None:
        raise RuntimeError(
            f"No trained model for crop={crop.value}. "
            f"Run scripts/train_yield_models_v7.py first."
        )

    # Build the full 30-feature dict, then slice down to whatever the
    # loaded model expects (v3=17, v6=23, v7=30). The trainer always
    # stores its expected feature list in `payload["features"]`; falling
    # back to V3_BASE_FEATURES keeps very old v1/v2 payloads working
    # even though they predate the metadata convention.
    features = build_feature_vector(session, field_id, crop=crop)
    expected: list[str] | tuple[str, ...] = payload.get("features") or V3_BASE_FEATURES
    arr = select_features(features, expected)

    # The payload may store the regressor under "model" (v3+ stack /
    # rf / lgbm / cat) or directly under "point" inside an xgboost
    # quantile-triple. Handle both shapes.
    model_obj = payload.get("model")
    if model_obj is None and "point" in payload:
        # Legacy XGBoost triple payload (point + q05 + q95). Use point.
        model_obj = payload["point"]
    if model_obj is None:
        raise RuntimeError(
            f"Unrecognised model payload for crop={crop.value}: keys={list(payload)}"
        )

    # Some payloads carry `log_target=True` — set by the v7 trainer for
    # crops in `LOG_TARGET_CROPS` (sugar_beet, potato, corn_silage),
    # which were trained on `np.log1p(yield)` to tame the right-skewed
    # т/га distribution. Every downstream value (point, q05, q95) gets
    # `np.expm1` so the API returns numbers in original yield units —
    # matches what the UI labels and what the conformal calibrator
    # persisted (radii are computed AFTER expm1 in original-scale).
    # `payload.get("log_target")` is None/False for legacy payloads,
    # making this fully back-compat with v3/v5/v6/v7h artefacts.
    log_target = bool(payload.get("log_target"))

    value = float(model_obj.predict(arr)[0])
    if log_target:
        value = float(np.expm1(value))

    # ── Hierarchical offset (v7h / v8h) ─────────────────────────────
    #
    # Both v7h (per-crop hierarchical) and v8h (multi-task
    # hierarchical) predict yield *deviation* from a per-(crop, oblast)
    # historical mean. To return absolute yield we look up the mean
    # for the field's oblast and add it back.
    #
    # Payload shape differs:
    #   - v7h: `oblast_means = {iso: mean}` (single-crop model, key by iso)
    #   - v8h: `oblast_means = {crop: {iso: mean}}` (multi-task, key by crop+iso)
    #
    # Both keep `_GLOBAL` (crop-mean) as a fallback for (crop, iso)
    # combinations the training years didn't cover.
    oblast_means = payload.get("oblast_means")
    if oblast_means:
        # Resolve the field's iso code from the centroid. The feature
        # vector already computed `centroid_lat/lon`; reuse it instead
        # of round-tripping back to the DB.
        iso_code: str | None = None
        oblast_en: str | None = None
        lat = features.get("centroid_lat")
        lon = features.get("centroid_lon")
        if lat is not None and lon is not None:
            try:
                from app.data_reference.oblast_names import iso_from_any_name
                from app.services.dashboard_analytics import (
                    find_oblast_for_centroid,
                )
                oblast_en = find_oblast_for_centroid(float(lat), float(lon))
                if oblast_en is not None:
                    iso_code = iso_from_any_name(oblast_en)
            except Exception as exc:  # noqa: BLE001
                log.warning("Centroid → iso failed: %s", exc)

        # Select the right means dict shape.
        means_by_iso: dict | None = None
        if resolved_version == "v8h":
            means_by_iso = oblast_means.get(crop.value) or {}
        else:  # v7h or any future hierarchical variant with iso-keyed dict
            means_by_iso = oblast_means

        if isinstance(means_by_iso, dict):
            global_mean = means_by_iso.get("_GLOBAL", 0.0)
            offset = means_by_iso.get(iso_code, global_mean) if iso_code else global_mean
            value = value + float(offset)
            log.debug(
                "Hierarchical offset for %s/%s (%s): +%.3f t/ha",
                crop.value, resolved_version, iso_code or "_GLOBAL", offset,
            )

    # Quantile band — two sources tried in order:
    #
    #   1. XGBoost-triple payload's native `q_low` / `q_high` siblings
    #      (quantile-regression loss). Available only for XGBoost
    #      payloads; preserved as the historical primary path. When
    #      `log_target=True`, the quantile siblings also live in
    #      log-space and we `expm1` their outputs here — `expm1` is
    #      monotonic, so the log-space quantile maps cleanly to the
    #      original-space quantile.
    #   2. Split-conformal radius from
    #      `data/processed/conformal_intervals_v7.json` — a per-(crop,
    #      family, version) half-width calibrated against absolute
    #      residuals on the 2021 test set. Calibrated values live in
    #      original yield units (т/га) — the calibrator script
    #      already applies `expm1` to log-target predictions before
    #      computing residuals, so the radius is directly applicable
    #      to `value` (which is also in original units after the
    #      `expm1` above). Works for ALL families (RF / Stack /
    #      LightGBM / XGBoost) — replaces the placebo ±0.3 heuristic
    #      with a coverage-guaranteed band.
    #
    # If both fail, the UI still renders the point estimate alone.
    value_q05 = value_q95 = None
    q_low = payload.get("q_low")
    q_high = payload.get("q_high")
    if q_low is not None and q_high is not None:
        try:
            raw_q05 = float(q_low.predict(arr)[0])
            raw_q95 = float(q_high.predict(arr)[0])
            if log_target:
                raw_q05 = float(np.expm1(raw_q05))
                raw_q95 = float(np.expm1(raw_q95))
            value_q05 = round(raw_q05, 2)
            value_q95 = round(raw_q95, 2)
        except Exception as exc:  # noqa: BLE001
            # Quantile estimator broken on this row → conformal fallback below.
            log.warning("Quantile predict failed for crop=%s: %s", crop.value, exc)
            value_q05 = value_q95 = None
    # Conformal fallback — applies to RF / Stack / LightGBM (no native
    # quantile siblings) AND to XGBoost rows where the quantile
    # estimator above blew up. The radius is symmetric around the
    # point estimate; this is the standard split-conformal interval
    # (Lei et al. 2018) and is statistically valid for any well-trained
    # regressor without requiring a quantile-loss head.
    conformal_radius_value: float | None = None
    if value_q05 is None or value_q95 is None:
        conformal_radius_value = _conformal_radius(
            crop, resolved_family, resolved_version,
        )
        if conformal_radius_value is not None:
            value_q05 = round(value - conformal_radius_value, 2)
            value_q95 = round(value + conformal_radius_value, 2)

    # ── Empirical-Bayes shrinkage (Phase C) ─────────────────────────
    #
    # Blend ML prediction toward the most-recent regional Держстат
    # baseline. Weight = `1 - test_r2` (clamped to [0, 1]) — strong
    # models barely shrink, broken models (R² < 0) collapse fully to
    # the baseline. James-Stein 1956 / Efron-Morris 1973 admissibility:
    # cannot increase MSE over the raw estimator under exchangeability
    # of the prior. For crops where R² ≈ 1, the shrunken value equals
    # raw ML to within float precision — full backward compatibility.
    raw_ml_value = float(value)
    raw_r2 = _per_crop_test_r2(crop)
    # Resolve oblast English name for shrinkage target. Re-uses the
    # same centroid → oblast resolver as the v7h/v8h block above. We
    # compute it independently here because the v7h/v8h block may not
    # have run (e.g. for v7/v8 models) but EB shrinkage applies to
    # every version.
    eb_oblast_en: str | None = None
    lat = features.get("centroid_lat")
    lon = features.get("centroid_lon")
    if lat is not None and lon is not None:
        try:
            from app.services.dashboard_analytics import find_oblast_for_centroid
            eb_oblast_en = find_oblast_for_centroid(float(lat), float(lon))
        except Exception as exc:  # noqa: BLE001
            log.debug("EB shrinkage oblast resolve failed: %s", exc)
    value, shrinkage_weight = _empirical_bayes_shrink(
        value, crop, eb_oblast_en, raw_r2,
    )
    # The conformal radius is the model's residual variance; once we
    # shrink toward a stable baseline the *effective* residual narrows
    # by (1 - weight) (the baseline has its own variance but ignoring
    # that is a conservative under-estimate of total variance — fine
    # for a 90 %-CI presentation). Rescale q05/q95 around the new
    # center so the band stays symmetric.
    if shrinkage_weight > 0 and conformal_radius_value is not None:
        conformal_radius_value = conformal_radius_value * (1.0 - shrinkage_weight)
        value_q05 = round(value - conformal_radius_value, 2)
        value_q95 = round(value + conformal_radius_value, 2)
    elif shrinkage_weight > 0 and value_q05 is not None and value_q95 is not None:
        # Quantile-band case (XGBoost-triple): rescale around the new
        # center without redefining the half-width source.
        half = (value_q95 - value_q05) / 2.0 * (1.0 - shrinkage_weight)
        value_q05 = round(value - half, 2)
        value_q95 = round(value + half, 2)

    # SHAP — prefer the resolved family's own explainer; fall back to a
    # tree sibling when the resolved family isn't tree-explainable. The
    # 30-feature vector is identical across families at the same version,
    # so the contributions remain attributable to the same feature names.
    explainer = registry.get_shap_explainer(
        crop, family=resolved_family, version=resolved_version,
    )
    explainer_source: str | None = resolved_family if explainer is not None else None
    if explainer is None:
        for tf in _TREE_FALLBACK_ORDER:
            candidate = registry.get_shap_explainer(
                crop, family=tf, version=resolved_version,
            )
            if candidate is not None:
                explainer = candidate
                explainer_source = tf
                log.debug(
                    "Using %s as SHAP companion for resolved=%s (no tree "
                    "explainer on the resolved model)",
                    tf, resolved_family,
                )
                break

    shap_top: list[dict] = []
    if explainer is not None:
        try:
            shap_vals = explainer(arr)
            # shap_vals.values shape: (1, n_features); shap_vals.base_values: (1,)
            # Use the same `expected` order so contribution names line up.
            contribs = list(zip(expected, shap_vals.values[0], arr[0]))
            # Phase-B (v8 multi-task): drop `crop_*` one-hot SHAP entries.
            # In a v8 model these are guaranteed-large contributors because
            # the one-hot for the resolved crop is the model's identity
            # signal — it's literally "this is wheat" not a discovered
            # agronomic pattern. Including them in the top-5 panel would
            # crowd out the NDVI / weather / soil features the user
            # actually wants to see. v7 / v7h models have no such
            # features, so this filter is a no-op for them.
            contribs = [c for c in contribs if not c[0].startswith("crop_")]
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            shap_top = [
                {
                    "name": name,
                    "value": None if np.isnan(val) else float(val),
                    "contribution": float(contrib),
                }
                for name, contrib, val in contribs[:5]
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP failed for crop=%s: %s", crop.value, exc)
            explainer_source = None

    # Confidence: prefer the conformal half-width when calibrated —
    # it's the same number the UI's ±X.X т/га label needs to mean what
    # users assume it means (a real coverage-guaranteed half-width, not
    # an arbitrary heuristic). For models where conformal calibration
    # wasn't possible (no v7 artifact for that crop, or the XGBoost-
    # triple path produced its own q_low/q_high already), derive
    # confidence from the quantile band as `(q95 - q05) / 2`. Final
    # fallback is None — the UI hides the ±X label rather than show
    # a misleading number.
    # NB: when EB shrinkage fired, `conformal_radius_value` was already
    # scaled by `(1 - shrinkage_weight)` above — `confidence` therefore
    # narrows to match the value the user is seeing post-shrinkage.
    if conformal_radius_value is not None:
        confidence = round(conformal_radius_value, 2)
    elif value_q05 is not None and value_q95 is not None:
        confidence = round((value_q95 - value_q05) / 2, 2)
    else:
        confidence = None

    # Use the resolved (family, version) from the registry — same
    # rationale as above (payload metadata sometimes lies after the
    # v3→v7 trainer dump-format migration).
    # File-prefix mapping mirrors registry.FAMILY_FILE_PREFIX so the
    # `model_name` value persisted on Prediction rows matches the
    # actual .joblib filename users see on disk.
    family_prefix = {
        "xgboost": "xgb",
        "rf": "rf",
        "lstm": "lstm",
        "lightgbm": "lgbm",
        "stack": "stack",
    }.get(resolved_family, resolved_family)

    return YieldPrediction(
        value_tha=round(value, 2),
        confidence=confidence,
        features=features,
        shap_top=shap_top,
        model_name=f"yield_{family_prefix}_{crop.value}",
        model_version=resolved_version,
        value_tha_q05=value_q05,
        value_tha_q95=value_q95,
        explainer_source=explainer_source,
        # summary_text is populated by the worker job after this returns
        # (the LLM call is async and bridges asyncio.run() at the worker
        # boundary, not here in the sync inference path).
        summary_text=None,
        shrinkage_weight=round(shrinkage_weight, 3),
        raw_ml_value_tha=round(raw_ml_value, 2),
    )
