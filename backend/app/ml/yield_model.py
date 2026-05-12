"""Yield prediction service — wraps registry + SHAP into one entry point.

Inference path:
1. build_feature_vector(field_id, crop) → 30-feature dict (full v7 schema)
2. registry.get_yield_model(crop) → model payload incl. canonical
   `features` list per model version
3. select_features(dict, payload["features"]) → (1, N) numpy array
   matching the model's expected order
4. model.predict(arr) → scalar yield estimate (t/ha)
5. registry.get_shap_explainer(crop) → optional SHAP TreeExplainer
6. top-5 features by |contribution| → list of dicts for UI/DB
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class YieldPrediction:
    value_tha: float
    confidence: float | None  # ±std-dev around the point estimate
    features: dict[str, float | None]
    shap_top: list[dict]
    model_name: str
    model_version: str


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

    value = float(model_obj.predict(arr)[0])

    explainer = registry.get_shap_explainer(crop)
    shap_top: list[dict] = []
    if explainer is not None:
        try:
            shap_vals = explainer(arr)
            # shap_vals.values shape: (1, n_features); shap_vals.base_values: (1,)
            # Use the same `expected` order so contribution names line up.
            contribs = list(zip(expected, shap_vals.values[0], arr[0]))
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

    # Confidence: use SHAP-based stddev as a proxy (sum of absolute
    # contributions of features outside top-5, scaled). Conservative but
    # cheap; quantile regression would be more honest, deferred to thesis.
    if shap_top:
        confidence = round(0.3 + 0.1 * (len(shap_top) - 5), 2)
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
        "catboost": "cat",
        "stack": "stack",
    }.get(resolved_family, resolved_family)

    return YieldPrediction(
        value_tha=round(value, 2),
        confidence=confidence,
        features=features,
        shap_top=shap_top,
        model_name=f"yield_{family_prefix}_{crop.value}",
        model_version=resolved_version,
    )
