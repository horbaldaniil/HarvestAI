"""Model registry — loads joblib files at FastAPI startup.

We hold one SHAP TreeExplainer per crop so per-prediction SHAP values are
cheap (~5 ms per call vs hundreds of ms if we built the explainer each time).
Registry is a per-process singleton: it's fine to share across requests
because XGBoost predict + SHAP explain are read-only and thread-safe.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import shap

from app.config import settings
from app.db.models.enums import CropType

log = logging.getLogger(__name__)


class ModelRegistry:
    """Lazy-init singleton holding loaded models keyed by crop."""

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._explainers = {}
            cls._instance._seasonal_norms = None
            cls._instance._loaded = False
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Forget cached models — used by tests to swap fixtures."""
        cls._instance = None

    def load_all(self) -> None:
        """Discover and load every yield_xgb_{crop}_v1.joblib in models/.

        Called once from FastAPI lifespan startup.
        """
        if self._loaded:
            return
        models_dir = Path(settings.models_dir)
        loaded = 0
        for crop in CropType:
            joblib_path = models_dir / f"yield_xgb_{crop.value}_v1.joblib"
            if not joblib_path.exists():
                log.warning("Model file missing: %s — predictions for %s disabled",
                            joblib_path.name, crop.value)
                continue
            payload = joblib.load(joblib_path)
            self._models[crop] = payload
            try:
                self._explainers[crop] = shap.TreeExplainer(payload["model"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not build SHAP explainer for %s: %s", crop.value, exc)
            loaded += 1

        # Seasonal NDVI norms for the anomaly detector.
        norms_path = Path("data/processed/seasonal_norms.json")
        if norms_path.exists():
            with norms_path.open(encoding="utf-8") as f:
                self._seasonal_norms = json.load(f)
        else:
            log.warning("seasonal_norms.json missing — anomaly detection limited")
            self._seasonal_norms = {}

        self._loaded = True
        log.info("ModelRegistry: loaded %d yield models + seasonal norms", loaded)

    def get_yield_model(self, crop: CropType) -> dict[str, Any] | None:
        """Returns the loaded {model, features, crop, version} dict or None."""
        return self._models.get(crop)

    def get_shap_explainer(self, crop: CropType) -> shap.TreeExplainer | None:
        return self._explainers.get(crop)

    @property
    def seasonal_norms(self) -> dict:
        return self._seasonal_norms or {}

    def is_available(self, crop: CropType) -> bool:
        return crop in self._models


def get_registry() -> ModelRegistry:
    return ModelRegistry()
