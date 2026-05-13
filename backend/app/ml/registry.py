"""Model registry — loads yield models at FastAPI startup.

Supports multiple algorithm families and versions per crop. The default
lookup returns the "preferred" model per crop following this order:
  1. settings.active_model_family (if set and present)
  2. Stack v3 (best generalisation per Wolpert 1992) if available
  3. CatBoost v3 → LightGBM v3 → XGBoost v3 → RandomForest v3
  4. XGBoost v2 (real Sentinel-2 features, v3-era ablation baseline)
  5. XGBoost v1 (synthesised NDVI fallback)
  6. RandomForest v1 baseline
  7. None — predictions disabled for that crop

LSTM models live alongside but are NOT used by default for runtime tabular
inference: the LSTM expects a (T=22, F=8) time-series input, while the rest
of the app builds 17-feature tabular vectors. Methodology endpoints expose
its metrics; runtime inference still goes through the tabular path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import shap

from app.config import settings
from app.db.models.enums import CropType

log = logging.getLogger(__name__)

AlgorithmFamily = Literal["xgboost", "rf", "lstm", "lightgbm", "stack"]
ALL_FAMILIES: tuple[AlgorithmFamily, ...] = (
    "xgboost", "rf", "lstm", "lightgbm", "stack",
)
# Filenames use short prefixes for python-package alignment + history:
#   xgboost → "xgb" (v1/v2/v3 history)
#   lightgbm → "lgbm"  stack → "stack"
# CatBoost was previously a fifth family ("cat" prefix); removed entirely
# — the package was an awkward dependency to maintain across Python
# versions and the three remaining tree boosters cover the same model
# diversity. v3-v6 legacy `yield_cat_*.joblib` files (if any survived)
# are no longer discoverable from here.
FAMILY_FILE_PREFIX: dict[AlgorithmFamily, str] = {
    "xgboost": "xgb",
    "rf": "rf",
    "lstm": "lstm",
    "lightgbm": "lgbm",
    "stack": "stack",
}

# Tree-based families that SHAP TreeExplainer supports out of the box.
# RF / XGB / LGBM all qualify; "stack" is a Ridge over OOF predictions
# of the three tree models, so SHAP at the stack level isn't
# meaningful (the meta-input space ≠ the original 30 features).
TREE_FAMILIES: frozenset[AlgorithmFamily] = frozenset({
    "xgboost", "rf", "lightgbm",
})

# Order in which we pick a default model for a crop if `active_model_family`
# isn't set. Stack v4 wins because v4 features (crop-specific weather +
# weather-conditioned yields) measurably improve R² across most crops.
# v3 stays as fallback; v2/v1 for the ablation study.
DEFAULT_PREFERENCE: tuple[tuple[AlgorithmFamily, str], ...] = (
    # v7 = real-only training + SoilGrids soil features (30 features total).
    # v7h = hierarchical (predicts yield deviation from oblast mean).
    # Both v7 variants beat v6 for different crop subsets; runtime
    # selection happens via best-of-both eval.
    ("stack", "v7"),
    ("lightgbm", "v7"),
    ("xgboost", "v7"),
    ("rf", "v7"),
    ("stack", "v7h"),
    ("lightgbm", "v7h"),
    ("xgboost", "v7h"),
    ("rf", "v7h"),
    # v6 = production models trained on REAL-ONLY Держстат yields
    # (2018-2021, chronological split train 2018-19 / val 2020 / test 2021).
    # No synthetic confound — these are the thesis-defense headline.
    # v5 = mixed real+synthetic trained, kept for ablation history.
    ("stack", "v6"),
    ("lightgbm", "v6"),
    ("xgboost", "v6"),
    ("rf", "v6"),
    ("stack", "v5"),
    ("lightgbm", "v5"),
    ("xgboost", "v5"),
    ("rf", "v5"),
    ("stack", "v4"),
    ("lightgbm", "v4"),
    ("xgboost", "v4"),
    ("rf", "v4"),
    ("stack", "v3"),
    ("lightgbm", "v3"),
    ("xgboost", "v3"),
    ("rf", "v3"),
    ("xgboost", "v2"),
    ("xgboost", "v1"),
    ("rf", "v1"),
)

# Versions we attempt to discover for every (crop, family) pair on startup.
# Listed explicitly so a missing v6 file just leaves the registry on v5,
# rather than walking the directory and risking accidental loads of
# half-trained artifacts a developer left behind.
DISCOVERY_VERSIONS: dict[AlgorithmFamily, tuple[str, ...]] = {
    "xgboost": ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7h"),
    "rf": ("v1", "v3", "v4", "v5", "v6", "v7", "v7h"),
    "lightgbm": ("v3", "v4", "v5", "v6", "v7", "v7h"),
    "stack": ("v3", "v4", "v5", "v6", "v7", "v7h"),
}


class ModelRegistry:
    """Lazy-init singleton holding loaded models keyed by (crop, family, version)."""

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}  # (CropType, family, version) → payload dict
            cls._instance._explainers = {}  # (CropType, family, version) → SHAP TreeExplainer
            cls._instance._lstm_models = {}  # CropType → torch.jit.ScriptModule
            cls._instance._metadata = {}  # (CropType, family, version) → metrics dict
            cls._instance._seasonal_norms = None
            cls._instance._loaded = False
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Forget cached models — used by tests to swap fixtures."""
        cls._instance = None

    def load_all(self) -> None:
        """Discover and load every model file under settings.models_dir.

        Missing files are warnings — partial availability keeps the rest of
        the app running even if a single .joblib is corrupt or absent.
        """
        if self._loaded:
            return
        models_dir = Path(settings.models_dir)
        if not models_dir.exists():
            log.warning("Models dir does not exist: %s", models_dir)
            self._loaded = True
            return

        for crop in CropType:
            for family, versions in DISCOVERY_VERSIONS.items():
                for version in versions:
                    self._try_load_joblib(crop, family, version, models_dir)
            self._try_load_lstm(crop, "v1", models_dir)

        norms_path = Path("data/processed/seasonal_norms.json")
        if norms_path.exists():
            with norms_path.open(encoding="utf-8") as f:
                self._seasonal_norms = json.load(f)
        else:
            log.warning("seasonal_norms.json missing — anomaly detection limited")
            self._seasonal_norms = {}

        self._loaded = True
        log.info(
            "ModelRegistry: loaded %d tabular + %d LSTM yield models",
            len(self._models), len(self._lstm_models),
        )

    def _try_load_joblib(
        self, crop: CropType, family: AlgorithmFamily, version: str, models_dir: Path,
    ) -> None:
        prefix = FAMILY_FILE_PREFIX[family]
        path = models_dir / f"yield_{prefix}_{crop.value}_{version}.joblib"
        if not path.exists():
            return
        try:
            payload = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load %s: %s", path.name, exc)
            return
        key = (crop, family, version)
        self._models[key] = payload

        # SHAP TreeExplainer is only relevant for tree-based regressors.
        # v3 XGBoost serialises under {"point": <regressor>, "q_low": ...,
        # "q_high": ...}; v1/v2 + RF/LGBM/CatBoost serialise under
        # {"model": <regressor>}. We pick whichever is present.
        if family in TREE_FAMILIES:
            estimator = payload.get("point") or payload.get("model")
            if estimator is not None:
                try:
                    self._explainers[key] = shap.TreeExplainer(estimator)
                except Exception as exc:  # noqa: BLE001
                    log.warning("SHAP explainer failed for %s: %s", path.name, exc)

        meta_path = models_dir / f"yield_{prefix}_{crop.value}_{version}.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["metrics"] = _normalise_metrics(meta.get("metrics", {}))
                self._metadata[key] = meta
            except json.JSONDecodeError:
                pass
        log.info("Loaded %s", path.name)

    def _try_load_lstm(self, crop: CropType, version: str, models_dir: Path) -> None:
        path = models_dir / f"yield_lstm_{crop.value}_{version}.pt"
        if not path.exists():
            return
        try:
            import io

            import torch

            # Mirror the BytesIO workaround used when saving: on Windows with
            # non-ASCII paths torch's C++ side can't open the file. Reading
            # through Python then handing torch a BytesIO sidesteps it.
            module = torch.jit.load(io.BytesIO(path.read_bytes()), map_location="cpu")
            module.eval()
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load %s: %s", path.name, exc)
            return
        self._lstm_models[crop] = module

        meta_path = models_dir / f"yield_lstm_{crop.value}_{version}.meta.json"
        if meta_path.exists():
            try:
                self._metadata[(crop, "lstm", version)] = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                pass
        log.info("Loaded %s", path.name)

    # ─── Lookup helpers ──────────────────────────────────────────

    def _resolve_default(self, crop: CropType) -> tuple[AlgorithmFamily, str] | None:
        configured = getattr(settings, "active_model_family", None)
        if configured:
            versions = sorted(
                v for (c, f, v) in self._models if c == crop and f == configured
            )
            if versions:
                return configured, versions[-1]
        for family, version in DEFAULT_PREFERENCE:
            if (crop, family, version) in self._models:
                return family, version
        return None

    def get_yield_model(
        self,
        crop: CropType,
        *,
        family: AlgorithmFamily | None = None,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """Return {model, features, crop, version, ...} or None.

        Without args, returns the preferred model per crop (XGBoost v2 → v1 → RF).
        Pass family/version to pick explicitly.
        """
        if family is None or version is None:
            resolved = self._resolve_default(crop)
            if resolved is None:
                return None
            family, version = resolved
        return self._models.get((crop, family, version))

    def get_shap_explainer(
        self,
        crop: CropType,
        *,
        family: AlgorithmFamily | None = None,
        version: str | None = None,
    ) -> shap.TreeExplainer | None:
        if family is None or version is None:
            resolved = self._resolve_default(crop)
            if resolved is None:
                return None
            family, version = resolved
        return self._explainers.get((crop, family, version))

    def get_lstm_model(self, crop: CropType):
        """Return the loaded TorchScript LSTM module for `crop`, or None."""
        return self._lstm_models.get(crop)

    def get_metadata(
        self,
        crop: CropType,
        family: AlgorithmFamily,
        version: str,
    ) -> dict | None:
        return self._metadata.get((crop, family, version))

    def list_available(self) -> list[dict[str, Any]]:
        """Inventory of loaded models — what /api/methodology consumes."""
        out: list[dict[str, Any]] = []
        for (crop, family, version), payload in self._models.items():
            meta = self._metadata.get((crop, family, version), {})
            out.append({
                "crop": crop.value,
                "family": family,
                "version": version,
                "metrics": meta.get("metrics"),
                "trained_at": meta.get("trained_at"),
                "feature_count": len(payload.get("features", [])),
            })
        for crop in self._lstm_models:
            meta = self._metadata.get((crop, "lstm", "v1"), {})
            out.append({
                "crop": crop.value,
                "family": "lstm",
                "version": "v1",
                "metrics": meta.get("metrics"),
                "trained_at": meta.get("trained_at"),
                "feature_count": 8,
            })
        return out

    @property
    def seasonal_norms(self) -> dict:
        return self._seasonal_norms or {}

    def is_available(self, crop: CropType) -> bool:
        return self._resolve_default(crop) is not None


def _normalise_metrics(metrics: dict) -> dict:
    """Convert legacy v1 flat-key metrics into the nested {train,val,test} shape
    that the new methodology UI + tests expect.

    v1 (scripts/train_yield_models.py) saved:
        {"rmse_test": 0.47, "mae_test": 0.38, "r2_test": 0.21,
         "rmse_val": ..., "n_train": 120, "n_val": 24, "n_test": 24}

    v2 (scripts/train_yield_models_v2.py) and lstm save:
        {"train": {"rmse": .., "mae": .., "r2": .., "n": ..},
         "val":   {...},
         "test":  {...}}

    If the metrics already look nested (has any of "train"/"val"/"test"), we
    pass through untouched.
    """
    if not metrics:
        return metrics
    if any(k in metrics for k in ("train", "val", "test")):
        return metrics

    out: dict = {}
    for split in ("train", "val", "test"):
        block: dict[str, float | int] = {}
        for short in ("r2", "mae", "rmse"):
            full = f"{short}_{split}"
            if full in metrics:
                block[short] = metrics[full]
        n_key = f"n_{split}"
        if n_key in metrics:
            block["n"] = metrics[n_key]
        if block:
            out[split] = block
    return out or metrics


def get_registry() -> ModelRegistry:
    return ModelRegistry()
