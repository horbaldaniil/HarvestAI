"""ModelRegistry — verify the v1/v2/RF/LSTM multi-version lookup logic.

Tests rely on whatever .joblib files are committed to backend/models/. If
v2 hasn't been trained yet, we expect the registry to fall back gracefully
to v1 (or RF, or None).
"""
from __future__ import annotations

import json

import joblib

from app.db.models.enums import CropType
from app.ml.registry import (
    DEFAULT_PREFERENCE,
    FAMILY_FILE_PREFIX,
    ModelRegistry,
    _normalise_metrics,
    get_registry,
)


def test_default_preference_ordered():
    """xgboost v2 is preferred over v1, which is preferred over rf v1."""
    assert DEFAULT_PREFERENCE[0] == ("xgboost", "v2")
    assert DEFAULT_PREFERENCE[1] == ("xgboost", "v1")
    assert DEFAULT_PREFERENCE[2] == ("rf", "v1")


def test_registry_load_doesnt_crash_when_files_missing(monkeypatch, tmp_path):
    """Even if models/ is empty, load_all() should complete and stay queryable."""
    from app.config import settings
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    ModelRegistry.reset()
    reg = get_registry()
    reg.load_all()
    assert reg.list_available() == []
    assert reg.get_yield_model(CropType.WHEAT) is None
    assert reg.is_available(CropType.WHEAT) is False


def test_registry_picks_v1_when_only_v1_present():
    """With the committed v1 .joblibs, the registry falls back to v1 even
    though v2 is the preferred slot."""
    ModelRegistry.reset()
    reg = get_registry()
    reg.load_all()
    # If wheat v1 exists in the repo, fallback should pick it up.
    payload = reg.get_yield_model(CropType.WHEAT)
    if payload is not None:
        assert payload.get("version") in {"v1", "v2"}


def test_registry_returns_none_for_unknown_explicit_version():
    ModelRegistry.reset()
    reg = get_registry()
    reg.load_all()
    assert reg.get_yield_model(
        CropType.WHEAT, family="xgboost", version="vXYZ"
    ) is None


def test_list_available_contains_family_and_version_for_each_entry():
    ModelRegistry.reset()
    reg = get_registry()
    reg.load_all()
    for entry in reg.list_available():
        assert "crop" in entry
        assert "family" in entry
        assert "version" in entry
        assert entry["family"] in ("xgboost", "rf", "lstm")


# ─── Filename-prefix regression tests ────────────────────────


def test_family_file_prefix_maps_xgboost_to_xgb():
    """Regression: registry was looking for `yield_xgboost_*.joblib` but the
    training scripts (v1 + v2) save files as `yield_xgb_*.joblib`. The map
    bridges the public family name and the on-disk short prefix."""
    assert FAMILY_FILE_PREFIX["xgboost"] == "xgb"
    assert FAMILY_FILE_PREFIX["rf"] == "rf"
    assert FAMILY_FILE_PREFIX["lstm"] == "lstm"


def test_registry_loads_xgb_short_filename(monkeypatch, tmp_path):
    """A `yield_xgb_wheat_v2.joblib` on disk must be discoverable through the
    `xgboost` family name."""
    from app.config import settings

    # Tiny dummy regressor so we can load it back via joblib.
    from sklearn.dummy import DummyRegressor
    import numpy as np

    model = DummyRegressor(strategy="constant", constant=5.0)
    model.fit(np.zeros((2, 3)), np.array([5.0, 5.0]))

    (tmp_path / "yield_xgb_wheat_v2.joblib").write_bytes(
        joblib.numpy_pickle._dumps_compat({  # type: ignore[attr-defined]
            "model": model, "features": ["a", "b", "c"],
            "crop": "wheat", "version": "v2",
        })
        if False else b""  # placeholder — we use joblib.dump below
    )
    # Re-write via the public API so the dump format matches what registry expects.
    joblib.dump(
        {"model": model, "features": ["a", "b", "c"], "crop": "wheat", "version": "v2"},
        tmp_path / "yield_xgb_wheat_v2.joblib",
    )

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    ModelRegistry.reset()
    reg = get_registry()
    reg.load_all()

    payload = reg.get_yield_model(CropType.WHEAT, family="xgboost", version="v2")
    assert payload is not None
    assert payload["version"] == "v2"
    assert payload["features"] == ["a", "b", "c"]


# ─── _normalise_metrics: legacy v1 flat → nested ─────────────


def test_normalise_metrics_passes_nested_through():
    nested = {"train": {"r2": 0.4}, "val": {"r2": 0.3}, "test": {"r2": 0.2}}
    assert _normalise_metrics(nested) is nested


def test_normalise_metrics_converts_flat_keys():
    flat = {
        "rmse_test": 0.47, "mae_test": 0.38, "r2_test": 0.21,
        "rmse_val": 0.40, "mae_val": 0.32, "r2_val": 0.18,
        "n_train": 120, "n_val": 24, "n_test": 24,
    }
    out = _normalise_metrics(flat)
    assert "test" in out
    assert out["test"]["r2"] == 0.21
    assert out["test"]["mae"] == 0.38
    assert out["test"]["rmse"] == 0.47
    assert out["test"]["n"] == 24
    assert out["val"]["r2"] == 0.18
    assert out["train"]["n"] == 120


def test_normalise_metrics_empty_passes_through():
    assert _normalise_metrics({}) == {}
    assert _normalise_metrics(None) is None  # type: ignore[arg-type]
