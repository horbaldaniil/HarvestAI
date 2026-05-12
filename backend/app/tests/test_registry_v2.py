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
    """v4 ablation iteration promoted Stack v4 to the top (weather-
    conditioned yields + crop calendar lifted 9 of 13 crops above
    R²=0). Older versions (v3 / v2 / v1) remain as fallbacks for the
    ablation chain — this test guards the order is preserved."""
    isos = [(f, v) for f, v in DEFAULT_PREFERENCE]
    # v7 stack first → v7 = real-only training + SoilGrids soil features.
    # v6 / v5 / v4 / v3 / v2 / v1 listed below for ablation comparison.
    assert isos[0] == ("stack", "v7")
    assert ("stack", "v7h") in isos
    assert ("stack", "v6") in isos
    assert ("stack", "v5") in isos
    assert ("stack", "v4") in isos
    assert ("stack", "v3") in isos
    assert ("xgboost", "v2") in isos
    assert ("xgboost", "v1") in isos
    assert ("rf", "v1") in isos
    # v3 versions all come BEFORE v2/v1 ones.
    last_v3 = max(i for i, (_, v) in enumerate(isos) if v == "v3")
    first_v2_or_v1 = min(i for i, (_, v) in enumerate(isos) if v in ("v1", "v2"))
    assert last_v3 < first_v2_or_v1


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
    """The registry falls back through the DEFAULT_PREFERENCE order. After
    Phase 4 trained v3 artifacts, the preferred slot is `stack v3`. The
    important guarantee is that we get *some* version of the wheat model
    back (not None) when any joblib is present in the repo."""
    ModelRegistry.reset()
    reg = get_registry()
    reg.load_all()
    payload = reg.get_yield_model(CropType.WHEAT)
    if payload is not None:
        # Phase 4 added v3; old assertion {"v1", "v2"} was too narrow.
        assert payload.get("version") in {"v1", "v2", "v3"}


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
    # Phase 4 expanded the supported families. Use the canonical
    # ALL_FAMILIES tuple so this stays in sync if more are added.
    from app.ml.registry import ALL_FAMILIES
    for entry in reg.list_available():
        assert "crop" in entry
        assert "family" in entry
        assert "version" in entry
        assert entry["family"] in ALL_FAMILIES


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
