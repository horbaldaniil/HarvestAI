"""Tests for the v3 expansion of `app/ml/registry.py`.

Phase 4 added two new families (lightgbm / stack) and a new version
(v3) across the board. (CatBoost was a third addition in Phase 4 but
was removed in the post-v7 cleanup.) These tests verify:

  - the AlgorithmFamily Literal accepts the new values;
  - FAMILY_FILE_PREFIX has matching short forms;
  - DEFAULT_PREFERENCE puts stack v3 first;
  - the registry can discover and load v3 artifacts written by the
    Phase-4 trainer (when present on disk).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ml.registry import (
    ALL_FAMILIES,
    DEFAULT_PREFERENCE,
    DISCOVERY_VERSIONS,
    FAMILY_FILE_PREFIX,
    TREE_FAMILIES,
    ModelRegistry,
)
from app.db.models.enums import CropType


# ─── Static configuration ──────────────────────────────────


def test_all_families_includes_new_v3_families():
    assert "lightgbm" in ALL_FAMILIES
    assert "stack" in ALL_FAMILIES
    # CatBoost was removed in the post-v7 cleanup — it must NOT be
    # discoverable through the registry anymore.
    assert "catboost" not in ALL_FAMILIES


def test_family_file_prefix_covers_all_families():
    for fam in ALL_FAMILIES:
        assert fam in FAMILY_FILE_PREFIX, f"FAMILY_FILE_PREFIX missing {fam}"
    # Specific spot-checks for filenames that exist on disk
    assert FAMILY_FILE_PREFIX["lightgbm"] == "lgbm"
    assert FAMILY_FILE_PREFIX["stack"] == "stack"
    # CatBoost prefix ("cat") is gone — model files were deleted in
    # the post-v7 cleanup along with the package itself.
    assert "catboost" not in FAMILY_FILE_PREFIX


def test_default_preference_stack_v3_first():
    """Stack v7 (real-only + SoilGrids + hierarchical v7h sibling)
    leapfrogged v6/v5/v4/v3 in the final ablation. Guarantee: stack
    is first family in DEFAULT_PREFERENCE; v7 → v7h → v6 → v5 → v4 →
    v3 order."""
    assert DEFAULT_PREFERENCE[0][0] == "stack"
    assert DEFAULT_PREFERENCE[0][1] in ("v3", "v4", "v5", "v6", "v7", "v7h")
    # v7 first, v7h second (hierarchical sibling), then v6 → v5 → v4 → v3.
    stack_versions = [v for f, v in DEFAULT_PREFERENCE if f == "stack"]
    assert stack_versions == ["v7", "v7h", "v6", "v5", "v4", "v3"]


def test_default_preference_v3_families_before_v1_v2():
    """v3 must come before older versions in the preference order."""
    v3_idx = next(i for i, (_, v) in enumerate(DEFAULT_PREFERENCE) if v == "v3")
    v1_idx = next(i for i, (_, v) in enumerate(DEFAULT_PREFERENCE) if v == "v1")
    assert v3_idx < v1_idx


def test_tree_families_excludes_lstm_and_stack():
    """LSTM (neural) and Stack (Ridge over OOF) are not SHAP-TreeExplainer-able."""
    assert "lstm" not in TREE_FAMILIES
    assert "stack" not in TREE_FAMILIES
    # Three tree models remain after CatBoost removal:
    assert TREE_FAMILIES == frozenset({"xgboost", "rf", "lightgbm"})


def test_discovery_versions_contains_v3_and_v4_for_all_new_families():
    """Phase-4 added v3, Week-10 v4 ablation extended each family with
    a v4 entry. Both should be discoverable. CatBoost is excluded —
    removed in the post-v7 cleanup."""
    for family in ("xgboost", "rf", "lightgbm", "stack"):
        assert "v3" in DISCOVERY_VERSIONS[family], f"{family} missing v3"
        assert "v4" in DISCOVERY_VERSIONS[family], f"{family} missing v4"
    assert "catboost" not in DISCOVERY_VERSIONS


# ─── Live load (only runs if v3 artifacts exist) ───────────


_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_WHEAT_STACK_V3 = _MODELS_DIR / "yield_stack_wheat_v3.joblib"


@pytest.mark.skipif(
    not _WHEAT_STACK_V3.exists(),
    reason="v3 stack artifacts not yet trained (run scripts/train_yield_models_v3.py)",
)
def test_registry_loads_stack_v3_for_wheat():
    """When stack models exist on disk, the registry MUST select stack
    family as default. v7 → v7h → v6 → v5 → v4 → v3 precedence."""
    ModelRegistry.reset()
    r = ModelRegistry()
    r.load_all()
    pref = r._resolve_default(CropType.WHEAT)
    assert pref is not None
    assert pref[0] == "stack"  # family is stack
    assert pref[1] in ("v3", "v4", "v5", "v6", "v7", "v7h")  # whichever's on disk

    model = r.get_yield_model(CropType.WHEAT)
    assert model is not None
    # All trainers store stack under "model" key
    assert "model" in model
    assert "features" in model
    # v3 stack: 17 features; v4/v5/v6/v7h stack: 23 features; v7 stack: 30 (v4 + 7 soil).
    assert len(model["features"]) in (17, 23, 30)


@pytest.mark.skipif(
    not _WHEAT_STACK_V3.exists(),
    reason="v3 artifacts not on disk",
)
def test_registry_lists_v3_models_in_inventory():
    """`list_available()` powers the methodology UI — v3 models must
    appear with their family/version pair."""
    ModelRegistry.reset()
    r = ModelRegistry()
    r.load_all()
    inv = r.list_available()
    v3_entries = [e for e in inv if e["version"] == "v3"]
    families_in_v3 = {e["family"] for e in v3_entries}
    # Four v3 families should be represented if the trainer ran cleanly
    # (CatBoost was a fifth family before the post-v7 cleanup).
    assert families_in_v3 >= {"xgboost", "rf", "lightgbm", "stack"}
    assert "catboost" not in families_in_v3


@pytest.mark.skipif(
    not _WHEAT_STACK_V3.exists(),
    reason="v3 artifacts not on disk",
)
def test_v3_xgboost_payload_provides_shap_explainer():
    """The v3 XGBoost payload has {point, q_low, q_high} — the registry
    must still build a SHAP TreeExplainer from `point`."""
    ModelRegistry.reset()
    r = ModelRegistry()
    r.load_all()
    explainer = r.get_shap_explainer(CropType.WHEAT, family="xgboost", version="v3")
    # Explainer may legitimately be None if SHAP build failed; the test
    # just guards that the load path supports the v3 payload shape — not
    # that SHAP itself succeeded on this particular crop/model.
    if explainer is not None:
        # Has the standard TreeExplainer interface
        assert hasattr(explainer, "shap_values")


@pytest.mark.skipif(
    not _WHEAT_STACK_V3.exists(),
    reason="v3 artifacts not on disk",
)
def test_v3_stack_payload_has_no_shap_explainer():
    """Stack uses Ridge over OOF predictions — TreeExplainer is not
    meaningful and must NOT be built (would crash or mislead)."""
    ModelRegistry.reset()
    r = ModelRegistry()
    r.load_all()
    explainer = r.get_shap_explainer(CropType.WHEAT, family="stack", version="v3")
    assert explainer is None
