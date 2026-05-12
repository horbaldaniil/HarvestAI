"""Tests for the Phase-4 metric panel — pure functions that don't
require trained joblib artifacts on disk."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_models import (
    _core_metrics,
    _crps_gaussian,
    _learning_curve,
    _loocv_oblast,
    _partial_dependence,
    _per_oblast_residuals,
    _pinball_loss,
    _repeated_kfold_variance,
    _unwrap_estimator,
)


# ─── Core metrics ──────────────────────────────────────────


def test_core_metrics_perfect_prediction_is_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = _core_metrics(y, y)
    assert m.rmse == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.mape == pytest.approx(0.0)


def test_core_metrics_handles_empty():
    m = _core_metrics(np.array([]), np.array([]))
    assert m.n == 0
    assert m.rmse is None
    assert m.r2 is None


def test_core_metrics_constant_target_returns_none_r2():
    """R² is undefined when y_true is constant — must return None
    rather than crash with `division by zero`."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 4.5, 5.5])
    m = _core_metrics(y_true, y_pred)
    assert m.r2 is None
    assert m.mae > 0  # other metrics still computable


def test_core_metrics_mape_correctness():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # MAPE = mean(|2/10|, |2/20|) * 100 = (0.2 + 0.1) / 2 * 100 = 15
    m = _core_metrics(y_true, y_pred)
    assert m.mape == pytest.approx(15.0, abs=0.01)


# ─── Pinball loss ──────────────────────────────────────────


def test_pinball_loss_at_q_05_penalises_under_prediction_more():
    """At q=0.05, predicting too high carries (1-0.05)=95% weight; too
    low only 5%. So under-prediction (y > y_pred) should have low loss."""
    y = np.array([1.0])
    high = _pinball_loss(y, np.array([2.0]), 0.05)   # over-prediction
    low = _pinball_loss(y, np.array([0.0]), 0.05)    # under-prediction
    assert high > low


def test_pinball_loss_at_q_95_inverse():
    y = np.array([1.0])
    high = _pinball_loss(y, np.array([2.0]), 0.95)   # over → low weight
    low = _pinball_loss(y, np.array([0.0]), 0.95)    # under → high weight
    assert low > high


def test_pinball_loss_perfect_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert _pinball_loss(y, y, 0.5) == pytest.approx(0.0)


# ─── CRPS (Gaussian approximation) ─────────────────────────


def test_crps_gaussian_zero_variance_collapses_to_mae():
    """With σ→0 the Gaussian CRPS converges to MAE."""
    y_true = np.array([3.0])
    mu = np.array([2.0])
    sigma = np.array([1e-3])
    # MAE = 1.0. CRPS should be ≈ 1.0 minus a small Gaussian-tail correction.
    crps = _crps_gaussian(y_true, mu, sigma)
    assert 0.9 < crps < 1.05


def test_crps_gaussian_perfect_prediction():
    """μ = y, σ = small → near-zero CRPS."""
    y = np.array([3.0])
    crps = _crps_gaussian(y, y, np.array([0.01]))
    assert crps < 0.05


# ─── LOOCV per oblast ─────────────────────────────────────


def test_loocv_oblast_returns_metric_object():
    """Smoke test: feed a tiny dataset → LOOCV runs without crashing
    and returns sensible n. We don't assert R² value because the
    factory we pass uses a 1-tree RandomForest (deterministic but trivial)
    on a 30-row toy set — the metric is mostly meaningless."""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed=42)
    rows = []
    for iso in ("UA-46", "UA-32", "UA-12", "UA-51"):
        for year in range(2017, 2024):
            rows.append({
                "iso_3166_2": iso,
                "oblast": iso,
                "year": year,
                "ndvi_peak": rng.uniform(0.5, 0.9),
                "ndvi_peak_week": rng.integers(20, 30),
                "ndvi_mean_may": rng.uniform(0.4, 0.7),
                "ndvi_mean_june": rng.uniform(0.5, 0.8),
                "ndvi_mean_july": rng.uniform(0.6, 0.9),
                "ndvi_mean_august": rng.uniform(0.4, 0.8),
                "ndvi_integral": rng.uniform(8, 14),
                "ndvi_std": rng.uniform(0.05, 0.15),
                "evi_peak": rng.uniform(0.4, 0.8),
                "ndwi_min": rng.uniform(-0.3, -0.1),
                "savi_peak": rng.uniform(0.5, 0.85),
                "precip_sum_apr_jul": rng.uniform(150, 350),
                "temp_mean_apr_jul": rng.uniform(15, 22),
                "heat_stress_days": int(rng.integers(0, 10)),
                "drought_dryspells": int(rng.integers(0, 15)),
                "centroid_lat": float(rng.uniform(46, 52)),
                "centroid_lon": float(rng.uniform(24, 38)),
                "yield_tha": float(rng.uniform(3.5, 5.0)),
            })
    df = pd.DataFrame(rows)
    factory = lambda: RandomForestRegressor(n_estimators=5, random_state=0)  # noqa: E731
    metrics = _loocv_oblast(factory, df)
    # We loop 4 oblasts, each holding out 7 rows → 28 predictions made.
    assert metrics.n == 28


# ─── RepeatedKFold variance ───────────────────────────────


def test_repeated_kfold_variance_returns_5_repeats():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        **{name: rng.uniform(size=50) for name in (
            "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
            "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
            "evi_peak", "ndwi_min", "savi_peak",
            "precip_sum_apr_jul", "temp_mean_apr_jul",
            "heat_stress_days", "drought_dryspells",
            "centroid_lat", "centroid_lon",
        )},
        "yield_tha": rng.uniform(2.0, 6.0, size=50),
    })
    factory = lambda: LinearRegression()  # noqa: E731
    res = _repeated_kfold_variance(factory, df, n_splits=5, n_repeats=3)
    assert res["n_folds"] == 15   # 5 × 3 = 15 fold-evaluations
    assert "r2_mean" in res
    assert "r2_std" in res


# ─── Learning curve ───────────────────────────────────────


def test_learning_curve_emits_one_point_per_fraction():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed=42)
    rows = []
    for year in range(2017, 2024):
        for _ in range(20):
            rows.append({
                **{name: rng.uniform() for name in (
                    "ndvi_peak", "ndvi_peak_week", "ndvi_mean_may", "ndvi_mean_june",
                    "ndvi_mean_july", "ndvi_mean_august", "ndvi_integral", "ndvi_std",
                    "evi_peak", "ndwi_min", "savi_peak",
                    "precip_sum_apr_jul", "temp_mean_apr_jul",
                    "heat_stress_days", "drought_dryspells",
                    "centroid_lat", "centroid_lon",
                )},
                "year": year,
                "yield_tha": rng.uniform(2.0, 6.0),
            })
    df = pd.DataFrame(rows)
    factory = lambda: RandomForestRegressor(n_estimators=10, random_state=0)  # noqa: E731
    points = _learning_curve(factory, df, sample_fractions=(0.25, 0.5, 1.0))
    assert len(points) == 3
    assert points[0]["fraction"] == 0.25
    assert points[-1]["fraction"] == 1.0
    # Each point must have all expected metrics
    for p in points:
        assert "test_r2" in p
        assert "n_train" in p


# ─── Per-oblast residuals ─────────────────────────────────


def test_per_oblast_residuals_returns_one_row_per_iso():
    import pandas as pd

    df = pd.DataFrame({
        "iso_3166_2": ["UA-46", "UA-46", "UA-32", "UA-12"],
        "oblast": ["Lviv", "Lviv", "Kyiv", "Dnipropetrovsk"],
    })
    y_pred = np.array([5.0, 6.0, 4.0, 7.0])
    y_true = np.array([5.5, 5.5, 4.5, 6.5])
    rows = _per_oblast_residuals(df, y_pred, y_true)
    assert len(rows) == 3
    by_iso = {r["iso_3166_2"]: r for r in rows}
    # Lviv: |5-5.5| + |6-5.5| → mean 0.5
    assert by_iso["UA-46"]["mean_abs_residual"] == pytest.approx(0.5, abs=1e-6)
    assert by_iso["UA-46"]["n"] == 2


# ─── _unwrap_estimator ────────────────────────────────────


def test_unwrap_estimator_handles_xgb_v3_payload():
    """v3 XGBoost stores {point, q_low, q_high} — unwrap returns `point`."""
    payload = {"point": "POINT_MODEL", "q_low": "LOW", "q_high": "HIGH"}
    assert _unwrap_estimator(payload) == "POINT_MODEL"


def test_unwrap_estimator_handles_model_key():
    payload = {"model": "RF_MODEL", "features": ["a", "b"]}
    assert _unwrap_estimator(payload) == "RF_MODEL"


def test_unwrap_estimator_passes_bare_estimator_through():
    """If someone passes a bare estimator (not a payload dict), return it."""
    class Stub:
        pass
    s = Stub()
    assert _unwrap_estimator(s) is s


# ─── Partial dependence (Friedman 2001) ────────────────────


def test_partial_dependence_returns_grid_per_feature():
    """For each requested top-feature we get a {grid, pdp} pair with
    the same length. Smoke test on a 1-tree RandomForest so it's
    deterministic and fast."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed=42)
    X = rng.uniform(size=(40, 17))  # 17 features matching FEATURE_NAMES
    y = X[:, 0] * 3 + X[:, 7] * 2 + rng.normal(scale=0.1, size=40)
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

    # ndvi_peak (idx 0) and ndvi_std (idx 7) are the two features the
    # synthetic target depends on; PDP should show clear gradient.
    out = _partial_dependence(model, X, ["ndvi_peak", "ndvi_std"])
    assert set(out.keys()) == {"ndvi_peak", "ndvi_std"}
    for feat, body in out.items():
        assert "grid" in body
        assert "pdp" in body
        assert len(body["grid"]) == len(body["pdp"])
        assert len(body["grid"]) > 1  # default grid_resolution is 30


def test_partial_dependence_skips_unknown_features():
    """Unknown feature names are silently dropped — caller's expected
    behaviour when permutation importance gives back something odd."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed=42)
    X = rng.uniform(size=(30, 17))
    y = rng.uniform(size=30)
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)

    out = _partial_dependence(model, X, ["ndvi_peak", "atlantis_feature"])
    assert "ndvi_peak" in out
    assert "atlantis_feature" not in out


def test_partial_dependence_empty_for_stacking_regressor():
    """Stack returns {} — sklearn partial_dependence reports meta-input
    dimensions, not the original 17 features, so curves would mislead."""
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed=42)
    X = rng.uniform(size=(30, 17))
    y = rng.uniform(size=30)
    stack = StackingRegressor(
        estimators=[
            ("rf1", RandomForestRegressor(n_estimators=5, random_state=0)),
            ("rf2", RandomForestRegressor(n_estimators=5, random_state=1)),
        ],
        final_estimator=Ridge(),
        cv=3,
    ).fit(X, y)

    out = _partial_dependence(stack, X, ["ndvi_peak"])
    assert out == {}


def test_partial_dependence_unwraps_xgboost_v3_payload():
    """If we hand it a {point: estimator} payload (v3 XGBoost shape),
    it must unwrap and still produce PDP curves."""
    import xgboost as xgb

    rng = np.random.default_rng(seed=42)
    X = rng.uniform(size=(50, 17))
    y = X[:, 0] * 3 + rng.normal(scale=0.1, size=50)
    point = xgb.XGBRegressor(n_estimators=20, random_state=0, verbosity=0).fit(X, y)
    payload = {"point": point, "q_low": point, "q_high": point}

    out = _partial_dependence(payload, X, ["ndvi_peak"])
    assert "ndvi_peak" in out
    assert len(out["ndvi_peak"]["grid"]) > 1
