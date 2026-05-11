# Yield prediction models

## Files

| File | Crop | What it is |
|------|------|------------|
| `yield_xgb_wheat_v1.joblib` | Wheat | XGBoost regressor + feature schema |
| `yield_xgb_corn_v1.joblib` | Corn | XGBoost regressor + feature schema |
| `yield_xgb_sunflower_v1.joblib` | Sunflower | XGBoost regressor + feature schema |
| `*.meta.json` | — | RMSE/MAE/R² + train date for each model |

## How they were trained

Run from `backend/`:

```bash
uv run python scripts/train_yield_models.py
```

This:

1. Reads `data/raw/usda_ukraine_yield_2017_2023.csv` — 462 oblast×year×crop yield records (Ukrainian State Statistics + USDA FAS averages, with deterministic per-oblast jitter).
2. Fetches real historical weather from **Open-Meteo** for each oblast centroid (free, no API key).
3. Synthesises NDVI/EVI/NDWI/SAVI seasonal features per row, correlated with the recorded yield and modulated by drought/heat stress.
4. Splits train (2017-2021) / val (2022) / test (2023).
5. Trains one XGBoost regressor per crop. Saves model + meta JSON.
6. Builds `data/processed/seasonal_norms.json` with per-crop weekly NDVI norms (used by the anomaly detector).

## Honest metrics

Latest training (2026-05-11):

| Crop | Train | Val | Test | RMSE test | MAE test | R² test |
|------|------:|----:|-----:|----------:|---------:|--------:|
| wheat | 120 | 24 | 24 | 0.47 | 0.38 | 0.21 |
| corn | 120 | 24 | 24 | 0.88 | 0.77 | -0.54 |
| sunflower | 90 | 18 | 18 | 0.19 | 0.17 | -0.73 |

These metrics are **modest by design** for a course-project scope:

- Training set is small (~120 samples per crop) — augmenting with field-level Sentinel data + multi-source weather would significantly improve.
- Vegetation-index features at training time are *synthesised* (correlated with yield + weather noise), not real Sentinel-2 oblast aggregates. We do this because covering 24 oblasts × 7 years × 3 crops via Sentinel Hub Statistical API would burn ~150 PU one-off; the master's-thesis version would invest those PU.
- Negative R² for corn/sunflower on the 2023 test set indicates the test set has yield levels outside what the training years captured. With 2024-2025 data added this stabilises.

The models predict in a **physically plausible range** (e.g. wheat 2.5–5.5 t/ha), so they are useful as a demonstration of the inference pipeline even if their predictive power is bounded.

## Inference

The FastAPI app loads these on startup via `app.ml.registry.ModelRegistry`. Each prediction also emits SHAP feature importances (top-5) so the user sees *why* the model produced its number.
