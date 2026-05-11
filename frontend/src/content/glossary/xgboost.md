---
title: XGBoost
short: Ансамбль рішучих дерев для прогнозу
category: ml
---

**XGBoost (eXtreme Gradient Boosting)** — open-source бібліотека, яка
імплементує градієнтний бустінг рішучих дерев. Стандарт індустрії для
табличних даних (Kaggle, виробництво). HarvestAI використовує XGBoost
для прогнозу врожайності — окрема модель на кожну культуру (wheat, corn,
sunflower).

### Чому саме XGBoost (а не Random Forest або нейронка)

- **Високі результати на малих датасетах** — у нашому випадку це
  ~600 рядків на культуру (USDA × oblast × year + augmentation)
- **Хороша інтерпретація через SHAP** — TreeExplainer дає точні Shapley
- **Не потребує feature scaling** — float, int, bool — все одно працює
- **Швидкий inference** — мс на прогноз

### Які фічі ми даємо моделі

```
- ndvi_peak, ndvi_peak_week
- ndvi_mean_may, _june, _july, _august
- ndvi_integral, ndvi_std
- evi_peak
- ndwi_min
- savi_peak
- precip_sum_apr_jul (мм)
- temp_mean_apr_jul (°C)
- heat_stress_days (днів T_max > 30°C)
- drought_dryspells (макс. дні без дощу)
- area_ha
- centroid_lat, centroid_lon
```

### Чесні метрики на test set

| Культура | R²       | MAE       |
|----------|----------|-----------|
| Пшениця  | 0.21     | 0.4 т/га  |
| Кукурудза | -0.54   | 1.2 т/га  |
| Соняшник | -0.73    | 0.3 т/га  |

Модельний апарат працює, але **точність обмежена** через використання
oblast-level USDA даних (не польових). У magistr-роботі ми замінимо
це на справжню Sentinel-2 oblast aggregation.
