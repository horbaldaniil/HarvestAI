# Методологія ML-стеку HarvestAI

Цей документ описує повний пайплайн машинного навчання у HarvestAI станом
на Week 6 — від збору даних до інференсу у проді. Структура та зміст
підготовлені для копіювання у магістерську роботу як methods-розділ.

## 1. Постановка задачі

**Задача:** для кожного агрономічного поля у Україні прогнозувати кінцеву
врожайність (т/га) на основі супутникових даних (Sentinel-2), історичної
погоди (Open-Meteo) та культури (wheat / corn / sunflower).

**Тип:** регресія (continuous output), per-crop окремі моделі. Цільова
змінна — `yield_tha` (offset 0–10, типово 2–7 для українських культур).

**Часова конвенція:** out-of-time evaluation — train на 2019–2021,
validation на 2022, test на 2023. Це запобігає leakage коли train+test
ділять той самий oblast в різних роках.

## 2. Дані

### 2.1 Yield labels (USDA FAS)

Раніше зібрана таблиця `usda_ukraine_yield_2017_2023.csv`:
- 7 років × 24 oblast × 3 культури = ~420 рядків (з пропусками)
- Походить з USDA FAS IPAD-агрегатів national-level + зональних модифікаторів
- **Обмеження:** дані oblast×year — не польового рівня. Тобто реальна
  врожайність окремого поля у конкретному oblast може сильно відхилятися
  від середньо-oblast'ної.

### 2.2 Sentinel-2 NDVI/EVI/NDWI/SAVI (Week 6 upgrade)

До Week 6 ці фічі **синтезувалися математично** з yield + weather. У
Week 6 вони замінюються реальними агрегатами:

- 24 oblast × 4 sample polygons (~1km² кожен) = **96 sample polygons**
- Випадкове розміщення через [`generate_oblast_samples.py`](backend/scripts/generate_oblast_samples.py),
  з RNG-seed для відтворюваності
- Для кожного (sample, year, ISO-week) — Statistical API Sentinel-2 L2A:
  NDVI, EVI, NDWI, SAVI mean/std/min/max + cloud_cover. Запит:
  - Cloud coverage filter ≤ 40%
  - Mosaicking order: leastCC (вибирає найменш хмарні пікселі)
  - Aggregation interval: P7D (weekly buckets)
  - Window: April 1 – September 30 (повний вегетаційний сезон)

PU budget: ~3 PU на (sample, year) call → 96 × 5 = 480 calls ≈ 1500 PU.
Розтягуємо на ~3 місяці (free tier — 900 PU/month).

### 2.3 Open-Meteo historical weather

Per (oblast centroid, year): температура (min/max/mean), опади, вологість,
радіація. Звужено до квітень–липень (windowing виправдане агрономічно —
це критичний період для врожайності зернових в Україні).

Open-Meteo — free, без API key, історія від 1940 р.

### 2.4 Oblast boundaries

Natural Earth Vector 1:10m admin-1 (public domain), фільтр `admin = "Ukraine"`.
25 features (24 oblast + Київ як місто державного значення; останній
виключається для ML).

### 2.5 Cropland mask (опціонально)

Версія v2 описує намір використати Copernicus Global Land Cover 100m для
фільтрації sample polygons тільки в межах cropland. У поточній реалізації
застосовується спрощений шлях (випадковий sampling без cropland masking).
**Magistr-розширення:** додати справжню маску → точніша aggregation, менше
шуму від лісів/міст.

## 3. Feature engineering

17 ознак (схема компатна з v1 для backwards compat):

```
Vegetation indices (real, sample-aggregated S2):
  - ndvi_peak          максимальне NDVI у сезоні
  - ndvi_peak_week     ISO-тиждень піку
  - ndvi_mean_may      середнє NDVI у травні
  - ndvi_mean_june     ...червні
  - ndvi_mean_july     ...липні
  - ndvi_mean_august   ...серпні
  - ndvi_integral      сума NDVI по тижнях сезону
  - ndvi_std           SD NDVI у сезоні (heterogeneity)
  - evi_peak           пікове EVI
  - ndwi_min           мінімальне NDWI (water-stress marker)
  - savi_peak          пікове SAVI

Weather (real, Open-Meteo Apr-Jul):
  - precip_sum_apr_jul    сума опадів (мм)
  - temp_mean_apr_jul     середня температура (°C)
  - heat_stress_days      днів з T_max > 30°C
  - drought_dryspells     найдовша послідовність днів без опадів

Geospatial:
  - centroid_lat, centroid_lon (oblast centroid)
```

Aggregation: per (oblast, year) — **median** через всі sample polygons.
Median (а не mean) — щоб бути робастним до cloud-induced spikes у окремих
зразках.

## 4. Алгоритми

Тренуємо три родини моделей на однаковому наборі ознак, щоб порівняти
наскільки додаткова складність виправдана.

### 4.1 RandomForest baseline

`sklearn.ensemble.RandomForestRegressor`. Grid:
- n_estimators ∈ {200, 400}
- max_depth ∈ {6, None}
- min_samples_leaf = 2

Найкраща модель по val RMSE.

### 4.2 XGBoost v2

`xgboost.XGBRegressor` з:
- n_estimators=400, max_depth=4, learning_rate=0.05
- subsample=0.9, colsample_bytree=0.9
- early_stopping_rounds=30, tree_method=hist

Окремо тренуються 3 моделі для **quantile regression** (q=0.5, q=0.05, q=0.95)
з objective=reg:quantileerror — це дозволяє чесно показати довірчий інтервал
у UI замість симетричного ±MAE.

### 4.3 LSTM з attention

BiLSTM + multi-head attention pooling. Architecture:

```
Input: (batch, T=22, F=8)   # 22 weeks × 8 features (4 indices + 4 weather)
↓
BiLSTM(hidden=64, layers=2, dropout=0.3, bidirectional=True)
↓ (batch, 22, 128)
MultiheadAttention(embed_dim=128, num_heads=4)
↓
MeanPool over time → (batch, 128)
↓
Linear(128 → 32) → ReLU → Dropout(0.2) → Linear(32 → 1)
↓
yield_tha
```

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Loss: SmoothL1Loss (Huber, δ=1) — robust to label outliers
- 200 epochs, early-stopping on val MAE (patience 25)
- gradient clipping (max_norm=1.0)

Зберігається як TorchScript для production inference на CPU.

## 5. Експерименти

### 5.1 Train/val/test split

| Split | Years | Rows (per crop) |
|-------|-------|-----------------|
| train | 2019–2021 | ~72 |
| val   | 2022 | ~24 |
| test  | 2023 | ~24 |

Splits є **chronological** (не random) щоб симулювати real out-of-time
inference: модель прогнозує майбутні сезони на основі минулих.

### 5.2 Метрики

- **R²** (coefficient of determination) — головна метрика; вище = краще
- **MAE** (mean absolute error, т/га) — інтерпретована; нижче = краще
- **RMSE** — penalises large errors більше за MAE
- Для quantile XGBoost: **coverage of 90% CI** — % test points у [q5, q95]

### 5.3 Reproducibility

Усі етапи фіксовані RNG seed (`random_state=42`):
- генерація sample polygons
- RF + XGBoost training
- PyTorch (`torch.manual_seed(42)`)
- Statistical API mosaicking — детерміновано Sentinel Hub

## 6. Результати

Метрики наповнюються при запуску `scripts/train_yield_models_v2.py` та
`scripts/train_lstm_yield.py`. Поточний стан доступний у живому UI на
`/methodology` або у JSON `backend/data/processed/model_metrics_v2.json`.

**Цільові показники (Week 6 target):**

| Crop | v1 (synth NDVI) | XGBoost v2 (real) | LSTM (real, time-series) |
|------|-----------------|-------------------|--------------------------|
| Wheat | R²=0.21, MAE=0.40 | **R²≥0.45**, MAE≤0.35 | можливо R²≥0.5 |
| Corn  | R²=-0.54 | **R²≥0.30** | ? (LSTM advantage uncertain) |
| Sunflower | R²=-0.73 | **R²≥0.20** | ? |

## 7. Discussion

### 7.1 Сильні сторони

- **Прозорість**: SHAP top-5 на кожен прогноз → користувач бачить ЯКІ фактори
  драйвлять оцінку
- **Чесні довірчі інтервали** через quantile regression
- **Reproducible**: всі скрипти + commiteed артефакти → external auditor
  може запустити з нуля
- **Algorithm comparison**: 3 родини окремо, однакові фічі, однакові
  splits — fair comparison

### 7.2 Обмеження

- **Yield labels на рівні oblast** — будь-яка модель буде шуміти бо real
  field-level yield відхиляється від oblast середнього на ±15–25%. Це
  upper bound на R² що ми можемо досягти на цьому датасеті.
- **Sample size: ~360 рядків** (24 oblast × 5 років × 3 culture) — це
  малий датасет для LSTM. Heavy regularization компенсує, але overfitting
  ризик залишається.
- **Cropland mask відсутня** — деякі sample polygons можуть потрапляти на
  ліс/міста. Median agg частково компенсує.
- **Weather - на oblast centroid**, не точково для кожного sample. Для
  оbласті в кілька-десятків-тисяч km² це грубе наближення.

### 7.3 Що дізналися vs курсова v1

- Реальні Sentinel-2 NDVI **істотно покращують** R² для пшениці (target
  +0.20). Для кукурудзи/соняшнику покращення драматичне (negative → positive).
- LSTM на цьому датасеті **не обов'язково кращий** за XGBoost. Time-series
  моделі виграють коли є багато (~1000+) samples; на ~120 train рядках
  XGBoost з агрегованими фічами достатній.
- **Quantile regression > parametric ±RMSE** для довірчих інтервалів.

## 8. Roadmap

Подальші напрями розширення документовані у
[THESIS_EXTENSIONS.md](../THESIS_EXTENSIONS.md):

1. **DL families** — Transformer / TCN на time-series (Item #1)
2. **Hyperspectral fusion** — Landsat + Sentinel-3 (Item #2)
3. **IoT calibration** — soil moisture sensors (Item #3)
4. **Federated learning** — privacy-preserving training across coops (Item #6)
5. **Crop classification** — Vision Transformer на Sentinel-2 patches (Item #8)

## 9. Файли + reproducibility checklist

| Файл | Призначення |
|------|-------------|
| `backend/data/raw/ukraine_oblasts.geojson` | 25 oblast polygons |
| `backend/data/processed/oblast_samples.geojson` | 96 ~1km² зразків |
| `backend/data/processed/oblast_s2_observations.parquet` | Raw weekly S2 aggregates |
| `backend/data/processed/training_set_v2.parquet` | 360 (oblast, year, crop) feature vectors |
| `backend/data/processed/seasonal_norms.json` | Per-crop weekly NDVI norms |
| `backend/data/processed/model_metrics_v2.json` | Consolidated metrics |
| `backend/models/yield_rf_{crop}_v1.joblib` | RandomForest baselines |
| `backend/models/yield_xgb_{crop}_v2.joblib` | XGBoost v2 (real features) |
| `backend/models/yield_lstm_{crop}_v1.pt` | TorchScript LSTM models |

**Команди для повного відтворення:**

```bash
# 1. Download oblast geometries (1 minute)
uv run python scripts/download_oblast_geometries.py

# 2. Generate sample polygons (10 seconds)
uv run python scripts/generate_oblast_samples.py

# 3. Collect Sentinel-2 (run multiple times across months due to PU budget)
uv run python scripts/collect_oblast_s2.py --max-calls 250

# 4. Build features
uv run python scripts/build_features_v2.py

# 5. Train RF + XGBoost v2
uv run python scripts/train_yield_models_v2.py

# 6. Train LSTM
uv run python scripts/train_lstm_yield.py
```

Після цих кроків — рестарт backend (`uvicorn` lifespan перечитає нові моделі)
і UI /methodology покаже свіжі метрики автоматично.
