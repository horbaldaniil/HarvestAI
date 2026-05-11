---
title: PU квота
short: Processing Units — ліміт Sentinel Hub
category: data-source
---

**PU (Processing Units)** — внутрішня одиниця обліку Sentinel Hub.
Кожен запит до Sentinel-2 коштує певну кількість PU. На безкоштовному
плані ліміт — **900 PU/місяць**.

### Як HarvestAI рахує PU

Кожен RQ-job, який звертається до Sentinel Hub:
1. Перевіряє `PuTracker.assert_can_spend(estimated_pu)` — це Redis-лічильник
2. Якщо вистачає — робить запит
3. Реальна вартість приходить у HTTP-хедері `X-ProcessingUnits-Spent`
4. `PuTracker.record(actual)` — оновлює лічильник

### Приблизні ціни

| Операція | PU |
|----------|-----|
| Statistical API: 1 рік × 4 індекси × 1 поле | ~2–4 PU |
| Process API: 1 heatmap (256×256 px) | ~3 PU |
| Statistical API: 2 роки × 4 індекси × 1 поле | ~5–8 PU |

### Що бачить користувач

- Якщо квота вичерпана → backend повертає 503 з повідомленням про
  ліміт. UI показує toast.
- На сторінці налаштувань (`/settings`) є віджет PU usage місяця.
- Дані зберігаються у Redis (live) + місячний snapshot у БД
  (`pu_usage_monthly`).

### Як економити

- Не клацайте heatmap на кожну дату — він кешується на диску
  (`var/rasters/`)
- Спостереження оновлюються максимум раз на 24 год
- HarvestAI хешує запити Sentinel Hub за `(geometry, date_range, indices)`
