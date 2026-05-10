# Архітектура HarvestAI

## Огляд

HarvestAI — це full-stack веб-застосунок, що поєднує супутникові дані Sentinel-2, машинне навчання та LLM для моніторингу сільгоспугідь та прогнозу врожайності зернових культур України.

## Діаграма компонентів

```
┌─────────────────────────────────────────────────────────────────┐
│                       Користувач (браузер)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTPS
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│   Frontend (React 18 + TypeScript + Vite)        Vercel         │
│   - React Router v6                                             │
│   - Leaflet + react-leaflet-draw (мапи, малювання полів)        │
│   - Recharts (часові ряди)                                      │
│   - shadcn/ui + Tailwind (UI)                                   │
│   - React Query (server state) + Zustand (auth)                 │
│   - i18next (uk локаль)                                         │
└────────────────────┬────────────────────────────────────────────┘
                     │ JWT в Authorization header
                     │ Refresh — httpOnly cookie
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│   Backend (FastAPI + Uvicorn)                    Railway        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Routers: /auth, /fields, /observations, /predictions,   │   │
│   │          /alerts, /chat, /reports, /dashboard            │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ Services: бізнес-логіка                                  │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ Integrations: Sentinel Hub, OpenAI, OpenWeather         │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ ML: yield_model, crop_model, anomaly, registry           │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ LLM: prompts, tools, orchestrator                        │   │
│   └─────────────────────────────────────────────────────────┘   │
└──────────┬──────────────┬──────────────────┬──────────────────┘
           │              │                  │
           ▼              ▼                  ▼
   ┌──────────────┐  ┌─────────┐  ┌─────────────────────┐
   │ PostgreSQL   │  │ Redis   │  │ RQ Worker           │
   │ + PostGIS    │  │ (queue+ │  │ - fetch_sentinel    │
   │              │  │  cache) │  │ - compute_alerts    │
   └──────────────┘  └────┬────┘  │ - generate_report   │
                          │       │ - train_model       │
                          └──────►└─────────┬───────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                       ┌────────────┐ ┌──────────┐ ┌──────────────┐
                       │ Sentinel   │ │ OpenAI   │ │ OpenWeather  │
                       │ Hub API    │ │ API      │ │ API          │
                       └────────────┘ └──────────┘ └──────────────┘
```

## Ключові архітектурні рішення

### 1. RQ (Redis Queue) для фонових задач
Sentinel-запити (5-30s), ML-тренування (хвилини), PDF-генерація — асинхронно. RQ простіший за Celery, працює на тому ж Docker-образі (тільки інша команда запуску).

### 2. Sentinel Hub: Statistical-API-first
Free tier = 1000 PU/місяць. Statistical API (~0.3 PU) для time-series, Process API (~1-3 PU) лише для heatmap.

### 3. ML-інференс in-process
joblib.load на старті — XGBoost швидкий, окремий сервіс не потрібен.

### 4. JWT + httpOnly refresh
Access у Authorization header, refresh у httpOnly secure SameSite=Lax cookie. Argon2id хешування.

### 5. Read-only LLM tools
Усі LLM tools перевіряють `field.user_id == current_user.id`. Жоден tool не мутує стан — захист від prompt injection.

## Потік даних: типовий NDVI-запит

1. Користувач малює полігон на Leaflet → frontend POSTs `/fields` → backend зберігає PostGIS geometry.
2. Користувач клікає "Оновити супутникові дані" → backend перевіряє кеш `satellite_observations` → міс → enqueue RQ job.
3. RQ worker викликає Sentinel Hub Statistical API з evalscript → отримує JSON з mean/std/percentiles → зберігає у БД.
4. Frontend через React Query опитує `/fields/{id}/observations` → рендерить часовий ряд (Recharts).
5. При кліку "Показати heatmap" на дату → backend викликає Process API → отримує TIFF → конвертує в PNG → зберігає в Railway Volume → Leaflet overlay.

## Потік: AI-чат

1. Користувач пише "Чому моя пшениця жовтіє?" → POST `/chat/messages` → backend будує messages list (system + history + user).
2. Виклик OpenAI з `tools=[list_user_fields, get_field_ndvi_history, get_field_weather, ...]`.
3. Якщо LLM повертає tool_call → backend виконує Python функцію (з фільтром по `user_id`) → додає `tool` message → повторний виклик.
4. Після max 5 ітерацій або фінальної відповіді → SSE стрім токенів на frontend.

## Безпека

- **Authentication:** JWT (HS256), Argon2id, refresh rotation
- **Authorization:** Усі query фільтрують по `user_id` з JWT context
- **LLM:** Per-user daily token budget, max_tokens=600, max_tool_calls=5
- **Sentinel Hub:** Monthly PU hard cap (`SH_MONTHLY_PU_LIMIT=900`)
- **Rate limiting:** slowapi на `/auth/login`, `/auth/register` (5/min/IP)
- **CORS:** Allowlist (тільки FRONTEND_ORIGIN)
- **Secrets:** Railway/Vercel env vars, ніколи в коді

## Деплоймент

Push to `main` → Railway автодеплой backend+worker, Vercel автодеплой frontend.

| Сервіс | Платформа |
|--------|-----------|
| FastAPI | Railway service |
| RQ Worker | Railway service (той самий image, інша команда) |
| PostgreSQL+PostGIS | Railway addon |
| Redis | Railway addon |
| Volume (models, PDFs) | Railway Volume |
| React SPA | Vercel |
