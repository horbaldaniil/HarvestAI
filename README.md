# 🌾 HarvestAI

**Веб-застосунок для супутникового моніторингу сільськогосподарських ресурсів та визначення врожайності за допомогою штучного інтелекту та машинного навчання**

Курсова робота 5 курс, спеціальність "Комп'ютерні технології"

---

## Що це таке?

HarvestAI — це інтелектуальна платформа для фермерів, яка поєднує:

- 🛰️ **Супутникові дані Sentinel-2** від ESA — глобальне покриття, оновлення кожні 5 днів
- 📊 **Вегетаційні індекси** — NDVI, EVI, NDWI, SAVI для оцінки стану посівів
- 🤖 **Машинне навчання** — прогноз врожайності для пшениці, кукурудзи та соняшнику (XGBoost моделі)
- 💬 **AI-помічник (GPT-4o-mini)** — фермер може запитати "чому моє поле жовтіє?" і отримати обґрунтовану відповідь з реальних даних
- ⚠️ **Виявлення аномалій** — автоматичне попередження про стрес рослин (посуха, шкідники)
- 📍 **Інтерактивна карта** — малювання полів полігонами, heatmap-візуалізація
- 📈 **Часові ряди та порівняння рік-до-року**
- 📄 **PDF-звіти** з мапами, графіками та рекомендаціями

---

## Архітектура

```
[React SPA + Leaflet]  <--JWT-->  [FastAPI]  <-->  [PostgreSQL + PostGIS]
                                      |
                              +-------+-------+
                              |               |
                          [Redis]         [RQ Worker]
                                              |
                                    [Sentinel Hub] [OpenAI] [OpenWeather]
```

Детальна архітектура — у [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Стек технологій

| Шар | Технологія |
|-----|-----------|
| Backend | Python 3.12 + FastAPI |
| Frontend | React 18 + TypeScript + Vite |
| База даних | PostgreSQL 16 + PostGIS 3.4 |
| Кеш / Черга | Redis + RQ |
| ML | XGBoost, scikit-learn |
| LLM | OpenAI GPT-4o-mini (function calling) |
| Супутник | Sentinel Hub API (Sentinel-2) |
| Погода | OpenWeatherMap API |
| Карти | Leaflet + react-leaflet-draw |
| Графіки | Recharts |
| UI | Tailwind CSS + shadcn/ui |
| PDF | WeasyPrint |
| Деплой | Railway (backend) + Vercel (frontend) |

---

## Швидкий старт (локально)

### Передумови
- Docker Desktop
- Python 3.12+
- Node.js 20+ та pnpm

### 1. Клонування
```bash
git clone <repo-url>
cd HarvestAI
```

### 2. Запуск інфраструктури (Postgres+PostGIS, Redis)
```bash
docker compose up -d
```

### 3. Backend
```bash
cd backend
cp .env.example .env  # заповнити SH_CLIENT_ID, OPENAI_API_KEY, etc.
uv sync               # або: pip install -e .
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

В окремому терміналі:
```bash
cd backend
rq worker default high low
```

### 4. Frontend
```bash
cd frontend
cp .env.example .env
pnpm install
pnpm dev
```

Відкрити: http://localhost:5173

---

## Структура проєкту

```
HarvestAI/
├── backend/          # FastAPI + ML + LLM
├── frontend/         # React + TypeScript
├── docker-compose.yml
├── ARCHITECTURE.md
├── THESIS_EXTENSIONS.md
└── README.md
```

---

## Документація

- [ARCHITECTURE.md](ARCHITECTURE.md) — детальна архітектура
- [THESIS_EXTENSIONS.md](THESIS_EXTENSIONS.md) — заділ під магістерську роботу

---

## Ліцензія

MIT (для академічних та навчальних цілей).

## Автор

Студент 5 курсу спеціальності "Комп'ютерні технології", ЛНУ ім. І. Франка.
