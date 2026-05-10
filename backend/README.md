# HarvestAI Backend

FastAPI + SQLAlchemy 2.0 (async) + PostGIS + RQ workers.

## Local setup

```bash
# 1. Install uv (if not yet)
# Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install deps
uv sync

# 3. Copy env
cp .env.example .env  # then fill in keys

# 4. Start Postgres+Redis (from project root)
cd .. && docker compose up -d && cd backend

# 5. Run migrations
uv run alembic upgrade head

# 6. Run dev server
uv run uvicorn app.main:app --reload --port 8000

# 7. (in another terminal) Run worker
uv run rq worker default high low
```

API docs: http://localhost:8000/docs

## Tests

```bash
uv run pytest -v --cov=app
```

## Make a migration

```bash
uv run alembic revision --autogenerate -m "add table foo"
uv run alembic upgrade head
```
