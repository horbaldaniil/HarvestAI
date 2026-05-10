# HarvestAI — convenience targets
# Працює на Linux/macOS/Windows (через git-bash або WSL).
# Для нативного PowerShell використовуйте окремі команди з README.

.PHONY: help up down logs db-shell redis-shell \
        be-install be-dev be-test be-migrate be-revision \
        fe-install fe-dev fe-build fe-lint \
        worker clean

help:
	@echo "Available targets:"
	@echo "  up           - Start Postgres + Redis via docker compose"
	@echo "  down         - Stop docker compose services"
	@echo "  logs         - Tail docker compose logs"
	@echo "  be-install   - Install backend dependencies (uv)"
	@echo "  be-dev       - Run FastAPI dev server"
	@echo "  be-test      - Run pytest"
	@echo "  be-migrate   - Apply Alembic migrations"
	@echo "  be-revision  - Generate Alembic migration (use NAME=desc)"
	@echo "  worker       - Run RQ worker"
	@echo "  fe-install   - Install frontend dependencies (pnpm)"
	@echo "  fe-dev       - Run Vite dev server"
	@echo "  fe-build     - Production build of frontend"
	@echo "  fe-lint      - ESLint + Prettier check"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

db-shell:
	docker compose exec db psql -U harvestai -d harvestai

redis-shell:
	docker compose exec redis redis-cli

# ─── Backend ───────────────────────────────────────────────────
be-install:
	cd backend && uv sync

be-dev:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

be-test:
	cd backend && uv run pytest -v

be-migrate:
	cd backend && uv run alembic upgrade head

be-revision:
	cd backend && uv run alembic revision --autogenerate -m "$(NAME)"

worker:
	cd backend && uv run rq worker default high low

# ─── Frontend ──────────────────────────────────────────────────
fe-install:
	cd frontend && pnpm install

fe-dev:
	cd frontend && pnpm dev

fe-build:
	cd frontend && pnpm build

fe-lint:
	cd frontend && pnpm lint

# ─── Cleanup ───────────────────────────────────────────────────
clean:
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/htmlcov
	rm -rf frontend/dist frontend/node_modules/.vite
