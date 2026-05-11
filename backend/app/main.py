"""FastAPI application entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import auth as auth_router
from app.routers import fields as fields_router
from app.routers import jobs as jobs_router
from app.routers import observations as observations_router
from app.routers import quota as quota_router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("harvestai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (env=%s)", settings.app_name, settings.app_env)
    settings.ensure_directories()
    yield
    log.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="HarvestAI API",
    description=(
        "Веб-застосунок для супутникового моніторингу сільськогосподарських "
        "ресурсів та визначення врожайності за допомогою AI/ML."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allowlist single frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router.router)
app.include_router(fields_router.router)
app.include_router(observations_router.router)
app.include_router(jobs_router.router)
app.include_router(quota_router.router)


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"message": "🌾 HarvestAI API. See /docs for OpenAPI."}
