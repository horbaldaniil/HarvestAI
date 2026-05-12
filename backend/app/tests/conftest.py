"""Pytest fixtures.

We use SQLite in-memory for unit-scope tests (auth, security) since neither
CITEXT (we register a SQLite-side compilation as TEXT) nor PostGIS (no
geospatial logic in v0) is exercised at this level. Integration tests that
need PostgreSQL+PostGIS bring up a real DB via testcontainers (added later).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles

import app.db.models  # noqa: F401  (registers ORM tables with Base.metadata)
from app.db.base import Base
from app.db.session import get_session
from app.main import app as fastapi_app


@compiles(CITEXT, "sqlite")
def _compile_citext_sqlite(type_, compiler, **kw):  # noqa: D401
    """Map CITEXT to plain TEXT when running on SQLite (tests)."""
    return "TEXT"


@compiles(BigInteger, "sqlite")
def _compile_bigint_sqlite(type_, compiler, **kw):  # noqa: D401
    """SQLite auto-increments only on INTEGER PRIMARY KEY, not BIGINT."""
    return "INTEGER"


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: D401
    """SQLite has no JSONB — fall back to TEXT. Reads/writes still work
    because SQLAlchemy serialises dicts as JSON strings."""
    return "TEXT"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Tables that require PostgreSQL/PostGIS specifics (Geography columns, ST_*
# generated columns) and therefore cannot be created on SQLite. We skip them
# for SQLite-based unit/integration tests; full coverage for these comes from
# integration tests run against a real PostgreSQL+PostGIS instance.
# satellite_observations FKs `fields`, so it must be skipped together.
# Anything that FKs to `fields` (and therefore inherits the PostGIS dependency)
# also has to be skipped. JSONB columns in `predictions` likewise need
# PostgreSQL.
_POSTGIS_ONLY_TABLES: frozenset[str] = frozenset({
    "fields",
    "satellite_observations",
    "predictions",
    "alerts",
    "weather_observations",
    # chat_sessions has a FK to `fields`, so it inherits the skip.
    "chat_sessions",
    "chat_messages",
    # generated_reports stores `params_json` as JSONB — PostgreSQL-only.
    "generated_reports",
})


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

    sqlite_safe_tables = [
        t for t in Base.metadata.sorted_tables if t.name not in _POSTGIS_ONLY_TABLES
    ]

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=sqlite_safe_tables)
        )

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    fastapi_app.dependency_overrides[get_session] = _override_session
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    fastapi_app.dependency_overrides.clear()
