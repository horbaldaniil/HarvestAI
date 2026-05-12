"""Application configuration loaded from environment variables.

Single source of truth — every module reads values from `settings`, never from
`os.environ` directly. This keeps the env contract documented and testable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Application ────────────────────────────────────────────
    app_name: str = "HarvestAI"
    app_env: str = "development"
    log_level: str = "INFO"

    # ─── Database ───────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://harvestai:harvestai_dev@localhost:5432/harvestai"
    )
    database_url_sync: str = Field(
        default="postgresql+psycopg2://harvestai:harvestai_dev@localhost:5432/harvestai"
    )

    # ─── Redis ──────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── JWT ────────────────────────────────────────────────────
    jwt_secret: str = "change_me_to_random_32_byte_hex_string"
    jwt_algorithm: str = "HS256"
    jwt_access_expire_minutes: int = 30
    jwt_refresh_expire_days: int = 14

    # ─── CORS ───────────────────────────────────────────────────
    frontend_origin: str = "http://localhost:5173"

    # ─── Sentinel Hub ───────────────────────────────────────────
    sh_client_id: str = ""
    sh_client_secret: str = ""
    sh_instance_id: str = ""
    sh_monthly_pu_limit: int = 900

    # ─── OpenAI ─────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 600
    openai_daily_token_budget: int = 100_000
    # Soft rate-limit per user, rolling 1-hour window. Set to 0 to disable.
    chat_rate_limit_per_hour: int = 20

    # ─── ML — Week 6 ────────────────────────────────────────────
    # Pin a specific algorithm family (xgboost / rf / lstm). Default empty =
    # registry picks the best available per crop (xgb v2 → v1 → rf v1).
    active_model_family: str = ""

    # ─── OpenWeather ────────────────────────────────────────────
    openweather_api_key: str = ""

    # ─── Storage ────────────────────────────────────────────────
    models_dir: Path = Path("./models")
    reports_dir: Path = Path("./var/reports")
    rasters_dir: Path = Path("./var/rasters")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist (idempotent)."""
        for path in (self.models_dir, self.reports_dir, self.rasters_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
