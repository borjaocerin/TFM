from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "docker-compose.yml").exists():
            return candidate
        if (candidate / "web" / "backend").exists() and (candidate / "modelos").exists():
            return candidate
        if (candidate / "backend").exists() and (candidate / "data").exists():
            return candidate
        if (
            (candidate / "backend").exists()
            and (candidate / "frontend").exists()
            and (candidate / "etl").exists()
        ):
            return candidate
    return start.parents[4]


BACKEND_APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = _find_repo_root(Path(__file__).resolve())


class Settings(BaseSettings):
    app_env: str = Field(default="dev", alias="APP_ENV")
    tz: str = Field(default="Europe/Madrid", alias="TZ")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    output_dir: Path = Field(default=REPO_ROOT / "data" / "out", alias="OUTPUT_DIR")
    model_dir: Path = Field(default=BACKEND_APP_DIR / "app" / "models" / "store", alias="MODEL_DIR")
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS"
    )
    fixtures_api_url: str | None = Field(
        default="https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id=4335",
        alias="FIXTURES_API_URL",
    )
    fixtures_api_key: str | None = Field(default=None, alias="FIXTURES_API_KEY")
    fixtures_api_host: str | None = Field(default=None, alias="FIXTURES_API_HOST")
    fixtures_api_timeout_sec: int = Field(default=15, alias="FIXTURES_API_TIMEOUT_SEC")
    fixtures_allow_csv_fallback: bool = Field(default=False, alias="FIXTURES_ALLOW_CSV_FALLBACK")
    odds_api_url: str | None = Field(default=None, alias="ODDS_API_URL")
    odds_api_key: str | None = Field(default=None, alias="ODDS_API_KEY")
    odds_api_sport_key: str = Field(default="soccer_spain_la_liga", alias="ODDS_API_SPORT_KEY")
    odds_api_regions: str = Field(default="eu", alias="ODDS_API_REGIONS")
    odds_api_markets: str = Field(default="h2h", alias="ODDS_API_MARKETS")
    odds_api_odds_format: str = Field(default="decimal", alias="ODDS_API_ODDS_FORMAT")
    odds_api_date_format: str = Field(default="iso", alias="ODDS_API_DATE_FORMAT")
    odds_api_bookmakers: str | None = Field(default=None, alias="ODDS_API_BOOKMAKERS")
    odds_api_timeout_sec: int = Field(default=15, alias="ODDS_API_TIMEOUT_SEC")

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @field_validator("data_dir", "output_dir", "model_dir", mode="before")
    @classmethod
    def _resolve_paths(cls, value: Path | str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (REPO_ROOT / path).resolve()

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]

    @property
    def backend_app_dir(self) -> Path:
        return BACKEND_APP_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
