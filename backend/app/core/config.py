from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_env: str = Field(default="dev", alias="APP_ENV")
    tz: str = Field(default="Europe/Madrid", alias="TZ")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    output_dir: Path = Field(default=REPO_ROOT / "data" / "out", alias="OUTPUT_DIR")
    model_dir: Path = Field(
        default=REPO_ROOT / "backend" / "app" / "models" / "store", alias="MODEL_DIR"
    )
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS"
    )

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
        return self.model_dir.parent.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
