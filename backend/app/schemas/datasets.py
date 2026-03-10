from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DatasetIngestRequest(BaseModel):
    historical: str = Field(..., description="Ruta al CSV historico base")
    football_data_dir: str = Field(..., description="Directorio con CSV de Football-Data")
    elo_csv: str | None = Field(default=None, description="Ruta opcional a ELO_RATINGS.csv")
    team_map: str | None = Field(default=None, description="Ruta opcional a team_name_map_es.json")
    windows: list[int] = Field(default_factory=lambda: [5, 10])

    @field_validator("windows")
    @classmethod
    def validate_windows(cls, windows: list[int]) -> list[int]:
        if not windows:
            raise ValueError("Debes indicar al menos una ventana rolling")
        cleaned = sorted(set(int(value) for value in windows if int(value) > 0))
        if not cleaned:
            raise ValueError("Las ventanas rolling deben ser enteros positivos")
        return cleaned


class DatasetIngestResponse(BaseModel):
    rows_total: int
    rows_by_season: dict[str, int]
    missing_pct_by_column: dict[str, float]
    columns: list[str]
    output_all: str
    output_model: str


class FixturesFeatureRequest(BaseModel):
    fixtures_csv: str = Field(..., description="Ruta al CSV de fixtures")
    historical_csv: str | None = Field(
        default=None,
        description="Ruta al historico enriquecido (por defecto out/laliga_enriched_all.csv)",
    )
    elo_csv: str | None = Field(default=None, description="Ruta opcional ELO")
    team_map: str | None = Field(default=None, description="Ruta opcional al mapa de equipos")
    windows: list[int] = Field(default_factory=lambda: [5, 10])

    @field_validator("windows")
    @classmethod
    def validate_windows(cls, windows: list[int]) -> list[int]:
        cleaned = sorted(set(int(value) for value in windows if int(value) > 0))
        if not cleaned:
            raise ValueError("Las ventanas rolling deben ser enteros positivos")
        return cleaned


class FixturesFeatureResponse(BaseModel):
    rows_total: int
    generated_columns: list[str]
    output_path: str
