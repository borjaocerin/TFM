from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

OddsKind = Literal["odds_avg", "odds_close"]


class PredictRequest(BaseModel):
    fixtures_enriched_path: str | None = Field(
        default=None,
        description="Ruta al CSV de fixtures enriquecidos",
    )
    fixtures: list[dict[str, Any]] | None = Field(
        default=None,
        description="Payload alternativo con fixtures ya enriquecidos",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "PredictRequest":
        if self.fixtures_enriched_path is None and not self.fixtures:
            raise ValueError("Debes enviar fixtures_enriched_path o fixtures en el payload")
        return self


class PredictResponse(BaseModel):
    rows: int
    output_csv: str
    predictions: list[dict[str, Any]]


class OddsCompareRequest(BaseModel):
    predictions_csv: str | None = Field(default=None, description="Ruta CSV con predicciones")
    predictions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Payload alternativo con predicciones + cuotas",
    )
    odds_kind: OddsKind = Field(default="odds_avg")
    value_threshold: float = Field(default=0.02)

    @model_validator(mode="after")
    def validate_source(self) -> "OddsCompareRequest":
        if self.predictions_csv is None and not self.predictions:
            raise ValueError("Debes enviar predictions_csv o predictions en el payload")
        return self


class OddsCompareResponse(BaseModel):
    rows: int
    metrics: dict[str, float | None]
    value_bets: list[dict[str, Any]]
    output_csv: str
