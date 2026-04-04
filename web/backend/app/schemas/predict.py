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


class UpcomingFixtureOption(BaseModel):
    fixture_id: str
    date: str
    home_team: str
    away_team: str
    label: str
    round: str | None = None
    p_H: float | None = None
    p_D: float | None = None
    p_A: float | None = None
    odds_avg_h: float | None = None
    odds_avg_d: float | None = None
    odds_avg_a: float | None = None
    best_ev: float | None = None
    best_ev_pick: str | None = None
    value_bet: bool | None = None


class UpcomingFixturesResponse(BaseModel):
    season_label: str
    source_path: str
    rows: int
    fixtures: list[UpcomingFixtureOption]
    error: str | None = None


class PredictUpcomingRequest(BaseModel):
    date: str
    home_team: str
    away_team: str
    round: str | None = None


class PredictUpcomingResponse(BaseModel):
    season_label: str
    selected_fixture: dict[str, str]
    prediction: dict[str, Any]
    market_odds: dict[str, Any] | None = None
    output_csv: str


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


class UpcomingOddsOption(BaseModel):
    fixture_id: str
    event_id: str
    date: str
    home_team: str
    away_team: str
    source: str
    bookmakers: int
    odds_avg_h: float | None = None
    odds_avg_d: float | None = None
    odds_avg_a: float | None = None
    odds_best_h: float | None = None
    odds_best_d: float | None = None
    odds_best_a: float | None = None


class UpcomingOddsResponse(BaseModel):
    sport_key: str
    source_path: str
    rows: int
    requests_remaining: str
    requests_used: str
    output_csv: str | None = None
    history_csv: str | None = None
    fetched_at_utc: str | None = None
    odds: list[UpcomingOddsOption]
