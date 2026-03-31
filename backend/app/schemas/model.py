from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CalibrationMethod = Literal["platt", "isotonic"]
SelectionMetric = Literal["log_loss", "accuracy", "f1_macro", "brier", "ece"]


class TrainRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Ruta al dataset de entrenamiento (por defecto out/laliga_enriched_model.csv)",
    )
    use_xgb: bool = Field(default=True, description="Activa XGBoost si esta instalado")
    use_catboost: bool = Field(default=True, description="Activa CatBoost si esta instalado")
    calibration: CalibrationMethod = Field(default="platt")
    selection_metric: SelectionMetric = Field(
        default="log_loss",
        description="Metrica principal para elegir el mejor modelo (log_loss se minimiza)",
    )
    min_season: int | None = Field(
        default=None,
        description="Temporada minima incluida para entrenar (ej. 2019)",
    )
    xg_poss_min_coverage_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Cobertura minima requerida (%% no nulo) por temporada para "
            "xg_last5_home/xg_last5_away/poss_last5_home/poss_last5_away"
        ),
    )


class TrainResponse(BaseModel):
    best_model: str
    metrics: dict[str, float]
    leaderboard: list[dict[str, Any]]
    model_path: str
    metadata_path: str
    reliability_plot: str | None = None
    metrics_report_path: str | None = None
    eda_missing_report_path: str | None = None
    selection_metric: SelectionMetric = Field(default="log_loss")


class ModelStatusResponse(BaseModel):
    model_available: bool
    metadata: dict[str, Any] | None = None
