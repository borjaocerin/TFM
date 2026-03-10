from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CalibrationMethod = Literal["platt", "isotonic"]


class TrainRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Ruta al dataset de entrenamiento (por defecto out/laliga_enriched_model.csv)",
    )
    use_xgb: bool = Field(default=False, description="Activa XGBoost si esta instalado")
    calibration: CalibrationMethod = Field(default="platt")


class TrainResponse(BaseModel):
    best_model: str
    metrics: dict[str, float]
    leaderboard: list[dict[str, Any]]
    model_path: str
    metadata_path: str
    reliability_plot: str | None = None
    metrics_report_path: str | None = None


class ModelStatusResponse(BaseModel):
    model_available: bool
    metadata: dict[str, Any] | None = None
