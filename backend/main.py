from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .api_data import get_fixtures
    from .predict import load_model_artifact, predict_match
    from .simulate import simulate_value_betting_round
    from .train import train_and_save_model
except ImportError:
    from api_data import get_fixtures
    from predict import load_model_artifact, predict_match
    from simulate import simulate_value_betting_round
    from train import train_and_save_model


app = FastAPI(
    title="LaLiga Predictor API",
    description="Backend para entrenamiento, prediccion y simulacion Value Betting.",
    version="0.1.0",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "dataset_clean.csv"
MODEL_PATH = BASE_DIR / "backend" / "model.pkl"


class TrainRequest(BaseModel):
    dataset_path: str = str(DATASET_PATH)


class PredictRequest(BaseModel):
    season: str = Field(..., examples=["2025/2026"])
    round: int = Field(..., ge=1, le=50)
    home_team: str
    away_team: str


class SimulateValueBettingRequest(BaseModel):
    season: str
    round: int = Field(..., ge=1, le=50)
    budget: float = Field(..., gt=0)
    odds_source: str = Field(default_factory=lambda: os.getenv("DEFAULT_ODDS_SOURCE", "api"))
    preferred_bookmaker: str | None = Field(default=None)


@app.post("/train")
def train_endpoint(request: TrainRequest) -> Dict[str, Any]:
    try:
        _, metrics = train_and_save_model(dataset_path=request.dataset_path, model_path=MODEL_PATH)
        return {"status": "ok", "model_path": str(MODEL_PATH), "metrics": metrics}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> Dict[str, Any]:
    try:
        result = predict_match(
            home_team=request.home_team,
            away_team=request.away_team,
            dataset_path=DATASET_PATH,
            model_path=MODEL_PATH,
        )
        return {
            "season": request.season,
            "round": request.round,
            **result,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/simulate-value-betting")
def simulate_endpoint(request: SimulateValueBettingRequest) -> Dict[str, Any]:
    if request.odds_source not in {"simulated", "api"}:
        raise HTTPException(status_code=400, detail="odds_source debe ser 'simulated' o 'api'")

    try:
        return simulate_value_betting_round(
            season=request.season,
            round_number=request.round,
            budget=request.budget,
            dataset_path=DATASET_PATH,
            model_path=MODEL_PATH,
            odds_source=request.odds_source,
            preferred_bookmaker=request.preferred_bookmaker,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/metrics")
def metrics_endpoint() -> Dict[str, Any]:
    try:
        artifact = load_model_artifact(MODEL_PATH)
        return {
            "best_model": artifact.get("best_model_name"),
            "training_rows": artifact.get("training_rows"),
            "metrics": artifact.get("metrics", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/feature-importance")
def feature_importance_endpoint(top_k: int = 12) -> Dict[str, Any]:
    try:
        artifact = load_model_artifact(MODEL_PATH)
        model = artifact["model"]
        feature_columns: List[str] = artifact.get("feature_columns", [])

        if hasattr(model, "feature_importances_"):
            raw = np.asarray(model.feature_importances_)
        elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
            raw = np.asarray(model.estimator.feature_importances_)
        elif hasattr(model, "coef_"):
            raw = np.abs(np.asarray(model.coef_)).mean(axis=0)
        elif hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
            raw = np.abs(np.asarray(model.estimator.coef_)).mean(axis=0)
        else:
            raw = np.zeros(len(feature_columns))

        order = np.argsort(raw)[::-1][:top_k]
        rows = [{"feature": feature_columns[i], "importance": float(raw[i])} for i in order]
        return {"top_features": rows}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/fixtures/{season}/{round_number}")
def fixtures_endpoint(season: str, round_number: int) -> Dict[str, Any]:
    try:
        fixtures = get_fixtures(season=season, round_number=round_number, dataset_path=DATASET_PATH)
        return {"season": season, "round": round_number, "fixtures": fixtures}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health_endpoint() -> Dict[str, Any]:
    return {"status": "ok"}
