from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.model import ModelStatusResponse, TrainRequest, TrainResponse
from app.services.train import get_active_model_status, train_and_calibrate

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest) -> TrainResponse:
    try:
        return TrainResponse(**train_and_calibrate(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/active", response_model=ModelStatusResponse)
def active_model() -> ModelStatusResponse:
    try:
        return ModelStatusResponse(**get_active_model_status())
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=404, detail=str(exc)) from exc
