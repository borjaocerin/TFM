from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.predict import predict_matches

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(**predict_matches(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
