from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictUpcomingRequest,
    PredictUpcomingResponse,
    UpcomingFixturesResponse,
)
from app.services.predict import list_upcoming_fixture_options, predict_matches, predict_selected_upcoming_match

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(**predict_matches(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/predict/options/upcoming", response_model=UpcomingFixturesResponse)
def upcoming_options() -> UpcomingFixturesResponse:
    try:
        return UpcomingFixturesResponse(**list_upcoming_fixture_options())
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict/upcoming", response_model=PredictUpcomingResponse)
def predict_upcoming(request: PredictUpcomingRequest) -> PredictUpcomingResponse:
    try:
        return PredictUpcomingResponse(**predict_selected_upcoming_match(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
