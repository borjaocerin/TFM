from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictUpcomingRequest,
    PredictUpcomingResponse,
    UpcomingFixturesResponse,
)
from app.services.predict import (
    list_upcoming_fixture_options,
    list_upcoming_fixture_options_with_value,
    predict_matches,
    predict_selected_upcoming_match,
)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(**predict_matches(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/predict/options/upcoming", response_model=UpcomingFixturesResponse)
def upcoming_options(
    include_value: bool = Query(default=False),
    value_threshold: float = Query(default=0.02, ge=0.0, le=1.0),
) -> UpcomingFixturesResponse:
    try:
        if include_value:
            return UpcomingFixturesResponse(
                **list_upcoming_fixture_options_with_value(value_threshold=value_threshold)
            )
        return UpcomingFixturesResponse(**list_upcoming_fixture_options())
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict/upcoming", response_model=PredictUpcomingResponse)
def predict_upcoming(request: PredictUpcomingRequest) -> PredictUpcomingResponse:
    try:
        return PredictUpcomingResponse(**predict_selected_upcoming_match(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
