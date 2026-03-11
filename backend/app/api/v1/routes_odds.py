from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.predict import OddsCompareRequest, OddsCompareResponse, UpcomingOddsResponse
from app.services.odds_api import fetch_upcoming_laliga_odds, load_team_map
from app.services.predict import compare_odds

router = APIRouter(prefix="/odds", tags=["odds"])


@router.post("/compare", response_model=OddsCompareResponse)
def odds_compare(request: OddsCompareRequest) -> OddsCompareResponse:
    try:
        return OddsCompareResponse(**compare_odds(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/upcoming", response_model=UpcomingOddsResponse)
def odds_upcoming(limit: int = Query(default=100, ge=1, le=400)) -> UpcomingOddsResponse:
    try:
        team_map = load_team_map()
        return UpcomingOddsResponse(**fetch_upcoming_laliga_odds(team_map=team_map, limit=limit))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
