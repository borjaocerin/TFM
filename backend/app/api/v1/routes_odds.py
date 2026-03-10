from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.predict import OddsCompareRequest, OddsCompareResponse
from app.services.predict import compare_odds

router = APIRouter(prefix="/odds", tags=["odds"])


@router.post("/compare", response_model=OddsCompareResponse)
def odds_compare(request: OddsCompareRequest) -> OddsCompareResponse:
    try:
        return OddsCompareResponse(**compare_odds(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
