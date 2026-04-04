from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.datasets import FixturesFeatureRequest, FixturesFeatureResponse
from app.services.datasets import build_fixtures_features

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/fixtures", response_model=FixturesFeatureResponse)
def features_for_fixtures(request: FixturesFeatureRequest) -> FixturesFeatureResponse:
    try:
        return FixturesFeatureResponse(**build_fixtures_features(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
