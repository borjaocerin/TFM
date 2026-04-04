from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.datasets import DatasetIngestRequest, DatasetIngestResponse
from app.services.datasets import ingest_datasets

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/ingest", response_model=DatasetIngestResponse)
def ingest(request: DatasetIngestRequest) -> DatasetIngestResponse:
    try:
        return DatasetIngestResponse(**ingest_datasets(request))
    except Exception as exc:  # pragma: no cover - error mapping
        raise HTTPException(status_code=400, detail=str(exc)) from exc
