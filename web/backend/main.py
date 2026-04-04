from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.routes_datasets import router as datasets_router
from app.api.v1.routes_features import router as features_router
from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_model import router as model_router
from app.api.v1.routes_odds import router as odds_router
from app.api.v1.routes_predict import router as predict_router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging(settings.log_level)

app = FastAPI(
    title="LaLiga 1X2 Predictor API",
    version="0.1.0",
    description=(
        "API para ingestar CSV locales, entrenar modelo multiclase 1X2 con "
        "calibracion de probabilidades y comparar contra cuotas de mercado."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(features_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(odds_router, prefix="/api/v1")

static_dir = settings.backend_app_dir / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
