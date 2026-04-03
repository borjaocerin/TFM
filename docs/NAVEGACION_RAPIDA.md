# Navegacion rapida del proyecto

Esta guia es para no perderse entre carpetas. No cambia la logica del proyecto: solo te dice por donde entrar.

## 1) Carpetas principales (las que mas vas a tocar)

- `backend/app/`: API y logica principal.
- `backend/tools/`: scripts operativos (ROI, cuotas, tareas programadas).
- `frontend/src/`: interfaz web.
- `data/`: datos de entrada y salidas de pipeline.
- `etl/`: construccion offline de datasets enriquecidos.

## 2) Flujo mental corto

1. Datos y fixtures en `data/`.
2. Features + modelo en `backend/app/services/`.
3. Endpoints en `backend/app/api/v1/`.
4. Pantallas en `frontend/src/routes/`.
5. Seguimiento de dinero/ROI en `out/roi/` y scripts de `backend/tools/`.

## 3) Si quieres seguir solo el dinero

- Script principal: `backend/tools/update_roi_jornada.bat`
- Refresco solo cuotas: `backend/tools/refresh_laliga_odds.py`
- Resumen final: `out/roi/roi_upcoming_summary.json`
- Detalle por partido: `out/roi/roi_upcoming_detail.csv`

## 4) Archivos generados que puedes ignorar en el dia a dia

- `data/out/*`
- `out/roi/odds_snapshots/*`
- `out/model_improvement/*`
- caches Python (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`)

Nota: el workspace ya los oculta visualmente en VS Code con `.vscode/settings.json`.
