# MEMORIA TÉCNICA COMPLETA DEL PROYECTO TFM (LaLiga 1X2 Predictor)

> Documento de transferencia técnica pensado para otra IA (o desarrollador) que necesite entender **cómo funciona realmente** el proyecto de extremo a extremo, qué piezas hay, cómo se ejecuta, qué genera y dónde tocar según el objetivo.

---

## 1) Resumen ejecutivo

Este repositorio implementa un sistema completo para predicción de partidos de LaLiga en formato **1X2**:

- `1` (victoria local), `X` (empate), `2` (victoria visitante).

Incluye:

1. **Pipeline de datos** (ingesta y enriquecimiento de histórico + fixtures).
2. **Feature engineering temporal** (rolling prepartido sin leakage).
3. **Entrenamiento y calibración probabilística** (platt/isotonic).
4. **Predicción para partidos próximos**.
5. **Comparación contra cuotas** y detección de **value bets** por EV.
6. **API FastAPI** para orquestar todo.
7. **Frontend React** para demo visual de selección de partido y análisis modelo vs mercado.
8. **Docker/Makefiles** para ejecución reproducible.

El objetivo funcional es doble:

- generar probabilidades útiles del modelo,
- y compararlas contra mercado para detectar posibles oportunidades (value).

---

## 2) Estructura del repositorio (visión funcional)

- `backend/`: API, servicios de ML/ETL, tests, almacenamiento del modelo.
- `frontend/`: SPA React para demo visual.
- `etl/`: CLI offline reutilizable para generar datasets enriquecidos sin levantar API.
- `data/`: datos de entrada/salida (histórico, football-data, ELO, fixtures, out).
- `infra/`: make/compose operativo (levantado, train, predict).
- `docker-compose.yml` (raíz): composición principal de servicios.

### Directorios críticos

- Modelo activo:
  - `backend/app/models/store/model.pkl`
  - `backend/app/models/store/metadata.json`
- Artefactos de salida:
  - `data/out/laliga_enriched_all.csv`
  - `data/out/laliga_enriched_model.csv`
  - `data/out/fixtures_enriched.csv`
  - `data/out/predictions.csv`
  - `data/out/predictions_with_odds.csv`
  - `data/out/laliga_upcoming_odds.csv`
  - `data/out/laliga_odds_history.csv`
  - `data/out/model_metrics.txt`
- Gráfico de calibración:
  - `backend/app/static/reliability_latest.png`

---

## 3) Arquitectura lógica (flujo completo)

### Flujo principal ideal

1. Ingesta de históricos (`/datasets/ingest`) para construir dataset enriquecido.
2. Entrenamiento (`/model/train`) con validación temporal + calibración.
3. Obtención de fixtures próximos (`/predict/options/upcoming`).
4. Predicción de partido seleccionado (`/predict/upcoming`) o batch (`/predict`).
5. Comparación con cuotas (`/odds/compare`) y ranking EV/value.

### Flujo real de la demo frontend actual

La app React actual está simplificada al flujo de partido:

- `/` → lista de partidos próximos (`DatasetsPage`),
- `/partido` → detalle de un partido y comparativa modelo vs mercado (`MatchPredictionPage`).

Las páginas legacy de entrenamiento/predicción/odds siguen existiendo en código (`TrainingPage`, `PredictPage`, etc.) pero la navegación principal se centra en la experiencia de partido único.

---

## 4) Backend FastAPI: composición y responsabilidades

Archivo de entrada: `backend/app/main.py`.

### Routers montados

- `GET /api/v1/health`
- `POST /api/v1/datasets/ingest`
- `POST /api/v1/features/fixtures`
- `POST /api/v1/model/train`
- `GET /api/v1/model/active`
- `POST /api/v1/predict`
- `GET /api/v1/predict/options/upcoming`
- `POST /api/v1/predict/upcoming`
- `POST /api/v1/odds/compare`
- `GET /api/v1/odds/upcoming`

### Middleware/config

- CORS configurable por `CORS_ORIGINS`.
- Carpeta `static` servida en `/static` para gráficos de calibración.
- Settings vía `pydantic-settings` (`backend/app/core/config.py`) con fallback a `.env`.

---

## 5) Contratos API (payloads y defaults importantes)

## 5.1 Ingesta

`POST /api/v1/datasets/ingest`

Campos importantes:

- `historical` (obligatorio)
- `football_data_dir` (obligatorio)
- `elo_csv` (opcional)
- `team_map` (opcional)
- `include_manual_results` (default `true`)
- `manual_results_json` (opcional)
- `windows` (default `[5,10]`, se limpian y ordenan)

Salida:

- resumen de filas, missing por columna, columnas generadas y rutas de salida.

## 5.2 Features para fixtures

`POST /api/v1/features/fixtures`

- `fixtures_csv` obligatorio.
- `historical_csv` opcional (si no, usa `data/out/laliga_enriched_all.csv`).
- `windows` default `[5,10]`.

## 5.3 Entrenamiento

`POST /api/v1/model/train`

- `dataset_path` opcional.
- `use_xgb` default `true`.
- `use_catboost` default `true`.
- `calibration`: `platt` o `isotonic` (default `platt`).

## 5.4 Predicción

`POST /api/v1/predict`

- requiere **uno** de:
  - `fixtures_enriched_path`, o
  - `fixtures` (lista de filas ya enriquecidas).

## 5.5 Opciones upcoming

`GET /api/v1/predict/options/upcoming`

query params:

- `include_value` (bool, default `false`)
- `value_threshold` (float, default `0.02`)

Si `include_value=true`, intenta adjuntar `p_H/p_D/p_A`, cuotas y ranking EV por fixture.

## 5.6 Predicción de partido concreto

`POST /api/v1/predict/upcoming`

- `date`
- `home_team`
- `away_team`
- `round` opcional

Devuelve predicción + odds de mercado (si disponibles) para ese partido.

## 5.7 Comparación de cuotas

`POST /api/v1/odds/compare`

- requiere `predictions_csv` o `predictions` inline.
- `odds_kind`: `odds_avg` o `odds_close`.
- `value_threshold` default `0.02`.

## 5.8 Cuotas live

`GET /api/v1/odds/upcoming?limit=100`

Consulta The Odds API, normaliza y persiste snapshot + histórico.

---

## 6) Ingesta y enriquecimiento de datos (detalle interno)

Servicio: `backend/app/services/datasets.py`.

### 6.1 Resolución de rutas

- Acepta rutas absolutas o relativas.
- Las relativas se resuelven desde raíz de repo (`settings.data_dir.parent`).

### 6.2 Normalización de histórico

- Renombra columnas heterogéneas (`Date/HomeTeam/AwayTeam`, `FTHG`, etc.) a formato interno.
- Canoniza nombres de equipos con:
  - mapa de equipos (`team_name_map_es.json`),
  - aliases manuales robustos (acentos, variantes).

### 6.3 Integración de resultados manuales

Si `include_manual_results=true`, intenta integrar partidos ya jugados desde JSON (`proximosPartidos.json`):

- parsea `score.ft` para inferir `result` y goles,
- descarta fechas futuras,
- rellena valores faltantes del histórico,
- añade filas nuevas si no existían,
- guarda histórico aumentado en `data/out/laliga_historical_augmented.csv`.

### 6.4 Unión con football-data

`services/football_data.py`:

- carga todos los CSV del directorio,
- mapea stats (`HS/AS/HST/AST`, etc.),
- construye odds apertura/cierre a partir de columnas candidatas (`AvgH`, `B365H`, ...),
- dedup por (`date`,`home_team`,`away_team`).

### 6.5 Features generadas

- diferenciales básicos (`xg_diff`, `sot_diff`, etc.);
- rolling por equipo y ventana con `shift(1)` (solo info prepartido):
  - `xg_last5_home`, `xg_last5_away`, ...
  - `points_last10_diff`, etc.;
- ELO interno y ELO externo.
- label objetivo `target` (`H=0, D=1, A=2`).

### 6.6 Salidas de ingesta

- `laliga_enriched_all.csv`: dataset amplio.
- `laliga_enriched_model.csv`: subset para modelado sin columnas de fuga explícita.

---

## 7) Feature engineering para fixtures futuros

Servicio principal: `enrich_fixtures()` en `services/features.py`.

Proceso:

1. Normaliza columnas del CSV de fixtures.
2. Canoniza nombres de equipos con `team_map`.
3. Concatena histórico + fixtures marcados como “predicción”.
4. Recalcula rolling/evolutivos temporalmente consistentes.
5. Añade ELO interno y diferenciales.
6. Filtra y devuelve solo las filas de fixtures.

Después, `build_fixtures_features()` añade también ELO externo (si hay archivo) y exporta `fixtures_enriched.csv`.

---

## 8) Entrenamiento, selección de modelo y calibración

Servicio: `backend/app/services/train.py`.

### 8.1 Carga de dataset

- Si no se da `dataset_path`, intenta `data/out/laliga_enriched_model.csv`.
- Si no existe, dispara ingesta automática desde rutas por defecto (`data/historical`, `data/football-data`, etc.).

### 8.2 Prevención de leakage en training

Excluye columnas no válidas para inferencia real:

- identificadores/contexto (`date`, equipos, goles reales, `result`, `target`),
- y señales postpartido o de fuga (`xg_home`, `goal_diff`, ...).

### 8.3 Candidatos de modelo

- `logreg`
- `random_forest`
- `extra_trees`
- `xgb` (si disponible y activado)
- `catboost` (si disponible y activado)

### 8.4 Validación

- `TimeSeriesSplit` dinámico: 2/3/5 folds según tamaño muestral.
- métricas: `log_loss`, `brier`, `ece`, `accuracy`, `f1_macro`.

### 8.5 Selección del mejor

Orden actual del leaderboard:

1. mayor `accuracy`,
2. menor `log_loss`,
3. mayor `f1_macro`,
4. menor `brier`.

### 8.6 Calibración y guardado

- `CalibratedClassifierCV` con `sigmoid` (`platt`) o `isotonic`.
- guarda:
  - `model.pkl`,
  - `metadata.json`,
  - `model_metrics.txt`,
  - `reliability_latest.png`.

---

## 9) Predicción y fallback automático

Servicio: `backend/app/services/predict.py`.

### 9.1 Predicción batch (`predict_matches`)

- recibe fixtures enriquecidos por path o payload.
- si no hay modelo activo, entrena automáticamente (`TrainRequest()` default).
- alinea columnas con `feature_columns` del modelo.
- predice `p_H, p_D, p_A`.
- exporta `data/out/predictions.csv`.

### 9.2 Predicción de partido seleccionado

`predict_selected_upcoming_match()`:

1. Garantiza histórico enriquecido (si falta, autoingesta).
2. Enriquce el partido solicitado.
3. Ejecuta inferencia.
4. Intenta añadir odds de mercado vía `find_fixture_odds`.

### 9.3 Obtención de “upcoming options”

Orden de fuentes para la lista de partidos:

1. JSON manual (`data/fixtures/proximosPartidos.json`),
2. API configurada de fixtures,
3. CSV fallback (si `FIXTURES_ALLOW_CSV_FALLBACK=true`),
4. The Odds API (solo partidos con cuota),
5. fallback demo (últimos históricos) para no dejar UI vacía.

También intenta rellenar `round` (jornada) por matching exacto/fuzzy con fuente manual.

---

## 10) Odds y value bets

### 10.1 Integración de The Odds API

Servicio: `backend/app/services/odds_api.py`.

- Usa `ODDS_API_KEY` o fallback local `oddapikey.txt`.
- Mercado `h2h` en decimal.
- Canoniza equipos y calcula:
  - `odds_avg_h/d/a` (media entre bookmakers),
  - `odds_best_h/d/a` (mejor cuota disponible).
- Persiste:
  - snapshot (`laliga_upcoming_odds.csv`),
  - histórico acumulado (`laliga_odds_history.csv`).

### 10.2 Comparativa modelo vs mercado

`services/evaluation.py`:

1. Convierte cuota a probabilidad implícita y normaliza margen.
2. Calcula EV por signo:
   - `ev_H = p_H * odd_h - 1`, etc.
3. Elige `best_ev_pick` por fila.
4. Marca `value_bet` si `best_ev > threshold`.
5. Si existe `target`, calcula retorno real (profit, ROI, hit rate).

---

## 11) Frontend React actual (estado real)

Archivo central: `frontend/src/App.tsx`.

### 11.1 Navegación activa

- `/` → `DatasetsPage`: lista de partidos, filtros por jornada/búsqueda.
- `/partido` → `MatchPredictionPage`: probabilidades + comparación vs mercado + guía de cuota objetivo.

Rutas legacy (`/training`, `/predict`, `/odds`) redirigen actualmente a `/`.

### 11.2 Lógica de pantalla principal

`DatasetsPage`:

- llama `getUpcomingFixtures()`,
- muestra partidos como cards con logos (fallback inteligente),
- permite filtrar por jornada y texto,
- navega al detalle del partido con query params.

### 11.3 Lógica de partido

`MatchPredictionPage`:

- llama `predictUpcomingFixture` con `date/home/away/round`,
- muestra `p_H/p_D/p_A`,
- si hay cuotas, muestra comparación completa y EV,
- si no hay cuotas, muestra “cuotas necesarias” para superar umbral value.

---

## 12) Configuración de entorno (variables clave)

Variables relevantes en backend/compose:

- `APP_ENV`, `TZ`, `LOG_LEVEL`
- `DATA_DIR`, `OUTPUT_DIR`, `MODEL_DIR`
- `CORS_ORIGINS`
- Fixtures API:
  - `FIXTURES_API_URL`
  - `FIXTURES_API_KEY`
  - `FIXTURES_API_HOST`
  - `FIXTURES_API_TIMEOUT_SEC`
  - `FIXTURES_ALLOW_CSV_FALLBACK`
- Odds API:
  - `ODDS_API_URL`
  - `ODDS_API_KEY`
  - `ODDS_API_SPORT_KEY` (default LaLiga)
  - `ODDS_API_REGIONS`, `ODDS_API_MARKETS`, `ODDS_API_ODDS_FORMAT`, `ODDS_API_DATE_FORMAT`
  - `ODDS_API_BOOKMAKERS`
  - `ODDS_API_TIMEOUT_SEC`

Frontend:

- `VITE_API_BASE_URL`
- `VITE_BACKEND_ORIGIN`

---

## 13) Ejecución operativa

## 13.1 Con Docker Compose (raíz)

```bash
docker compose up --build
```

Servicios:

- API en `http://localhost:8000`
- Front en `http://localhost:5173`

## 13.2 Makefiles

Raíz delega a `infra/Makefile`:

- `make up`
- `make down`
- `make logs`
- `make etl`
- `make train`
- `make predict`

`make train`/`make predict` usan `curl` contra API local.

---

## 14) Dependencias técnicas

Backend (Python 3.11+):

- FastAPI, Uvicorn, Pydantic v2
- Pandas, NumPy
- scikit-learn
- matplotlib
- joblib
- opcionales integradas en pipeline: XGBoost y CatBoost
- QA: pytest, ruff, black, mypy

Frontend:

- React 18 + TypeScript
- Vite
- Axios
- Zustand
- Recharts
- Vitest + Testing Library

---

## 15) Puntos críticos / comportamiento no obvio

1. **Autoentrenamiento**: si no hay `model.pkl`, un `predict` puede disparar entrenamiento.
2. **Autoingesta**: algunos flujos (`train`, `predict/upcoming`) intentan generar outputs faltantes automáticamente.
3. **Fallbacks múltiples de fixtures**: la UI puede mostrar datos demo si no hay fuentes live válidas.
4. **Normalización de equipos agresiva**: hay mapeo manual extenso para salvar diferencias de naming.
5. **Leaderboard prioriza accuracy antes de log_loss** (es decisión importante del criterio de “mejor modelo”).
6. **Integración manual de resultados** por defecto activada en ingesta (`include_manual_results=true`).

---

## 16) Estado de testing

Existe suite de tests backend en `backend/tests/` (health, features, evaluación, odds, rutas de predict, etc.).

Para ejecutar:

```bash
cd backend
pytest -q
```

---

## 17) Guía para otra IA: cómo trabajar bien sobre este repo

Si vas a pasar este proyecto a otra IA, lo ideal es pedirle siempre con este contexto:

1. “Respeta arquitectura actual y no inventes nuevos flujos.”
2. “Antes de tocar nada, lee `backend/app/services/predict.py`, `datasets.py`, `train.py` y `frontend/src/routes/MatchPredictionPage.tsx`.”
3. “Mantén compatibilidad con los outputs en `data/out/` y con los endpoints actuales.”
4. “No rompas los fallbacks de fixtures/odds.”
5. “No metas leakage temporal en features.”

Prompt base recomendado para otra IA:

> “Este repo predice 1X2 de LaLiga con FastAPI + React. Revisa la memoria técnica (`memoria.md`) y aplica cambios mínimos. Mantén los endpoints y artefactos actuales. No cambies contratos sin justificación. Si tocas features o train, preserva no-leakage temporal y calibration outputs.”

---

## 18) Checklist rápido de diagnóstico (cuando algo falla)

1. ¿Está levantada la API (`/api/v1/health`)?
2. ¿Existen datos base en `data/historical` y `data/football-data`?
3. ¿Hay `ODDS_API_KEY` o `oddapikey.txt` si se esperan cuotas live?
4. ¿Se generó `data/out/laliga_enriched_all.csv`?
5. ¿Existe `backend/app/models/store/model.pkl`?
6. ¿Los nombres de equipos están mapeados correctamente en `team_name_map_es.json`?

---

## 19) Conclusión funcional

El proyecto está diseñado para operar como pipeline robusto de predicción futbolística con degradación controlada (fallbacks) para demo y operación local.

La combinación de:

- features temporales,
- calibración,
- evaluación probabilística,
- y comparación directa contra mercado,

permite usarlo tanto como demo académica/profesional como base para iteración analítica posterior.
