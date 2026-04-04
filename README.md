# Info Proyecto

Guia detallada del monorepo para prediccion 1X2 de LaLiga con probabilidades calibradas, comparacion contra cuotas de mercado y demo web profesional.

## 1. Vision y objetivo

Este proyecto implementa un flujo completo de analitica aplicada a futbol para un TFM:

1. Ingestar datos historicos y cuotas desde CSV locales.
2. Construir features pre-partido (sin fuga temporal).
3. Entrenar un modelo multiclase `H/D/A` con validacion temporal.
4. Calibrar probabilidades para mejorar calidad probabilistica.
5. Predecir fixtures futuros.
6. Comparar el modelo contra cuotas de mercado y detectar value bets.
7. Exponer todo por API + frontend React para demo.

El sistema evita scraping automatico y permite usar APIs oficiales de fixtures/cuotas con fallback configurable.

## 2. Estado actual (implementado)

Actualmente el repositorio ya incluye:

1. Backend FastAPI modular con endpoints de salud, ingest, features, entrenamiento, prediccion y odds.
2. ETL CLI offline (`etl/build_laliga_enriched.py`) reutilizando logica del backend.
3. Frontend React + TypeScript (Vite) con 4 pantallas principales.
4. Persistencia de artefactos (`model.pkl`, `metadata.json`) y grafica de calibracion (`reliability_latest.png`).
5. Dockerfiles para backend, frontend y etl.
6. `docker-compose` para entorno local reproducible.
7. Comandos `make` para operaciones comunes.
8. QA base: `ruff`, `black`, `mypy`, `pytest`, `eslint`, `prettier`, `vitest`.

## 3. Arquitectura funcional

Flujo alto nivel:

1. Usuario coloca CSV en `data/`.
2. ETL/API genera datasets enriquecidos en `data/out/`.
3. API entrena y calibra modelo usando `data/out/laliga_enriched_model.csv`.
4. Modelo activo se guarda en `backend/app/models/store/`.
5. API predice fixtures enriquecidos y exporta `data/out/predictions.csv`.
6. API compara predicciones con cuotas y exporta `data/out/predictions_with_odds.csv`.
7. Frontend consume API y muestra datos, metricas, predicciones y value bets.

## 4. Estructura del monorepo

```text
TFM/
  backend/
    app/
      api/v1/                 # Rutas FastAPI
      core/                   # Configuracion y logging
      services/               # Logica de negocio ETL/ML/evaluacion
      models/store/           # model.pkl y metadata.json
      schemas/                # Contratos Pydantic
      tasks/                  # Scheduler opcional
      static/                 # reliability_latest.png
    tests/                    # Tests backend
    pyproject.toml            # Ruff/Black/Mypy/Pytest config
    requirements.txt
    Dockerfile
  etl/
    build_laliga_enriched.py  # CLI offline
    team_name_map_es.json
    requirements.txt
    Dockerfile
  frontend/
    src/
      routes/                 # Datasets, Training, Predict, Odds
      components/
      lib/
    package.json
    tsconfig.json
    vite.config.ts
    Dockerfile
  infra/
    docker-compose.yml
    Makefile
  data/
    historical/
    football-data/
    elo/
    fixtures/
    out/
  docs/screens/
  .env.example
  docker-compose.yml
  Makefile
  README.md
```

## 5. Datos de entrada esperados

### 5.1 Historico base

Ruta recomendada: `data/historical/laliga_merged_matches.csv`

Columnas clave (minimo util):

1. `date`
2. `home_team`
3. `away_team`
4. `home_goals`
5. `away_goals`
6. `result` (si no existe se infiere)

Columnas recomendadas para mejor rendimiento:

1. `season`
2. `xg_home`, `xg_away`
3. `xga_home`, `xga_away`
4. `poss_home`, `poss_away`
5. `sh_home`, `sh_away`
6. `sot_home`, `sot_away`

Nota de actualizacion automatica:

- En `POST /api/v1/datasets/ingest`, si existe `data/fixtures/proximosPartidos.json`, se agregan al historico los partidos ya jugados (incluyendo fechas de 2026) para mejorar el entrenamiento.
- Solo se incorporan encuentros con marcador final (`score.ft`) y fecha no futura para evitar leakage.

### 5.2 Football-data por temporada

Ruta recomendada: `data/football-data/*.csv`

Se usan columnas tipo:

1. `Date`, `HomeTeam`, `AwayTeam`
2. Stats: `HS`, `AS`, `HST`, `AST`, `Referee`, `Attendance`, etc.
3. Odds apertura: `AvgH`, `AvgD`, `AvgA` (y/o bookies alternativos)
4. Odds cierre: `AvgCH`, `AvgCD`, `AvgCA` (si existen)

### 5.3 ELO opcional

Ruta recomendada: `data/elo/ELO_RATINGS.csv`

Columnas requeridas:

1. `Date`
2. `Club`
3. `Elo`

### 5.4 Fixtures futuros

Ruta recomendada: `data/fixtures/fixtures.csv`

Campos esperados:

1. `Date`
2. `HomeTeam`
3. `AwayTeam`
4. Odds opcionales (si quieres comparar al final)

## 6. Ingenieria de features y principios temporales

La logica prioriza evitar leakage temporal:

1. Rolling por equipo con `shift(1)` (solo info hasta `t-1`).
2. Ventanas configurables (por defecto `5,10`).
3. Diferenciales home-away para variables clave.
4. ELO por snapshot historico mas reciente anterior al partido.

Features relevantes:

1. Diferenciales directos: `xg_diff`, `xga_diff`, `poss_diff`, `sh_diff`, `sot_diff`.
2. Rolling por equipo: `xg_last5`, `xg_last10`, `points_last5`, etc.
3. Diferenciales rolling: `xg_last5_diff`, `points_last10_diff`, etc.
4. ELO: `elo_home`, `elo_away`, `elo_diff`.

## 7. Modelado y calibracion

### 7.1 Tarea

Clasificacion multiclase 1X2:

1. `H -> 0`
2. `D -> 1`
3. `A -> 2`

### 7.2 Estrategia de validacion

1. `TimeSeriesSplit` (2/3/5 splits segun tamano de muestra).
2. Ranking principal por `log_loss` (menor mejor).
3. Desempate por `accuracy` (mayor mejor).

### 7.3 Modelos candidatos

1. Logistic Regression multinomial.
2. Random Forest.
3. XGBoost opcional (`use_xgb=true` y dependencia instalada).

### 7.4 Calibracion

Opciones disponibles:

1. `platt` (sigmoid)
2. `isotonic`

Salidas de entrenamiento:

1. `backend/app/models/store/model.pkl`
2. `backend/app/models/store/metadata.json`
3. `backend/app/static/reliability_latest.png`

## 8. API completa

Base URL por defecto: `http://localhost:8000/api/v1`

Documentacion OpenAPI:

1. Swagger UI: `http://localhost:8000/docs`
2. OpenAPI JSON: `http://localhost:8000/openapi.json`

### 8.1 Salud

`GET /health`

Respuesta:

```json
{"status": "ok"}
```

### 8.2 Ingesta

`POST /datasets/ingest`

Payload ejemplo:

```json
{
  "historical": "data/historical/laliga_merged_matches.csv",
  "football_data_dir": "data/football-data",
  "elo_csv": "data/elo/ELO_RATINGS.csv",
  "team_map": "etl/team_name_map_es.json",
  "windows": [5, 10]
}
```

Efecto:

1. Crea `data/out/laliga_enriched_all.csv`
2. Crea `data/out/laliga_enriched_model.csv`

### 8.3 Features para fixtures

`POST /features/fixtures`

Payload ejemplo:

```json
{
  "fixtures_csv": "data/fixtures/fixtures.csv",
  "historical_csv": "data/out/laliga_enriched_all.csv",
  "elo_csv": "data/elo/ELO_RATINGS.csv",
  "team_map": "etl/team_name_map_es.json",
  "windows": [5, 10]
}
```

Efecto:

1. Crea `data/out/fixtures_enriched.csv`

### 8.4 Entrenamiento

`POST /model/train`

Payload ejemplo:

```json
{
  "dataset_path": "data/out/laliga_enriched_model.csv",
  "use_xgb": false,
  "calibration": "platt"
}
```

Retorna:

1. `best_model`
2. `metrics` (`log_loss`, `brier`, `ece`, `accuracy`, `f1_macro`)
3. `leaderboard`
4. rutas de `model.pkl` y `metadata.json`
5. `reliability_plot`

### 8.5 Estado del modelo activo

`GET /model/active`

Retorna si hay modelo entrenado y su metadata.

### 8.6 Prediccion

`POST /predict`

Payload ejemplo:

```json
{
  "fixtures_enriched_path": "data/out/fixtures_enriched.csv"
}
```

Efecto:

1. Crea `data/out/predictions.csv`
2. Devuelve `p_H`, `p_D`, `p_A` por partido

### 8.7 Comparacion odds

`POST /odds/compare`

Payload ejemplo:

```json
{
  "predictions_csv": "data/out/predictions.csv",
  "odds_kind": "odds_avg",
  "value_threshold": 0.02
}
```

Efecto:

1. Calcula probabilidades implicitas del mercado ajustando margen.
2. Compara `log_loss` y `brier` (si existe `target`).
3. Marca value bets por `EV = p_model * odd - 1`.
4. Si existe `target`, calcula rentabilidad real (`value_bets_profit`, `value_bets_roi`, `value_bets_hit_rate`).
5. Exporta `data/out/predictions_with_odds.csv`.

### 8.8 Cuotas live (The Odds API)

`GET /odds/upcoming?limit=100`

Efecto:

1. Consulta The Odds API para partidos upcoming de LaLiga.
2. Devuelve cuotas `1/X/2` en formato decimal (`odds_avg_*` y `odds_best_*`).
3. Expone cabeceras de cuota consumida: `requests_remaining` y `requests_used`.
4. Guarda snapshot actual en `data/out/laliga_upcoming_odds.csv` y acumula historico en `data/out/laliga_odds_history.csv`.

## 9. Frontend (demo)

URL: `http://localhost:5173`

Pantallas:

1. Datasets: formulario de ingesta + tablas de calidad.
2. Entrenamiento: seleccion de calibracion/modelo + metricas + reliability plot.
3. Prediccion: generar `fixtures_enriched` y luego predecir.
4. Odds: comparar predicciones con cuotas y listar value bets.

Detalles UX actuales:

1. Header con estado de modelo activo.
2. Mensajes tipo toast para exito/error.
3. Base i18n ES/EN.
4. Layout responsive para desktop y movil.

## 10. Ejecucion rapida (5 minutos)

### 10.1 Requisitos

1. Docker Desktop activo.
2. Opcional: `make`, `curl` en host.

### 10.2 Pasos

1. Copiar entorno:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

2. Colocar CSV en rutas `data/*`.

3. Levantar servicios:

```bash
docker compose up --build
```

4. Ejecutar ETL + train + predict:

```bash
cd infra
make etl
make train
make predict
```

5. Abrir:

1. `http://localhost:8000/docs`
2. `http://localhost:5173`

## 11. Comandos operativos

### 11.1 Make raiz

```bash
make up
make etl
make train
make predict
make logs
make down
```

### 11.2 Make en `infra/`

```bash
cd infra
make up
make etl
make train
make predict
make logs
make down
```

### 11.3 Alternativas sin make

ETL:

```bash
docker compose run --rm etl --hist data/historical/laliga_merged_matches.csv --fdata_dir data/football-data --elo data/elo/ELO_RATINGS.csv --fixtures data/fixtures/fixtures.csv --windows 5,10
```

Train:

```bash
curl -X POST "http://localhost:8000/api/v1/model/train" -H "Content-Type: application/json" -d "{\"dataset_path\":\"data/out/laliga_enriched_model.csv\",\"use_xgb\":false,\"calibration\":\"platt\"}"
```

Predict:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" -H "Content-Type: application/json" -d "{\"fixtures_enriched_path\":\"data/out/fixtures_enriched.csv\"}"
```

## 12. Variables de entorno

Plantilla en `.env.example`:

1. `APP_ENV=dev`
2. `TZ=Europe/Madrid`
3. `LOG_LEVEL=INFO`
4. `DATA_DIR=./data`
5. `OUTPUT_DIR=./data/out`
6. `MODEL_DIR=./backend/app/models/store`
7. `CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
8. `VITE_API_BASE_URL=http://localhost:8000/api/v1`
9. `VITE_BACKEND_ORIGIN=http://localhost:8000`
10. `FIXTURES_API_URL=https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id=4335`
11. `FIXTURES_API_KEY=`
12. `FIXTURES_ALLOW_CSV_FALLBACK=false`
13. `ODDS_API_KEY=`
14. `ODDS_API_SPORT_KEY=soccer_spain_la_liga`
15. `ODDS_API_REGIONS=eu`
16. `ODDS_API_MARKETS=h2h`
17. `ODDS_API_ODDS_FORMAT=decimal`

## 13. Calidad, tests y estandares

Objetivo de calidad:

1. Codigo tipado y legible.
2. Linting y formato consistentes.
3. Tests de humo y metricas core.
4. Base para escalar cobertura a >= 80%.

Backend:

```bash
cd backend
python -m pip install -r requirements.txt
ruff check .
black --check .
mypy app tests
pytest
```

Frontend:

```bash
cd frontend
npm install
npm run lint
npm run test
npm run build
```

## 14. Troubleshooting

### 14.1 Error de columnas faltantes

1. Verifica nombres reales en tus CSV.
2. Revisa mapeo `etl/team_name_map_es.json`.
3. Comprueba que `date/home_team/away_team` existen tras normalizacion.

### 14.2 No aparece `reliability_latest.png`

1. Asegura que `POST /model/train` termino correctamente.
2. Revisa permisos de escritura en `backend/app/static/`.

### 14.3 `predict` falla por features faltantes

1. Ejecuta antes `POST /features/fixtures`.
2. Usa como entrada `data/out/fixtures_enriched.csv`.

### 14.4 `odds/compare` falla por cuotas faltantes

1. El CSV debe contener `odds_avg_h/d/a` o `odds_close_h/d/a`.
2. Ajusta `odds_kind` segun columnas disponibles.

### 14.5 Problemas en Windows

1. Si no tienes `make`, usa comandos `docker compose` y `curl` directos.
2. Si no tienes `curl`, usa Swagger UI en `/docs` para ejecutar endpoints.

### 14.6 The Odds API sin cuotas

1. Verifica `ODDS_API_KEY` o el fichero `oddapikey.txt` en la raiz del repo.
2. Revisa que `ODDS_API_SPORT_KEY` sea `soccer_spain_la_liga`.
3. Comprueba cuota de peticiones en `requests_remaining`.

## 15. Seguridad, legal y gobierno de datos

1. No hay scraping automatico por defecto.
2. El proyecto trabaja con CSV locales del usuario.
3. Cualquier conector externo debe tratarse como plugin opcional desactivado por defecto.
4. No se deben commitear secretos reales en `.env`.
5. Revisar licencias de datasets antes de publicacion academica.

## 16. Definition of Done (checklist operativo)

1. `docker compose up` levanta API + frontend + etl.
2. `POST /api/v1/datasets/ingest` genera ambos CSV enriquecidos.
3. `POST /api/v1/model/train` guarda `model.pkl` + `metadata.json`.
4. `POST /api/v1/predict` devuelve `p_H`, `p_D`, `p_A`.
5. `POST /api/v1/odds/compare` calcula EV y value bets.
6. Frontend muestra datasets, metricas, predicciones y comparativa.

## 17. Capturas para memoria TFM

Guardar imagenes en `docs/screens/`:

1. `docs/screens/datasets.png`
2. `docs/screens/training.png`
3. `docs/screens/predict.png`
4. `docs/screens/odds.png`

## 18. Roadmap recomendado

1. Backtest rodante por jornada (train hasta `t`, evalua `t+1`).
2. Pagina de explicabilidad (importance, permutation, PDP).
3. Exportes adicionales de curvas de calibracion y reportes PDF.
4. Autenticacion ligera para proteger `/model/train` y `/predict`.
5. Pipeline CI local con ejecucion automatica de lint/tests/build.

## 19. Nota final

Este README (Info Proyecto) documenta el estado funcional actual del monorepo y sirve como base de defensa tecnica del TFM.

