# LaLiga Predictor - Base TFM

Proyecto base para predecir resultados 1X2 de LaLiga con Machine Learning, exponer servicios con FastAPI y visualizar resultados en Streamlit, incluyendo simulacion de estrategia **Value Betting**.

## Lo Que Tienes Que Hacer Tu (Checklist)

1. Conseguir un CSV historico de LaLiga (2013-2023 o similar).
2. Verificar que el CSV tiene al menos estas columnas: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`.
3. Guardar el archivo como `data/dataset_clean.csv`.
4. Levantar el proyecto (recomendado con Docker): `docker compose up --build`.
5. Entrenar el modelo en `http://localhost:8501` (pestana "Metricas modelo") o por API (`POST /train`).
6. Probar predicciones en la pestana "Prediccion partido".
7. Ejecutar simulacion de value betting y guardar capturas/metricas para la memoria del TFM.

## Como Encontrar Un CSV De LaLiga

Fuentes recomendadas (academicas y faciles de justificar):

1. Kaggle (buscar: `laliga matches`, `spanish league results`, `football-data`).
2. Football-Data.co.uk (historicos por temporada, muy usado en investigacion aplicada).
3. Zenodo (datasets con DOI, ideal para citar en TFM).
4. GitHub (repos de datos historicos de futbol; revisa licencia antes de usar).

Consejos para elegir dataset:

1. Prioriza datasets con fecha de partido y goles finalizados.
2. Mejor si incluyen jornada, tiros, tiros a puerta y posesion.
3. Revisa licencia de uso para incluirla en la memoria.
4. Evita mezclar competiciones o paises si tu objetivo es solo LaLiga.

## Validacion Rapida Del CSV Antes De Entrenar

Comprobaciones minimas:

1. No hay fechas vacias en `date`.
2. `home_goals` y `away_goals` son numericos.
3. Los nombres de equipos son consistentes (por ejemplo, no mezclar `FC Barcelona` con `Barcelona`).
4. El rango temporal es correcto (ejemplo: 2013-2023).

Si el CSV usa otros nombres de columna, esta base intenta inferirlos automaticamente (por ejemplo `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`).

## Formato Minimo Esperado

Columnas obligatorias:

- `date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`

Columnas opcionales recomendadas (mejoran el modelo):

- `season`, `round`, `FTR`
- `HS`, `AS`, `HST`, `AST`
- `home_possession`, `away_possession`

Ejemplo de una fila:

```csv
date,season,round,home_team,away_team,home_goals,away_goals,FTR,HS,AS,HST,AST
2022-10-16,2022/2023,9,Barcelona,Real Madrid,3,1,H,15,8,7,4
```

## Estructura

```text
laliga_predictor/
├── backend/
│   ├── main.py
│   ├── train.py
│   ├── preprocessing.py
│   ├── predict.py
│   ├── simulate.py
│   ├── api_data.py
│   ├── odds_data.py
│   ├── requirements.txt
│   └── model.pkl (se genera tras /train)
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── dataset_clean.csv (debes colocarlo)
│   └── README.md
└── README.md
```

## Que incluye la base

- Pipeline de features con forma reciente, goles, tiros y ELO.
- Entrenamiento de multiples modelos con validacion temporal (`TimeSeriesSplit`).
- Metricas: `accuracy`, `f1_macro`, `log_loss`, `brier_score`, `ECE`.
- Seleccion automatica del mejor modelo y calibracion de probabilidades.
- Endpoints FastAPI para entrenar, predecir, leer metricas y simular value betting.
- Interfaz Streamlit por pestañas para consumo del backend.

## Ejecucion local

### 1) Backend

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`

### 2) Frontend

```bash
cd frontend
python -m pip install -r requirements.txt
streamlit run app.py
```

## Levantar Todo Con Docker

Requisito: Docker Desktop instalado y en ejecucion.

1) Opcional: crea `.env` en la raiz con tu API key.

```bash
API_FOOTBALL_KEY=tu_api_key
ODDS_API_KEY=tu_api_key_theoddsapi
PREFERRED_BOOKMAKER=bet365
DEFAULT_ODDS_SOURCE=api
```

2) Arranca backend + frontend:

```bash
docker compose up --build
```

3) URLs:
- Backend API: `http://localhost:8000/docs`
- Frontend Streamlit: `http://localhost:8501`

4) Parar servicios:

```bash
docker compose down
```

Notas:
- El backend usa `data/dataset_clean.csv` montado por volumen (`./data:/app/data`).
- Si no existe modelo, primero entrena desde la pestaña de metricas o via `POST /train`.
- Para cuotas reales, configura `ODDS_API_KEY` y usa `odds_source=api` en simulacion.

## Dataset esperado

Coloca tu CSV historico en `data/dataset_clean.csv`.

Columnas minimas:
- `date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`

Columnas recomendadas:
- `season`, `round`, `FTR`
- `HS`, `AS`, `HST`, `AST`
- `home_possession`, `away_possession`

## Endpoints principales

- `POST /train`
- `POST /predict`
- `POST /simulate-value-betting`
- `GET /metrics`
- `GET /feature-importance`
- `GET /fixtures/{season}/{round}`

## Mejores opciones para plantear el TFM (recomendado)

1. Define dos objetivos separados:
- Objetivo A: calidad predictiva (logloss/calibracion).
- Objetivo B: utilidad economica (ROI en backtest).

2. Usa validacion walk-forward estricta por temporada:
- Entrena en temporadas anteriores y valida en la siguiente.
- Evita fuga temporal de informacion.

3. Incluye baseline fuerte y simple:
- Baseline 1: probabilidad por frecuencia historica 1X2.
- Baseline 2: ELO + logistic regression.
- Tus modelos complejos deben superar ambos.

4. Trata calibracion como bloque obligatorio:
- Muestra reliability plots, Brier y ECE.
- En apuestas importa mas una probabilidad bien calibrada que accuracy pura.

5. Simulacion de apuestas con fricciones realistas:
- Margen bookmaker, limites de stake, varianza.
- Reporta curva de capital, max drawdown, Sharpe simplificado.

6. Interpretable first:
- SHAP global + local en partidos de interes.
- Compara interpretabilidad con importancia por permutacion.

7. Reproducibilidad academica:
- Semillas fijas, versionado de datos/modelos, experiment tracking.
- Guarda resultados por corrida en CSV/JSON para anexos.

## Roadmap sugerido en 4 fases

1. Fase datos:
- Limpieza final, control de calidad y diccionario de variables.

2. Fase modelado:
- Comparativa modelos + calibracion + seleccion final.

3. Fase producto:
- FastAPI + Streamlit + visualizaciones.

4. Fase evaluacion TFM:
- Backtesting value betting, analisis de riesgo, conclusiones y limites.

## Limitaciones actuales de esta base

- SHAP aun no esta integrado en endpoint dedicado.
- Las cuotas reales dependen de la cobertura de `TheOddsAPI` y de disponer de API key valida.
- Explorador historico en Streamlit esta en modo placeholder.

## Mejoras aplicadas en esta iteracion

- Contenerizacion completa con `backend/Dockerfile`, `frontend/Dockerfile` y `docker-compose.yml`.
- Endpoint de salud `GET /health` para monitoreo y arranque ordenado.
- Frontend preparado para Docker con `BACKEND_URL` por variable de entorno.
- `.dockerignore` para reducir contexto de build y acelerar imagenes.
- Integracion de cuotas reales con TheOddsAPI (`ODDS_API_KEY`) para Value Betting con `odds_source=api`.

Estas tres piezas son las siguientes a implementar para tener una version defendible de TFM.
