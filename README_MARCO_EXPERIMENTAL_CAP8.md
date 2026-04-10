# 8. Definicion y ejecucion del marco experimental

Este documento resume como esta implementado en codigo el marco experimental del proyecto.

## 8.1 Flujo general de ejecucion experimental

El flujo operativo completo se ejecuta en esta secuencia:

1. Ingesta y enriquecimiento historico.
2. Generacion de features para fixtures futuros.
3. Entrenamiento y comparativa de candidatos con validacion temporal.
4. Seleccion del mejor modelo segun metrica principal.
5. Entrenamiento final y calibracion del modelo ganador.
6. Persistencia de artefactos de modelo y reportes.
7. Prediccion de partidos y comparacion con cuotas.

Vias de ejecucion disponibles:

1. API FastAPI (`/api/v1/datasets/ingest`, `/api/v1/features/fixtures`, `/api/v1/model/train`, `/api/v1/predict`, `/api/v1/odds/compare`).
2. Scripts y tareas operativas (`make` en raiz o en `infra/`, y utilidades en `web/backend/tools`).

## 8.2 Ejecucion de la validacion temporal

La validacion de candidatos se realiza con `TimeSeriesSplit` sobre datos ordenados cronologicamente por fecha.

Regla de numero de splits:

1. 2 splits si el dataset tiene menos de 60 filas.
2. 3 splits si tiene entre 60 y 149 filas.
3. 5 splits si tiene 150 filas o mas.

Mecanica de evaluacion:

1. En cada fold se entrena con indices temporales anteriores y se valida en bloque posterior.
2. Se obtienen probabilidades por fold.
3. Se concatenan todas las predicciones out-of-fold.
4. Las metricas se calculan sobre el conjunto concatenado, no por mejor fold puntual.

Consecuencia metodologica:

1. No hay mezcla pasado/futuro en la comparativa principal.
2. No se usa validacion aleatoria clasica (`shuffle=True`) para seleccionar el modelo.

## 8.3 Entrenamiento y evaluacion de modelos

Familias de modelos candidatas:

1. Baselines: `dummy_prior`, `dummy_most_frequent`, `dummy_uniform`.
2. Regresion logistica: `logreg`, `logreg_balanced`.
3. Ensambles: `random_forest`, `extra_trees`, `hist_gb`, `voting_soft`.
4. Opcionales por dependencia: `xgb` y `catboost`.

Preprocesado aplicado:

1. Imputacion (`SimpleImputer`) en todos los candidatos.
2. Escalado (`StandardScaler`) en las ramas logisticas.
3. Ejecucion mediante `Pipeline` para evitar leakage de transformaciones en validacion.

Metricas evaluadas en todos los candidatos:

1. `log_loss`
2. `brier`
3. `ece`
4. `accuracy`
5. `f1_macro`

Notas de robustez del entrenamiento:

1. Candidatos que fallan en algun punto se omiten y el proceso continua.
2. Si todos fallan, se devuelve error explicito de validacion temporal.

## 8.4 Seleccion del modelo final

La seleccion se hace sobre el leaderboard de metricas CV temporales.

Criterio principal:

1. Metrica configurable (`selection_metric`) con valor por defecto `log_loss`.

Jerarquia de ordenacion:

1. Si la metrica principal es de minimizacion (`log_loss`, `brier`, `ece`), se ordena ascendente por esa metrica.
2. En empate, se apoya en `accuracy`, `f1_macro` y `brier`.
3. Si la metrica principal es de maximizacion (`accuracy`, `f1_macro`), se invierte el orden principal y se desempata con `log_loss`, `f1_macro` y `brier`.

Baselines en la decision:

1. Los baselines participan en el leaderboard.
2. No existe veto formal para impedir que un baseline sea elegido si queda primero.
3. Ademas se calcula mejora relativa frente a baseline como anotacion del reporte.

## 8.5 Entrenamiento final y calibracion

Una vez seleccionado el mejor candidato:

1. Se reconstruye su pipeline.
2. Se ajusta en el conjunto completo filtrado de entrenamiento.
3. Se aplica calibracion probabilistica del modelo final.

Metodos de calibracion soportados:

1. `platt` (sigmoid).
2. `isotonic`.

CV interna para calibracion:

1. `cv=3` cuando la clase minoritaria tiene al menos 3 muestras.
2. `cv=2` cuando la clase minoritaria tiene al menos 2 muestras.
3. Sin calibracion (`none`) si no hay soporte minimo por clase.

Importante:

1. La calibracion se aplica despues de la seleccion final.
2. No se comparan todos los candidatos ya calibrados entre si.
3. La CV de calibracion es interna y no temporal explicita.

## 8.6 Generacion y persistencia de artefactos

Artefactos principales que genera el flujo:

1. `data/out/laliga_historical_augmented.csv`.
2. `data/out/laliga_enriched_all.csv`.
3. `data/out/laliga_enriched_model.csv`.
4. `data/out/fixtures_enriched.csv`.
5. `data/out/predictions.csv`.
6. `data/out/predictions_with_odds.csv`.
7. `data/out/model_metrics.txt`.
8. `data/out/eda_missing_report.json`.
9. `web/backend/app/static/reliability_latest.png`.
10. `web/backend/app/models/store/model.pkl`.
11. `web/backend/app/models/store/metadata.json`.

Contenido util de metadata:

1. Timestamp de entrenamiento.
2. Dataset usado.
3. Columnas de features.
4. Modelo ganador y metrica de seleccion.
5. Metricas CV y fit_metrics.
6. Leaderboard completo.
7. Rutas a reportes y grafica de calibracion.

## 8.7 Consideraciones practicas de ejecucion

Consideraciones de entorno y operacion:

1. `xgboost` y `catboost` son opcionales; si no estan disponibles, no se incluyen como candidatos.
2. Existen semillas fijas (`random_state=42`) en varios modelos para mejorar reproducibilidad.
3. `predict` incluye fallback de cold-start: si no hay modelo activo, dispara entrenamiento automatico baseline-friendly (`use_xgb=false`, `use_catboost=false`).
4. Si falta el dataset de entrenamiento por defecto, el servicio puede intentar generarlo via ingesta automatica si existen fuentes base.
5. Las rutas API mapean errores a HTTP 400/404 con mensaje explicito.

Operacion recomendada para ejecucion completa:

1. Levantar servicios (`make up` o `docker compose -f infra/docker-compose.yml up --build`).
2. Ejecutar ingesta (`/api/v1/datasets/ingest`).
3. Generar features de fixtures (`/api/v1/features/fixtures`).
4. Entrenar (`/api/v1/model/train`).
5. Predecir (`/api/v1/predict`).
6. Comparar cuotas (`/api/v1/odds/compare`).

Soporte para analisis adicional:

1. `web/backend/tools/backtest_walkforward.py` implementa un backtest walk-forward separado para analisis de estrategias.
2. Scripts en `web/backend/tools` permiten generar figuras y reportes para la memoria.

## Resumen ejecutivo del capitulo 8

1. El marco experimental esta implementado como pipeline reproducible de ingesta, validacion temporal, seleccion, calibracion y persistencia.
2. La comparativa de modelos se apoya en metricas probabilisticas y ranking configurable, con `log_loss` como criterio por defecto.
3. La calibracion se realiza solo sobre el modelo final, con control de soporte por clase.
4. El sistema deja trazabilidad de resultados mediante artefactos versionables en `data/out` y `model_store`.
5. Existen mecanismos practicos de robustez para entorno real (fallbacks, candidatos opcionales, manejo de errores y comandos operativos).
