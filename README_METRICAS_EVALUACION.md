# Metricas de evaluacion del modelo

Este documento resume el esquema de metricas que usa el proyecto para comparar modelos y reportar resultados.

## 1. Metrica principal para seleccionar modelos

Si. La metrica principal es `log_loss`.

1. La seleccion del mejor modelo se hace por una metrica configurable.
2. En la configuracion por defecto del proyecto, esa metrica es `log_loss`.
3. `log_loss` se minimiza, por eso es la referencia principal para ranking de modelos.

En el codigo, la API acepta como criterio de seleccion `log_loss`, `accuracy`, `f1_macro`, `brier` y `ece`, pero el valor por defecto es `log_loss`.

## 2. Metricas que se calculan siempre

Si. Las metricas que se calculan para todos los modelos candidatos son estas cinco:

1. `log_loss`
2. `accuracy`
3. `f1_macro`
4. `brier`
5. `ece` (Expected Calibration Error)

No aparece `ROC-AUC` en el flujo de entrenamiento actual.

## 3. Como se agregan las metricas entre folds

No se selecciona el mejor modelo por el mejor fold puntual.

1. El esquema temporal genera varios folds.
2. En cada fold se obtienen predicciones probabilisticas sobre el bloque de validacion.
3. Esas predicciones se concatenan para formar un unico conjunto out-of-fold.
4. Las metricas finales se calculan sobre ese conjunto agregado.

Por tanto, la comparacion entre modelos usa una agregacion global de todos los folds temporales, no la media aritmetica fold a fold.

## 4. Sobre que tipo de salida se calculan las metricas

Si, la separacion es esta:

1. `log_loss`, `brier` y `ece` se calculan sobre probabilidades predichas.
2. `accuracy` y `f1_macro` se calculan sobre la clase predicha final.

Eso es coherente con la forma en la que se evalua un clasificador probabilistico multiclase 1X2.

## 5. Cuanto entra la calibracion en las metricas

Aqui hay dos momentos distintos:

1. Durante la comparacion temporal de modelos, las metricas se calculan sobre la salida probabilistica del pipeline evaluado en validacion temporal.
2. Una vez elegido el mejor modelo, se entrena la version final calibrada y se vuelven a calcular metricas sobre el ajuste final como `fit_metrics`.

Importante:

- Las metricas de seleccion sirven para comparar modelos entre si.
- Las metricas de `fit_metrics` son solo descriptivas y se calculan sobre el modelo final ya calibrado.

En otras palabras:

1. `log_loss`, `brier` y `ece` ya forman parte de la evaluacion temporal de seleccion.
2. Tras calibrar el mejor modelo, esas metricas se vuelven a medir solo como referencia del modelo final entrenado en todo el conjunto.

## 6. Metricas de seleccion vs metricas descriptivas

Si, esta distincion existe y es importante dejarla clara.

### Metricas de seleccion

1. `log_loss`
2. `accuracy`
3. `f1_macro`
4. `brier`
5. `ece`

Estas metricas se usan para construir el leaderboard y elegir el mejor modelo segun la metrica principal configurada.

### Metricas descriptivas

1. Las `fit_metrics` del modelo final calibrado.
2. La grafica de reliability.
3. El reporte final `model_metrics.txt`.

Estas salidas ayudan a interpretar el modelo ya entrenado, pero no sustituyen la comparacion temporal principal.

## Resumen corto para memoria

1. Metrica principal por defecto: `log_loss`.
2. Metricas siempre calculadas: `log_loss`, `accuracy`, `f1_macro`, `brier`, `ece`.
3. No se usa `ROC-AUC` en el flujo actual.
4. Las metricas se agregan sobre predicciones out-of-fold concatenadas, no por el mejor fold.
5. `log_loss`, `brier` y `ece` trabajan con probabilidades; `accuracy` y `f1_macro`, con clases finales.
6. La calibracion final genera metricas descriptivas adicionales, separadas de la seleccion temporal.
