# Calibracion de probabilidades

Este documento describe como esta implementada la calibracion en el proyecto y como se interpreta en la memoria del TFM.

## 1. Se calibra el modelo o todos los modelos

Solo se calibra el modelo final seleccionado.

1. Durante la comparacion de candidatos no se calibra cada modelo por separado.
2. Primero se evaluan los candidatos con validacion temporal.
3. Despues se elige el mejor modelo.
4. Finalmente se calibra ese modelo ganador.

Por tanto, la calibracion forma parte del tramo final del flujo, no de la comparacion entre todos los candidatos.

## 2. Metodos de calibracion usados

Se usan dos metodos, segun la opcion elegida en la peticion de entrenamiento:

1. `platt` -> calibracion tipo sigmoid.
2. `isotonic` -> regresion isotonica.

En el codigo, el metodo concreto se activa con el parametro `calibration` del entrenamiento.

## 3. Momento exacto de aplicacion en el flujo

El orden real del flujo es este:

1. Comparacion de modelos con validacion temporal.
2. Seleccion del mejor modelo, normalmente por `log_loss`.
3. Entrenamiento final del mejor modelo con todo el conjunto disponible.
4. Calibracion de probabilidades del modelo final.

Ese es el comportamiento actual del codigo.

## 4. La calibracion usa validacion interna

Si. La calibracion usa `CalibratedClassifierCV`.

1. Si hay suficientes ejemplos por clase, se usa `cv=3`.
2. Si no, se usa `cv=2`.
3. Si aun asi no hay suficientes muestras por clase, el modelo final se deja sin calibrar.

Importante para explicarlo bien:

- Ese CV de calibracion es interno.
- No es un split temporal tipo walk-forward.
- No es la misma validacion temporal usada para comparar candidatos.

## 5. Que metricas se usan para evaluar el efecto de la calibracion

El proyecto guarda las siguientes salidas sobre el modelo final calibrado:

1. `log_loss`
2. `brier`
3. `ece`
4. `accuracy`
5. `f1_macro`
6. `reliability diagram`

Si quieres centrar el capitulo en calibracion, las metricas principales son:

1. `brier`
2. `ece`
3. `log_loss`
4. la grafica de fiabilidad

No hay otras metricas especificas de calibracion como `ROC-AUC` en este flujo.

## 6. Como se presenta la calibracion en resultados

La calibracion se presenta de forma pragmatica, no con un estudio exhaustivo antes/despues sobre un hold-out temporal separado.

Lo que si genera el codigo es:

1. Las metricas de seleccion del mejor modelo antes de calibrar, obtenidas en validacion temporal.
2. Las `fit_metrics` del modelo final ya calibrado, calculadas sobre el conjunto completo de entrenamiento.
3. La grafica de fiabilidad (`reliability diagram`).

Interpretacion correcta para la memoria:

- El bloque de seleccion de modelo muestra la calidad comparativa temporal.
- El bloque de calibracion muestra la calidad probabilistica del modelo final entrenado.
- No hay un benchmark formal before/after sobre el mismo hold-out temporal en el codigo actual.

## 7. Resumen corto para memoria

1. Solo se calibra el modelo final seleccionado.
2. Se usan Platt scaling (`sigmoid`) e isotonic regression.
3. La calibracion va despues de la seleccion del mejor modelo.
4. La calibracion usa CV interna con `cv=2` o `cv=3`, no temporal.
5. Se reportan `log_loss`, `brier`, `ece`, `accuracy`, `f1_macro` y la reliability diagram.
6. No hay comparativa formal before/after sobre un hold-out separado; se reporta el modelo final calibrado y sus metricas descriptivas.
