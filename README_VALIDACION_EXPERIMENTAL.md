# Validacion experimental del modelo (TFM)

Este documento describe el esquema de validacion que usa el codigo del proyecto para entrenamiento, comparacion de modelos y control de data leakage.

## 1. La validacion es temporal

Si. La validacion principal es temporal y respeta el orden cronologico de los partidos.

1. El dataset se ordena por fecha.
2. En cada fold se entrena con partidos anteriores.
3. Se valida con partidos posteriores.

Implementacion en codigo:

- modelos/services/train.py (orden cronologico por date_dt)
- modelos/services/train.py (uso de TimeSeriesSplit)
- modelos/services/train.py (iteracion train_idx/test_idx)

## 2. Numero de splits temporales

El numero de splits no es fijo: se adapta al tamano del dataset.

1. 2 splits si hay menos de 60 filas.
2. 3 splits si hay entre 60 y 149 filas.
3. 5 splits si hay 150 filas o mas.

En consecuencia, el esquema es temporal por bloques de partidos en orden cronologico, no por temporadas completas.

Implementacion en codigo:

- modelos/services/train.py (regla de 2, 3 o 5 splits)

## 3. Mismo esquema para todos los modelos

Si. Todos los modelos candidatos se comparan bajo exactamente el mismo esquema de validacion temporal.

Esto incluye los modelos baseline y los modelos avanzados (por ejemplo, dummy, regresion logistica, random forest, extra trees, hist gradient boosting, voting soft, y opcionalmente xgboost/catboost cuando estan disponibles).

Implementacion en codigo:

- modelos/services/train.py (bucle unico de candidatos, todos evaluados con la misma funcion de CV temporal)

## 4. Validacion cruzada aleatoria clasica

No en la comparativa principal de modelos.

1. No se usa train_test_split con barajado aleatorio para seleccionar el mejor modelo.
2. No se usa cross_val_score aleatorio sin control temporal para el ranking principal.

Punto fuerte metodologico: la comparativa central evita mezclar pasado y futuro.

## 5. Donde se ajustan imputacion, escalado y calibracion

### 5.1 Imputacion y escalado

Se ajustan dentro del conjunto de entrenamiento de cada fold, mediante Pipeline.

Esto evita que el bloque de validacion influya en el ajuste de transformaciones (control de leakage en preprocesado).

### 5.2 Calibracion

Aqui hay un matiz relevante:

1. Tras elegir el mejor modelo por validacion temporal, se calibra con CalibratedClassifierCV.
2. Esa calibracion se entrena sobre el conjunto de entrenamiento completo disponible.
3. Su CV interna es cv=2 o cv=3 (segun soporte por clase), y no es un esquema temporal explicito tipo walk-forward.

Conclusion metodologica:

1. La seleccion de modelos y metricas comparativas principales si son temporales.
2. La calibracion final no sigue aun una particion temporal explicita.

## Resumen ejecutivo para memoria

1. Validacion principal temporal con TimeSeriesSplit.
2. Splits dinamicos (2/3/5) segun tamano muestral.
3. Esquema uniforme para todos los modelos.
4. Sin validacion aleatoria clasica en la comparativa principal.
5. Preprocesado dentro de fold para reducir data leakage.
6. Calibracion final con CV interna no temporal (mejora futura: calibracion temporal anidada).
