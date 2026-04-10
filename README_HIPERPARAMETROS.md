# Ajuste de hiperparametros y seleccion de modelos

Este documento describe como esta implementado el ajuste de hiperparametros en el proyecto. La conclusion principal es simple: no hay una busqueda exhaustiva de hiperparametros; el codigo usa configuraciones fijas y razonadas por familia de modelo, y luego selecciona el mejor candidato con validacion temporal.

## 1. Hay ajuste explicito de hiperparametros

No hay tuning exhaustivo tipo grid search, random search u Optuna.

1. No se usa GridSearchCV.
2. No se usa RandomizedSearchCV.
3. No se usa una optimizacion automatica de hiperparametros.
4. Los modelos se instancian con valores fijos elegidos manualmente.

Por tanto, la respuesta correcta para la memoria es: ajuste limitado y manual, no tuning exhaustivo.

## 2. En que modelos se ajustan hiperparametros

Si se fijan algunos hiperparametros por familia de modelo, aunque no exista un proceso de busqueda.

### Regresion logistica

Se usan parametros concretos como:

1. `max_iter`
2. `class_weight` en la variante balanceada

No se realiza busqueda sobre `C`, `solver` o combinaciones alternativas.

### Random Forest

Se fijan parametros como:

1. `n_estimators`
2. `max_depth`
3. `min_samples_leaf`
4. `n_jobs`
5. `random_state`

### Extra Trees

Se fijan parametros como:

1. `n_estimators`
2. `min_samples_leaf`
3. `class_weight`
4. `n_jobs`
5. `random_state`

### HistGradientBoosting

Se fijan parametros como:

1. `learning_rate`
2. `max_depth`
3. `max_iter`
4. `random_state`

### XGBoost y CatBoost, si estan disponibles

Tambien se cargan con configuraciones fijas, por ejemplo:

1. `n_estimators` o `iterations`
2. `learning_rate`
3. `max_depth` o `depth`
4. `subsample` y `colsample_bytree` en XGBoost

### Baselines Dummy

No hay ajuste real de hiperparametros.

1. `dummy_prior`
2. `dummy_most_frequent`
3. `dummy_uniform`

## 3. Usa validacion temporal para ajustar hiperparametros

No, porque no existe una fase de tuning independiente.

1. El esquema temporal se usa para comparar candidatos de modelo.
2. Los hiperparametros no se optimizan con una busqueda interna.
3. Cada familia entra directamente con su configuracion fija dentro del mismo proceso de validacion temporal.

En otras palabras: la validacion temporal si existe, pero no hay una capa adicional de tuning sobre esa validacion.

## 4. Metrica usada para el ajuste

No hay una metrica de tuning en sentido estricto porque no hay optimizacion de hiperparametros.

Lo que si existe es la metrica principal para seleccionar el mejor modelo entre candidatos:

1. Por defecto, `log_loss`.
2. El proyecto tambien permite `accuracy`, `f1_macro`, `brier` y `ece` como criterio de seleccion.

Si necesitas una frase breve para la memoria: el criterio principal de seleccion es `log_loss`, pero no se usa para una busqueda automatica de hiperparametros, sino para ordenar candidatos ya definidos.

## 5. El ajuste es exhaustivo o limitado

Es claramente limitado.

1. No hay exploracion masiva del espacio de hiperparametros.
2. No hay combinatoria automatica sobre muchas opciones.
3. Las configuraciones son pequenas, fijas y razonadas.

Esto encaja mejor con un enfoque practico y reproducible para TFM.

## 6. Relacion con el modelo final

El orden real del flujo es este:

1. Primero se definen candidatos con hiperparametros fijos.
2. Luego se evalua cada candidato con validacion temporal.
3. Despues se elige el mejor segun la metrica principal.
4. Finalmente se calibra ese modelo ganador.

Importante:

1. No se hace tuning separado solo para los modelos fuertes.
2. Tampoco se hace una fase de ajuste previa y luego otra de seleccion.
3. Todos los candidatos relevantes compiten bajo el mismo esquema temporal.

## Resumen para memoria

1. No hay Grid Search ni Random Search.
2. Hay configuracion manual y fija de hiperparametros por familia de modelo.
3. La seleccion final usa validacion temporal, no tuning automatizado.
4. `log_loss` es la referencia principal para ordenar candidatos.
5. El ajuste es limitado, no exhaustivo.
6. El modelo final sale de la comparativa de candidatos ya configurados, no de una optimizacion previa del espacio de hiperparametros.
