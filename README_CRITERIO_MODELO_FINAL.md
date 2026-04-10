# 7.8 Criterio de seleccion del modelo final

Este documento recoge como se decide el modelo final en el codigo actual y como redactarlo con precision metodologica.

## 1. Criterio principal de seleccion

Si, el criterio principal por defecto es minimizar log loss en validacion temporal.

1. La seleccion se hace con una metrica principal configurable.
2. El valor por defecto es log loss.
3. Si no se cambia ese parametro, el modelo ganador es el de menor log loss.

Por tanto, en la redaccion del TFM: no se prioriza accuracy o F1 como criterio principal salvo que se configure explicitamente otra metrica.

## 2. Factores adicionales considerados

Ademas de la metrica principal, el codigo incorpora criterios secundarios de desempate en el ranking:

1. Accuracy.
2. F1-macro.
3. Brier.

Tambien hay una condicion operativa:

1. Candidatos que fallan durante evaluacion se omiten para mantener robustez del entrenamiento.

Importante para no sobreafirmar:

1. No hay un criterio formal explicito de simplicidad o interpretabilidad en la funcion de seleccion.
2. No hay una penalizacion formal por complejidad del modelo.
3. No hay una metrica explicita de integracion web en la decision automatica.

## 3. Jerarquia entre criterios

Si, hay jerarquia clara en codigo.

Caso habitual del proyecto, cuando la metrica principal es log loss:

1. Primero: menor log loss.
2. Segundo: mayor accuracy.
3. Tercero: mayor F1-macro.
4. Cuarto: menor Brier.

Si se cambia la metrica principal a otra, esa metrica pasa a primer lugar y el resto actua como apoyo en el orden definido.

## 4. Pueden ganar los baselines

Si, tecnicamente pueden.

1. Los baselines (dummy) participan en el mismo leaderboard.
2. No existe una regla en codigo que los excluya del modelo final.
3. No existe un umbral obligatorio de mejora sobre baseline para permitir seleccionar el ganador.

En la practica, se usan como referencia, pero el sistema no impide que uno sea elegido si queda primero por la metrica de seleccion.

## 5. Papel de la calibracion en la decision

La calibracion se aplica despues de elegir el mejor candidato.

1. Primero se comparan candidatos en validacion temporal.
2. Luego se selecciona el mejor por la metrica principal.
3. Despues se calibra ese modelo final.

Por tanto:

1. No se comparan entre si todos los candidatos ya calibrados.
2. La calibracion no decide el ganador del leaderboard.

## 6. Criterios de exclusion explicitos

No hay una lista formal extensa de exclusiones por calidad de probabilidades o colapso de clase.

Lo que si existe:

1. Si un candidato lanza excepcion durante evaluacion, se descarta de la comparativa.
2. Si ningun candidato puede evaluarse, el entrenamiento falla con error explicito.

No existe en el codigo actual:

1. Una regla declarada del tipo eliminar modelos por inestabilidad de log loss entre folds.
2. Una regla declarada del tipo eliminar modelos por mala calibracion minima.
3. Una regla declarada del tipo eliminar modelos por complejidad.

## Texto corto recomendado para memoria

1. El modelo final se elige por menor log loss en validacion temporal.
2. Accuracy, F1-macro y Brier actuan como criterios de apoyo para desempatar.
3. Los baselines participan como referencia y, en teoria, podrian ser elegidos al no existir veto explicito.
4. La calibracion se aplica despues de seleccionar el ganador, no durante la comparativa entre candidatos.
5. No hay politicas de exclusion avanzadas mas alla de descartar candidatos que fallan al evaluarse.
