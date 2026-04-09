# Capitulo 6. Datos: recopilacion, tratamiento y preparacion para modelado

Este documento recoge, de forma estructurada, el proceso seguido para la recopilacion y el tratamiento de los datos utilizados en el proyecto. Se describen las fuentes de informacion, las tareas de limpieza y transformacion, la ingenieria de caracteristicas y la preparacion final del conjunto de datos para su uso en modelos predictivos 1X2 de LaLiga.

La calidad del dato es un factor determinante en el rendimiento del sistema. Por ello, el pipeline implementado prioriza tres principios:

1. Consistencia semantica entre fuentes heterogeneas.
2. Trazabilidad de cada transformacion.
3. Prevencion de fuga temporal de informacion (data leakage).

## 6.1 Fuentes de datos

El proyecto integra varias fuentes complementarias, cada una con una funcion especifica dentro del flujo de datos:

1. Historico base de partidos.
2. Datos football-data por temporada (estadisticas y cuotas).
3. Resultados manuales recientes en formato JSON.
4. Ratings ELO (externos u obtenidos internamente).
5. Fixtures futuros para inferencia.

### 6.1.0 Procedencia y trazabilidad de las fuentes

En la medida en que el repositorio lo permite, la procedencia de cada bloque de datos es la siguiente:

1. Historico base consolidado en el propio proyecto a partir de CSV locales en data/historical/laliga_merged_matches.csv.
2. Football-Data.co.uk para estadisticas por temporada, cuotas y metadatos del partido. La referencia de columnas del ETL apunta a la documentacion oficial: https://www.football-data.co.uk/notes.txt.
3. The Sports DB como ejemplo de API de fixtures futuros configurable en la aplicacion, visible en la configuracion del proyecto para generar partidos upcoming.
4. The Odds API para cuotas live y snapshots de mercado, usada para obtener probabilidades 1X2 y almacenar historicos de cuotas.
5. ELO snapshots tipo ClubElo / ratings consolidados, si estan disponibles en data/elo/ELO_RATINGS.csv; en caso contrario, el sistema calcula un ELO interno secuencial.
6. Resultados manuales de apoyo en data/fixtures/proximosPartidos.json, utilizados para incorporar partidos ya disputados y completar marcadores pendientes.

### 6.1.1 Historico base

Archivo principal: data/historical/laliga_merged_matches.csv.

Este CSV esta estructurado originalmente a nivel equipo-partido (no a nivel partido unico), con campos como:

- date, season, team, opponent, venue.
- gf, ga, result.
- xg, xga, poss, sh, sot.
- Metadatos contextuales (attendance, referee, etc.).

En el estado actual del repositorio, el fichero historico contiene 4700 registros y 29 columnas (conteo observado en entorno local del proyecto).

### 6.1.2 Football-data por temporada

Directorio: data/football-data/*.csv.

Aporta una segunda capa de informacion orientada a contexto competitivo y mercado:

- Estadisticas del encuentro (tiros, tiros a puerta, tarjetas, corners, etc.).
- Cuotas de apertura y cierre (promedios o fuentes alternativas).
- Variables utiles para contraste con predicciones del modelo.

La documentacion del ETL referencia expresamente la nota oficial de columnas de Football-Data.co.uk.

### 6.1.3 Resultados manuales de partidos recientes

Archivo: data/fixtures/proximosPartidos.json.

Aunque su nombre indica proximos partidos, tambien puede contener marcadores finales (score.ft) y de descanso (score.ht). El pipeline los utiliza para:

1. Completar resultados faltantes del historico.
2. Incorporar partidos ya jugados que aun no estaban consolidados en el CSV principal.
3. Mantener actualizado el entrenamiento con informacion reciente sin esperar a una nueva descarga completa.

### 6.1.4 Ratings ELO

Archivo opcional: data/elo/ELO_RATINGS.csv.

Se utiliza como fuente externa de fuerza relativa por equipo y fecha. Si el fichero no existe, el sistema genera un ELO interno partido a partido para no perder esta senal de fuerza relativa.

### 6.1.5 Fixtures de prediccion

Entrada habitual: data/fixtures/fixtures.csv.

Contiene partidos futuros (fecha, local, visitante y cuotas opcionales) sobre los que se calculan features pre-partido para generar probabilidades 1X2.

## 6.2 Descripcion del conjunto de datos

El flujo de ingestion produce dos datasets principales en data/out:

1. laliga_enriched_all.csv: dataset enriquecido completo (incluye variables de partido y variables derivadas).
2. laliga_enriched_model.csv: subconjunto orientado a modelado, sin variables bloqueadas por riesgo de fuga.

Adicionalmente, se generan:

- laliga_historical_augmented.csv: historico tras normalizacion y aumento con resultados manuales.
- fixtures_enriched.csv: dataset de partidos futuros listo para inferencia.

### 6.2.1 Variables nucleares

Tras la normalizacion y enriquecimiento, el dataset consolida los siguientes bloques:

1. Identificacion del partido: date, season, home_team, away_team.
2. Variables de resultado: home_goals, away_goals, result, target.
3. Estadisticas de contexto y mercado: odds_avg_*, odds_close_*, stats de football-data.
4. Variables ELO: elo_home, elo_away, elo_diff.
5. Variables rolling pre-partido: metricas last5/last10 por equipo y en diferencial.

### 6.2.2 Variable objetivo

La variable objetivo se define como clasificacion multiclase:

- H (victoria local) -> 0
- D (empate) -> 1
- A (victoria visitante) -> 2

Si el campo result no esta informado pero existen goles finales, el sistema lo infiere automaticamente.

## 6.3 Analisis exploratorio de datos (EDA)

El EDA se centra en calidad y cobertura de variables, no solo en estadistica descriptiva basica. El pipeline genera un informe automatico (eda_missing_report.json) con:

1. Porcentaje de missing global por columna.
2. Cobertura por temporada para features criticas (xG/posesion rolling).
3. Trazabilidad de filtros aplicados antes del entrenamiento (min_season y umbral de cobertura).

Este enfoque permite responder preguntas clave para el TFM:

- En que temporadas la informacion es suficientemente completa.
- Que variables tienen mayor riesgo de ruido o ausencia.
- Como impacta el filtrado en el tamano final de muestra.

### 6.3.1 Evidencias recomendadas para memoria

Para documentar visualmente el EDA en el TFM, es recomendable incluir:

1. Curva de cobertura por temporada (xg_last5_home/away, poss_last5_home/away).
2. Diagrama de flujo del pipeline de datos y entrenamiento.
3. Tabla comparativa antes/despues de filtros.

El repositorio incluye scripts para generar estas figuras en docs/screens.

### 6.3.2 Interpretacion de metricas de calidad pre-EDA

En la figura de calidad pre-EDA (5 dimensiones), la metrica de Validez puede aparecer en torno a 25% (por ejemplo, 25.2%). Esto no implica necesariamente que el 75% de los registros presentes sean incorrectos, sino que la definicion aplicada es estricta y exige que cada fila cumpla simultaneamente todas las reglas de dominio consideradas.

Reglas usadas en Validez (antes del tratamiento):

1. home_goals y away_goals deben ser numericos y >= 0.
2. poss_home y poss_away deben estar en [0, 100].
3. sot_home <= sh_home y sot_away <= sh_away.
4. pk_home <= pkatt_home y pk_away <= pkatt_away.

La metrica se calcula como porcentaje de filas que satisfacen todas las reglas a la vez. Por tanto, si una parte importante de columnas de eventos (posesion, tiros, penaltis) viene vacia en el historico crudo, esas filas fallan la validacion conjunta y el porcentaje baja de forma notable.

En el estado actual observado para el historico pre-EDA, las columnas de eventos presentan aproximadamente 74.76% de valores nulos, lo que explica que la Validez conjunta quede cerca de 25.24%.

Conclusiones para interpretar correctamente el resultado:

1. Integridad y Validez no deben confundirse: la primera mide completitud/coherencia minima y la segunda reglas de dominio.
2. Una Validez baja en pre-EDA puede deberse sobre todo a ausencia de datos, no a contradicciones logicas en los datos disponibles.
3. Tras limpieza e imputacion controlada, la cobertura y el cumplimiento de reglas mejoran en el dataset enriquecido.

## 6.4 Limpieza de datos

La limpieza se implementa con reglas explicitas y reproducibles, agrupadas en cinco etapas.

### 6.4.1 Normalizacion de esquema

Se unifican nombres de columnas procedentes de fuentes distintas:

- Date/date -> date
- HomeTeam/home_team -> home_team
- AwayTeam/away_team -> away_team
- FTHG/FTAG -> home_goals/away_goals
- FTR/result -> result

Esto permite operar con un contrato de datos estable a lo largo de todo el pipeline.

### 6.4.2 Estandarizacion de equipos

Se aplica una canonizacion de nombres de club mediante:

1. Mapeo configurable (team_name_map_es.json).
2. Alias manuales para variantes frecuentes.
3. Normalizacion textual (minusculas, eliminacion de tildes y puntuacion).

Con ello se reduce la perdida de joins por diferencias ortograficas entre fuentes.

### 6.4.3 Normalizacion temporal

Las fechas se parsean y homogeneizan en formato ISO (YYYY-MM-DD). Para formatos ambiguos se aplica estrategia robusta de parseo y se descartan registros no interpretables.

### 6.4.4 Reglas de completado y validacion

Se aplican imputaciones deterministas cuando hay coherencia semantica:

- Si xg_home falta y xga_away existe, se usa xga_away como aproximacion.
- Si poss_home falta y poss_away existe, se deduce como 100 - poss_away (y viceversa).
- Se eliminan filas sin claves minimas (date, home_team, away_team).

### 6.4.5 Integridad de resultados

Cuando existe marcador final valido, se valida o reconstruye result. Para datos manuales en JSON:

1. Solo se incorporan partidos con score.ft numerico.
2. Se excluyen fechas futuras (cutoff temporal).
3. Se deduplican claves date-home-away.

## 6.5 Ingenieria de caracteristicas

La ingenieria de caracteristicas busca representar forma deportiva, fortaleza relativa y contexto de mercado sin fuga temporal.

### 6.5.1 Features diferenciales directas

Se construyen diferenciales local-visitante:

- xg_diff, xga_diff, poss_diff, sh_diff, sot_diff, goal_diff.

Estas variables capturan ventaja relativa entre equipos en el partido observado.

### 6.5.2 Features rolling pre-partido

Se genera una tabla longitudinal por equipo y se calculan ventanas moviles (por defecto 5 y 10 partidos), siempre con shift(1), para que cada partido solo vea informacion historica previa.

Ejemplos:

- xg_last5_home, xg_last5_away.
- points_last10_home, points_last10_away.
- xg_last5_diff, points_last10_diff.

### 6.5.3 Features ELO

Se incorporan:

- elo_home y elo_away.
- elo_diff como senal principal de fortaleza relativa.

Si no hay fichero ELO externo, se calcula un ELO interno con actualizacion secuencial por partido (incluyendo ventaja de campo y ajuste por margen de goles).

### 6.5.4 Features de cuotas

Desde football-data se consolidan cuotas de apertura y cierre:

- odds_avg_h, odds_avg_d, odds_avg_a.
- odds_close_h, odds_close_d, odds_close_a.

Estas variables permiten comparar probabilidad modelo vs probabilidad implita del mercado.

## 6.6 Codificacion y normalizacion

### 6.6.1 Codificacion de la variable objetivo

La etiqueta de clase se codifica a enteros:

- H -> 0
- D -> 1
- A -> 2

### 6.6.2 Seleccion de variables para entrenamiento

Para evitar leakage se excluyen explicitamente columnas de partido ya finalizado (por ejemplo goles y estadisticas del propio encuentro). El dataset de modelado conserva unicamente:

1. Variables numericas validas.
2. Features rolling pre-partido.
3. Features ELO.
4. Features de cuotas.

### 6.6.3 Imputacion y escalado

Durante entrenamiento, la transformacion se ejecuta dentro del pipeline de sklearn:

- Imputacion mediana para valores faltantes (SimpleImputer).
- Escalado estandar (StandardScaler) cuando el estimador lo requiere (especialmente regresion logistica).

El ajuste de estas transformaciones se realiza exclusivamente sobre el conjunto de entrenamiento de cada split temporal.

## 6.7 Preparacion final para el modelado

La preparacion final integra calidad de dato, control temporal y persistencia de artefactos.

### 6.7.1 Filtros previos a entrenamiento

Se aplican filtros configurables:

1. min_season: elimina temporadas antiguas no deseadas.
2. xg_poss_min_coverage_pct: conserva solo temporadas con cobertura minima de features criticas.

### 6.7.2 Validacion temporal

La seleccion de modelo se basa en TimeSeriesSplit (2, 3 o 5 particiones segun tamano muestral), priorizando log_loss y usando accuracy/f1_macro como apoyo.

### 6.7.3 Calibracion probabilistica

Tras seleccionar el mejor estimador, se aplica calibracion:

- Platt (sigmoid) o isotonic.
- Si no hay suficientes muestras por clase para calibrar con CV, se mantiene modelo no calibrado.

### 6.7.4 Artefactos finales generados

El proceso deja trazabilidad completa en:

1. data/out/laliga_enriched_all.csv
2. data/out/laliga_enriched_model.csv
3. data/out/eda_missing_report.json
4. data/out/model_metrics.txt
5. backend/app/models/store/model.pkl
6. backend/app/models/store/metadata.json
7. backend/app/static/reliability_latest.png

## Texto sugerido para cerrar el capitulo en la memoria

La fase de datos se ha implementado como un pipeline reproducible y orientado a evitar fuga de informacion temporal. La integracion de fuentes heterogeneas (historico base, football-data, ELO y resultados manuales) se realiza mediante reglas de normalizacion y validacion consistentes, mientras que la ingenieria de caracteristicas prioriza senales pre-partido (rolling, diferenciales y mercado). Este enfoque permite construir un dataset robusto para modelado probabilistico 1X2, mejorando la fiabilidad del entrenamiento y de la evaluacion posterior.

## Como usar este README para redactar el TFM

1. Copia cada subseccion (6.1 a 6.7) en tu memoria y adapta el estilo al formato de tu universidad.
2. Sustituye o amplia cifras descriptivas con la ultima ejecucion del pipeline si actualizas datos.
3. Inserta en anexos las figuras de docs/screens y los informes de data/out como evidencia de reproducibilidad.
4. Anade una tabla final con limitaciones (missing residual, cobertura desigual por temporada, dependencia de calidad de odds) y mejoras futuras.
