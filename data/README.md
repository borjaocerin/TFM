Estructura de datos esperada para el monorepo:

- `historical/laliga_merged_matches.csv` -> historico base por partido
- `football-data/*.csv` -> CSV por temporada con odds/stats
- `elo/ELO_RATINGS.csv` -> ELO opcional
- `fixtures/fixtures.csv` -> partidos futuros
- `out/` -> salidas del ETL y predicciones

Todo el flujo funciona con CSV locales. No hay scraping automatico por defecto.

