# ROI tracking (apuestas jornada)

Este directorio guarda:

- `roi_upcoming_summary.json`: resumen de ROI esperado y (si hay resultados) ROI real.
- `roi_upcoming_detail.csv`: detalle por partido (EV, pick, stake, bankroll esperado).
- `roi_expected_profit_by_match.png`: gráfico beneficio esperado por partido.
- `roi_cumulative_bankroll.png`: gráfico bankroll acumulado esperado.
- `settlement_realized.csv`: liquidación real (ganado/perdido) cuando existan marcadores finales.
- `odds_snapshots/`: snapshots históricos de cuotas descargadas.
- `update_roi_jornada.bat`: ejecución rápida (doble clic) para recalcular todo.
- Integración automática de resultados reales desde API (TheSportsDB) para no cargar marcadores a mano.

## Qué decirle al asistente para actualizar todo

Usa este mensaje:

"Actualiza ROI de próximas jornadas: vuelve a leer cuotas, recalcula predicciones con el modelo, regenera CSV y gráficos, y liquida si ya hay resultados."

## Comando directo (terminal)

```bash
python backend/tools/compute_upcoming_roi.py
```

## Automatizar para cada jornada (Windows)

1. Ejecuta manualmente con doble clic:

```bat
backend\tools\update_roi_jornada.bat
```

2. Para programarlo automáticamente cada día a las 09:00:

```powershell
powershell -ExecutionPolicy Bypass -File backend/tools/register_weekly_roi_task.ps1
```

Esto crea la tarea `TFM_ROI_Update_Jornada` en el Programador de tareas.

## Cómo ver si habríamos ganado o perdido esta jornada

1. Revisa `settlement_realized.csv`.
2. Columnas clave:
   - `bet_won` (`True`/`False`)
   - `realized_profit_eur` (beneficio por apuesta: gana `odds-1`, pierde `-1`)
   - `realized_bankroll_cumulative_eur`
3. En `roi_upcoming_summary.json` verás:
    - `next_jornada_expected_bankroll_eur`
    - `next_jornada_realized_bankroll_eur`
   - `settled_bets`
   - `realized_total_profit_eur`
   - `realized_roi`

Si todavía no hay resultados finales en `data/fixtures/proximosPartidos.json`, esas métricas saldrán como `null`.

## Fuente de resultados reales (automática)

1. Primero intenta leer resultados reales desde API externa.
2. Después aplica cualquier resultado manual de `data/fixtures/proximosPartidos.json` (si existe) como prioridad final.

Si no hay Internet o la API no responde, la liquidación seguirá funcionando con resultados manuales.
