@echo off
setlocal
cd /d "%~dp0\..\.."
set MPLBACKEND=Agg

echo [1/4] Actualizando cuotas LaLiga (snapshot + historico)...
python backend\tools\refresh_laliga_odds.py --limit 200
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Fallo al actualizar cuotas de LaLiga.
  exit /b %ERRORLEVEL%
)

echo [2/4] Recalculando ROI esperado...
python backend\tools\compute_upcoming_roi.py
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Fallo al recalcular ROI de jornada.
  exit /b %ERRORLEVEL%
)

echo [3/4] Reaplicando settlement real (Semana 29)...
python backend\tools\apply_semana29_results.py
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Fallo al reaplicar resultados reales de Semana 29.
  exit /b %ERRORLEVEL%
)

echo [4/4] Validando resumen final...
python -c "import json,pathlib; p=pathlib.Path('out/roi/roi_upcoming_summary.json'); d=json.loads(p.read_text(encoding='utf-8')); print('[OK] settled_bets=', d.get('settled_bets'), 'realized_profit=', d.get('realized_total_profit_eur'), 'realized_roi=', d.get('realized_roi'))"
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Fallo al validar out\roi\roi_upcoming_summary.json
  exit /b %ERRORLEVEL%
)

echo [OK] Pipeline completado: recompute + reapply + validacion.
endlocal

