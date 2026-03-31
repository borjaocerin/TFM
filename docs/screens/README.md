Guarda aqui las capturas de la demo:
- datasets.png
- training.png
- predict.png
- odds.png

Figuras generadas para memoria tecnica:
- model_primary_metric_comparison.png
- model_leaderboard_cv.png
- model_metrics_cv_vs_fit.png
- eda_missing_coverage_by_season.png
- eda_pipeline_flow.png



En este trabajo, la métrica principal de selección de modelos es log_loss, porque el objetivo no es solo acertar el signo 1X2, sino estimar probabilidades fiables para cada resultado (H, D, A). A diferencia de accuracy, que únicamente evalúa si la clase más probable coincide con la real, log_loss valora toda la distribución de probabilidad y penaliza con fuerza los errores cometidos con alta confianza. Por tanto, un modelo con menor log_loss no solo clasifica razonablemente bien, sino que además produce probabilidades mejor calibradas, lo que resulta especialmente relevante en escenarios de decisión probabilística (por ejemplo, análisis de valor esperado en apuestas). En consecuencia, en el pipeline de entrenamiento se selecciona el mejor modelo minimizando log_loss, manteniendo accuracy, f1_macro, brier y ece como métricas complementarias para evaluar rendimiento discriminativo y calibración.