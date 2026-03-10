import { useMemo, useState } from "react";

import { MetricsChart } from "../components/charts/MetricsChart";
import { TrainingForm } from "../components/forms/TrainingForm";
import { BACKEND_ORIGIN, TrainPayload, TrainResponse, trainModel } from "../lib/api";
import { useAppStore } from "../lib/state/appStore";

function formatMetric(value: number | undefined): string {
  return value === undefined ? "-" : value.toFixed(4);
}

export function TrainingPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const setToast = useAppStore((state) => state.setToast);
  const setModelMetadata = useAppStore((state) => state.setModelMetadata);

  const handleSubmit = async (payload: TrainPayload) => {
    setLoading(true);
    try {
      const response = await trainModel(payload);
      setResult(response);
      setModelMetadata({
        best_model: response.best_model,
        metrics: response.metrics,
        reliability_plot: response.reliability_plot,
        trained_at: new Date().toISOString()
      });
      setToast(`Modelo entrenado: ${response.best_model}`);
    } catch (error) {
      setToast(`Error al entrenar: ${String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  const metrics = useMemo(() => result?.metrics ?? {}, [result]);

  return (
    <>
      <TrainingForm onSubmit={handleSubmit} loading={loading} />

      {result && (
        <>
          <div className="panel">
            <h3>Resultado entrenamiento</h3>
            <p>
              Mejor modelo: <strong>{result.best_model}</strong>
            </p>
            <p>model.pkl: {result.model_path}</p>
            <p>metadata.json: {result.metadata_path}</p>
            <p>metrics txt: {result.metrics_report_path ?? "-"}</p>
          </div>

          <div className="panel">
            <h3>Metricas clave</h3>
            <div className="metric-grid">
              <div className="metric-card">
                <span>log_loss</span>
                <strong>{formatMetric(metrics.log_loss)}</strong>
              </div>
              <div className="metric-card">
                <span>brier</span>
                <strong>{formatMetric(metrics.brier)}</strong>
              </div>
              <div className="metric-card">
                <span>ECE</span>
                <strong>{formatMetric(metrics.ece)}</strong>
              </div>
              <div className="metric-card">
                <span>accuracy</span>
                <strong>{formatMetric(metrics.accuracy)}</strong>
              </div>
              <div className="metric-card">
                <span>f1_macro</span>
                <strong>{formatMetric(metrics.f1_macro)}</strong>
              </div>
            </div>
          </div>

          <MetricsChart metrics={metrics} />

          {result.reliability_plot && (
            <div className="panel">
              <h3>Curva de calibracion</h3>
              <img
                src={`${BACKEND_ORIGIN}${result.reliability_plot}`}
                alt="Reliability diagram"
                style={{ width: "100%", maxWidth: "860px", borderRadius: "12px" }}
              />
            </div>
          )}

          <div className="panel">
            <h3>Leaderboard modelos</h3>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>log_loss</th>
                  <th>brier</th>
                  <th>ece</th>
                  <th>accuracy</th>
                  <th>f1_macro</th>
                </tr>
              </thead>
              <tbody>
                {result.leaderboard.map((row, idx) => (
                  <tr key={idx}>
                    <td>{String(row.model)}</td>
                    <td>{Number(row.log_loss).toFixed(4)}</td>
                    <td>{Number(row.brier).toFixed(4)}</td>
                    <td>{Number(row.ece).toFixed(4)}</td>
                    <td>{Number(row.accuracy).toFixed(4)}</td>
                    <td>{Number(row.f1_macro).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
