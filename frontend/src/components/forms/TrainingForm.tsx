import { FormEvent, useState } from "react";

import type { TrainPayload } from "../../lib/api";

type Props = {
  onSubmit: (payload: TrainPayload) => Promise<void>;
  loading: boolean;
};

export function TrainingForm({ onSubmit, loading }: Props) {
  const [datasetPath, setDatasetPath] = useState("data/out/laliga_enriched_model.csv");
  const [useXgb, setUseXgb] = useState(false);
  const [calibration, setCalibration] = useState<"platt" | "isotonic">("platt");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await onSubmit({
      dataset_path: datasetPath,
      use_xgb: useXgb,
      calibration
    });
  };

  return (
    <form className="panel" onSubmit={submit}>
      <h2>2) Entrenamiento y calibracion</h2>
      <div className="grid-form">
        <div className="field">
          <label>Dataset path</label>
          <input value={datasetPath} onChange={(event) => setDatasetPath(event.target.value)} />
        </div>

        <div className="field">
          <label>Calibracion</label>
          <select
            value={calibration}
            onChange={(event) => setCalibration(event.target.value as "platt" | "isotonic")}
          >
            <option value="platt">Platt (sigmoid)</option>
            <option value="isotonic">Isotonica</option>
          </select>
        </div>

        <div className="field">
          <label>Modelo XGBoost</label>
          <select
            value={useXgb ? "yes" : "no"}
            onChange={(event) => setUseXgb(event.target.value === "yes")}
          >
            <option value="no">No</option>
            <option value="yes">Si (si esta instalado)</option>
          </select>
        </div>
      </div>
      <div className="actions">
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Entrenando..." : "Entrenar modelo"}
        </button>
      </div>
    </form>
  );
}
