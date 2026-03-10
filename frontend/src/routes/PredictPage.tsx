import { useState } from "react";

import { PredictForm } from "../components/forms/PredictForm";
import { PredictionsTable } from "../components/tables/PredictionsTable";
import {
  FixturesFeaturesPayload,
  PredictPayload,
  PredictResponse,
  buildFixturesFeatures,
  predict
} from "../lib/api";
import { useAppStore } from "../lib/state/appStore";

export function PredictPage() {
  const [buildLoading, setBuildLoading] = useState(false);
  const [predictLoading, setPredictLoading] = useState(false);
  const [featuresOutputPath, setFeaturesOutputPath] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const setToast = useAppStore((state) => state.setToast);

  const handleBuildFeatures = async (payload: FixturesFeaturesPayload) => {
    setBuildLoading(true);
    try {
      const response = await buildFixturesFeatures(payload);
      setFeaturesOutputPath(response.output_path);
      setToast(`Features OK: ${response.rows_total} fixtures enriquecidos`);
    } catch (error) {
      setToast(`Error creando features: ${String(error)}`);
    } finally {
      setBuildLoading(false);
    }
  };

  const handlePredict = async (payload: PredictPayload) => {
    setPredictLoading(true);
    try {
      const response = await predict(payload);
      setPredictionResult(response);
      setToast(`Prediccion OK: ${response.rows} partidos`);
    } catch (error) {
      setToast(`Error en prediccion: ${String(error)}`);
    } finally {
      setPredictLoading(false);
    }
  };

  return (
    <>
      <PredictForm
        onBuildFeatures={handleBuildFeatures}
        onPredict={handlePredict}
        loadingBuild={buildLoading}
        loadingPredict={predictLoading}
      />

      {featuresOutputPath && (
        <div className="panel">
          <h3>Features de fixtures generadas</h3>
          <p>{featuresOutputPath}</p>
        </div>
      )}

      {predictionResult && (
        <>
          <div className="panel">
            <h3>Salida de prediccion</h3>
            <p>rows: {predictionResult.rows}</p>
            <p>csv: {predictionResult.output_csv}</p>
          </div>
          <PredictionsTable rows={predictionResult.predictions} />
        </>
      )}
    </>
  );
}
