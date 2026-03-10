import { useEffect, useMemo, useState } from "react";

import {
  PredictUpcomingResponse,
  UpcomingFixtureOption,
  getUpcomingFixtures,
  predictUpcomingFixture
} from "../lib/api";
import { useAppStore } from "../lib/state/appStore";

export function DatasetsPage() {
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [fixtures, setFixtures] = useState<UpcomingFixtureOption[]>([]);
  const [seasonLabel, setSeasonLabel] = useState<string>("");
  const [sourcePath, setSourcePath] = useState<string>("");
  const [selectedFixtureId, setSelectedFixtureId] = useState<string>("");
  const [predictionResult, setPredictionResult] = useState<PredictUpcomingResponse | null>(null);
  const setToast = useAppStore((state) => state.setToast);

  const selectedFixture = useMemo(
    () => fixtures.find((fixture) => fixture.fixture_id === selectedFixtureId) ?? null,
    [fixtures, selectedFixtureId]
  );

  const loadFixtures = async () => {
    setLoadingOptions(true);
    try {
      const response = await getUpcomingFixtures();
      setFixtures(response.fixtures);
      setSeasonLabel(response.season_label);
      setSourcePath(response.source_path);

      if (response.fixtures.length > 0) {
        setSelectedFixtureId(response.fixtures[0].fixture_id);
        setToast(`Cargados ${response.fixtures.length} partidos futuros`);
      } else {
        setSelectedFixtureId("");
        setToast(
          "No hay partidos futuros en CSV ni en la API configurada. Revisa FIXTURES_API_URL o tus datos locales."
        );
      }
    } catch (error) {
      setToast(`Error cargando partidos: ${String(error)}`);
    } finally {
      setLoadingOptions(false);
    }
  };

  const handlePredictSelected = async () => {
    if (!selectedFixture) {
      setToast("Selecciona un partido para predecir");
      return;
    }

    setLoadingPredict(true);
    try {
      const response = await predictUpcomingFixture({
        date: selectedFixture.date,
        home_team: selectedFixture.home_team,
        away_team: selectedFixture.away_team
      });
      setPredictionResult(response);
      setToast("Prediccion completada");
    } catch (error) {
      setToast(`Error al predecir: ${String(error)}`);
    } finally {
      setLoadingPredict(false);
    }
  };

  useEffect(() => {
    void loadFixtures();
  }, []);

  const predictionRecord = predictionResult?.prediction ?? {};
  const pH = Number(predictionRecord["p_H"] ?? 0);
  const pD = Number(predictionRecord["p_D"] ?? 0);
  const pA = Number(predictionRecord["p_A"] ?? 0);

  const bestLabel =
    pH >= pD && pH >= pA
      ? "1 (Local)"
      : pD >= pH && pD >= pA
        ? "X (Empate)"
        : "2 (Visitante)";

  return (
    <>
      <div className="panel">
        <h2>Prediccion rapida de partido</h2>
        <p className="small-note">
          Temporada activa: <strong>{seasonLabel || "-"}</strong>
        </p>
        <p className="small-note">
          Fuente de partidos: <code>{sourcePath || "(sin fuente detectada)"}</code>
        </p>

        <div className="field">
          <label>Partidos de LaLiga pendientes</label>
          <select
            value={selectedFixtureId}
            onChange={(event) => setSelectedFixtureId(event.target.value)}
            disabled={fixtures.length === 0 || loadingOptions}
          >
            {fixtures.map((fixture) => (
              <option key={fixture.fixture_id} value={fixture.fixture_id}>
                {fixture.label}
              </option>
            ))}
          </select>
        </div>

        <div className="actions">
          <button className="btn" type="button" onClick={() => void loadFixtures()} disabled={loadingOptions}>
            {loadingOptions ? "Cargando..." : "Recargar partidos"}
          </button>
          <button
            className="btn btn-alt"
            type="button"
            onClick={() => void handlePredictSelected()}
            disabled={!selectedFixture || loadingPredict}
          >
            {loadingPredict ? "Prediciendo..." : "Predecir partido seleccionado"}
          </button>
        </div>
      </div>

      {predictionResult && (
        <>
          <div className="panel">
            <h3>Resultado de prediccion</h3>
            <p>
              <strong>{predictionResult.selected_fixture.home_team}</strong> vs{" "}
              <strong>{predictionResult.selected_fixture.away_team}</strong>
            </p>
            <p>Fecha: {predictionResult.selected_fixture.date}</p>
            <p>Pronostico principal: {bestLabel}</p>
            <p>CSV de salida: {predictionResult.output_csv}</p>
          </div>

          <div className="panel">
            <h3>Probabilidades 1X2</h3>
            <div className="metric-grid">
              <div className="metric-card">
                <span>p_H (1)</span>
                <strong>{pH.toFixed(3)}</strong>
              </div>
              <div className="metric-card">
                <span>p_D (X)</span>
                <strong>{pD.toFixed(3)}</strong>
              </div>
              <div className="metric-card">
                <span>p_A (2)</span>
                <strong>{pA.toFixed(3)}</strong>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
