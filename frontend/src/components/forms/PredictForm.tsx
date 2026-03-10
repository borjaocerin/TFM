import { FormEvent, useState } from "react";

type BuildFeaturesPayload = {
  fixtures_csv: string;
  historical_csv?: string;
  elo_csv?: string;
  team_map?: string;
  windows: number[];
};

type PredictPayload = {
  fixtures_enriched_path?: string;
};

type Props = {
  onBuildFeatures: (payload: BuildFeaturesPayload) => Promise<void>;
  onPredict: (payload: PredictPayload) => Promise<void>;
  loadingBuild: boolean;
  loadingPredict: boolean;
};

export function PredictForm({ onBuildFeatures, onPredict, loadingBuild, loadingPredict }: Props) {
  const [fixturesCsv, setFixturesCsv] = useState("data/fixtures/fixtures.csv");
  const [historicalCsv, setHistoricalCsv] = useState("data/out/laliga_enriched_all.csv");
  const [eloCsv, setEloCsv] = useState("data/elo/ELO_RATINGS.csv");
  const [teamMap, setTeamMap] = useState("etl/team_name_map_es.json");
  const [windows, setWindows] = useState("5,10");
  const [fixturesEnrichedPath, setFixturesEnrichedPath] = useState("data/out/fixtures_enriched.csv");

  const buildSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const parsedWindows = windows
      .split(",")
      .map((value: string) => Number(value.trim()))
      .filter((value: number) => Number.isFinite(value) && value > 0);
    await onBuildFeatures({
      fixtures_csv: fixturesCsv,
      historical_csv: historicalCsv,
      elo_csv: eloCsv || undefined,
      team_map: teamMap || undefined,
      windows: parsedWindows
    });
  };

  const predictSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await onPredict({ fixtures_enriched_path: fixturesEnrichedPath });
  };

  return (
    <div className="content">
      <form className="panel" onSubmit={buildSubmit}>
        <h2>3) Features para fixtures futuros</h2>
        <div className="grid-form">
          <div className="field">
            <label>fixtures.csv</label>
            <input value={fixturesCsv} onChange={(event) => setFixturesCsv(event.target.value)} />
          </div>
          <div className="field">
            <label>Historico enriquecido</label>
            <input value={historicalCsv} onChange={(event) => setHistoricalCsv(event.target.value)} />
          </div>
          <div className="field">
            <label>ELO (opcional)</label>
            <input value={eloCsv} onChange={(event) => setEloCsv(event.target.value)} />
          </div>
          <div className="field">
            <label>Mapa equipos</label>
            <input value={teamMap} onChange={(event) => setTeamMap(event.target.value)} />
          </div>
          <div className="field">
            <label>Ventanas rolling</label>
            <input value={windows} onChange={(event) => setWindows(event.target.value)} />
          </div>
        </div>
        <div className="actions">
          <button className="btn" type="submit" disabled={loadingBuild}>
            {loadingBuild ? "Calculando..." : "Generar fixtures_enriched"}
          </button>
        </div>
      </form>

      <form className="panel" onSubmit={predictSubmit}>
        <h3>4) Prediccion 1X2</h3>
        <div className="field">
          <label>fixtures_enriched.csv</label>
          <input
            value={fixturesEnrichedPath}
            onChange={(event) => setFixturesEnrichedPath(event.target.value)}
          />
        </div>
        <div className="actions">
          <button className="btn btn-alt" type="submit" disabled={loadingPredict}>
            {loadingPredict ? "Prediciendo..." : "Predecir"}
          </button>
        </div>
      </form>
    </div>
  );
}
