import { FormEvent, useState } from "react";

import type { DatasetIngestPayload } from "../../lib/api";

type Props = {
  onSubmit: (payload: DatasetIngestPayload) => Promise<void>;
  loading: boolean;
};

export function DatasetIngestForm({ onSubmit, loading }: Props) {
  const [historical, setHistorical] = useState("data/historical/laliga_merged_matches.csv");
  const [footballDataDir, setFootballDataDir] = useState("data/football-data");
  const [eloCsv, setEloCsv] = useState("data/elo/ELO_RATINGS.csv");
  const [teamMap, setTeamMap] = useState("etl/team_name_map_es.json");
  const [windows, setWindows] = useState("5,10");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const parsedWindows = windows
      .split(",")
      .map((value: string) => Number(value.trim()))
      .filter((value: number) => Number.isFinite(value) && value > 0);

    await onSubmit({
      historical,
      football_data_dir: footballDataDir,
      elo_csv: eloCsv || undefined,
      team_map: teamMap || undefined,
      windows: parsedWindows
    });
  };

  return (
    <form className="panel" onSubmit={submit}>
      <h2>1) Ingesta de datasets</h2>
      <div className="grid-form">
        <div className="field">
          <label>CSV historico</label>
          <input value={historical} onChange={(event) => setHistorical(event.target.value)} />
        </div>

        <div className="field">
          <label>Directorio football-data</label>
          <input value={footballDataDir} onChange={(event) => setFootballDataDir(event.target.value)} />
        </div>

        <div className="field">
          <label>CSV ELO (opcional)</label>
          <input value={eloCsv} onChange={(event) => setEloCsv(event.target.value)} />
        </div>

        <div className="field">
          <label>Mapa nombres equipos</label>
          <input value={teamMap} onChange={(event) => setTeamMap(event.target.value)} />
        </div>

        <div className="field">
          <label>Ventanas rolling</label>
          <input value={windows} onChange={(event) => setWindows(event.target.value)} />
        </div>
      </div>

      <div className="actions">
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Procesando..." : "Lanzar ingest"}
        </button>
      </div>
      <p className="small-note">
        Recomendado: monta los CSV locales dentro de `data/` para trabajar igual en local y Docker.
      </p>
    </form>
  );
}
