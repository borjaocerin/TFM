import { FormEvent, useState } from "react";

import type { OddsComparePayload } from "../../lib/api";

type Props = {
  onSubmit: (payload: OddsComparePayload) => Promise<void>;
  loading: boolean;
};

export function OddsCompareForm({ onSubmit, loading }: Props) {
  const [predictionsCsv, setPredictionsCsv] = useState("data/out/predictions.csv");
  const [oddsKind, setOddsKind] = useState<"odds_avg" | "odds_close">("odds_avg");
  const [threshold, setThreshold] = useState("0.02");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await onSubmit({
      predictions_csv: predictionsCsv,
      odds_kind: oddsKind,
      value_threshold: Number(threshold)
    });
  };

  return (
    <form className="panel" onSubmit={submit}>
      <h2>5) Comparativa con cuotas</h2>
      <div className="grid-form">
        <div className="field">
          <label>predictions.csv</label>
          <input value={predictionsCsv} onChange={(event) => setPredictionsCsv(event.target.value)} />
        </div>

        <div className="field">
          <label>Tipo de cuotas</label>
          <select
            value={oddsKind}
            onChange={(event) => setOddsKind(event.target.value as "odds_avg" | "odds_close")}
          >
            <option value="odds_avg">Apertura promedio</option>
            <option value="odds_close">Cierre promedio</option>
          </select>
        </div>

        <div className="field">
          <label>Umbral value bet</label>
          <input value={threshold} onChange={(event) => setThreshold(event.target.value)} />
        </div>
      </div>

      <div className="actions">
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Comparando..." : "Comparar modelo vs mercado"}
        </button>
      </div>
    </form>
  );
}
