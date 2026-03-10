import { useState } from "react";

import { DatasetIngestForm } from "../components/forms/DatasetIngestForm";
import { DataQualityTable } from "../components/tables/DataQualityTable";
import {
  DatasetIngestPayload,
  DatasetIngestResponse,
  ingestDatasets
} from "../lib/api";
import { useAppStore } from "../lib/state/appStore";

export function DatasetsPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DatasetIngestResponse | null>(null);
  const setToast = useAppStore((state) => state.setToast);

  const handleSubmit = async (payload: DatasetIngestPayload) => {
    setLoading(true);
    try {
      const response = await ingestDatasets(payload);
      setResult(response);
      setToast(`Ingest OK: ${response.rows_total} partidos procesados`);
    } catch (error) {
      setToast(`Error en ingest: ${String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <DatasetIngestForm onSubmit={handleSubmit} loading={loading} />
      {result && (
        <>
          <div className="panel">
            <h3>Salida generada</h3>
            <p>all: {result.output_all}</p>
            <p>model: {result.output_model}</p>
          </div>
          <DataQualityTable
            rowsBySeason={result.rows_by_season}
            missingPct={result.missing_pct_by_column}
          />
        </>
      )}
    </>
  );
}
