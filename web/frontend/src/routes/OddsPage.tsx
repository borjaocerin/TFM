import { useMemo, useState } from "react";

import { OddsCompareForm } from "../components/forms/OddsCompareForm";
import { ValueBetsTable } from "../components/tables/ValueBetsTable";
import {
  OddsComparePayload,
  OddsCompareResponse,
  compareOdds
} from "../lib/api";
import { useAppStore } from "../lib/state/appStore";

export function OddsPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OddsCompareResponse | null>(null);
  const setToast = useAppStore((state) => state.setToast);

  const handleSubmit = async (payload: OddsComparePayload) => {
    setLoading(true);
    try {
      const response = await compareOdds(payload);
      setResult(response);
      setToast(`Comparativa completada. Value bets: ${response.value_bets.length}`);
    } catch (error) {
      let errorMsg = "Error desconocido";
      if (error instanceof Error) {
        errorMsg = error.message;
        if ("response" in error && error.response && typeof error.response === "object" && "data" in error.response) {
          const data = error.response.data as Record<string, unknown>;
          if ("detail" in data) {
            errorMsg = String(data.detail);
          }
        }
      }
      setToast(`Error en comparativa: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  const metricsRows = useMemo(() => Object.entries(result?.metrics ?? {}), [result]);

  return (
    <>
      <OddsCompareForm onSubmit={handleSubmit} loading={loading} />

      {result && (
        <>
          <div className="panel">
            <h3>Metricas modelo vs mercado</h3>
            <table>
              <thead>
                <tr>
                  <th>Metrica</th>
                  <th>Valor</th>
                </tr>
              </thead>
              <tbody>
                {metricsRows.map(([metric, value]) => (
                  <tr key={metric}>
                    <td>{metric}</td>
                    <td>{value === null ? "-" : Number(value).toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small-note">CSV combinado exportado en: {result.output_csv}</p>
          </div>

          <ValueBetsTable rows={result.value_bets} />
        </>
      )}
    </>
  );
}
