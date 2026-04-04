import { useMemo, useState } from "react";

type Props = {
  rows: Array<Record<string, unknown>>;
};

export function ValueBetsTable({ rows }: Props) {
  const [sortOrder, setSortOrder] = useState<"ev_desc" | "ev_asc">("ev_desc");

  const sortedRows = useMemo(() => {
    const toComparableEv = (row: Record<string, unknown>): number => {
      const ev = Number(row.best_ev ?? Number.NaN);
      return Number.isFinite(ev) ? ev : Number.NEGATIVE_INFINITY;
    };

    return [...rows].sort((leftRow, rightRow) => {
      const leftEv = toComparableEv(leftRow);
      const rightEv = toComparableEv(rightRow);

      if (sortOrder === "ev_asc") {
        return leftEv - rightEv;
      }

      return rightEv - leftEv;
    });
  }, [rows, sortOrder]);

  if (rows.length === 0) {
    return (
      <div className="panel">
        <h3>Value bets</h3>
        <p>No hay value bets por encima del umbral seleccionado.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>Value bets detectadas</h3>
      <div className="field">
        <label htmlFor="value-bets-order">Ordenar por value bet</label>
        <select
          id="value-bets-order"
          value={sortOrder}
          onChange={(event) => setSortOrder(event.target.value as "ev_desc" | "ev_asc")}
        >
          <option value="ev_desc">Mejor EV → peor EV</option>
          <option value="ev_asc">Peor EV → mejor EV</option>
        </select>
      </div>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Home</th>
            <th>Away</th>
            <th>Pick</th>
            <th>EV</th>
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, index) => (
            <tr key={index}>
              <td>{String(row.date ?? "-")}</td>
              <td>{String(row.home_team ?? "-")}</td>
              <td>{String(row.away_team ?? "-")}</td>
              <td>
                <span className="tag-value">{String(row.best_ev_pick ?? "-")}</span>
              </td>
              <td>{Number(row.best_ev ?? 0).toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
