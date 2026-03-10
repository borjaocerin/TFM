type Props = {
  rows: Array<Record<string, unknown>>;
};

export function ValueBetsTable({ rows }: Props) {
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
          {rows.map((row, index) => (
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
