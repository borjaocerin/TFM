type Props = {
  rows: Array<Record<string, unknown>>;
};

export function PredictionsTable({ rows }: Props) {
  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="panel">
      <h3>Predicciones</h3>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Home</th>
            <th>Away</th>
            <th>p_H</th>
            <th>p_D</th>
            <th>p_A</th>
            <th>ELO diff</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 100).map((row, index) => (
            <tr key={index}>
              <td>{String(row.date ?? "-")}</td>
              <td>{String(row.home_team ?? "-")}</td>
              <td>{String(row.away_team ?? "-")}</td>
              <td>{Number(row.p_H ?? 0).toFixed(3)}</td>
              <td>{Number(row.p_D ?? 0).toFixed(3)}</td>
              <td>{Number(row.p_A ?? 0).toFixed(3)}</td>
              <td>{row.elo_diff ? Number(row.elo_diff).toFixed(2) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small-note">Mostrando hasta 100 filas para mantener la UI fluida.</p>
    </div>
  );
}
