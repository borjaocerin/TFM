type Props = {
  rowsBySeason: Record<string, number>;
  missingPct: Record<string, number>;
};

export function DataQualityTable({ rowsBySeason, missingPct }: Props) {
  const missingTop = Object.entries(missingPct)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15);

  return (
    <div className="panel">
      <h3>Calidad de datos</h3>

      <div className="content">
        <section>
          <h4>Filas por temporada</h4>
          <table>
            <thead>
              <tr>
                <th>Season</th>
                <th>Rows</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(rowsBySeason).map(([season, rows]) => (
                <tr key={season}>
                  <td>{season}</td>
                  <td>{rows}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section>
          <h4>Missing (%) top columnas</h4>
          <table>
            <thead>
              <tr>
                <th>Column</th>
                <th>Missing %</th>
              </tr>
            </thead>
            <tbody>
              {missingTop.map(([column, pct]) => (
                <tr key={column}>
                  <td>{column}</td>
                  <td>{pct.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
