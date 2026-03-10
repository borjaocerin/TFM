import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const colors = ["#0b7a75", "#1c9a93", "#2d6a9f", "#d97635", "#bc5f2a"];

type Props = {
  metrics: Record<string, number>;
};

export function MetricsChart({ metrics }: Props) {
  const rows = Object.entries(metrics).map(([name, value]) => ({ name, value }));
  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="panel">
      <h3>Metricas del modelo</h3>
      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ top: 20, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#0b7a75">
              {rows.map((row, index) => (
                <Cell key={`${row.name}-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
