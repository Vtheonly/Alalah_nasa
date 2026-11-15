import type { Kpi } from "../types";
import { fmt } from "../format";

export function KpiGrid({ items }: { items: Kpi[] }) {
  if (!items.length) return null;
  return (
    <div className="kpi-grid">
      {items.map((k) => (
        <div key={k.label} className={`kpi ${k.tone ?? ""}`}>
          <div className="kpi-value">{fmt(k.value, k.format)}</div>
          <div className="kpi-label">{k.label}</div>
        </div>
      ))}
    </div>
  );
}

export function Stat({ value, label, accent }: { value: string; label: string; accent?: string }) {
  return (
    <div className="stat">
      <div className="stat-value" style={accent ? { color: accent } : undefined}>{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
