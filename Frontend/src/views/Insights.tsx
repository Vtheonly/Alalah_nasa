import { useEffect, useState } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { api } from "../api";
import type { Insights, Region } from "../types";
import Chart from "../components/Chart";
import { compact, FLAG_LABEL, fmt, MONTHS } from "../format";

const ATTR_COLORS: Record<string, string> = {
  REAL_FIRE_CHANGE: "#f97316", SENSOR_ARTIFACT_CORRECTED: "#a78bfa",
  WITHIN_NORMAL_VARIABILITY: "#475569", AMBIGUOUS_OVERLAPPING_CI: "#f87171",
};

export default function InsightsView({ args }: { args: string[] }) {
  const [regions, setRegions] = useState<Region[]>([]);
  const iso = args[0]?.toUpperCase() ?? null;
  const [ins, setIns] = useState<Insights | null>(null);
  useEffect(() => { api.regions().then(setRegions).catch(() => undefined); }, []);
  useEffect(() => {
    setIns(null);
    const ac = new AbortController();
    api.insights(iso, ac.signal).then(setIns).catch(() => undefined);
    return () => ac.abort();
  }, [iso]);

  const t = ins?.trend;
  return (
    <div className="stack-lg">
      <div className="page-head">
        <div>
          <h1>Scientific insights</h1>
          <p className="muted">What the harmonized record says: long-term trends, the fire season, and the most extreme months, with their attribution.</p>
        </div>
        <div className="chips">
          <a href="#/insights" className={!iso ? "chip active" : "chip"}>All countries</a>
          {regions.map((r) => <a key={r.iso} href={`#/insights/${r.iso}`} className={iso === r.iso ? "chip active" : "chip"}>{r.iso}</a>)}
        </div>
      </div>
      {!ins ? <div className="skeleton" style={{ height: 420 }} /> : (
        <>
          <div className="insight-kpis">
            <div className="kpi big">
              {(t?.pct_per_decade ?? 0) >= 0 ? <TrendingUp /> : <TrendingDown />}
              <div className="kpi-value">{t?.pct_per_decade === null ? "—" : `${(t?.pct_per_decade ?? 0) > 0 ? "+" : ""}${t?.pct_per_decade}%`}<small>/decade</small></div>
              <div className="kpi-label">Fire-energy trend (Theil–Sen) · {t?.significant ? "significant" : "not significant"} (Mann–Kendall p = {fmt(t?.mann_kendall_p, "p")})</div>
            </div>
            <div className="kpi big">
              <div className="kpi-value">{ins.seasonality.peak_month ? MONTHS[ins.seasonality.peak_month - 1] : "—"}</div>
              <div className="kpi-label">Peak fire month (2003–2020)</div>
            </div>
            <div className="kpi big">
              <div className="kpi-value">{ins.seasonality.season_length_months}<small> months</small></div>
              <div className="kpi-label">Fire-season length (≥30% of peak)</div>
            </div>
          </div>
          <div className="grid-2">
            <div className="card"><Chart spec={ins.charts.annual} height={320} /></div>
            <div className="card"><Chart spec={ins.charts.seasonal} height={320} /></div>
          </div>
          <div className="card">
            <div className="card-head"><h2>Most extreme months</h2><span className="muted">monthly fire energy vs the 2003–2020 median for that calendar month, with how hexagons were attributed</span></div>
            <table className="table">
              <thead><tr><th>Month</th><th>Energy</th><th>× usual</th><th>Attribution of hexagons</th></tr></thead>
              <tbody>
                {ins.anomalies.map((a) => (
                  <tr key={`${a.year}-${a.month}`}>
                    <td><b>{MONTHS[a.month - 1]} {a.year}</b></td>
                    <td>{compact(a.fre_tj)} TJ</td>
                    <td><span className="ratio">×{fmt(a.ratio_to_median, "ratio")}</span></td>
                    <td>
                      <div className="attr-bar">
                        {Object.entries(a.attribution).map(([k, v]) => (
                          <i key={k} style={{ width: `${v * 100}%`, background: ATTR_COLORS[k] ?? "#334155" }} title={`${FLAG_LABEL[k] ?? k}: ${(v * 100).toFixed(0)}%`} />
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="legend">
              {Object.entries(ATTR_COLORS).map(([k, c]) => <span key={k} className="legend-stop"><i style={{ background: c }} />{FLAG_LABEL[k]}</span>)}
            </div>
          </div>
          {ins.by_country.length > 0 && (
            <div className="card">
              <div className="card-head"><h2>Country comparison</h2></div>
              <div className="country-compare">
                {ins.by_country.map((c) => (
                  <a key={c.iso} href={`#/insights/${c.iso}`} className="cc">
                    <span className="iso">{c.iso}</span>
                    <b>{compact(c.fre_tj_per_year)} TJ/yr</b>
                    <span className={(c.trend_pct_per_decade ?? 0) >= 0 ? "up" : "down"}>
                      {(c.trend_pct_per_decade ?? 0) >= 0 ? "▲" : "▼"} {Math.abs(c.trend_pct_per_decade ?? 0).toFixed(0)}%/decade
                    </span>
                    <small className="muted">{(c.trend_p ?? 1) < 0.05 ? "significant" : "not significant"} · p={fmt(c.trend_p, "p")}</small>
                  </a>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
