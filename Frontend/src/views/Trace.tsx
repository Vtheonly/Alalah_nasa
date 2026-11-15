import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, Search, XCircle } from "lucide-react";
import { api } from "../api";
import type { CellTrace } from "../types";
import Chart from "../components/Chart";
import { PLATFORM_COLORS } from "../colors";
import { FLAG_LABEL, fmt, MONTHS } from "../format";
import { go } from "../App";

export default function Trace({ args }: { args: string[] }) {
  const [h3, setH3] = useState(args[0] ?? "");
  const [cell, setCell] = useState<CellTrace | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const year = args[1] ? Number(args[1]) : null;
  const month = args[2] ? Number(args[2]) : null;

  useEffect(() => {
    setErr(null);
    if (!args[0]) { setCell(null); return; }
    const ac = new AbortController();
    api.cell(args[0], year, month, ac.signal).then(setCell).catch((e: Error) => setErr(e.message));
    return () => ac.abort();
  }, [args[0], year, month]); // eslint-disable-line react-hooks/exhaustive-deps

  const spec = useMemo(() => cell && ({
    kind: "band" as const, title: "This hexagon over 26 years: harmonized estimate, 90% interval and raw detections",
    x: cell.series.x, yLabel: "daily-mean FRP (MW)",
    series: [
      { name: "5th", role: "band_low", data: cell.series.q05 },
      { name: "90% interval", role: "band_high", data: cell.series.q95 },
      { name: "Harmonized", role: "harmonized", data: cell.series.q50 },
      { name: "Raw detections", role: "raw", axis: 1, data: cell.series.raw_count },
    ],
    markers: [{ x: `${cell.year}-${String(cell.month).padStart(2, "0")}`, label: "selected" }],
  }), [cell]);

  const r = cell?.month_record;
  const rec = cell?.reconciliation;
  return (
    <div className="stack-lg">
      <div className="page-head">
        <div>
          <h1>Traceability</h1>
          <p className="muted">Follow any harmonized number back to the satellite detections it was built from: every pixel, its sensor, time, weights and flags.</p>
        </div>
        <form className="search" onSubmit={(e) => { e.preventDefault(); if (h3) go(`trace/${h3.trim()}`); }}>
          <Search size={15} />
          <input value={h3} onChange={(e) => setH3(e.target.value)} placeholder="H3 cell id (pick one on the Explorer map)" />
        </form>
      </div>
      {!args[0] && <div className="card empty">Choose a hexagon on the <a className="link" href="#/explorer">Explorer</a> map, or enter an H3 id above.</div>}
      {err && <div className="card empty">Could not load this cell: {err}</div>}
      {cell && (
        <>
          <div className="trace-head card">
            <div><small className="muted">Hexagon</small><code>{cell.h3_index}</code></div>
            <div><small className="muted">Where</small><b>{cell.region ?? cell.country_iso}</b><span className="muted"> {cell.lat.toFixed(2)}°, {cell.lon.toFixed(2)}°</span></div>
            <div><small className="muted">Month</small><b>{MONTHS[cell.month - 1]} {cell.year}</b></div>
            <div><small className="muted">Fire regime (pyrome)</small><b>#{cell.pyrome_id}</b></div>
          </div>
          {spec && (
            <div className="card">
              <Chart spec={spec} height={300} />
            </div>
          )}
          {r && (
            <div className="trace-flow">
              <div className="card flow-step">
                <div className="eyebrow">1 · Raw satellite evidence</div>
                <b className="big-num">{cell.detections.length}</b><span className="muted"> detections this month</span>
                <div className="platform-bars">
                  {cell.by_platform.map((p) => (
                    <div key={p.platform} className="pbar">
                      <span style={{ color: PLATFORM_COLORS[p.platform] }}>{p.platform}</span>
                      <i style={{ width: `${(100 * p.n) / Math.max(1, cell.detections.length)}%`, background: PLATFORM_COLORS[p.platform] }} />
                      <small>{p.n} · {fmt(p.frp_mw, "mw")}</small>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card flow-step">
                <div className="eyebrow">2 · Evidence quality</div>
                <div className="side-grid">
                  <div><small>Days observed</small><b>{fmt(r.measured_fraction, "pct0")}</b></div>
                  <div><small>Cloud cover</small><b>{fmt(r.mean_cloud_frac, "pct0")}</b></div>
                </div>
                <div className={`flag flag-${r.quality_flag}`}>{FLAG_LABEL[r.quality_flag]}</div>
                <small className="muted">Satellites flying: {r.active_platforms.join(", ")}</small>
              </div>
              <div className="card flow-step">
                <div className="eyebrow">3 · Harmonized estimate</div>
                <b className="big-num">{fmt(r.phi_q50_mw, "mw")}</b>
                <div className="muted">90% interval {fmt(r.phi_q05_mw, "mw")} – {fmt(r.phi_q95_mw, "mw")}</div>
                <div className="muted">×{fmt(r.anomaly_ratio, "ratio")} vs usual · {FLAG_LABEL[r.attribution_flag]}</div>
              </div>
              <div className={`card flow-step recon ${rec?.match ? "ok" : "ko"}`}>
                <div className="eyebrow">4 · Reconciliation</div>
                {rec?.match ? <CheckCircle2 /> : <XCircle />}
                <p>{rec?.biomass_detections_in_product1} vegetation-fire detections in the provenance table = {rec?.raw_count_in_product2} raw detections in the monthly product{rec?.match ? "" : " (mismatch)"}.</p>
              </div>
            </div>
          )}
          <div className="card">
            <div className="card-head"><h2>Every detection behind this month</h2><span className="muted">Product 1 · obs_harmonized_provenance</span></div>
            <div className="table-wrap">
              <table className="table dense">
                <thead><tr><th>Satellite</th><th>Date</th><th>UTC</th><th>Local h</th><th>FRP</th><th>Pixel</th><th>View</th><th>Conf</th><th>Bow-tie w</th><th>Reliab.</th><th>Gain</th><th>Used</th></tr></thead>
                <tbody>
                  {cell.detections.map((d, i) => (
                    <tr key={i} className={d.is_biomass ? "" : "excluded"}>
                      <td><span className="dotp" style={{ background: PLATFORM_COLORS[String(d.platform)] }} />{String(d.platform)}</td>
                      <td>{String(d.acq_date).slice(0, 10)}</td>
                      <td>{String(d.acq_hhmm).padStart(4, "0")}</td>
                      <td>{fmt(Number(d.solar_hour), "hour")}</td>
                      <td>{fmt(Number(d.frp_mw), "mw")}</td>
                      <td>{fmt(Number(d.pixel_area_km2), "km2")}</td>
                      <td>{Number(d.vza_deg).toFixed(0)}°</td>
                      <td>{String(d.conf_raw)}</td>
                      <td>{Number(d.bowtie_weight).toFixed(2)}</td>
                      <td>{Number(d.reliability_prob).toFixed(2)}</td>
                      <td>{Number(d.forward_gain_c).toFixed(2)}</td>
                      <td>{d.is_biomass ? "✓" : d.is_static_industrial ? "flare" : d.is_sama_filtered ? "SAMA" : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
