import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { Clapperboard, Map as MapIcon, Microscope, Pause, Play } from "lucide-react";
import { api } from "../api";
import type { CalendarCell, HexCollection, HexFeature, Region } from "../types";
import HexMap, { LEGENDS, type MapMetric } from "../components/HexMap";
import ReplayPanel from "../components/ReplayPanel";
import Chart from "../components/Chart";
import { C } from "../colors";
import { compact, FLAG_LABEL, fmt, MONTHS } from "../format";
import { go } from "../App";

const METRICS: Array<{ id: MapMetric; label: string }> = [
  { id: "frp", label: "Fire power" },
  { id: "anomaly", label: "Anomaly" },
  { id: "uncertainty", label: "Uncertainty" },
  { id: "evidence", label: "Evidence" },
];

function calendarOption(cells: CalendarCell[], sel: { year: number; month: number }): EChartsOption {
  const years = Array.from(new Set(cells.map((c) => c.year))).sort();
  const data = cells.map((c) => [c.month - 1, years.indexOf(c.year), c.fre_tj ?? 0]);
  const vals = cells.map((c) => c.fre_tj ?? 0).filter((v) => v > 0).sort((a, b) => a - b);
  const hi = vals.length ? vals[Math.floor(vals.length * 0.97)] : 1;
  return {
    grid: { left: 44, right: 12, top: 8, bottom: 46 },
    tooltip: {
      backgroundColor: "#0f172a", borderColor: "#1e293b", textStyle: { color: "#e2e8f0", fontSize: 12 },
      formatter: (p: unknown) => {
        const v = (p as { value: number[] }).value;
        const c = cells.find((x) => x.year === years[v[1]] && x.month === v[0] + 1);
        return c ? `<b>${MONTHS[c.month - 1]} ${c.year}</b><br/>Energy: ${compact(c.fre_tj)} TJ<br/>Raw detections: ${compact(c.raw_count)}<br/>Observed: ${fmt(c.measured_fraction, "pct0")}<br/>${FLAG_LABEL[c.quality_flag] ?? c.quality_flag}` : "";
      },
    },
    xAxis: { type: "category", data: MONTHS, axisLabel: { color: C.muted, fontSize: 10 }, axisLine: { show: false }, axisTick: { show: false } },
    yAxis: { type: "category", data: years.map(String), axisLabel: { color: C.muted, fontSize: 9 }, axisLine: { show: false }, axisTick: { show: false }, inverse: true },
    visualMap: { min: 0, max: hi, calculable: false, orient: "horizontal", left: "center", bottom: 0, itemHeight: 140, itemWidth: 10, textStyle: { color: C.muted, fontSize: 10 }, inRange: { color: ["#0f1d33", "#7c2d12", "#ea580c", "#fbbf24", "#fef3c7"] }, text: ["high", "low"] },
    series: [{
      type: "heatmap", data, itemStyle: { borderColor: "#0b1220", borderWidth: 1 },
      markPoint: { symbol: "rect", symbolSize: [16, 10], itemStyle: { color: "transparent", borderColor: "#fff", borderWidth: 2 }, data: [{ name: "selected", coord: [sel.month - 1, years.indexOf(sel.year)] }] },
    }],
  };
}

export default function Explorer({ args }: { args: string[] }) {
  const [regions, setRegions] = useState<Region[]>([]);
  const iso = (args[0] ?? "BRA").toUpperCase();
  const region = regions.find((r) => r.iso === iso) ?? regions[0];
  const [cal, setCal] = useState<CalendarCell[]>([]);
  const [ym, setYm] = useState<{ year: number; month: number } | null>(null);
  const [metric, setMetric] = useState<MapMetric>("frp");
  const [hex, setHex] = useState<HexCollection | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState<"monthly" | "replay">("monthly");

  useEffect(() => { api.regions().then(setRegions).catch(() => undefined); }, []);
  useEffect(() => {
    if (!region) return;
    const ac = new AbortController();
    setCal([]); setSel(null);
    api.calendar(region, "phi_q50_mw", ac.signal).then((c) => {
      setCal(c);
      const recent = c.filter((x) => x.year >= 2018 && x.year <= 2025);
      const best = recent.reduce((a, b) => ((b.fre_tj ?? 0) > (a?.fre_tj ?? -1) ? b : a), recent[0]);
      if (best) setYm({ year: best.year, month: best.month });
    }).catch(() => undefined);
    return () => ac.abort();
  }, [region?.iso]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!region || !ym) return;
    const ac = new AbortController();
    api.mapHex(region, ym.year, ym.month, "anomaly_ratio", ac.signal).then(setHex).catch(() => undefined);
    return () => ac.abort();
  }, [region?.iso, ym?.year, ym?.month]); // eslint-disable-line react-hooks/exhaustive-deps

  const months = useMemo(() => cal.map((c) => ({ year: c.year, month: c.month })), [cal]);
  const idx = ym ? months.findIndex((m) => m.year === ym.year && m.month === ym.month) : -1;
  useEffect(() => {
    if (!playing || !months.length) return;
    const t = window.setInterval(() => {
      setYm((cur) => {
        const i = cur ? months.findIndex((m) => m.year === cur.year && m.month === cur.month) : 0;
        return months[(i + 1) % months.length];
      });
    }, 900);
    return () => window.clearInterval(t);
  }, [playing, months]);

  const cur = idx >= 0 ? cal[idx] : null;
  const selFeat: HexFeature | undefined = hex?.features.find((f) => f.properties.h3_index === sel);
  const series = useMemo(() => ({
    kind: "band" as const, title: `${region?.name ?? ""}: monthly harmonized fire power with 90% interval`,
    x: cal.map((c) => `${c.year}-${String(c.month).padStart(2, "0")}`), yLabel: "mean daily FRP per hexagon (MW)",
    series: [
      { name: "5th pct", role: "band_low", data: cal.map((c) => c.q05) },
      { name: "90% interval", role: "band_high", data: cal.map((c) => c.q95) },
      { name: "Harmonized (median)", role: "harmonized", data: cal.map((c) => c.value) },
      { name: "Raw detections", role: "raw", axis: 1, data: cal.map((c) => c.raw_count) },
    ],
    markers: [{ x: "2012-01", label: "VIIRS" }, { x: "2018-04", label: "NOAA-20" }, { x: "2023-03", label: "NOAA-21" }],
  }), [cal, region?.name]);

  if (!region) return <div className="skeleton" style={{ height: 400 }} />;
  return (
    <div className="stack-lg">
      <div className="page-head">
        <div>
          <h1>Explorer</h1>
          <p className="muted">Harmonized fire activity on 36 km² hexagons. Click a hexagon to trace it back to individual satellite detections.</p>
        </div>
        <div className="chips">
          {regions.map((r) => (
            <a key={r.iso} href={`#/explorer/${r.iso}`} className={r.iso === iso ? "chip active" : "chip"}>{r.name.split(" (")[0]}</a>
          ))}
        </div>
      </div>

      <div className="explorer">
        <div className="card map-card">
          <div className="mode-tabs">
            <button className={mode === "monthly" ? "active" : ""} onClick={() => setMode("monthly")}><MapIcon size={14} /> Monthly map</button>
            <button className={mode === "replay" ? "active" : ""} onClick={() => { setPlaying(false); setMode("replay"); }}>
              <Clapperboard size={14} /> Replay {ym ? `${MONTHS[ym.month - 1]} ${ym.year}` : ""}
            </button>
          </div>
          {mode === "replay" && ym ? (
            <ReplayPanel region={region} year={ym.year} month={ym.month} onSelect={setSel} />
          ) : (<>
          <div className="map-toolbar">
            <div className="seg">
              {METRICS.map((m) => (
                <button key={m.id} className={metric === m.id ? "active" : ""} onClick={() => setMetric(m.id)}>{m.label}</button>
              ))}
            </div>
            <div className="player">
              <button className="icon-btn" onClick={() => setPlaying((p) => !p)} aria-label="play">{playing ? <Pause size={15} /> : <Play size={15} />}</button>
              <input type="range" min={0} max={Math.max(0, months.length - 1)} value={Math.max(0, idx)}
                onChange={(e) => { setPlaying(false); setYm(months[Number(e.target.value)]); }} />
              <b className="ym">{ym ? `${MONTHS[ym.month - 1]} ${ym.year}` : "…"}</b>
            </div>
          </div>
          <HexMap data={hex} bbox={region.bbox} metric={metric} selected={sel} onSelect={setSel} fitKey={region.iso} />
          <div className="legend">
            <span>{LEGENDS[metric].label}</span>
            {LEGENDS[metric].stops.map(([c, l]) => <span key={l} className="legend-stop"><i style={{ background: c }} />{l}</span>)}
            <span className="muted">{hex ? `${hex.meta.n} hexagons` : ""}</span>
          </div>
          </>)}
        </div>
        <aside className="card side">
          <h3>{cur ? `${MONTHS[cur.month - 1]} ${cur.year}` : "—"} · {region.iso}</h3>
          <div className="side-grid">
            <div><small>Energy (FRE)</small><b>{compact(cur?.fre_tj)} TJ</b></div>
            <div><small>Raw detections</small><b>{compact(cur?.raw_count)}</b></div>
            <div><small>Days observed</small><b>{fmt(cur?.measured_fraction, "pct0")}</b></div>
            <div><small>vs usual</small><b>×{fmt(cur?.anomaly_ratio, "ratio")}</b></div>
          </div>
          <div className={`flag flag-${cur?.quality_flag ?? "NOMINAL"}`}>{FLAG_LABEL[cur?.quality_flag ?? ""] ?? "—"}</div>
          <hr />
          {selFeat ? (
            <div className="sel-cell">
              <small className="muted">Selected hexagon</small>
              <code>{selFeat.properties.h3_index}</code>
              <div className="side-grid">
                <div><small>Fire power</small><b>{fmt(selFeat.properties.phi_q50_mw, "mw")}</b></div>
                <div><small>90% interval</small><b>±{compact(selFeat.properties.width / 2)} MW</b></div>
                <div><small>Raw pixels</small><b>{compact(selFeat.properties.raw_count)}</b></div>
                <div><small>Observed</small><b>{fmt(selFeat.properties.measured_fraction, "pct0")}</b></div>
              </div>
              <div className="muted small">{FLAG_LABEL[selFeat.properties.attribution_flag] ?? ""}</div>
              <button className="btn primary block" onClick={() => go(`trace/${selFeat.properties.h3_index}/${ym?.year}/${ym?.month}`)}>
                <Microscope size={15} /> Trace to satellite detections
              </button>
            </div>
          ) : <p className="muted small">Click any hexagon on the map to inspect it.</p>}
        </aside>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <h2>Burning calendar</h2>
            <span className="muted">click a month to select it, then replay it</span>
            {ym && <button className="btn small-btn" onClick={() => { setPlaying(false); setMode("replay"); window.scrollTo({ top: 0, behavior: "smooth" }); }}>
              <Clapperboard size={14} /> Replay {MONTHS[ym.month - 1]} {ym.year}
            </button>}
          </div>
          {cal.length ? (
            <ReactECharts option={calendarOption(cal, ym ?? { year: 0, month: 0 })} style={{ height: 520 }} notMerge
              onEvents={{ click: (p: { value: number[] }) => { const years = Array.from(new Set(cal.map((c) => c.year))).sort(); setPlaying(false); setYm({ year: years[p.value[1]], month: p.value[0] + 1 }); } }} />
          ) : <div className="skeleton" style={{ height: 520 }} />}
        </div>
        <div className="card">
          <div className="card-head"><h2>Time series</h2><span className="muted">raw detections (bars) jump with each new satellite; the harmonized line does not</span></div>
          {cal.length ? <Chart spec={series} height={500} /> : <div className="skeleton" style={{ height: 500 }} />}
        </div>
      </div>
    </div>
  );
}
