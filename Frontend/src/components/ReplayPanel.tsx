import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { Pause, Play, SkipBack } from "lucide-react";
import { api } from "../api";
import type { Region, ReplayData } from "../types";
import ReplayMap from "./ReplayMap";
import { C, PLATFORM_COLORS } from "../colors";
import { compact, MONTHS } from "../format";

interface Props {
  region: Region;
  year: number;
  month: number;
  onSelect: (h3: string) => void;
}

const SPEEDS = [0.5, 1, 2, 4];

export default function ReplayPanel({ region, year, month, onSelect }: Props) {
  const [step, setStep] = useState<"day" | "hour">("day");
  const [data, setData] = useState<ReplayData | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showDet, setShowDet] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setData(null); setErr(null); setFrame(0); setPlaying(false);
    api.replay(region.iso, year, month, step, ac.signal)
      .then((d) => { setData(d); setPlaying(d.frames.length > 0); })
      .catch((e: Error) => { if (e.name !== "AbortError") setErr(e.message); });
    return () => ac.abort();
  }, [region.iso, year, month, step]);

  useEffect(() => {
    if (!playing || !data?.frames.length) return;
    const base = step === "day" ? 650 : 260;
    const t = window.setInterval(() => {
      setFrame((f) => {
        if (f + 1 >= data.frames.length) { setPlaying(false); return f; }
        return f + 1;
      });
    }, base / speed);
    return () => window.clearInterval(t);
  }, [playing, data, speed, step]);

  const perFrame = useMemo(() => {
    if (!data) return { det: [] as number[], frp: [] as number[], cum: [] as number[] };
    const det = new Array(data.frames.length).fill(0);
    const frp = new Array(data.frames.length).fill(0);
    for (const d of data.detections) { det[d[0]] += 1; frp[d[0]] += d[3]; }
    const cum: number[] = [];
    det.reduce((a, b, i) => (cum[i] = a + b), 0);
    return { det, frp, cum };
  }, [data]);

  const burningNow = useMemo(() => {
    if (!data) return 0;
    const d0 = (data.frames[frame]?.day ?? 1) - 1;
    return data.daily.filter((r) => r[0] === d0 && (r[2] ?? 0) > 0.05).length;
  }, [data, frame]);

  const platformsNow = useMemo(() => {
    const m = new Map<string, number>();
    if (!data) return m;
    for (const d of data.detections) if (d[0] === frame) {
      const p = data.platforms[d[4]];
      m.set(p, (m.get(p) ?? 0) + 1);
    }
    return m;
  }, [data, frame]);

  const strip: EChartsOption | null = data && {
    grid: { left: 40, right: 12, top: 10, bottom: 22 },
    tooltip: { trigger: "axis", backgroundColor: "#0f172a", borderColor: "#1e293b", textStyle: { color: "#e2e8f0", fontSize: 12 } },
    xAxis: { type: "category", data: data.frames.map((f) => f.label), axisLabel: { show: false }, axisTick: { show: false }, axisLine: { lineStyle: { color: C.axis } } },
    yAxis: { type: "value", splitLine: { lineStyle: { color: C.grid } }, axisLabel: { color: C.muted, fontSize: 10, formatter: (v: number) => compact(v, 0) } },
    series: [{
      type: "bar", name: "detections", data: perFrame.det.map((v, i) => ({ value: v, itemStyle: { color: i === frame ? "#fef08a" : i < frame ? "#f97316" : "#334155" } })),
      barCategoryGap: "15%",
      markLine: { silent: true, symbol: "none", lineStyle: { color: "#fef08a" }, label: { show: false }, data: [{ xAxis: frame }] },
    }],
  };

  const cur = data?.frames[frame];
  return (
    <div className="replay">
      <div className="map-toolbar">
        <div className="seg">
          <button className={step === "day" ? "active" : ""} onClick={() => setStep("day")}>By day</button>
          <button className={step === "hour" ? "active" : ""} onClick={() => setStep("hour")}>By overpass hour</button>
        </div>
        <div className="player">
          <button className="icon-btn" onClick={() => { setFrame(0); setPlaying(true); }} aria-label="restart"><SkipBack size={14} /></button>
          <button className="icon-btn" onClick={() => setPlaying((p) => !p)} aria-label="play">{playing ? <Pause size={15} /> : <Play size={15} />}</button>
          <input type="range" min={0} max={Math.max(0, (data?.frames.length ?? 1) - 1)} value={frame}
            onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }} />
          <select className="speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
          <b className="ym wide">{cur ? cur.label : `${MONTHS[month - 1]} ${year}`}</b>
        </div>
      </div>
      <div className="replay-body">
        <div className="replay-map">
          {data ? <ReplayMap data={data} frame={frame} trail={step === "day" ? 0.72 : 0.94} showDetections={showDet} onSelect={onSelect} />
            : <div className="skeleton" style={{ height: 520 }}>{err && <p className="empty">{err}</p>}</div>}
          {data && data.frames.length === 0 && <div className="replay-empty">No satellite detections in {MONTHS[month - 1]} {year} for {region.name}.</div>}
        </div>
        <aside className="replay-side">
          <div className="side-grid">
            <div><small>Detections this {step === "day" ? "day" : "overpass"}</small><b>{compact(perFrame.det[frame] ?? 0)}</b></div>
            <div><small>Hexagons burning today</small><b>{compact(burningNow)}</b></div>
            <div><small>Detections so far</small><b>{compact(perFrame.cum[frame] ?? 0)}</b></div>
            <div><small>Raw FRP this frame</small><b>{compact(perFrame.frp[frame] ?? 0)} MW</b></div>
          </div>
          <div className="platform-now">
            <small className="muted">Satellites observing now</small>
            {[...platformsNow.entries()].map(([p, n]) => (
              <span key={p} className="pchip" style={{ borderColor: PLATFORM_COLORS[p], color: PLATFORM_COLORS[p] }}>{p} · {n}</span>
            ))}
            {platformsNow.size === 0 && <span className="muted small">none (cloud, no overpass or no fire)</span>}
          </div>
          <label className="toggle"><input type="checkbox" checked={showDet} onChange={(e) => setShowDet(e.target.checked)} /> Show raw detections</label>
          <p className="muted small">
            Hexagons glow with the <b>harmonized daily fire power</b> and fade over the following {step === "day" ? "days" : "overpasses"} (heat trail).
            Dots and the heat haze are the <b>raw satellite detections</b> of the current {step === "day" ? "day" : "hour"}, coloured by satellite.
            {step === "hour" && " The harmonized state is daily; hours are the actual satellite overpass times."}
          </p>
          {data?.truncated.detections && <p className="muted small">Showing the {compact(data.detections.length)} strongest detections of the month.</p>}
          {data?.truncated.cells && <p className="muted small">Showing the {data.cells.length} most active hexagons.</p>}
        </aside>
      </div>
      {strip && (
        <div className="strip">
          <ReactECharts option={strip} style={{ height: 90 }} notMerge onEvents={{ click: (p: { dataIndex: number }) => { setPlaying(false); setFrame(p.dataIndex); } }} />
        </div>
      )}
    </div>
  );
}
