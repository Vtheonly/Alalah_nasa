import { useEffect, useMemo, useRef, useState } from "react";
import Globe, { type GlobeMethods } from "react-globe.gl";
import * as THREE from "three";
import { Pause, Play, Satellite } from "lucide-react";
import { api } from "../api";
import type { GlobeData, SatelliteMeta } from "../types";
import { PLATFORM_COLORS } from "../colors";
import { activeAt, greatCircleKm, orbitPath, subpoint } from "../orbits";
import { compact, MONTHS } from "../format";

const SPEEDS = [1, 60, 300, 1200];
const MONTH_LIST: Array<{ y: number; m: number }> = [];
for (let y = 2000; y <= 2026; y++) for (let m = 1; m <= 12; m++) {
  if ((y === 2000 && m < 11) || (y === 2026 && m > 10)) continue;
  MONTH_LIST.push({ y, m });
}

function satelliteObject(color: string): THREE.Object3D {
  const g = new THREE.Group();
  const mat = new THREE.MeshLambertMaterial({ color: "#e2e8f0", emissive: new THREE.Color(color), emissiveIntensity: 0.6 });
  g.add(new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.1, 1.8), mat));
  const panel = new THREE.MeshLambertMaterial({ color: "#1e3a8a", emissive: new THREE.Color("#1d4ed8"), emissiveIntensity: 0.5, side: THREE.DoubleSide });
  for (const x of [-2.4, 2.4]) {
    const p = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 1.2), panel);
    p.position.set(x, 0, 0);
    g.add(p);
  }
  const glow = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 16), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.25 }));
  g.add(glow);
  g.scale.setScalar(2.4);          // globe radius = 100 units: make spacecraft visible
  return g;
}

export default function GlobeView() {
  const globeRef = useRef<GlobeMethods | undefined>(undefined);
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 640 });
  const [mi, setMi] = useState(MONTH_LIST.findIndex((x) => x.y === 2020 && x.m === 9));
  const [data, setData] = useState<GlobeData | null>(null);
  const [speed, setSpeed] = useState(300);
  const [running, setRunning] = useState(true);
  const [follow, setFollow] = useState<string>("");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const ym = MONTH_LIST[mi];
  const isoDate = `${ym.y}-${String(ym.m).padStart(2, "0")}-15`;
  const [simT, setSimT] = useState(() => Date.UTC(ym.y, ym.m - 1, 15, 6));
  const [lit, setLit] = useState<Uint8Array>(new Uint8Array(0));

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: Math.max(520, Math.min(760, el.clientWidth * 0.62)) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    api.globe(ym.y, ym.m, ac.signal).then(setData).catch(() => undefined);
    setSimT((t) => Date.UTC(ym.y, ym.m - 1, 15) + (t % 86_400_000));
    return () => ac.abort();
  }, [ym.y, ym.m]);

  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.pointOfView({ lat: -5, lng: -40, altitude: 2.5 }, 0);
    const c = g.controls() as unknown as { autoRotate: boolean; autoRotateSpeed: number };
    c.autoRotate = true;
    c.autoRotateSpeed = 0.35;
  }, [data !== null]); // eslint-disable-line react-hooks/exhaustive-deps

  // simulation clock (~30 fps)
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      if (dt > 33) {
        last = now;
        setSimT((t) => t + dt * speed);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, speed]);

  const sats = useMemo<SatelliteMeta[]>(() => (data?.satellites ?? []).filter((s) => activeAt(s, isoDate) && !hidden.has(s.platform)), [data, isoDate, hidden]);
  const positions = useMemo(() => sats.map((s) => ({ s, ...subpoint(s, simT), alt: s.altitude_km / 6371 })), [sats, simT]);
  const pathsKey = Math.floor(simT / 120_000);   // refresh orbit paths every 2 simulated minutes
  const paths = useMemo(() => sats.map((s) => ({ name: s.platform, color: PLATFORM_COLORS[s.platform], pts: orbitPath(s, simT) })), [sats, pathsKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const ringsKey = Math.floor(simT / 20_000);
  const rings = useMemo(() => positions.map((p) => ({ lat: p.lat, lng: p.lng, color: PLATFORM_COLORS[p.s.platform], maxR: p.s.swath_km / 2 / 111 })), [ringsKey, sats]); // eslint-disable-line react-hooks/exhaustive-deps

  // hotspots inside a satellite's swath light up (refreshed ~3x / s)
  useEffect(() => {
    if (!data) return;
    const id = window.setInterval(() => {
      const arr = new Uint8Array(data.hotspots.length);
      data.hotspots.forEach((h, i) => {
        for (const p of positions) if (greatCircleKm(h[1], h[2], p.lat, p.lng) < p.s.swath_km / 2) { arr[i] = 1; break; }
      });
      setLit(arr);
    }, 330);
    return () => window.clearInterval(id);
  }, [data, positions]);

  useEffect(() => {
    if (!follow) return;
    const p = positions.find((x) => x.s.platform === follow);
    if (p) globeRef.current?.pointOfView({ lat: p.lat, lng: p.lng, altitude: 1.6 }, 0);
  }, [follow, positions]);

  const maxLog = useMemo(() => Math.max(1, ...(data?.hotspots ?? []).map((h) => Math.log1p(h[4] ?? 0))), [data]);
  const points = useMemo(() => (data?.hotspots ?? []).map((h, i) => ({
    lat: h[1], lng: h[2], h3: h[0], phi: h[4] ?? 0, iso: h[3],
    alt: 0.004 + 0.07 * (Math.log1p(h[4] ?? 0) / maxLog),
    color: lit[i] ? "#fffbe6" : (Math.log1p(h[4] ?? 0) / maxLog > 0.6 ? "#ff4d1a" : "#f97316"),
  })), [data, lit, maxLog]);
  const nLit = lit.reduce((a, b) => a + b, 0);
  const objFor = useMemo(() => new Map<string, THREE.Object3D>(), []);

  const utc = new Date(simT);
  return (
    <div className="stack-lg">
      <div className="page-head">
        <div>
          <h1>Live constellation</h1>
          <p className="muted">The five fire satellites on their sun-synchronous orbits over NASA Blue Marble imagery. When a satellite's swath sweeps over a fire hexagon it lights up: that is one observation feeding the harmonized record.</p>
        </div>
      </div>
      <div className="globe-layout">
        <div className="card globe-card" ref={wrap}>
          <div className="globe-hud">
            <div className="player">
              <button className="icon-btn" onClick={() => setRunning((r) => !r)} aria-label="play">{running ? <Pause size={15} /> : <Play size={15} />}</button>
              <input type="range" min={0} max={MONTH_LIST.length - 1} value={mi} onChange={(e) => setMi(Number(e.target.value))} />
              <b className="ym wide">{MONTHS[ym.m - 1]} {ym.y} · {utc.toISOString().slice(11, 16)} UTC</b>
            </div>
            <div className="seg">
              {SPEEDS.map((s) => <button key={s} className={speed === s ? "active" : ""} onClick={() => setSpeed(s)}>{s === 1 ? "real time" : `${s}×`}</button>)}
            </div>
          </div>
          <Globe
            ref={globeRef}
            width={size.w}
            height={size.h}
            backgroundColor="#00000000"
            globeImageUrl="/textures/earth-blue-marble.jpg"
            bumpImageUrl="/textures/earth-topology.png"
            backgroundImageUrl="/textures/night-sky.png"
            showAtmosphere
            atmosphereColor="#7dd3fc"
            atmosphereAltitude={0.18}
            objectsData={positions}
            objectLat="lat"
            objectLng="lng"
            objectAltitude="alt"
            objectFacesSurfaces
            objectThreeObject={(d: object) => {
              const p = d as { s: SatelliteMeta };
              let o = objFor.get(p.s.platform);
              if (!o) { o = satelliteObject(PLATFORM_COLORS[p.s.platform]); objFor.set(p.s.platform, o); }
              return o;
            }}
            pathsData={paths}
            pathPoints="pts"
            pathPointLat={(p: number[]) => p[0]}
            pathPointLng={(p: number[]) => p[1]}
            pathPointAlt={(p: number[]) => p[2]}
            pathColor={(d: object) => [`${(d as { color: string }).color}00`, (d as { color: string }).color, `${(d as { color: string }).color}00`]}
            pathStroke={1.2}
            pathDashLength={0.06}
            pathDashGap={0.012}
            pathDashAnimateTime={9000}
            pathTransitionDuration={0}
            ringsData={rings}
            ringColor={(d: object) => (t: number) => `${(d as { color: string }).color}${Math.round((1 - t) * 160).toString(16).padStart(2, "0")}`}
            ringMaxRadius="maxR"
            ringPropagationSpeed={6}
            ringRepeatPeriod={900}
            pointsData={points}
            pointLat="lat"
            pointLng="lng"
            pointAltitude="alt"
            pointColor="color"
            pointRadius={0.3}
            pointsMerge
            pointsTransitionDuration={0}
            htmlElementsData={positions}
            htmlLat="lat"
            htmlLng="lng"
            htmlAltitude={(d: object) => (d as { alt: number }).alt + 0.02}
            htmlElement={(d: object) => {
              const p = d as { s: SatelliteMeta };
              const el = document.createElement("div");
              el.className = "sat-label";
              el.style.color = PLATFORM_COLORS[p.s.platform];
              el.textContent = p.s.platform;
              return el;
            }}
          />
          <div className="globe-foot">
            <span><i className="dotp" style={{ background: "#f97316" }} /> fire hexagon (height = harmonized fire power)</span>
            <span><i className="dotp" style={{ background: "#fffbe6" }} /> inside a satellite swath now: <b>{nLit}</b></span>
            <span className="muted">{data ? `${compact(data.hotspots.length)} most active hexagons of ${MONTHS[ym.m - 1]} ${ym.y}` : "loading…"}</span>
          </div>
        </div>
        <aside className="globe-side">
          <div className="card">
            <div className="card-head"><h2><Satellite size={16} /> Satellites flying in {ym.y}</h2></div>
            <div className="sat-list">
              {(data?.satellites ?? []).map((s) => {
                const on = activeAt(s, isoDate);
                return (
                  <div key={s.platform} className={`sat-card ${on ? "" : "off"}`} style={{ borderColor: on ? PLATFORM_COLORS[s.platform] : undefined }}>
                    <div className="sat-top">
                      <label className="toggle">
                        <input type="checkbox" disabled={!on} checked={on && !hidden.has(s.platform)}
                          onChange={() => setHidden((h) => { const n = new Set(h); if (n.has(s.platform)) n.delete(s.platform); else n.add(s.platform); return n; })} />
                        <b style={{ color: PLATFORM_COLORS[s.platform] }}>{s.platform}</b>
                      </label>
                      <span className="pchip">{s.sensor} · {s.pixel_m} m</span>
                      {on && <button className="link-btn" onClick={() => setFollow(follow === s.platform ? "" : s.platform)}>{follow === s.platform ? "unfollow" : "follow"}</button>}
                    </div>
                    <div className="sat-facts">
                      <span>{s.altitude_km} km</span><span>{s.crossing}</span><span>swath {compact(s.swath_km)} km</span><span>{s.period_min} min/orbit</span>
                    </div>
                    <small className="muted">{on ? `${compact(s.n_detections ?? 0)} detections in the record (${s.record_first?.slice(0, 7)} → ${s.record_last?.slice(0, 7)})` : `not flying yet (launch ${s.launch.slice(0, 7)})`}{s.end && isoDate > s.end ? " · ended" : ""}</small>
                  </div>
                );
              })}
            </div>
          </div>
          <p className="muted small">Orbits use each mission's nominal parameters (altitude, inclination, period, equator-crossing time); positions along the orbit are illustrative, not real-time tracking. Earth imagery: NASA Blue Marble.</p>
        </aside>
      </div>
    </div>
  );
}
