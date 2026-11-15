import { useEffect, useMemo, useRef, useState } from "react";
import Globe, { type GlobeMethods } from "react-globe.gl";
import * as THREE from "three";
import { Pause, Play, Satellite } from "lucide-react";
import { api } from "../api";
import type { GlobeData, SatelliteMeta } from "../types";
import { PLATFORM_COLORS } from "../colors";
import { activeAt, bearingDeg, destination, greatCircleKm, orbitRing, subpoint, subsolarPoint } from "../orbits";
import { makeClouds, makeEarthMaterial, makeSpacecraft } from "../earthMaterial";
import { compact, MONTHS } from "../format";

const SPEEDS = [1, 60, 300, 1200];
const MONTH_LIST: Array<{ y: number; m: number }> = [];
for (let y = 2000; y <= 2026; y++) for (let m = 1; m <= 12; m++) {
  if ((y === 2000 && m < 11) || (y === 2026 && m > 10)) continue;
  MONTH_LIST.push({ y, m });
}

type Layer = { id: string; kind: "sat" | "fan" | "strip"; s: SatelliteMeta };

function rgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
}

function fanObject(color: string): THREE.Object3D {
  const g = new THREE.Group();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(9), 3));
  g.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.09, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending })));
  const edge = new THREE.BufferGeometry();
  edge.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(12), 3));
  g.add(new THREE.LineSegments(edge, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.28 })));
  return g;
}

const STRIP_N = 48;
function stripObject(color: string): THREE.Object3D {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(STRIP_N * 2 * 3), 3));
  const alpha = new Float32Array(STRIP_N * 2);
  for (let i = 0; i < STRIP_N; i++) { alpha[2 * i] = alpha[2 * i + 1] = 1 - i / (STRIP_N - 1); }
  geo.setAttribute("alpha", new THREE.Float32BufferAttribute(alpha, 1));
  const idx: number[] = [];
  for (let i = 0; i < STRIP_N - 1; i++) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  geo.setIndex(idx);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
    uniforms: { color: { value: new THREE.Color(color) } },
    vertexShader: "attribute float alpha; varying float vA; void main(){ vA = alpha; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
    fragmentShader: "uniform vec3 color; varying float vA; void main(){ gl_FragColor = vec4(color, 0.22 * vA * vA); }",
  });
  return new THREE.Mesh(geo, mat);
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
  // 15:00 UTC: the Americas (and their fires) in daylight on first view
  const [simT, setSimT] = useState(() => Date.UTC(ym.y, ym.m - 1, 15, 15));
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
    g.pointOfView({ lat: -8, lng: -55, altitude: 2.3 }, 0);
    const c = g.controls() as unknown as { autoRotate: boolean; autoRotateSpeed: number };
    c.autoRotate = true;
    c.autoRotateSpeed = 0.12;
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
  // closed orbit rings, fixed relative to the Sun, fading behind each satellite
  const ringTick = Math.floor(simT / 20_000);
  const paths = useMemo(() => sats.map((s) => {
    const pts = orbitRing(s, simT);
    const c = PLATFORM_COLORS[s.platform];
    return { name: s.platform, pts, colors: pts.map((_, k) => rgba(c, 0.95 * (1 - k / pts.length) ** 1.6)) };
  }), [sats, ringTick]); // eslint-disable-line react-hooks/exhaustive-deps

  // spacecraft, scan fans and scanned strips: stable objects updated in place
  const layers = useMemo<Layer[]>(() => sats.flatMap((s) => (["strip", "fan", "sat"] as const).map((kind) => ({ id: `${s.platform}-${kind}`, kind, s }))), [sats]);
  const simRef = useRef(simT);
  simRef.current = simT;
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
    alt: 0.002 + 0.035 * (Math.log1p(h[4] ?? 0) / maxLog),
    color: lit[i] ? "#fff7d6" : (Math.log1p(h[4] ?? 0) / maxLog > 0.6 ? "#ff5a1f" : "#ff9a3c"),
  })), [data, lit, maxLog]);
  const nLit = lit.reduce((a, b) => a + b, 0);
  const earth = useMemo(() => makeEarthMaterial(), []);
  const cloudsRef = useRef<ReturnType<typeof makeClouds> | null>(null);
  const onReady = () => {
    const g = globeRef.current;
    if (!g || cloudsRef.current) return;
    const clouds = makeClouds(g.getGlobeRadius());
    cloudsRef.current = clouds;
    g.scene().add(clouds.mesh);
  };
  // the ready callback can fire before the ref is attached: retry briefly
  useEffect(() => {
    let n = 0;
    const id = window.setInterval(() => {
      onReady();
      if (cloudsRef.current || ++n > 50) window.clearInterval(id);
    }, 200);
    return () => {
      window.clearInterval(id);
      if (cloudsRef.current) { globeRef.current?.scene().remove(cloudsRef.current.mesh); cloudsRef.current = null; }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // Sun direction for the simulated instant (date + time of day)
  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    const sp = subsolarPoint(simT);
    const c = g.getCoords(sp.lat, sp.lng, 0);
    const dir = new THREE.Vector3(c.x, c.y, c.z).normalize();
    earth.setSun(dir);
    cloudsRef.current?.setSun(dir);
    if (cloudsRef.current) cloudsRef.current.mesh.rotation.y = -Math.PI / 2 + (simT / 3_600_000) * 0.002;
  }, [simT, earth]);

  const updateLayer = (obj: THREE.Object3D, d: object) => {
    const g = globeRef.current;
    if (!g) return;
    const L = d as Layer;
    const t = simRef.current;
    const alt = L.s.altitude_km / 6371;
    const p = subpoint(L.s, t);
    const ahead = subpoint(L.s, t + 20_000);
    const head = bearingDeg(p.lat, p.lng, ahead.lat, ahead.lng);
    const xyz = (lat: number, lng: number, a: number) => { const c = g.getCoords(lat, lng, a); return new THREE.Vector3(c.x, c.y, c.z); };
    const P = xyz(p.lat, p.lng, alt);
    if (L.kind === "sat") {
      obj.position.copy(P);
      obj.up.copy(P.clone().normalize());
      obj.lookAt(xyz(ahead.lat, ahead.lng, alt));
      return;
    }
    const half = L.s.swath_km / 2;
    if (L.kind === "fan") {
      const left = destination(p.lat, p.lng, head - 90, half);
      const right = destination(p.lat, p.lng, head + 90, half);
      const Lp = xyz(left[0], left[1], 0.002), Rp = xyz(right[0], right[1], 0.002);
      const tri = (obj.children[0] as THREE.Mesh).geometry.getAttribute("position") as THREE.BufferAttribute;
      tri.setXYZ(0, P.x, P.y, P.z); tri.setXYZ(1, Lp.x, Lp.y, Lp.z); tri.setXYZ(2, Rp.x, Rp.y, Rp.z);
      tri.needsUpdate = true;
      const ed = (obj.children[1] as THREE.LineSegments).geometry.getAttribute("position") as THREE.BufferAttribute;
      ed.setXYZ(0, P.x, P.y, P.z); ed.setXYZ(1, Lp.x, Lp.y, Lp.z); ed.setXYZ(2, P.x, P.y, P.z); ed.setXYZ(3, Rp.x, Rp.y, Rp.z);
      ed.needsUpdate = true;
      (obj.children[0] as THREE.Mesh).geometry.computeBoundingSphere();
      (obj.children[1] as THREE.LineSegments).geometry.computeBoundingSphere();
      return;
    }
    // scanned strip: the ground swath of the last ~12 simulated minutes
    const pos = (obj as THREE.Mesh).geometry.getAttribute("position") as THREE.BufferAttribute;
    for (let i = 0; i < STRIP_N; i++) {
      const ti = t - (i / (STRIP_N - 1)) * 720_000;
      const a = subpoint(L.s, ti), b = subpoint(L.s, ti + 20_000);
      const hb = bearingDeg(a.lat, a.lng, b.lat, b.lng);
      const l = destination(a.lat, a.lng, hb - 90, half), r = destination(a.lat, a.lng, hb + 90, half);
      const lp = xyz(l[0], l[1], 0.003), rp = xyz(r[0], r[1], 0.003);
      pos.setXYZ(2 * i, lp.x, lp.y, lp.z); pos.setXYZ(2 * i + 1, rp.x, rp.y, rp.z);
    }
    pos.needsUpdate = true;
    (obj as THREE.Mesh).geometry.computeBoundingSphere();
  };

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
            backgroundImageUrl="/textures/night-sky.png"
            globeMaterial={earth.material}
            onGlobeReady={onReady}
            showAtmosphere
            atmosphereColor="#5fa8ff"
            atmosphereAltitude={0.14}
            showGraticules={false}
            pathsData={paths}
            pathPoints="pts"
            pathPointLat={(p: number[]) => p[0]}
            pathPointLng={(p: number[]) => p[1]}
            pathPointAlt={(p: number[]) => p[2]}
            pathColor={(d: object) => (d as { colors: string[] }).colors}
            pathResolution={2}
            pathTransitionDuration={0}
            customLayerData={layers.slice()}
            customThreeObject={(d: object) => {
              const L = d as Layer;
              const c = PLATFORM_COLORS[L.s.platform];
              return L.kind === "sat" ? makeSpacecraft(c) : L.kind === "fan" ? fanObject(c) : stripObject(c);
            }}
            customThreeObjectUpdate={updateLayer}
            pointsData={points}
            pointLat="lat"
            pointLng="lng"
            pointAltitude="alt"
            pointColor="color"
            pointRadius={0.13}
            pointResolution={6}
            pointsMerge
            pointsTransitionDuration={0}
            htmlElementsData={positions}
            htmlLat="lat"
            htmlLng="lng"
            htmlAltitude={(d: object) => (d as { alt: number }).alt + 0.015}
            htmlElement={(d: object) => {
              const p = d as { s: SatelliteMeta };
              const el = document.createElement("div");
              el.className = "sat-label";
              el.innerHTML = `<i style="background:${PLATFORM_COLORS[p.s.platform]}"></i>${p.s.platform}`;
              return el;
            }}
          />
          <div className="globe-foot">
            <span><i className="dotp" style={{ background: "#ff9a3c" }} /> fire hexagon (height = harmonized fire power)</span>
            <span className="muted">fan = instrument scan · glowing band = ground strip imaged in the last 12 min</span>
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
