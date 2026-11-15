import { useEffect, useMemo, useRef } from "react";
import maplibregl, { type Map as MLMap } from "maplibre-gl";
import type { ReplayData } from "../types";
import { STYLE } from "./HexMap";
import { PLATFORM_COLORS } from "../colors";

interface Props {
  data: ReplayData;
  frame: number;
  trail: number;            // per-frame decay of the heat afterglow (0..1)
  showDetections: boolean;
  onSelect?: (h3: string) => void;
}

const EMPTY = { type: "FeatureCollection" as const, features: [] };

/** Heat state at frame f: max over earlier frames of the (normalized)
 *  harmonized daily fire power, decayed by trail^(age in frames). */
const BY_DAY = new WeakMap<ReplayData, Map<number, Array<[number, number]>>>();

function byDayOf(data: ReplayData) {
  let m = BY_DAY.get(data);
  if (!m) {
    m = new Map();
    for (const [d0, c, q50] of data.daily) {
      if (q50 === null) continue;
      (m.get(d0) ?? m.set(d0, []).get(d0)!).push([c, q50]);
    }
    BY_DAY.set(data, m);
  }
  return m;
}

export function heatAt(data: ReplayData, f: number, trail: number, norm: number): Float32Array {
  const heat = new Float32Array(data.cells.length);
  const byDay = byDayOf(data);
  for (let i = 0; i <= f && i < data.frames.length; i++) {
    for (let c = 0; c < heat.length; c++) heat[c] *= trail;
    // in hour mode only frames with detections exist; the daily state of
    // the frame's day lights the hexagons
    for (const [c, v] of byDay.get(data.frames[i].day - 1) ?? []) {
      const x = Math.log1p(v) / norm;
      if (x > heat[c]) heat[c] = x;
    }
  }
  return heat;
}

export default function ReplayMap({ data, frame, trail, showDetections, onSelect }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const ready = useRef(false);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;

  const norm = useMemo(() => {
    const v = data.daily.map((r) => Math.log1p(r[2] ?? 0)).sort((a, b) => a - b);
    return Math.max(0.5, v.length ? v[Math.floor(v.length * 0.98)] : 1);
  }, [data]);

  const bounds = useMemo<[number, number, number, number] | null>(() => {
    if (!data.polygons.length) return null;
    let a = 180, b = 90, c = -180, d = -90;
    for (const ring of data.polygons) for (const [x, y] of ring) {
      a = Math.min(a, x); b = Math.min(b, y); c = Math.max(c, x); d = Math.max(d, y);
    }
    return [a, b, c, d];
  }, [data]);

  useEffect(() => {
    if (!host.current) return;
    const map = new maplibregl.Map({ container: host.current, style: STYLE, center: [0, 0], zoom: 2, attributionControl: { compact: true } });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource("hex", { type: "geojson", data: EMPTY });
      map.addSource("det", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "hex-heat", type: "fill", source: "hex", paint: {
        "fill-color": ["interpolate", ["linear"], ["get", "heat"],
          0, "#1e293b", 0.08, "#7c2d12", 0.3, "#c2410c", 0.55, "#f97316", 0.8, "#fbbf24", 1, "#fef9c3"],
        "fill-opacity": ["interpolate", ["linear"], ["get", "heat"], 0, 0.12, 0.05, 0.35, 0.3, 0.75, 1, 0.95],
      } });
      map.addLayer({ id: "hex-edge", type: "line", source: "hex", paint: {
        "line-color": ["case", [">", ["get", "live"], 0], "rgba(254,240,138,0.85)", "rgba(15,23,42,0.5)"],
        "line-width": ["case", [">", ["get", "live"], 0], 1.1, 0.3],
      } });
      map.addLayer({ id: "det-heat", type: "heatmap", source: "det", paint: {
        "heatmap-weight": ["*", ["min", 1, ["/", ["ln", ["+", 1, ["get", "frp"]]], 5]], ["-", 1, ["/", ["get", "age"], 4]]],
        "heatmap-intensity": 1.2,
        "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 3, 8, 8, 26],
        "heatmap-opacity": 0.75,
        "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
          0, "rgba(0,0,0,0)", 0.2, "rgba(234,88,12,0.5)", 0.5, "#f97316", 0.8, "#fbbf24", 1, "#ffffff"],
      } });
      map.addLayer({ id: "det-now", type: "circle", source: "det", filter: ["==", ["get", "age"], 0], paint: {
        "circle-radius": ["interpolate", ["linear"], ["ln", ["+", 1, ["get", "frp"]]], 0, 2, 5, 6],
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#fff", "circle-stroke-width": 0.8, "circle-opacity": 0.95,
      } });
      map.on("click", "hex-heat", (e) => { const f = e.features?.[0]; if (f) selectRef.current?.(String(f.properties?.h3_index)); });
      map.on("mouseenter", "hex-heat", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "hex-heat", () => { map.getCanvas().style.cursor = ""; });
      ready.current = true;
      map.fire("fh-ready");
    });
    return () => { ready.current = false; map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !bounds) return;
    const fit = () => map.fitBounds(bounds, { padding: 40, duration: 800, maxZoom: 8 });
    if (ready.current) fit(); else map.once("fh-ready", fit);
  }, [bounds]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      const heat = heatAt(data, frame, trail, norm);
      const today = new Set<number>();
      const day0 = (data.frames[frame]?.day ?? 1) - 1;
      for (const [d0, c, q50] of data.daily) if (d0 === day0 && (q50 ?? 0) > 0.05) today.add(c);
      (map.getSource("hex") as maplibregl.GeoJSONSource).setData({
        type: "FeatureCollection",
        features: data.cells.map((h3, i) => ({
          type: "Feature" as const,
          properties: { h3_index: h3, heat: Math.min(1.2, heat[i]), live: today.has(i) ? 1 : 0 },
          geometry: { type: "Polygon" as const, coordinates: [data.polygons[i]] },
        })),
      });
      const det = showDetections
        ? data.detections.filter((d) => d[0] <= frame && d[0] > frame - 4).map((d) => ({
            type: "Feature" as const,
            properties: { frp: d[3], age: frame - d[0], color: PLATFORM_COLORS[data.platforms[d[4]]] ?? "#fff" },
            geometry: { type: "Point" as const, coordinates: [d[1], d[2]] },
          }))
        : [];
      (map.getSource("det") as maplibregl.GeoJSONSource).setData({ type: "FeatureCollection", features: det });
    };
    if (ready.current) apply(); else map.once("fh-ready", apply);
  }, [data, frame, trail, norm, showDetections]);

  return <div ref={host} className="hexmap" />;
}
