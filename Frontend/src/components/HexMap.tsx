import { useEffect, useRef } from "react";
import maplibregl, { type Map as MLMap, type StyleSpecification } from "maplibre-gl";
import type { HexCollection } from "../types";

export type MapMetric = "frp" | "anomaly" | "uncertainty" | "evidence";

// Inline style: if the basemap tiles cannot load (offline judging laptop),
// the hexagons still render on a dark background.
const STYLE: StyleSpecification = {
  version: 8,
  sources: {
    carto: {
      type: "raster",
      tiles: ["https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
              "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors © CARTO",
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0b1220" } },
    { id: "carto", type: "raster", source: "carto", paint: { "raster-opacity": 0.9 } },
  ],
};

const COLOR_EXPR: Record<MapMetric, unknown> = {
  frp: ["interpolate", ["linear"], ["ln", ["+", 1, ["get", "phi_q50_mw"]]],
    0, "#1e3a5f", 1.5, "#f59e0b", 3, "#f97316", 4.5, "#ef4444", 6, "#fde68a"],
  anomaly: ["interpolate", ["linear"], ["ln", ["max", 0.05, ["get", "anomaly_ratio"]]],
    -2, "#2563eb", -0.5, "#60a5fa", 0, "#e2e8f0", 0.7, "#f97316", 2, "#dc2626"],
  uncertainty: ["interpolate", ["linear"], ["/", ["get", "width"], ["max", 0.1, ["get", "phi_q50_mw"]]],
    0.2, "#34d399", 0.8, "#fbbf24", 1.6, "#f87171"],
  evidence: ["match", ["get", "quality_flag"],
    "NOMINAL", "#34d399", "CLIMATOLOGY_CLOUD_FALLBACK", "#fbbf24", "ABSTAIN_LOW_EVIDENCE", "#f87171", "#94a3b8"],
};

interface Props {
  data: HexCollection | null;
  bbox: [number, number, number, number];
  metric: MapMetric;
  selected: string | null;
  onSelect: (h3: string) => void;
  fitKey: string;            // re-fit to the data extent when this changes
}

function dataBounds(data: HexCollection | null): [number, number, number, number] | null {
  if (!data?.features.length) return null;
  let a = 180, b = 90, c = -180, d = -90;
  for (const f of data.features) for (const [x, y] of f.geometry.coordinates[0]) {
    a = Math.min(a, x); b = Math.min(b, y); c = Math.max(c, x); d = Math.max(d, y);
  }
  return [a, b, c, d];
}

export default function HexMap({ data, bbox, metric, selected, onSelect, fitKey }: Props) {
  const fitted = useRef<string | null>(null);
  const host = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const ready = useRef(false);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;

  useEffect(() => {
    if (!host.current) return;
    const map = new maplibregl.Map({ container: host.current, style: STYLE, bounds: bbox, fitBoundsOptions: { padding: 30 }, attributionControl: { compact: true } });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource("hex", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({ id: "hex-fill", type: "fill", source: "hex",
        paint: { "fill-color": COLOR_EXPR.frp as never, "fill-opacity": 0.82 } });
      map.addLayer({ id: "hex-line", type: "line", source: "hex",
        paint: { "line-color": "rgba(15,23,42,0.6)", "line-width": 0.4 } });
      map.addLayer({ id: "hex-selected", type: "line", source: "hex", filter: ["==", ["get", "h3_index"], ""],
        paint: { "line-color": "#ffffff", "line-width": 2.5 } });
      map.on("click", "hex-fill", (e) => {
        const f = e.features?.[0];
        if (f) selectRef.current(String(f.properties?.h3_index));
      });
      map.on("mouseenter", "hex-fill", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "hex-fill", () => { map.getCanvas().style.cursor = ""; });
      ready.current = true;
      map.fire("fh-ready");
    });
    return () => { ready.current = false; map.remove(); mapRef.current = null; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      (map.getSource("hex") as maplibregl.GeoJSONSource | undefined)?.setData((data ?? { type: "FeatureCollection", features: [] }) as never);
      const b = dataBounds(data);
      if (b && fitted.current !== fitKey) {   // zoom to where the hexagons are
        fitted.current = fitKey;
        map.fitBounds(b, { padding: 50, duration: 900, maxZoom: 8 });
      }
    };
    if (ready.current) apply(); else map.once("fh-ready", apply);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => map.setPaintProperty("hex-fill", "fill-color", COLOR_EXPR[metric] as never);
    if (ready.current) apply(); else map.once("fh-ready", apply);
  }, [metric]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => map.setFilter("hex-selected", ["==", ["get", "h3_index"], selected ?? ""]);
    if (ready.current) apply(); else map.once("fh-ready", apply);
  }, [selected]);


  return <div ref={host} className="hexmap" />;
}

export const LEGENDS: Record<MapMetric, { label: string; stops: Array<[string, string]> }> = {
  frp: { label: "Daily-mean fire radiative power", stops: [["#1e3a5f", "~0"], ["#f59e0b", "3 MW"], ["#f97316", "20"], ["#ef4444", "90"], ["#fde68a", "400+"]] },
  anomaly: { label: "vs 2003-2020 median for this month", stops: [["#2563eb", "×0.1"], ["#60a5fa", "×0.6"], ["#e2e8f0", "×1"], ["#f97316", "×2"], ["#dc2626", "×7+"]] },
  uncertainty: { label: "Relative 90% interval width", stops: [["#34d399", "narrow"], ["#fbbf24", "medium"], ["#f87171", "wide"]] },
  evidence: { label: "Evidence quality", stops: [["#34d399", "nominal"], ["#fbbf24", "climatological cloud"], ["#f87171", "abstained"]] },
};
