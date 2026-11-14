import { useEffect, useRef } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";
import type { FireCollection, FireCellProperties, Region } from "./data";

interface MapViewProps {
  region: Region;
  cells: FireCollection;
  selectedCell: FireCellProperties | null;
  onCellSelect: (properties: FireCellProperties) => void;
}

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export default function MapView({
  region,
  cells,
  selectedCell,
  onCellSelect,
}: MapViewProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const cellClickRef = useRef(onCellSelect);
  cellClickRef.current = onCellSelect;

  useEffect(() => {
    if (!hostRef.current) return;
    const map = new maplibregl.Map({
      container: hostRef.current,
      style: MAP_STYLE,
      center: region.center,
      zoom: region.zoom,
      attributionControl: false,
      cooperativeGestures: true,
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");

    map.on("load", () => {
      map.setProjection({ type: "globe" });
      map.setFog({
        color: "rgb(16, 23, 26)",
        "high-color": "rgb(26, 48, 62)",
        "space-color": "rgb(7, 12, 16)",
        "horizon-blend": 0.08,
      });

      map.addSource("fire-cells", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      const firstSymbol = map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
      map.addLayer({
        id: "fire-cell-fill",
        type: "fill",
        source: "fire-cells",
        paint: {
          "fill-color": [
            "interpolate",
            ["linear"],
            ["coalesce", ["get", "value"], 0],
            0, "#f2b66d",
            0.18, "#ef864e",
            0.48, "#e65d42",
            0.78, "#d83735",
            1, "#a91f32",
          ],
          "fill-opacity": [
            "interpolate",
            ["linear"],
            ["coalesce", ["get", "value"], 0],
            0, 0.12,
            0.12, 0.44,
            1, 0.82,
          ],
        },
      }, firstSymbol);
      map.addLayer({
        id: "fire-cell-outline",
        type: "line",
        source: "fire-cells",
        paint: {
          "line-color": "rgba(255, 228, 194, 0.58)",
          "line-width": 0.7,
          "line-opacity": 0.58,
        },
      }, firstSymbol);
      map.on("click", "fire-cell-fill", (event) => {
        const properties = event.features?.[0]?.properties;
        if (properties) cellClickRef.current(properties as FireCellProperties);
      });
      map.on("mouseenter", "fire-cell-fill", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "fire-cell-fill", () => {
        map.getCanvas().style.cursor = "";
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const fitRegion = () => {
      map.fitBounds(
        [[region.bbox[0], region.bbox[1]], [region.bbox[2], region.bbox[3]]],
        { padding: { top: 72, right: 56, bottom: 86, left: 56 }, duration: 900, maxZoom: region.zoom },
      );
    };
    if (map.isStyleLoaded()) fitRegion();
    else map.once("load", fitRegion);
  }, [region]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const updateCells = () => {
      const source = map.getSource("fire-cells") as maplibregl.GeoJSONSource | undefined;
      source?.setData(cells);
    };
    if (map.getSource("fire-cells")) updateCells();
    else map.once("load", updateCells);
  }, [cells]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer("fire-cell-outline")) return;
    map.setFilter("fire-cell-outline", null);
    if (selectedCell?.h3_index === undefined) return;
    map.setFilter("fire-cell-outline", [
      "==",
      ["to-string", ["get", "h3_index"]],
      String(selectedCell.h3_index),
    ]);
  }, [selectedCell]);

  return <div ref={hostRef} className="map-canvas" role="application" aria-label="Interactive globe and fire segmentation map" />;
}
