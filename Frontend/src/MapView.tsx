import { useEffect, useRef } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";
import type { FireCollection, FireCellProperties, FireFeature, Region } from "./data";

function featureKey(feature: FireFeature): string {
  return String(feature.properties.h3_index ?? "");
}

function numberProperty(feature: FireFeature, property: string): number {
  const value = feature.properties[property];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function interpolateCollections(
  previous: FireCollection,
  next: FireCollection,
  progress: number,
): FireCollection {
  const previousByIndex = new Map(
    previous.features.map((feature) => [featureKey(feature), feature]),
  );
  const nextIndexes = new Set(next.features.map(featureKey));
  const features = next.features.map((feature) => {
    const prior = previousByIndex.get(featureKey(feature));
    const startValue = prior ? numberProperty(prior, "value") : 0;
    const startHeat = prior ? numberProperty(prior, "heat_score") : 0;
    return {
      ...feature,
      properties: {
        ...feature.properties,
        value: startValue + (numberProperty(feature, "value") - startValue) * progress,
        heat_score: startHeat +
          (numberProperty(feature, "heat_score") - startHeat) * progress,
        transition_progress: prior ? 1 : progress,
      },
    };
  });

  for (const feature of previous.features) {
    const index = featureKey(feature);
    if (nextIndexes.has(index)) continue;
    features.push({
      ...feature,
      properties: {
        ...feature.properties,
        value: numberProperty(feature, "value") * (1 - progress),
        heat_score: numberProperty(feature, "heat_score") * (1 - progress),
        transition_progress: 1 - progress,
      },
    });
  }
  return { type: "FeatureCollection", features };
}

interface MapViewProps {
  region: Region;
  focused: boolean;
  cells: FireCollection;
  selectedCell: FireCellProperties | null;
  onCellSelect: (properties: FireCellProperties) => void;
}

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export default function MapView({
  region,
  focused,
  cells,
  selectedCell,
  onCellSelect,
}: MapViewProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const previousCellsRef = useRef<FireCollection | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const projectionRef = useRef<"globe" | "mercator">("globe");
  const hasBeenFocusedRef = useRef(false);
  const cellClickRef = useRef(onCellSelect);
  cellClickRef.current = onCellSelect;

  useEffect(() => {
    if (!hostRef.current) return;
    const map = new maplibregl.Map({
      container: hostRef.current,
      style: MAP_STYLE,
      center: [0, 18],
      zoom: 1.15,
      pitch: 0,
      attributionControl: false,
      cooperativeGestures: true,
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");

    map.on("load", () => {
      map.setProjection({ type: "globe" });

      map.addSource("fire-cells", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      const firstSymbol = map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
      map.addLayer({
        id: "fire-cell-extrusion",
        type: "fill-extrusion",
        source: "fire-cells",
        paint: {
          "fill-extrusion-color": [
            "interpolate",
            ["linear"],
            ["coalesce", ["get", "value"], 0],
            0, "#f2b66d",
            0.18, "#ef864e",
            0.48, "#e65d42",
            0.78, "#d83735",
            1, "#a91f32",
          ],
          "fill-extrusion-height": [
            "interpolate",
            ["linear"],
            [
              "*",
              ["coalesce", ["get", "heat_score"], 0],
              ["coalesce", ["get", "transition_progress"], 1],
            ],
            0, 0,
            0.08, 250,
            1, 6500,
          ],
          "fill-extrusion-base": 0,
          "fill-extrusion-opacity": 0.9,
          "fill-extrusion-vertical-gradient": true,
        },
      }, firstSymbol);
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
            "*",
            [
              "interpolate",
              ["linear"],
              ["coalesce", ["get", "value"], 0],
              0, 0.12,
              0.12, 0.44,
              1, 0.82,
            ],
            ["coalesce", ["get", "transition_progress"], 1],
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
          "line-opacity": [
            "*",
            0.58,
            ["coalesce", ["get", "transition_progress"], 1],
          ],
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
    let projectionTimer: number | undefined;
    const moveToRegion = () => {
      if (projectionRef.current === "globe") {
        map.fitBounds(
          [[region.bbox[0], region.bbox[1]], [region.bbox[2], region.bbox[3]]],
          {
            padding: { top: 52, right: 48, bottom: 62, left: 48 },
            duration: 2300,
            maxZoom: region.zoom,
            pitch: 48,
            bearing: -14,
            curve: 1.55,
            speed: 0.62,
            essential: true,
          },
        );
        projectionTimer = window.setTimeout(() => {
          if (mapRef.current !== map) return;
          map.setProjection({ type: "mercator" });
          projectionRef.current = "mercator";
        }, 1850);
        return;
      }
      map.fitBounds(
        [[region.bbox[0], region.bbox[1]], [region.bbox[2], region.bbox[3]]],
        {
          padding: { top: 52, right: 48, bottom: 62, left: 48 },
          duration: 1500,
          maxZoom: region.zoom,
          pitch: 48,
          bearing: -14,
          curve: 1.55,
          speed: 0.62,
          essential: true,
        },
      );
    };
    if (focused) {
      hasBeenFocusedRef.current = true;
      if (map.isStyleLoaded()) moveToRegion();
      else map.once("load", moveToRegion);
    } else if (hasBeenFocusedRef.current) {
      if (projectionRef.current === "mercator") {
        map.setProjection({ type: "globe" });
        projectionRef.current = "globe";
      }
      hasBeenFocusedRef.current = false;
      map.easeTo({
        center: [0, 18],
        zoom: 1.15,
        pitch: 0,
        bearing: 0,
        duration: 1700,
        essential: true,
      });
    }
    return () => {
      if (projectionTimer !== undefined) window.clearTimeout(projectionTimer);
    };
  }, [region, focused]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const updateCells = () => {
      const source = map.getSource("fire-cells") as maplibregl.GeoJSONSource | undefined;
      if (!source) return;
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      const previous = previousCellsRef.current;
      if (!previous?.features.length || !cells.features.length) {
        source.setData(cells);
        previousCellsRef.current = cells;
        return;
      }

      const started = performance.now();
      const duration = 700;
      const animate = (now: number) => {
        const linearProgress = Math.min(1, (now - started) / duration);
        const progress = linearProgress * linearProgress * (3 - 2 * linearProgress);
        const frame = interpolateCollections(previous, cells, progress);
        source.setData(frame);
        previousCellsRef.current = frame;
        if (linearProgress < 1) {
          animationFrameRef.current = window.requestAnimationFrame(animate);
        } else {
          animationFrameRef.current = null;
          previousCellsRef.current = cells;
        }
      };
      animationFrameRef.current = window.requestAnimationFrame(animate);
    };
    if (map.getSource("fire-cells")) updateCells();
    else map.once("load", updateCells);
    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
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
