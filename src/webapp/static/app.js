/* Burning Activity Calendar — vanilla ES6 + MapLibre GL JS + Apache ECharts.
 * No build step, no framework: plain modules of functions wired to the DOM.
 */
"use strict";

const PRESETS = {
  "Chilecito ARG 2026": [-68.0, -29.6, -67.1, -28.7],
  "Kabylie DZA": [3.8, 36.3, 5.5, 37.0],
  "Hassi Messaoud Gas Flares DZA": [5.5, 31.3, 6.6, 32.2],
  "Alaska Boreal USA": [-156.0, 63.0, -142.0, 67.5],
  "Black Summer AUS": [147.0, -37.8, 151.0, -34.5],
  "Kalimantan Peat IDN": [112.5, -3.3, 115.2, -1.5],
};

const VIRIDIS = [
  "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
  "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
];
const CIVIDIS = [
  "#00204d", "#31446b", "#666970", "#958f78", "#cab969",
  "#ecdf2f", "#f4ea53", "#ffd325",
];

const $ = (id) => document.getElementById(id);

function setStatus(text) { $("status").textContent = text; }

function bboxFromInputs() {
  return [
    parseFloat($("min_lon").value), parseFloat($("min_lat").value),
    parseFloat($("max_lon").value), parseFloat($("max_lat").value),
  ];
}

function setBbox(bbox) {
  $("min_lon").value = bbox[0]; $("min_lat").value = bbox[1];
  $("max_lon").value = bbox[2]; $("max_lat").value = bbox[3];
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
  }
  return res.json();
}

/* ------------------------------------------------------------------ */
/* Map: ROI drawing + hexagon layer                                    */
/* ------------------------------------------------------------------ */
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      basemap: {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
        ],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors, © CARTO",
      },
    },
    layers: [
      { id: "base", type: "raster", source: "basemap" },
    ],
  },
  center: [-67.5, -29.16],
  zoom: 6.5,
});
map.addControl(new maplibregl.NavigationControl());

let drawMode = false;
let drawStart = null;
let roiGeo = null;

map.on("load", () => {
  map.addSource("hex", { type: "geojson", data: emptyFC() });
  map.addLayer({
    id: "hex-fill", type: "fill", source: "hex",
    paint: {
      "fill-color": ["get", "color"],
      "fill-opacity": 0.75,
      "fill-outline-color": "#ffffff33",
    },
  });
  map.addSource("roi", { type: "geojson", data: emptyFC() });
  map.addLayer({
    id: "roi-outline", type: "line", source: "roi",
    paint: { "line-color": "#f0c441", "line-width": 2 },
  });
  refreshHexLayer();
});

function emptyFC() { return { type: "FeatureCollection", features: [] }; }

map.on("mousedown", (e) => {
  if (!drawMode) return;
  drawStart = e.lngLat;
  e.preventDefault();
});
map.on("mousemove", (e) => {
  if (!drawMode || !drawStart) return;
  updateRoiBox(drawStart, e.lngLat);
});
map.on("mouseup", (e) => {
  if (!drawMode || !drawStart) return;
  const box = updateRoiBox(drawStart, e.lngLat);
  drawStart = null;
  if (box) {
    setBbox(box);
    setStatus(`AOI set to [${box.map((v) => v.toFixed(2)).join(", ")}]`);
    refreshAll();
  }
});

function updateRoiBox(a, b) {
  const west = Math.min(a.lng, b.lng), east = Math.max(a.lng, b.lng);
  const south = Math.min(a.lat, b.lat), north = Math.max(a.lat, b.lat);
  if (east - west < 1e-4 || north - south < 1e-4) return null;
  const box = [west, south, east, north];
  map.getSource("roi").setData({
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [[
        [west, south], [east, south], [east, north], [west, north],
        [west, south],
      ]],
    },
    properties: {},
  });
  return box;
}

function paletteColors() {
  return $("palette").value === "cividis" ? CIVIDIS : VIRIDIS;
}

function colorFor(value, vmin, vmax) {
  const colors = paletteColors();
  const t = vmax > vmin ? (value - vmin) / (vmax - vmin) : 0.5;
  const idx = Math.min(
    colors.length - 1,
    Math.max(0, Math.round(t * (colors.length - 1)))
  );
  return colors[idx];
}

async function refreshHexLayer() {
  const bbox = bboxFromInputs();
  const year = parseInt($("end_year").value, 10);
  const month = 8; // peak-season default for the map layer
  const colorBy = $("metric").value === "anomaly_ratio"
    ? "anomaly_ratio" : "phi_q50_mw";
  try {
    const fc = await fetchJSON(
      `/api/map_hex?min_lon=${bbox[0]}&min_lat=${bbox[1]}` +
      `&max_lon=${bbox[2]}&max_lat=${bbox[3]}` +
      `&year=${year}&month=${month}&color_by=${colorBy}`
    );
    const values = fc.features.map(
      (f) => f.properties[colorBy === "anomaly_ratio"
        ? "anomaly_ratio" : "phi_q50_mw"]
    );
    const vmin = Math.min(...values, 0);
    const vmax = Math.max(...values, 1e-9);
    for (const f of fc.features) {
      const key = colorBy === "anomaly_ratio"
        ? "anomaly_ratio" : "phi_q50_mw";
      f.properties.color = colorFor(f.properties[key], vmin, vmax);
    }
    map.getSource("hex").setData(fc);
    $("legend-metric").textContent =
      colorBy === "anomaly_ratio" ? "anomaly ratio" : "harmonized flux q50";
  } catch (err) {
    setStatus(`map: ${err.message}`);
  }
}

map.on("click", "hex-fill", (e) => {
  const p = e.features[0].properties;
  new maplibregl.Popup()
    .setLngLat(e.lngLat)
    .setHTML(
      `<b>H3 ${p.h3_index}</b><br/>flux q50: ${p.phi_q50_mw.toFixed(2)} MW` +
      `<br/>anomaly ratio: ${p.anomaly_ratio.toFixed(2)}` +
      `<br/>hist. median: ${p.hist_median.toFixed(2)} MW` +
      `<br/>quality: ${p.quality_flag}`
    )
    .addTo(map);
});

/* ------------------------------------------------------------------ */
/* Calendar heatmap (Year x Month) with abstain hatching               */
/* ------------------------------------------------------------------ */
const calendarChart = echarts.init($("calendar"));
const timeseriesChart = echarts.init($("timeseries"));

function monthLabel(m) {
  return ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"][m - 1];
}

async function refreshCalendar() {
  const bbox = bboxFromInputs();
  const startYear = parseInt($("start_year").value, 10);
  const endYear = parseInt($("end_year").value, 10);
  try {
    const data = await fetchJSON(
      `/api/calendar?min_lon=${bbox[0]}&min_lat=${bbox[1]}` +
      `&max_lon=${bbox[2]}&max_lat=${bbox[3]}` +
      `&start_year=${startYear}&end_year=${endYear}` +
      `&metric=${$("metric").value}`
    );
    const cells = data.cells;
    const byYearMonth = new Map();
    for (const c of cells) {
      const key = `${c.year}-${c.month}`;
      const agg = byYearMonth.get(key) || { v: 0, n: 0, abstain: false };
      agg.v += c.value; agg.n += 1;
      if (c.quality_flag === "ABSTAIN_LOW_EVIDENCE") agg.abstain = true;
      byYearMonth.set(key, agg);
    }
    const years = [];
    for (let y = startYear; y <= endYear; y++) years.push(y);
    const points = [];
    const borders = [];
    let vmax = 0;
    for (const [key, agg] of byYearMonth) {
      const [y, m] = key.split("-").map(Number);
      const v = agg.v / Math.max(agg.n, 1);
      vmax = Math.max(vmax, v);
      points.push([m - 1, y - startYear, v, agg.abstain]);
    }
    for (const p of points) {
      if (p[3] && $("hatch_abstain").checked) borders.push(p);
    }
    calendarChart.setOption({
      tooltip: {
        position: "top",
        formatter: (p) =>
          `${p.data[1] + startYear}-${monthLabel(p.data[0] + 1)}` +
          `<br/>value: ${p.data[2].toFixed(3)}` +
          (p.data[3] ? "<br/><b>ABSTAIN_LOW_EVIDENCE</b>" : ""),
      },
      grid: { top: 8, left: 44, right: 8, bottom: 44 },
      xAxis: {
        type: "category",
        data: Array.from({ length: 12 }, (_, i) => monthLabel(i + 1)),
        splitArea: { show: true },
      },
      yAxis: {
        type: "category",
        data: years.map(String),
        splitArea: { show: true },
      },
      visualMap: {
        min: 0, max: Math.max(vmax, 1e-9),
        calculable: true,
        orient: "horizontal",
        left: "center", bottom: 0,
        inRange: { color: paletteColors() },
      },
      series: [
        {
          type: "heatmap",
          data: points,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 6, shadowColor: "#000" } },
          itemStyle: {
            borderColor: "#ffffff22",
            borderWidth: 1,
          },
        },
        {
          // Diagonal-hatch overlay for ABSTAIN_LOW_EVIDENCE months: drawn
          // as striped scatter markers above the heatmap tiles.
          type: "scatter",
          symbol: "rect",
          symbolSize: [14, 14],
          data: borders.map((p) => [p[0], p[1], p[2]]),
          itemStyle: { color: "rgba(0,0,0,0)", borderColor: "#ff5252", borderWidth: 2 },
          tooltip: {
            formatter: () => "ABSTAIN_LOW_EVIDENCE (hatched border)",
          },
        },
      ],
    });
    setStatus(`calendar: ${data.meta.n_tiles} tiles loaded`);
  } catch (err) {
    setStatus(`calendar: ${err.message}`);
  }
}

/* ------------------------------------------------------------------ */
/* Raw vs harmonized time series with conformal CI ribbons             */
/* ------------------------------------------------------------------ */
async function refreshTimeseries() {
  const bbox = bboxFromInputs();
  const startYear = parseInt($("start_year").value, 10);
  const endYear = parseInt($("end_year").value, 10);
  try {
    const data = await fetchJSON(
      `/api/timeseries?min_lon=${bbox[0]}&min_lat=${bbox[1]}` +
      `&max_lon=${bbox[2]}&max_lat=${bbox[3]}` +
      `&start_year=${startYear}&end_year=${endYear}`
    );
    const s = data.series;
    const months = s.map((d) => d.month);
    const markLines = data.constellation_events.map((ev) => ({
      xAxis: ev,
      label: { formatter: ev, fontSize: 10, rotate: 90 },
      lineStyle: { color: "#f4a", type: "dashed", width: 1 },
    }));
    const series = [
      {
        name: "harmonized q50 (MW)", type: "line", data: s.map((d) => d.phi_q50_mw),
        showSymbol: false, lineStyle: { color: "#f0c441", width: 2 },
        markLine: { data: markLines, symbol: "none", silent: true },
      },
      {
        name: "raw hotspot counts", type: "line",
        yAxisIndex: 1, data: s.map((d) => d.raw_count),
        showSymbol: false, lineStyle: { color: "#4fc3f7", width: 1.4 },
      },
    ];
    if ($("show_uncertainty").checked && s.length) {
      series.push({
        name: "conformal CI", type: "line", data: s.map((d) => d.phi_q95_mw),
        lineStyle: { opacity: 0 }, stack: "ci", symbol: "none",
        silent: true,
      });
      series.push({
        name: "CI band", type: "line",
        data: s.map((d) => d.phi_q95_mw - d.phi_q05_mw),
        lineStyle: { opacity: 0 }, stack: "ci", symbol: "none",
        areaStyle: { color: "rgba(240,196,65,0.18)" },
        silent: true,
      });
    }
    timeseriesChart.setOption({
      tooltip: { trigger: "axis" },
      legend: { top: 0, textStyle: { color: "#93a1b0" } },
      grid: { left: 52, right: 52, top: 30, bottom: 56 },
      xAxis: { type: "category", data: months, axisLabel: { rotate: 45 } },
      yAxis: [
        { type: "value", name: "MW" },
        { type: "value", name: "raw counts" },
      ],
      dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 8 }],
      series,
    });
  } catch (err) {
    setStatus(`timeseries: ${err.message}`);
  }
}

function refreshAll() {
  refreshHexLayer();
  refreshCalendar();
  refreshTimeseries();
}

/* ------------------------------------------------------------------ */
/* Wiring                                                              */
/* ------------------------------------------------------------------ */
const presetsBox = $("presets");
for (const [name, bbox] of Object.entries(PRESETS)) {
  const btn = document.createElement("button");
  btn.textContent = name;
  btn.addEventListener("click", () => {
    for (const b of presetsBox.children) b.classList.remove("active");
    btn.classList.add("active");
    setBbox(bbox);
    map.fitBounds(
      [[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 40 }
    );
    refreshAll();
  });
  presetsBox.appendChild(btn);
}

$("btn-apply").addEventListener("click", refreshAll);
$("btn-draw").addEventListener("click", () => {
  drawMode = !drawMode;
  $("btn-draw").textContent = drawMode ? "Drawing… (drag)" : "Draw bbox";
  setStatus(drawMode
    ? "Drag a rectangle on the map to define the AOI"
    : "Ready.");
});
$("metric").addEventListener("change", refreshAll);
$("palette").addEventListener("change", refreshAll);
$("show_uncertainty").addEventListener("change", refreshTimeseries);
$("hatch_abstain").addEventListener("change", refreshCalendar);
$("btn-export").addEventListener("click", () => {
  const bbox = bboxFromInputs();
  const url =
    `/api/export.csv?min_lon=${bbox[0]}&min_lat=${bbox[1]}` +
    `&max_lon=${bbox[2]}&max_lat=${bbox[3]}` +
    `&start_year=${$("start_year").value}&end_year=${$("end_year").value}`;
  window.open(url, "_blank");
});

window.addEventListener("resize", () => {
  calendarChart.resize();
  timeseriesChart.resize();
});

map.on("load", () => { setStatus("Map ready — pick a spotlight or draw an AOI."); });
refreshAll();
