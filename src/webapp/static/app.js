/* Burning Activity Calendar frontend — vanilla ES6 + MapLibre GL JS + ECharts. */
"use strict";

const PRESETS = [
  { key: "chilecito", label: "Chilecito ARG 2026", bbox: [-68.0, -29.6, -67.1, -28.7] },
  { key: "kabylie", label: "Kabylie DZA", bbox: [3.8, 36.3, 5.5, 37.0] },
  { key: "hassi", label: "Hassi Messaoud Gas Flares DZA", bbox: [5.5, 31.3, 6.6, 32.2] },
  { key: "alaska", label: "Alaska Boreal USA", bbox: [-156.0, 63.0, -142.0, 67.5] },
  { key: "blacksummer", label: "Black Summer AUS", bbox: [147.0, -37.8, 151.0, -34.5] },
  { key: "kalimantan", label: "Kalimantan Peat IDN", bbox: [112.5, -3.3, 115.2, -1.5] },
];

const PALETTES = {
  viridis: ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
  cividis: ["#00204d", "#31446b", "#666970", "#958f78", "#cab969", "#ffea46"],
};

const state = { bbox: PRESETS[0].bbox.slice(), palette: "viridis" };

/* ---------------- map ---------------- */
const map = new maplibregl.Map({
  container: "map",
  style: { version: 8, sources: {}, layers: [
    { id: "bg", type: "background", paint: { "background-color": "#101526" } },
  ] },
  center: [(state.bbox[0] + state.bbox[2]) / 2, (state.bbox[1] + state.bbox[3]) / 2],
  zoom: 6,
});
map.addControl(new maplibregl.NavigationControl());
map.on("load", () => { addHexLayer(); refresh(); });

function addHexLayer() {
  if (map.getSource("hexes")) return;
  map.addSource("hexes", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({
    id: "hex-fill", type: "fill", source: "hexes",
    paint: { "fill-color": ["get", "color"], "fill-opacity": 0.72,
             "fill-outline-color": "#1c2237" },
  });
}

async function refreshMap() {
  const [w, s, e, n] = state.bbox;
  const params = new URLSearchParams({ min_lon: w, min_lat: s, max_lon: e, max_lat: n, year: 2026, month: 8 });
  const fc = await fetch(`/api/map_hex?${params}`).then((r) => r.json());
  const vals = fc.features.map((f) => f.properties.anomaly_ratio);
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 1);
  const pal = PALETTES[state.palette];
  for (const f of fc.features) {
    const t = Math.min(1, Math.max(0, (f.properties.anomaly_ratio - lo) / (hi - lo + 1e-9)));
    f.properties.color = pal[Math.min(pal.length - 1, Math.floor(t * pal.length))];
  }
  map.getSource("hexes").setData(fc);
  map.fitBounds([[w, s], [e, n]], { padding: 40, duration: 400 });
}

/* ---------------- calendar heatmap (Year x Month) ---------------- */
const calChart = echarts.init(document.getElementById("calendar"));
const tsChart = echarts.init(document.getElementById("timeseries"));

function hatchFlag(c) {
  return c.quality_flag === "ABSTAIN_LOW_EVIDENCE" || (c.measured_fraction !== null && c.measured_fraction < 0.4);
}

async function refresh() {
  const [w, s, e, n] = state.bbox;
  const params = new URLSearchParams({ min_lon: w, min_lat: s, max_lon: e, max_lat: n, start_year: 2000, end_year: 2026 });
  const res = await fetch(`/api/calendar?${params}`).then((r) => r.json());
  const cells = res.cells;
  const years = [...new Set(cells.map((c) => c.year))].sort();
  const vals = cells.map((c) => Math.log10(1 + (c.value ?? 0)));
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0.1);
  const pal = PALETTES[state.palette];
  const data = [];
  const hatched = [];
  for (const c of cells) {
    const t = Math.min(1, Math.max(0, (Math.log10(1 + (c.value ?? 0)) - lo) / (hi - lo + 1e-9)));
    const i = years.indexOf(c.year);
    data.push([c.month - 1, i, Math.log10(1 + (c.value ?? 0)).toFixed(3),
               pal[Math.min(pal.length - 1, Math.floor(t * pal.length))]]);
    if (hatchFlag(c)) hatched.push({ coord: [c.month - 1, i] });
  }
  calChart.setOption({
    title: { text: "Burning Activity Calendar (log10 Phi, MW)", textStyle: { color: "#e8eaf6", fontSize: 13 } },
    tooltip: { position: "top" },
    grid: { left: 46, right: 20, top: 34, bottom: 46 },
    xAxis: { type: "category", data: ["J","F","M","A","M","J","J","A","S","O","N","D"],
             axisLabel: { color: "#9aa2c0" } },
    yAxis: { type: "category", data: years.map(String), axisLabel: { color: "#9aa2c0" } },
    visualMap: { show: false, min: lo, max: hi, inRange: { color: pal } },
    series: [
      { type: "heatmap", data, label: { show: false },
        itemStyle: { borderColor: "#171b2e", borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: "#ffb703", borderWidth: 2 } } },
      { type: "scatter", data: hatched, symbolSize: 7, symbol: "rect",
        itemStyle: { color: "transparent", borderColor: "#ff5d8f", borderWidth: 1.4 } },
    ],
  });
  const ts = await fetch(`/api/timeseries?${params}`).then((r) => r.json());
  const marks = ts.constellation_events.filter((ev) => ts.months.includes(ev.month))
    .map((ev) => ({ xAxis: ev.month, label: { formatter: ev.label.split(" ")[0], fontSize: 9 }, lineStyle: { color: "#ff5d8f", type: "dashed" } }));
  // Conformal 90% CI ribbon: an invisible q05 base + stacked (q95 - q05)
  // band.  Linear 'value' axes (log axes break on zero-flux months).
  const lower = ts.q05.map((v) => (v == null ? null : Math.max(0, v)));
  const band = ts.q95.map((v, i) => (v == null || lower[i] == null ? null : Math.max(0, v - lower[i])));
  tsChart.setOption({
    title: { text: "Harmonized vs raw monthly series (conformal 90% CI)", textStyle: { color: "#e8eaf6", fontSize: 13 } },
    tooltip: {
      trigger: "axis",
      formatter: (items) => {
        const i = items[0].dataIndex;
        const f = (x) => (x == null ? "-" : Number(x).toFixed(2));
        return `${ts.months[i]}<br/>Phi q50: ${f(ts.values[i])} MW` +
          `<br/>90% CI: [${f(ts.q05[i])}, ${f(ts.q95[i])}] MW` +
          `<br/>2003-2020 median: ${f(ts.hist_median[i])} MW` +
          `<br/>raw detections: ${f(ts.raw_counts[i])}`;
      },
    },
    legend: { data: ["Harmonized Phi (MW)", "90% CI", "Raw counts"], textStyle: { color: "#9aa2c0" }, top: 24 },
    grid: { left: 52, right: 52, top: 56, bottom: 34 },
    xAxis: { type: "category", data: ts.months, axisLabel: { color: "#9aa2c0", fontSize: 9 } },
    yAxis: [{ type: "value", name: "MW", min: 0, axisLabel: { color: "#9aa2c0" } },
            { type: "value", name: "count", min: 0, axisLabel: { color: "#9aa2c0" }, splitLine: { show: false } }],
    series: [
      { name: "ci-base", type: "line", data: lower, stack: "ci", symbol: "none",
        lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
      { name: "90% CI", type: "line", data: band, stack: "ci", symbol: "none",
        lineStyle: { opacity: 0 }, areaStyle: { color: "#ffb703", opacity: 0.22 }, silent: true },
      { name: "Harmonized Phi (MW)", type: "line", data: ts.values, showSymbol: false,
        lineStyle: { color: "#ffb703", width: 2 },
        markLine: { data: marks, symbol: "none", silent: true } },
      { name: "Raw counts", type: "line", yAxisIndex: 1, data: ts.raw_counts,
        showSymbol: false, lineStyle: { color: "#5d8fff", width: 1.4, opacity: 0.8 } },
    ],
  }, true);
  refreshMap();
}

/* ---------------- controls ---------------- */
const presetDiv = document.getElementById("preset-buttons");
for (const p of PRESETS) {
  const b = document.createElement("button");
  b.textContent = p.label;
  b.dataset.key = p.key;
  b.onclick = () => { applyBbox(p.bbox); setActive(b); };
  presetDiv.appendChild(b);
}
function setActive(btn) {
  for (const x of presetDiv.children) x.classList.remove("active");
  if (btn) btn.classList.add("active");
}
function applyBbox(bbox) {
  state.bbox = bbox.slice();
  const f = document.getElementById("bbox-form");
  [f.min_lon, f.min_lat, f.max_lon, f.max_lat].forEach((el, i) => (el.value = bbox[i]));
  refresh();
}
document.getElementById("bbox-form").onsubmit = (ev) => {
  ev.preventDefault();
  const f = ev.target;
  applyBbox([+f.min_lon.value, +f.min_lat.value, +f.max_lon.value, +f.max_lat.value]);
  setActive(null);
};
document.getElementById("palette").onchange = (ev) => { state.palette = ev.target.value; refresh(); };
document.getElementById("download-csv").onclick = () => {
  const [w, s, e, n] = state.bbox;
  const url = `/api/export.csv?min_lon=${w}&min_lat=${s}&max_lon=${e}&max_lat=${n}`;
  const a = document.createElement("a");
  a.href = url; a.download = "burning_activity_calendar.csv"; a.click();
};

/* ROI drawing: shift-drag box on the map */
let roiPoints = [];
map.on("click", (ev) => {
  if (!ev.originalEvent.shiftKey) return;
  roiPoints.push([ev.lngLat.lng, ev.lngLat.lat]);
  if (roiPoints.length === 2) {
    const [[x1, y1], [x2, y2]] = roiPoints;
    applyBbox([Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)]);
    roiPoints = [];
  }
});
document.getElementById("draw-roi").onclick = () => {
  alert("Shift-click two corners on the map to set the ROI bounding box.");
};
window.addEventListener("resize", () => { calChart.resize(); tsChart.resize(); });
setActive(presetDiv.firstChild);
