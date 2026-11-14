import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  CalendarDays,
  ChevronDown,
  Database,
  Flame,
  Globe2,
  Layers3,
  LocateFixed,
  MapPin,
  Pause,
  Play,
  Search,
  Sparkles,
} from "lucide-react";
import {
  clamp,
  MONTH_COUNT,
  MONTH_NAMES,
  monthLabel,
  REGIONS,
  selectedMonthString,
  type FireCellProperties,
  type MapMetric,
  type MonthlyMetric,
  type Region,
  type RegionData,
} from "./data";
import { loadRegionData, loadRegionMap, mapFeaturesForMetric } from "./api";
import type { FireCollection } from "./data";
import MapView from "./MapView";

const INITIAL_MONTH = Math.min(
  MONTH_COUNT - 1,
  Math.max(0, (new Date().getFullYear() - 2000) * 12 + new Date().getMonth()),
);

function compact(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: digits,
  }).format(value);
}

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function sourceLabel(
  data: RegionData | null,
  mapSource: "live" | "demo",
): "LIVE" | "MIXED" | "DEMO" {
  if (!data) return "DEMO";
  const values = [data.sources.calendar, data.sources.timeseries, mapSource];
  if (values.every((source) => source === "live")) return "LIVE";
  if (values.every((source) => source === "demo")) return "DEMO";
  return "MIXED";
}

function currentRecord(series: MonthlyMetric[], month: string): MonthlyMetric | null {
  return series.find((item) => item.month === month) ??
    [...series].reverse().find((item) => item.value !== null) ??
    null;
}

function buildCalendarOption(data: RegionData | null): EChartsOption {
  const cells = data?.calendar ?? [];
  const years = Array.from({ length: 27 }, (_, index) => String(2000 + index));
  const values = cells
    .map((cell) => Math.log10(1 + Math.max(0, cell.value ?? 0)))
    .filter(Number.isFinite);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  return {
    animationDuration: 450,
    grid: { left: 52, right: 16, top: 10, bottom: 30 },
    tooltip: {
      trigger: "item",
      backgroundColor: "#172023",
      borderColor: "#39484a",
      textStyle: { color: "#f6f5f0", fontSize: 11 },
      formatter: (raw) => {
        const point = raw as { value?: [number, number, number] };
        const [month, yearIndex] = point.value ?? [0, 0, 0];
        const cell = cells.find((item) => item.year === 2000 + yearIndex && item.month === month + 1);
        return `<b>${MONTH_NAMES[month]} ${2000 + yearIndex}</b><br/>Harmonized FRP: ${compact(cell?.value)} MW<br/>Raw detections: ${compact(cell?.rawCount, 0)}`;
      },
    },
    xAxis: {
      type: "category",
      data: MONTH_NAMES,
      axisLine: { lineStyle: { color: "#354244" } },
      axisTick: { show: false },
      axisLabel: { color: "#8e9b9b", fontSize: 10, margin: 11 },
      splitArea: { show: true, areaStyle: { color: ["#151d1f", "#151d1f"] } },
    },
    yAxis: {
      type: "category",
      data: years,
      inverse: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: "#899596", fontSize: 9, margin: 9 },
      splitArea: { show: true, areaStyle: { color: ["#151d1f", "#151d1f"] } },
    },
    visualMap: {
      show: false,
      min,
      max: max === min ? min + 1 : max,
      inRange: { color: ["#263437", "#b65b3b", "#f0bb73"] },
    },
    series: [{
      type: "heatmap",
      data: cells
        .filter((cell) => cell.value !== null)
        .map((cell) => [
          cell.month - 1,
          cell.year - 2000,
          Math.log10(1 + Math.max(0, cell.value ?? 0)),
        ]),
      itemStyle: {
        borderColor: "#111819",
        borderWidth: 2,
        borderRadius: 2,
      },
      emphasis: {
        itemStyle: { borderColor: "#f0bb73", borderWidth: 1.5 },
      },
    }],
    graphic: [{
      type: "text",
      right: 16,
      bottom: 5,
      style: { text: "Lower activity", fill: "#738181", font: "10px sans-serif" },
    }],
    aria: { enabled: true },
  };
}

function calendarEvents(onSelect: (index: number) => void) {
  return {
    click: (params: { value?: [number, number, number] }) => {
      if (params.value) onSelect((params.value[1] * 12) + params.value[0]);
    },
  };
}

function buildSeriesOption(series: MonthlyMetric[], selectedMonth: string): EChartsOption {
  const months = series.map((item) => item.month);
  const hasInterval = series.some((item) => item.q05 !== null && item.q95 !== null);
  const pointLabel = (index: number) => monthLabel(months[index] ?? "", false);
  return {
    animationDuration: 500,
    grid: { left: 56, right: 48, top: 24, bottom: 48 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#172023",
      borderColor: "#39484a",
      textStyle: { color: "#f6f5f0", fontSize: 11 },
      formatter: (raw) => {
        const items = raw as Array<{ dataIndex: number }>;
        const record = series[items[0]?.dataIndex];
        if (!record) return "";
        const interval = record.q05 !== null && record.q95 !== null
          ? `<br/>90% interval: ${compact(record.q05)}–${compact(record.q95)} MW`
          : "";
        return `<b>${pointLabel(items[0].dataIndex)}</b><br/>Harmonized FRP: ${compact(record.value)} MW${interval}<br/>Raw detections: ${compact(record.rawCount, 0)}`;
      },
    },
    legend: {
      top: 0,
      right: 4,
      itemWidth: 10,
      itemHeight: 8,
      textStyle: { color: "#91a0a0", fontSize: 10 },
      data: ["Harmonized FRP", ...(hasInterval ? ["90% interval"] : []), "Raw detections"],
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: months,
      axisLine: { lineStyle: { color: "#354244" } },
      axisTick: { show: false },
      axisLabel: {
        color: "#879493",
        fontSize: 9,
        interval: Math.max(1, Math.floor(months.length / 7)),
        formatter: (value: string | number) => String(value).slice(0, 4),
      },
    },
    yAxis: [
      {
        type: "value",
        name: "MW",
        nameTextStyle: { color: "#899595", fontSize: 9, padding: [0, 0, 0, -30] },
        axisLabel: { color: "#879493", fontSize: 9 },
        splitLine: { lineStyle: { color: "rgba(130, 151, 149, 0.1)" } },
      },
      {
        type: "value",
        name: "Detections",
        nameTextStyle: { color: "#899595", fontSize: 9 },
        axisLabel: { color: "#879493", fontSize: 9 },
        splitLine: { show: false },
      },
    ],
    series: [
      ...(hasInterval ? [
        {
          name: "_interval base",
          type: "line" as const,
          stack: "confidence",
          data: series.map((item) => item.q05),
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          tooltip: { show: false },
          silent: true,
        },
        {
          name: "90% interval",
          type: "line" as const,
          stack: "confidence",
          data: series.map((item) =>
            item.q05 !== null && item.q95 !== null ? item.q95 - item.q05 : null),
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(229, 171, 106, 0.16)" },
          tooltip: { show: false },
          silent: true,
        },
      ] : []),
      {
        name: "Harmonized FRP",
        type: "line",
        data: series.map((item) => item.value),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { color: "#ecad6b", width: 2 },
        itemStyle: { color: "#ecad6b" },
        areaStyle: { color: "rgba(236, 173, 107, 0.07)" },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#f1e7d4", opacity: 0.4, type: "dashed" },
          data: [{ xAxis: selectedMonth }],
        },
      },
      {
        name: "Raw detections",
        type: "line",
        yAxisIndex: 1,
        data: series.map((item) => item.rawCount),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { color: "#7da69d", width: 1, opacity: 0.72 },
      },
    ],
    aria: { enabled: true },
  };
}

export default function App() {
  const [regionId, setRegionId] = useState(REGIONS[0].id);
  const [monthIndexValue, setMonthIndexValue] = useState(INITIAL_MONTH);
  const [mapMetric, setMapMetric] = useState<MapMetric>("frp");
  const [data, setData] = useState<RegionData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMapLoading, setIsMapLoading] = useState(true);
  const [mapCells, setMapCells] = useState<FireCollection | null>(null);
  const [mapSource, setMapSource] = useState<"live" | "demo">("demo");
  const [mapNotice, setMapNotice] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<FireCellProperties | null>(null);
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [activeSection, setActiveSection] = useState("overview");
  const [showDataInfo, setShowDataInfo] = useState(false);
  const region = REGIONS.find((item) => item.id === regionId) ?? REGIONS[0];
  const selectedMonth = selectedMonthString(monthIndexValue);
  const selectedRecord = useMemo(
    () => currentRecord(data?.series ?? [], selectedMonth),
    [data, selectedMonth],
  );
  const renderedCells = useMemo(
    () => {
      const currentCells = mapCells ?? data?.fireCells;
      return currentCells
        ? mapFeaturesForMetric(currentCells, mapMetric)
        : { type: "FeatureCollection" as const, features: [] };
    },
    [data, mapCells, mapMetric],
  );
  const filteredRegions = REGIONS.filter((item) =>
    `${item.name} ${item.country} ${item.iso}`.toLowerCase().includes(search.trim().toLowerCase()));
  const source = sourceLabel(data, mapSource);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setIsMapLoading(true);
    setData(null);
    setMapCells(null);
    setMapSource("demo");
    setMapNotice(null);
    setSelectedCell(null);
    const timer = window.setTimeout(() => {
      void loadRegionData(region).then((result) => {
        if (!active) return;
        setData(result);
        setIsLoading(false);
      });
    }, 280);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [region.id]);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setIsMapLoading(true);
      void loadRegionMap(region, selectedMonth, controller.signal).then((result) => {
        if (!active) return;
        setMapCells(result.fireCells);
        setMapSource(result.source);
        setMapNotice(result.notice ?? null);
        setIsMapLoading(false);
      });
    }, 240);
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [region.id, selectedMonth]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setMonthIndexValue((previous) => (previous + 1) % MONTH_COUNT);
    }, 850);
    return () => window.clearInterval(timer);
  }, [playing]);

  const chartSeries = data?.series ?? [];
  const calendarOption = useMemo(
    () => buildCalendarOption(data),
    [data],
  );
  const seriesOption = useMemo(
    () => buildSeriesOption(chartSeries, selectedMonth),
    [chartSeries, selectedMonth],
  );
  const mapMetricLabel = {
    frp: "Fire radiative power",
    anomaly: "Fire anomaly",
    uncertainty: "Uncertainty width",
  }[mapMetric];
  const selectedCellValue = selectedCell?.phi_q50_mw;

  function chooseRegion(item: Region) {
    setRegionId(item.id);
    setSearch(item.name);
    setSearchOpen(false);
    setActiveSection("overview");
  }

  function jumpTo(section: string) {
    setActiveSection(section);
    document.getElementById(section)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" onClick={() => jumpTo("overview")}>
          <span className="brand-mark"><Flame size={20} fill="currentColor" /></span>
          <span className="brand-copy">
            <strong>EMBER ATLAS</strong>
            <small>FIRE INTELLIGENCE</small>
          </span>
        </a>

        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Workspace">
          <button className={activeSection === "overview" ? "nav-link active" : "nav-link"} onClick={() => jumpTo("overview")}>
            <Globe2 size={17} /> Overview
          </button>
          <button className={activeSection === "activity" ? "nav-link active" : "nav-link"} onClick={() => jumpTo("activity")}>
            <BarChart3 size={17} /> Fire activity
          </button>
          <button className={activeSection === "calendar" ? "nav-link active" : "nav-link"} onClick={() => jumpTo("calendar")}>
            <CalendarDays size={17} /> Activity calendar
          </button>
        </nav>

        <div className="region-heading">
          <div className="sidebar-caption">STUDY AREAS</div>
        </div>
        <div className="region-list">
          {REGIONS.map((item) => (
            <button
              key={item.id}
              className={regionId === item.id ? "region-link selected" : "region-link"}
              onClick={() => chooseRegion(item)}
              aria-current={regionId === item.id ? "location" : undefined}
            >
              <span className={`region-dot ${item.id}`} />
              <span className="region-name">{item.name}<small>{item.country}</small></span>
              <span className="region-count">{item.iso}</span>
            </button>
          ))}
        </div>

        <div className="sidebar-spacer" />
        <div className="model-card">
          <div className="model-icon"><Sparkles size={15} /></div>
          <div><strong>Harmonized record</strong><small>MODIS · VIIRS · 2000–2026</small></div>
          <span className="model-status" />
        </div>
        <div className="sidebar-foot">
          <span className="connection-dot" /> Satellite observation workspace
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>Global fire activity</strong></div>
          <div className="topbar-actions">
            <div className={`data-status ${source.toLowerCase()}`}>
              <span className="status-dot" />
              {isLoading ? "Checking sources" : source === "LIVE" ? "API data" : source === "MIXED" ? "API + demo" : "Demo data"}
            </div>
            <div className="topbar-divider" />
            <button
              className="icon-button"
              aria-label="Data and method information"
              title="Data and method information"
              aria-expanded={showDataInfo}
              onClick={() => setShowDataInfo((visible) => !visible)}
            >
              <Database size={16} />
            </button>
            <div className="user-avatar" aria-label="Fire Harmonizer project">FH</div>
          </div>
          {showDataInfo && (
            <div className="data-popover" role="dialog" aria-label="Data source details">
              <strong>Data &amp; method</strong>
              <p>MODIS and VIIRS observations are harmonized into one monthly fire record with uncertainty and quality flags.</p>
              <div><span>Calendar / series</span><b>{data?.sources.timeseries === "live" ? "API" : "Demo fallback"}</b></div>
              <div><span>Map segmentation</span><b>{mapSource === "live" ? "H3 · API" : "Illustrative grid"}</b></div>
              <small>Demo values are synthetic and are not satellite observations.</small>
            </div>
          )}
        </header>

        <main className="page-content" id="overview">
          <section className="page-intro">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> EARTH OBSERVATION · MONTHLY MONITORING</div>
              <h1>Fire, in <em>perspective.</em></h1>
              <p>Explore a consistent view of fire activity, harmonized across satellite generations.</p>
            </div>
            <div className="intro-controls">
              <div className="search-wrap">
                <Search size={16} />
                <input
                  type="search"
                  value={search}
                  placeholder="Find a study area"
                  aria-label="Find a study area"
                  onFocus={() => setSearchOpen(true)}
                  onChange={(event) => { setSearch(event.target.value); setSearchOpen(true); }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && filteredRegions[0]) chooseRegion(filteredRegions[0]);
                    if (event.key === "Escape") setSearchOpen(false);
                  }}
                />
                <ChevronDown size={14} className="search-chevron" />
                {searchOpen && (
                  <div className="search-results">
                    {filteredRegions.length ? filteredRegions.map((item) => (
                      <button key={item.id} onMouseDown={(event) => event.preventDefault()} onClick={() => chooseRegion(item)}>
                        <MapPin size={14} /><span>{item.name}<small>{item.country} · {item.iso}</small></span>
                      </button>
                    )) : <div className="empty-search">No study areas match that search.</div>}
                  </div>
                )}
              </div>
              <div className="date-button" title="Current selected month">
                <CalendarDays size={15} /> {monthLabel(selectedMonth)} <ChevronDown size={13} />
              </div>
            </div>
          </section>

          <section className="metric-row" aria-label="Selected month summary">
            <MetricCard
              label="Harmonized fire power"
              value={compact(selectedRecord?.value)}
              unit="MW"
              icon={<Flame size={17} />}
              detail={`Regional mean · ${monthLabel(selectedRecord?.month ?? selectedMonth)}`}
              trend={selectedRecord?.anomaly}
            />
            <MetricCard
              label="Raw fire detections"
              value={compact(selectedRecord?.rawCount, 0)}
              unit="observations"
              icon={<Activity size={17} />}
              detail="Across active satellite passes"
            />
            <MetricCard
              label="Cells segmented"
              value={compact(mapCells?.features.length ?? data?.fireCells.features.length, 0)}
              unit="map cells"
              icon={<Layers3 size={17} />}
              detail={mapSource === "live" ? "H3 cells from DuckDB" : "Illustrative hexagonal grid"}
            />
            <MetricCard
              label="Measured coverage"
              value={pct(selectedRecord?.measuredFraction)}
              unit=""
              icon={<LocateFixed size={17} />}
              detail={selectedRecord?.qualityFlag?.replaceAll("_", " ").toLowerCase() ?? "90% uncertainty intervals"}
            />
          </section>

          {data?.notices.length || mapNotice ? (
            <div className="notice-bar" role="status">
              <Database size={15} />
              <span>{data?.notices[0] ?? mapNotice}</span>
              {(data?.notices.length ?? 0) + (mapNotice ? 1 : 0) > 1 &&
                <span className="notice-extra">+{(data?.notices.length ?? 0) + (mapNotice ? 1 : 0) - 1} source notes</span>}
            </div>
          ) : null}

          <section className="map-section panel" aria-labelledby="map-title">
            <div className="panel-heading map-heading">
              <div>
                <div className="panel-kicker"><span className="live-pulse" /> HARMONIZED FIRE MAP</div>
                <h2 id="map-title">{region.name}<span className="heading-country"> · {region.country}</span></h2>
                <p>{region.description} <span className="meta-separator">·</span> {monthLabel(selectedMonth)}</p>
              </div>
              <div className="map-header-actions">
                <div className="map-layer-toggle" aria-label="Fire layer">
                  {([
                    ["frp", "FRP"],
                    ["anomaly", "Anomaly"],
                    ["uncertainty", "Uncertainty"],
                  ] as const).map(([metric, label]) => (
                    <button
                      key={metric}
                      className={mapMetric === metric ? "on" : ""}
                      onClick={() => setMapMetric(metric)}
                      aria-pressed={mapMetric === metric}
                    >{label}</button>
                  ))}
                </div>
              </div>
            </div>

            <div className="map-frame">
              <MapView
                region={region}
                cells={renderedCells}
                selectedCell={selectedCell}
                onCellSelect={setSelectedCell}
              />
              <div className="map-top-chip"><Globe2 size={13} /> Interactive 3D globe</div>
              <div className="map-region-chip"><span className="region-dot active" /> {region.iso} · {region.name}</div>
              <div className="map-legend">
                <div className="legend-heading">{mapMetricLabel}<span>LOW <i /> HIGH</span></div>
                <div className="legend-gradient" />
                <div className="legend-units">{mapMetric === "frp" ? "Fire radiative power · MW" : mapMetric === "anomaly" ? "Relative to seasonal baseline" : "90% interval width"}</div>
              </div>
              {(isLoading || isMapLoading) && <div className="map-loading"><span className="spinner" /> Updating regional fire layer…</div>}
            </div>
            <div className="map-footer">
              <span><span className="foot-dot" />Country borders and coastlines from OpenStreetMap</span>
              <span>{mapSource === "live" ? "H3 segmentation · API data" : "Synthetic segmentation · illustrative only"}</span>
              <span>Drag to rotate <b>·</b> Scroll to zoom <b>·</b> Select a cell for details</span>
            </div>
          </section>

          <section className="timeline-panel" aria-label="Monthly timeline">
            <button className="play-button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause timeline" : "Play timeline"}>
              {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
            </button>
            <div className="timeline-copy"><strong>{monthLabel(selectedMonth)}</strong><small>Move through the monthly record</small></div>
            <input
              type="range"
              min={0}
              max={MONTH_COUNT - 1}
              value={monthIndexValue}
              onChange={(event) => setMonthIndexValue(Number(event.target.value))}
              aria-label="Selected month from January 2000 to December 2026"
              style={{ "--range-progress": `${(monthIndexValue / (MONTH_COUNT - 1)) * 100}%` } as CSSProperties}
            />
            <span className="timeline-year">2000</span><span className="timeline-year">2026</span>
            <button className="now-button" onClick={() => setMonthIndexValue(INITIAL_MONTH)}>LATEST <ArrowDownRight size={13} /></button>
          </section>

          <section className="chart-grid" id="activity">
            <article className="panel chart-panel series-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-kicker">LONG-TERM SIGNAL</div>
                  <h2>Fire activity over time</h2>
                  <p>Harmonized fire power and raw satellite detections</p>
                </div>
                <div className="chart-chip"><span className="chart-dot" /> 90% interval where available</div>
              </div>
              {isLoading
                ? <ChartSkeleton />
                : <ReactECharts option={seriesOption} notMerge lazyUpdate className="echart series-chart" opts={{ renderer: "canvas" }} />}
              <div className="chart-footnote"><span><span className="series-key harmonized" /> Harmonized FRP</span><span><span className="series-key detections" /> Raw detections</span><span>Satellite seam adjusted in harmonized series</span></div>
            </article>

            <article className="panel chart-panel calendar-panel" id="calendar">
              <div className="panel-heading">
                <div>
                  <div className="panel-kicker">SEASONAL PATTERN</div>
                  <h2>Activity calendar</h2>
                  <p>Monthly fire power · 2000–2026</p>
                </div>
                <span className="calendar-unit">log MW</span>
              </div>
              {isLoading
                ? <ChartSkeleton compact />
                : <ReactECharts
                    option={calendarOption}
                    onEvents={calendarEvents(setMonthIndexValue)}
                    notMerge
                    lazyUpdate
                    className="echart calendar-chart"
                    opts={{ renderer: "canvas" }}
                  />}
              <div className="calendar-footnote"><span>Jan</span><span>Each row is a year · select a month to inspect</span><span>Dec</span></div>
            </article>
          </section>

          {selectedCell && (
            <section className="selected-cell panel" aria-live="polite">
              <div className="selected-cell-icon"><Layers3 size={17} /></div>
              <div><span className="panel-kicker">SELECTED FIRE SEGMENT</span><strong>{selectedCell.h3_index ? "H3 cell selected" : "Demonstration cell"}</strong></div>
              <div className="cell-detail"><small>Harmonized FRP</small><strong>{compact(selectedCellValue)} MW</strong></div>
              <div className="cell-detail"><small>Anomaly ratio</small><strong>{selectedCell.anomaly_ratio ? `${selectedCell.anomaly_ratio.toFixed(2)}×` : "—"}</strong></div>
              <div className="cell-detail"><small>Quality</small><strong>{selectedCell.quality_flag?.replaceAll("_", " ") ?? "Nominal"}</strong></div>
              <button className="icon-button close-cell" onClick={() => setSelectedCell(null)} aria-label="Close selected cell">×</button>
            </section>
          )}

          <footer className="page-footer">
            <span>EMBER ATLAS <b>·</b> A harmonized view of satellite fire observations</span>
            <span>MODIS + VIIRS <b>·</b> Values shown with measured coverage and uncertainty</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  unit: string;
  icon: ReactNode;
  detail: string;
  trend?: number | null;
}

function MetricCard({ label, value, unit, icon, detail, trend }: MetricCardProps) {
  const up = trend !== null && trend !== undefined && trend > 1.15;
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span className="metric-icon">{icon}</span>
        {trend !== null && trend !== undefined && (
          <span className={`trend-pill ${up ? "elevated" : "steady"}`}>
            {up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
            {trend.toFixed(2)}×
          </span>
        )}
      </div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}<span>{unit}</span></div>
      <div className="metric-detail">{detail}</div>
    </article>
  );
}

function ChartSkeleton({ compact: small = false }: { compact?: boolean }) {
  return <div className={small ? "chart-skeleton compact" : "chart-skeleton"}><span /><span /><span /><span /><span /><div className="skeleton-line" /></div>;
}
