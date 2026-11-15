import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type { ChartSpec } from "../types";
import { C, roleColor } from "../colors";
import { compact } from "../format";

const baseAxis = {
  axisLine: { lineStyle: { color: C.axis } },
  axisLabel: { color: C.muted, fontSize: 11 },
  splitLine: { lineStyle: { color: C.grid } },
  nameTextStyle: { color: C.muted, fontSize: 11 },
};

function markerLine(spec: ChartSpec) {
  if (!spec.markers?.length) return undefined;
  return {
    silent: true,
    symbol: "none",
    lineStyle: { color: "rgba(226,232,240,0.35)", type: "dashed" as const, width: 1 },
    label: { color: C.muted, fontSize: 10, formatter: "{b}", position: "insideEndTop" as const },
    data: spec.markers.map((m) => ({ name: m.label, xAxis: String(m.x) })),
  };
}

export function chartOption(spec: ChartSpec): EChartsOption {
  const x = (spec.x ?? []).map(String);
  const hasAxis1 = spec.series.some((s) => s.axis === 1);
  const grid = { left: 56, right: hasAxis1 ? 56 : 18, top: 36, bottom: spec.xLabel ? 52 : 36 };
  const legend = { top: 0, textStyle: { color: C.text, fontSize: 11 }, icon: "roundRect", itemWidth: 12, itemHeight: 8 };
  const tooltip = { trigger: "axis" as const, backgroundColor: "#0f172a", borderColor: "#1e293b", textStyle: { color: "#e2e8f0", fontSize: 12 } };

  if (spec.kind === "scatter") {
    const pts = spec.series[0]?.data as Array<[number, number]>;
    const vals = pts.flat().filter((v) => Number.isFinite(v));
    const hi = vals.length ? Math.max(...vals) : 1;
    return {
      grid: { left: 56, right: 18, top: 24, bottom: 48 },
      tooltip: { trigger: "item", backgroundColor: "#0f172a", textStyle: { color: "#e2e8f0" } },
      xAxis: { type: "value", name: spec.xLabel, nameLocation: "middle", nameGap: 30, ...baseAxis },
      yAxis: { type: "value", name: spec.yLabel, ...baseAxis },
      series: [
        { type: "scatter", data: pts, symbolSize: 4, itemStyle: { color: C.harmonized, opacity: 0.45 } },
        { type: "line", data: [[0, 0], [hi, hi]], symbol: "none", lineStyle: { color: C.muted, type: "dashed" }, tooltip: { show: false } },
      ],
    };
  }

  const yAxis: EChartsOption["yAxis"] = [
    { type: "value", name: spec.yLabel, ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: (v: number) => compact(v, 0) } },
  ];
  if (hasAxis1) {
    (yAxis as object[]).push({ type: "value", name: "detections", ...baseAxis, splitLine: { show: false }, axisLabel: { ...baseAxis.axisLabel, formatter: (v: number) => compact(v, 0) } });
  }

  const series: EChartsOption["series"] = [];
  const low = spec.series.find((s) => s.role === "band_low");
  const high = spec.series.find((s) => s.role === "band_high");
  if (spec.kind === "band" && low && high) {
    series.push({ name: low.name, type: "line", data: low.data as number[], stack: "band", symbol: "none", lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true });
    series.push({
      name: high.name, type: "line", stack: "band", symbol: "none", lineStyle: { opacity: 0 },
      areaStyle: { color: C.band },
      data: (high.data as number[]).map((v, i) => (v === null || low.data[i] === null ? null : (v as number) - (low.data[i] as number))),
    });
  }
  spec.series.forEach((s, i) => {
    if (s.role === "band_low" || s.role === "band_high") return;
    const color = roleColor(s.role, i);
    const isBar = spec.kind === "bar" || (spec.kind === "band" && s.axis === 1);
    series.push({
      name: s.name,
      type: isBar ? "bar" : "line",
      data: s.data as number[],
      yAxisIndex: s.axis === 1 ? 1 : 0,
      stack: s.stack,
      smooth: spec.kind !== "bar" && x.length > 60 ? 0.2 : false,
      symbol: x.length > 40 ? "none" : "circle",
      symbolSize: 5,
      barMaxWidth: 28,
      itemStyle: { color, opacity: s.axis === 1 ? 0.35 : 1, borderRadius: isBar && !s.stack ? [3, 3, 0, 0] : 0 },
      // stacked areas are fills only: an outline would trace the stack top
      // even where the series is zero (e.g. a satellite before launch)
      lineStyle: { color, width: spec.kind === "area" ? 0 : s.role === "harmonized" ? 2.6 : 1.8, type: s.role === "trend" ? "dashed" : "solid" },
      areaStyle: spec.kind === "area" ? { color, opacity: 0.85 } : undefined,
      markLine: i === spec.series.findIndex((t) => t.role !== "band_low" && t.role !== "band_high") ? markerLine(spec) : undefined,
    });
  });

  const legendNames = spec.series.filter((s) => s.role !== "band_low").map((s) => s.name);
  return {
    animationDuration: 600,
    grid,
    legend: { ...legend, data: legendNames },
    tooltip,
    xAxis: { type: "category", data: x, name: spec.xLabel, nameLocation: "middle", nameGap: 30, ...baseAxis, splitLine: { show: false } },
    yAxis,
    dataZoom: x.length > 120 ? [{ type: "inside" }, { type: "slider", height: 16, bottom: 4, borderColor: "transparent", textStyle: { color: C.muted } }] : undefined,
    series,
  };
}

export default function Chart({ spec, height = 280 }: { spec: ChartSpec; height?: number }) {
  return (
    <figure className="chart">
      <figcaption>{spec.title}</figcaption>
      <ReactECharts option={chartOption(spec)} style={{ height, width: "100%" }} notMerge lazyUpdate />
    </figure>
  );
}
