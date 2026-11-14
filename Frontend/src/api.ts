import {
  clamp,
  type CalendarCell,
  type FireCellProperties,
  type FireCollection,
  type FireFeature,
  type MapMetric,
  type MonthlyMetric,
  type Region,
  type RegionData,
} from "./data";
import { makeDemoData, makeDemoMap } from "./demoData";

interface ApiCalendarCell {
  year: number;
  month: number;
  value: number | null;
  q05?: number | null;
  q95?: number | null;
  raw_count?: number | null;
  anomaly_ratio?: number | null;
  measured_fraction?: number | null;
  quality_flag?: string | null;
}

interface ApiCalendarResponse {
  cells: ApiCalendarCell[];
}

interface ApiTimeseriesResponse {
  months: string[];
  values: Array<number | null>;
  raw_counts?: Array<number | null>;
  q05?: Array<number | null>;
  q95?: Array<number | null>;
}

interface ApiMapResponse {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: FireFeature["geometry"];
    properties: {
      h3_index?: number | string;
      phi_q50_mw?: number | null;
      anomaly_ratio?: number | null;
      width?: number | null;
    };
  }>;
}

interface RequestResult<T> {
  value: T | null;
  error?: string;
}

function apiUrl(path: string, region: Region): string {
  const [minLon, minLat, maxLon, maxLat] = region.bbox;
  const params = new URLSearchParams({
    min_lon: String(minLon),
    min_lat: String(minLat),
    max_lon: String(maxLon),
    max_lat: String(maxLat),
  });
  if (path === "calendar" || path === "timeseries") {
    params.set("start_year", "2000");
    params.set("end_year", "2026");
  }
  if (path === "calendar") params.set("country_iso", region.iso);
  return `/api/${path}?${params.toString()}`;
}

async function request<T>(url: string, signal?: AbortSignal): Promise<RequestResult<T>> {
  try {
    const response = await fetch(url, { headers: { Accept: "application/json" }, signal });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`API returned ${response.status}${detail ? `: ${detail.slice(0, 140)}` : ""}`);
    }
    return { value: (await response.json()) as T };
  } catch (error) {
    return {
      value: null,
      error: error instanceof Error ? error.message : "Unable to reach the fire data API",
    };
  }
}

function toCalendarCell(cell: ApiCalendarCell): CalendarCell {
  return {
    year: cell.year,
    month: cell.month,
    value: cell.value,
    q05: cell.q05 ?? null,
    q95: cell.q95 ?? null,
    rawCount: cell.raw_count ?? null,
    anomaly: cell.anomaly_ratio ?? null,
    measuredFraction: cell.measured_fraction ?? null,
    qualityFlag: cell.quality_flag ?? undefined,
  };
}

function seriesFromCalendar(cells: CalendarCell[]): MonthlyMetric[] {
  return cells.map((cell) => ({
    month: `${cell.year}-${String(cell.month).padStart(2, "0")}`,
    value: cell.value,
    q05: cell.q05 ?? null,
    q95: cell.q95 ?? null,
    rawCount: cell.rawCount,
    anomaly: cell.anomaly,
    measuredFraction: cell.measuredFraction,
    qualityFlag: cell.qualityFlag,
  }));
}

function normalizeTimeseries(response: ApiTimeseriesResponse): MonthlyMetric[] {
  return response.months.map((month, index) => ({
    month,
    value: response.values[index] ?? null,
    q05: response.q05?.[index] ?? null,
    q95: response.q95?.[index] ?? null,
    rawCount: response.raw_counts?.[index] ?? null,
    anomaly: null,
    measuredFraction: null,
  }));
}

function normalizeMap(response: ApiMapResponse): FireCollection {
  const features: FireFeature[] = response.features.map((feature) => {
    const properties: FireCellProperties = {
      h3_index: feature.properties.h3_index,
      phi_q50_mw: feature.properties.phi_q50_mw ?? undefined,
      anomaly_ratio: feature.properties.anomaly_ratio ?? undefined,
      width: feature.properties.width ?? undefined,
      frp_score: Math.max(0, feature.properties.phi_q50_mw ?? 0),
      anomaly_score: Math.max(0, feature.properties.anomaly_ratio ?? 0),
      uncertainty_score: Math.max(0, feature.properties.width ?? 0),
      value: Math.max(0, feature.properties.phi_q50_mw ?? 0),
    };
    return { type: "Feature", geometry: feature.geometry, properties };
  });

  for (const key of ["frp_score", "anomaly_score", "uncertainty_score"] as const) {
    const values = features.map((feature) => feature.properties[key]);
    const max = Math.max(1, ...values);
    for (const feature of features) feature.properties[key] /= max;
  }
  return { type: "FeatureCollection", features };
}

export function mapFeaturesForMetric(data: FireCollection, metric: MapMetric): FireCollection {
  const property = {
    frp: "frp_score",
    anomaly: "anomaly_score",
    uncertainty: "uncertainty_score",
  }[metric];
  const rawValues = data.features.map((feature) => feature.properties[property]);
  const scale = Math.max(0.01, ...rawValues);
  return {
    type: "FeatureCollection",
    features: data.features.map((feature) => ({
      ...feature,
      properties: {
        ...feature.properties,
        value: clamp(feature.properties[property] / scale, 0, 1),
      },
    })),
  };
}

export async function loadRegionData(region: Region): Promise<RegionData> {
  const [calendarResult, timeseriesResult] = await Promise.all([
    request<ApiCalendarResponse>(apiUrl("calendar", region)),
    request<ApiTimeseriesResponse>(apiUrl("timeseries", region)),
  ]);
  const demo = makeDemoData(region, "2000-01");
  const liveCalendar = calendarResult.value?.cells?.length
    ? calendarResult.value.cells.map(toCalendarCell)
    : null;
  const timeseries = timeseriesResult.value;
  const liveSeries = timeseries?.months.length &&
    timeseries.months.length === timeseries.values.length
    ? normalizeTimeseries(timeseries)
    : null;
  const calendar = liveCalendar ?? (liveSeries
    ? liveSeries.map((item) => ({
        year: Number(item.month.slice(0, 4)),
        month: Number(item.month.slice(5, 7)),
        value: item.value,
        q05: item.q05,
        q95: item.q95,
        rawCount: item.rawCount,
        anomaly: item.anomaly,
        measuredFraction: item.measuredFraction,
      }))
    : demo.calendar);
  const series = liveSeries
    ? liveSeries.map((item) => {
        const calendarCell = liveCalendar?.find((cell) =>
          cell.year === Number(item.month.slice(0, 4)) &&
          cell.month === Number(item.month.slice(5, 7)));
        return {
          ...item,
          anomaly: calendarCell?.anomaly ?? item.anomaly,
          measuredFraction: calendarCell?.measuredFraction ?? item.measuredFraction,
          qualityFlag: calendarCell?.qualityFlag ?? item.qualityFlag,
        };
      })
    : liveCalendar
      ? seriesFromCalendar(liveCalendar)
      : demo.series;
  const notices = [
    calendarResult.error && `Calendar API: ${calendarResult.error}`,
    timeseriesResult.error && `Time-series API: ${timeseriesResult.error}`,
    !liveCalendar && !liveSeries && !calendarResult.error && !timeseriesResult.error &&
      "No fire records were returned for this area; showing demonstration activity.",
  ].filter((notice): notice is string => Boolean(notice));

  return {
    series,
    calendar,
    fireCells: demo.fireCells,
    sources: {
      calendar: liveCalendar || liveSeries ? "live" : "demo",
      timeseries: liveSeries || liveCalendar ? "live" : "demo",
      map: "demo",
    },
    notices,
  };
}

export async function loadRegionMap(region: Region, month: string, signal?: AbortSignal): Promise<{
  fireCells: FireCollection;
  source: "live" | "demo";
  notice?: string;
}> {
  const url = `${apiUrl("map_hex", region)}&year=${month.slice(0, 4)}&month=${Number(month.slice(5, 7))}`;
  const result = await request<ApiMapResponse>(url, signal);
  if (result.value?.features?.length) {
    return { fireCells: normalizeMap(result.value), source: "live" };
  }
  return {
    fireCells: makeDemoMap(region, month),
    source: "demo",
    notice: result.error ??
      "No segmented cells were returned for this month; showing demonstration segmentation.",
  };
}
