export type MapMetric = "frp" | "anomaly" | "uncertainty";

export interface Region {
  id: string;
  name: string;
  country: string;
  iso: string;
  bbox: [number, number, number, number];
  center: [number, number];
  zoom: number;
  description: string;
  hotspots: Array<[number, number, number]>;
}

export interface MonthlyMetric {
  month: string;
  value: number | null;
  q05: number | null;
  q95: number | null;
  rawCount: number | null;
  anomaly: number | null;
  measuredFraction: number | null;
  qualityFlag?: string | null;
}

export interface CalendarCell {
  year: number;
  month: number;
  value: number | null;
  q05?: number | null;
  q95?: number | null;
  rawCount: number | null;
  anomaly: number | null;
  measuredFraction: number | null;
  qualityFlag?: string | null;
}

export interface FireCellProperties {
  [key: string]: number | string | undefined;
  h3_index?: number | string;
  phi_q50_mw?: number;
  anomaly_ratio?: number;
  width?: number;
  frp_score: number;
  anomaly_score: number;
  uncertainty_score: number;
  value: number;
  raw_count?: number;
  quality_flag?: string;
  measured_fraction?: number;
  attribution_flag?: string;
}

export interface FireFeature {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
  properties: FireCellProperties;
}

export interface FireCollection {
  type: "FeatureCollection";
  features: FireFeature[];
}

export interface RegionData {
  series: MonthlyMetric[];
  calendar: CalendarCell[];
  fireCells: FireCollection;
  sources: {
    calendar: "live" | "demo";
    timeseries: "live" | "demo";
    map: "live" | "demo";
  };
  notices: string[];
}

export const REGIONS: Region[] = [
  {
    id: "chilecito",
    name: "Chilecito",
    country: "Argentina",
    iso: "ARG",
    bbox: [-69.3, -30.2, -66.3, -27.4],
    center: [-67.5, -29.2],
    zoom: 5.2,
    description: "Andean foothills · South America",
    hotspots: [[-67.5, -29.2, 1], [-68.5, -28.9, 0.45], [-66.9, -29.8, 0.3]],
  },
  {
    id: "amazon",
    name: "Amazon Basin",
    country: "Brazil",
    iso: "BRA",
    bbox: [-63, -13, -57, -8],
    center: [-60, -10.5],
    zoom: 4.4,
    description: "Amazon rainforest · South America",
    hotspots: [[-61.2, -9.1, 1], [-59.4, -11.2, 0.75], [-58.2, -9.4, 0.45]],
  },
  {
    id: "kabylie",
    name: "Kabylie",
    country: "Algeria",
    iso: "DZA",
    bbox: [2.4, 35, 6.8, 38],
    center: [4.6, 36.3],
    zoom: 5.2,
    description: "Tell Atlas · North Africa",
    hotspots: [[4.7, 36.6, 1], [3.4, 36, 0.4], [5.4, 36.9, 0.35]],
  },
  {
    id: "alaska",
    name: "Alaska",
    country: "United States",
    iso: "USA",
    bbox: [-151, 62, -145, 66],
    center: [-148, 64],
    zoom: 4.2,
    description: "Boreal forest · North America",
    hotspots: [[-150, 64.5, 1], [-147.2, 63.5, 0.6], [-149, 65.2, 0.45]],
  },
  {
    id: "black-summer",
    name: "Black Summer",
    country: "Australia",
    iso: "AUS",
    bbox: [146, -38, 151, -34],
    center: [148.5, -36],
    zoom: 4.7,
    description: "New South Wales · Oceania",
    hotspots: [[149.5, -34.3, 1], [147.8, -35.6, 0.75], [150, -36.8, 0.35]],
  },
  {
    id: "kalimantan",
    name: "Kalimantan",
    country: "Indonesia",
    iso: "IDN",
    bbox: [111, -3.2, 116.5, 1],
    center: [113.7, -1.1],
    zoom: 4.7,
    description: "Borneo peatlands · Southeast Asia",
    hotspots: [[113.5, -1.6, 1], [114.7, -0.2, 0.6], [112.1, -2, 0.4]],
  },
];

export const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export const MONTH_COUNT = 27 * 12;

export function monthLabel(month: string, short = false): string {
  const [year, value] = month.split("-").map(Number);
  return `${MONTH_NAMES[value - 1]}${short ? "" : ` ${year}`}`;
}

export function selectedMonthString(index: number): string {
  const year = 2000 + Math.floor(index / 12);
  const month = (index % 12) + 1;
  return `${year}-${String(month).padStart(2, "0")}`;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
