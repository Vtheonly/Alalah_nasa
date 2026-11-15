// Types of the backend API. The frontend renders whatever provider the
// server has selected; provenance is data, never a UI branch.

export interface Provenance {
  source: "REAL_ENGINE" | "SYNTHETIC_DEMO" | string;
  is_synthetic: boolean;
  validated: boolean;
  label: string;
  reason: string;
}

export interface Status {
  provenance: Provenance;
  dataset: Record<string, unknown>;
  config: Record<string, unknown>;
  real_engine: { available: boolean; validated?: boolean; reason: string; failed_checks?: string[] };
  validation_gate: { passed: boolean; critical: string[]; failed: string[] };
  constellation_events: Array<{ month: string; label: string }>;
}

export interface ChartSeries {
  name: string;
  role: string;
  data: Array<number | null | [number | null, number | null]>;
  stack?: string;
  axis?: number;
}

export interface ChartSpec {
  kind: "line" | "bar" | "area" | "band" | "scatter";
  title: string;
  x?: Array<string | number>;
  xLabel?: string;
  yLabel?: string;
  series: ChartSeries[];
  markers?: Array<{ x: string | number; label: string }>;
  h3_index?: string;
}

export interface Kpi {
  label: string;
  value: number | null;
  format?: string;
  tone?: "good" | "bad";
}

export interface Region {
  iso: string;
  name: string;
  bbox: [number, number, number, number];
  center: [number, number];
  n_cells: number;
  spotlights: Array<{ key: string; label: string; bbox: number[] }>;
}

export interface Overview {
  provenance: Provenance;
  n_cells: number;
  first_year: number;
  last_year: number;
  n_detections: number | null;
  countries: Array<{ iso: string; name: string; n_cells: number; fre_tj_per_year: number | null }>;
  platforms: Array<{ platform: string; sensor: string; n_detections: number; first: string; last: string }>;
}

export interface ChallengeSummary {
  id: string;
  icon: string;
  title: string;
  problem: string;
  solution: string;
  pipeline: string[];
}

export interface ChallengeDetail extends ChallengeSummary {
  provenance: Provenance;
  kpis: Kpi[];
  charts: ChartSpec[];
}

export interface ValidationCheck {
  id: string;
  title: string;
  question: string;
  value: number | null;
  threshold: string;
  status: "pass" | "fail" | "unavailable";
  critical: boolean;
  format: string;
  detail: Record<string, unknown> | null;
  evidence: ChartSpec | null;
}

export interface Validation {
  provenance: Provenance;
  checks: ValidationCheck[];
  gate: { passed: boolean; critical: string[]; failed: string[] };
  reference: string | null;
  diagnostics_available: boolean;
}

export interface Insights {
  provenance: Provenance;
  trend: { pct_per_decade: number | null; mann_kendall_p: number | null; significant: boolean };
  seasonality: { peak_month: number | null; season_length_months: number };
  charts: { annual: ChartSpec; seasonal: ChartSpec };
  anomalies: Array<{ year: number; month: number; fre_tj: number | null; ratio_to_median: number | null; attribution: Record<string, number> }>;
  by_country: Array<{ iso: string; name: string; fre_tj_per_year: number | null; trend_pct_per_decade: number | null; trend_p: number | null }>;
}

export interface CalendarCell {
  year: number;
  month: number;
  value: number | null;
  q05: number | null;
  q95: number | null;
  raw_count: number | null;
  anomaly_ratio: number | null;
  measured_fraction: number | null;
  quality_flag: string;
  attribution_flag: string;
  fre_tj: number | null;
}

export interface HexFeature {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: {
    h3_index: string;
    phi_q50_mw: number;
    anomaly_ratio: number;
    width: number;
    quality_flag: string;
    attribution_flag: string;
    raw_count: number;
    fre_tj: number;
    measured_fraction: number;
    [k: string]: number | string;
  };
}

export interface HexCollection {
  type: "FeatureCollection";
  features: HexFeature[];
  meta: { year: number; month: number; n: number };
}

export interface CellTrace {
  provenance: Provenance;
  h3_index: string;
  lat: number;
  lon: number;
  country_iso: string;
  region: string | null;
  pyrome_id: number | null;
  year: number;
  month: number;
  series: { x: string[]; q05: (number | null)[]; q50: (number | null)[]; q95: (number | null)[]; raw_count: number[] };
  month_record: null | {
    phi_q05_mw: number; phi_q50_mw: number; phi_q95_mw: number; raw_count: number; raw_frp_mw: number;
    anomaly_ratio: number; measured_fraction: number; mean_cloud_frac: number;
    quality_flag: string; attribution_flag: string; active_platforms: string[];
  };
  detections: Array<Record<string, string | number | boolean | null>>;
  by_platform: Array<{ platform: string; n: number; frp_mw: number; biomass: number }>;
  reconciliation: null | { biomass_detections_in_product1: number; raw_count_in_product2: number; match: boolean; n_detections_total: number };
}

export interface ReplayData {
  provenance: Provenance;
  country_iso: string | null;
  year: number;
  month: number;
  step: "day" | "hour";
  frames: Array<{ key: string; label: string; day: number }>;
  cells: string[];
  polygons: number[][][];
  daily: Array<[number, number, number | null, number | null]>;     // [day0, cell, q50, q95]
  detections: Array<[number, number, number, number, number]>;      // [frame, lon, lat, frp, platform]
  platforms: string[];
  truncated: { cells: boolean; detections: boolean };
  n_detections_month: number;
}

export interface SatelliteMeta {
  platform: string;
  sensor: string;
  pixel_m: number;
  altitude_km: number;
  inclination_deg: number;
  period_min: number;
  ltan_h: number;
  crossing: string;
  swath_km: number;
  launch: string;
  end: string | null;
  phase_deg: number;
  note: string;
  record_first: string | null;
  record_last: string | null;
  n_detections: number | null;
}

export interface GlobeData {
  provenance: Provenance;
  year: number;
  month: number;
  satellites: SatelliteMeta[];
  hotspots: Array<[string, number, number, string, number | null, number | null, number | null, number]>;
}
