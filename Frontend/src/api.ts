import type {
  CalendarCell, CellTrace, ReplayData, ChallengeDetail, ChallengeSummary, HexCollection,
  Insights, Overview, Region, Status, Validation,
} from "./types";

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { headers: { Accept: "application/json" }, signal });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status} ${text.slice(0, 160)}`);
  }
  return (await r.json()) as T;
}

const q = (params: Record<string, string | number | null | undefined>) => {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") s.set(k, String(v));
  return s.toString();
};

export const api = {
  status: (s?: AbortSignal) => get<Status>("/api/status", s),
  overview: (s?: AbortSignal) => get<Overview>("/api/overview", s),
  regions: (s?: AbortSignal) => get<{ regions: Region[] }>("/api/regions", s).then((r) => r.regions),
  challenges: (s?: AbortSignal) => get<{ challenges: ChallengeSummary[] }>("/api/challenges", s).then((r) => r.challenges),
  challenge: (id: string, iso: string | null, s?: AbortSignal) =>
    get<ChallengeDetail>(`/api/challenges/${id}?${q({ country_iso: iso })}`, s),
  validation: (s?: AbortSignal) => get<Validation>("/api/validation", s),
  insights: (iso: string | null, s?: AbortSignal) => get<Insights>(`/api/insights?${q({ country_iso: iso })}`, s),
  calendar: (r: Region, metric: string, s?: AbortSignal) =>
    get<{ cells: CalendarCell[] }>(`/api/calendar?${q({
      min_lon: r.bbox[0], min_lat: r.bbox[1], max_lon: r.bbox[2], max_lat: r.bbox[3],
      start_year: 2000, end_year: 2026, metric, country_iso: r.iso,
    })}`, s).then((x) => x.cells),
  mapHex: (r: Region, year: number, month: number, colorBy: string, s?: AbortSignal) =>
    get<HexCollection>(`/api/map_hex?${q({
      min_lon: r.bbox[0], min_lat: r.bbox[1], max_lon: r.bbox[2], max_lat: r.bbox[3],
      year, month, color_by: colorBy, country_iso: r.iso,
    })}`, s),
  replay: (iso: string, year: number, month: number, step: "day" | "hour", s?: AbortSignal) =>
    get<ReplayData>(`/api/replay?${q({ country_iso: iso, year, month, step })}`, s),
  cell: (h3: string, year: number | null, month: number | null, s?: AbortSignal) =>
    get<CellTrace>(`/api/cell/${h3}?${q({ year, month })}`, s),
};
