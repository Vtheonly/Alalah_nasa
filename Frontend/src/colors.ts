// One semantic palette for every chart: raw sensor signals are warm,
// harmonized results are cool, satellites have fixed identities.
export const C = {
  raw: "#fb923c",
  raw2: "#facc15",
  harmonized: "#38bdf8",
  band: "rgba(56, 189, 248, 0.18)",
  trend: "#c084fc",
  grid: "rgba(148, 163, 184, 0.12)",
  axis: "#64748b",
  text: "#cbd5e1",
  muted: "#94a3b8",
  good: "#34d399",
  bad: "#f87171",
  warn: "#fbbf24",
};

export const PLATFORM_COLORS: Record<string, string> = {
  Terra: "#f59e0b",
  Aqua: "#22d3ee",
  SNPP: "#a78bfa",
  "NOAA-20": "#34d399",
  "NOAA-21": "#f472b6",
};

export const ROLE_COLORS: Record<string, string> = {
  raw: C.raw,
  raw2: C.raw2,
  harmonized: C.harmonized,
  trend: C.trend,
  "flag:NOMINAL": "#34d399",
  "flag:CLIMATOLOGY_CLOUD_FALLBACK": "#fbbf24",
  "flag:ABSTAIN_LOW_EVIDENCE": "#f87171",
  "flag:flare": "#fb923c",
  "flag:sama": "#a78bfa",
  "flag:other": "#94a3b8",
};

export function roleColor(role: string, i = 0): string {
  if (role.startsWith("platform:")) return PLATFORM_COLORS[role.slice(9)] ?? "#94a3b8";
  return ROLE_COLORS[role] ?? ["#38bdf8", "#fb923c", "#a78bfa", "#34d399", "#f472b6"][i % 5];
}
