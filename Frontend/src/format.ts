export const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function fmt(value: number | null | undefined, format = "num"): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  switch (format) {
    case "pct": return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
    case "pct0": return `${(value * 100).toFixed(0)}%`;
    case "pct1": return `${(value * 100).toFixed(1)}%`;
    case "p": return value < 0.001 ? (value === 0 ? "< 1e-6" : value.toExponential(1)) : value.toFixed(3);
    case "mw": return `${value.toFixed(1)} MW`;
    case "km2": return `${value.toFixed(2)} km²`;
    case "hour": {
      const h = Math.floor(value);
      const m = Math.round((value - h) * 60);
      return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    }
    case "int": return Math.round(value).toLocaleString("en-US");
    case "ratio": return value.toFixed(value < 10 ? 2 : 1);
    default: return compact(value);
  }
}

export function compact(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (Number.isInteger(v) && a < 1e4) return v.toLocaleString("en-US");
  if (a >= 1e9) return `${(v / 1e9).toFixed(digits)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(digits)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(digits)}k`;
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(2);
}

export const FLAG_LABEL: Record<string, string> = {
  NOMINAL: "Nominal evidence",
  CLIMATOLOGY_CLOUD_FALLBACK: "Cloud from climatology",
  ABSTAIN_LOW_EVIDENCE: "Abstained: too little evidence",
  REAL_FIRE_CHANGE: "Real fire change",
  SENSOR_ARTIFACT_CORRECTED: "Sensor artefact corrected",
  WITHIN_NORMAL_VARIABILITY: "Within normal variability",
  AMBIGUOUS_OVERLAPPING_CI: "Ambiguous",
};
