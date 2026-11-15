// Circular sun-synchronous orbit model driven by each satellite's nominal
// parameters (served by /api/globe): altitude, inclination, period and the
// local time of the ascending node.  The orbital plane is fixed relative to
// the Sun, so the sub-satellite point is computed in a Sun-fixed frame and
// shifted by the sub-solar longitude.  Along-track phase is illustrative.
import type { SatelliteMeta } from "./types";

const D2R = Math.PI / 180;
export const EARTH_R_KM = 6371;

export function subpoint(s: SatelliteMeta, tMs: number): { lat: number; lng: number } {
  const T = s.period_min * 60_000;
  const u = 2 * Math.PI * (((tMs / T) % 1) + 1) % (2 * Math.PI) + s.phase_deg * D2R;
  const inc = s.inclination_deg * D2R;
  const node = (s.ltan_h - 12) * 15 * D2R;
  const lat = Math.asin(Math.sin(inc) * Math.sin(u));
  const lonSun = node + Math.atan2(Math.cos(inc) * Math.sin(u), Math.cos(u));
  const utcH = ((tMs / 3_600_000) % 24 + 24) % 24;
  const subsolar = -15 * (utcH - 12);
  let lng = lonSun / D2R + subsolar;
  lng = ((lng + 540) % 360) - 180;
  return { lat: lat / D2R, lng };
}

export function orbitPath(s: SatelliteMeta, tMs: number, n = 140): Array<[number, number, number]> {
  const T = s.period_min * 60_000;
  const alt = s.altitude_km / EARTH_R_KM;
  const out: Array<[number, number, number]> = [];
  for (let k = 0; k <= n; k++) {
    const p = subpoint(s, tMs - T / 2 + (k / n) * T);
    out.push([p.lat, p.lng, alt]);
  }
  return out;
}

export function greatCircleKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const a = Math.sin(((lat2 - lat1) * D2R) / 2) ** 2
    + Math.cos(lat1 * D2R) * Math.cos(lat2 * D2R) * Math.sin(((lon2 - lon1) * D2R) / 2) ** 2;
  return 2 * EARTH_R_KM * Math.asin(Math.min(1, Math.sqrt(a)));
}

export function activeAt(s: SatelliteMeta, isoDate: string): boolean {
  const first = s.record_first ?? s.launch;
  return first <= isoDate && (!s.end || isoDate <= s.end);
}

/** Instantaneous orbit ring (closed circle fixed relative to the Sun),
 *  starting at the satellite and going backwards one full revolution. */
export function orbitRing(s: SatelliteMeta, tMs: number, n = 160): Array<[number, number, number]> {
  const T = s.period_min * 60_000;
  const u0 = 2 * Math.PI * (((tMs / T) % 1) + 1) % (2 * Math.PI) + s.phase_deg * D2R;
  const inc = s.inclination_deg * D2R;
  const node = (s.ltan_h - 12) * 15 * D2R;
  const utcH = ((tMs / 3_600_000) % 24 + 24) % 24;
  const subsolar = -15 * (utcH - 12);
  const alt = s.altitude_km / EARTH_R_KM;
  const out: Array<[number, number, number]> = [];
  for (let k = 0; k <= n; k++) {
    const u = u0 - (k / n) * 2 * Math.PI;
    const lat = Math.asin(Math.sin(inc) * Math.sin(u)) / D2R;
    let lng = (node + Math.atan2(Math.cos(inc) * Math.sin(u), Math.cos(u))) / D2R + subsolar;
    lng = ((lng + 540) % 360) - 180;
    out.push([lat, lng, alt]);
  }
  return out;
}

export function bearingDeg(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const p1 = lat1 * D2R, p2 = lat2 * D2R, dl = (lon2 - lon1) * D2R;
  const y = Math.sin(dl) * Math.cos(p2);
  const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
  return Math.atan2(y, x) / D2R;
}

export function destination(lat: number, lon: number, bearing: number, km: number): [number, number] {
  const d = km / EARTH_R_KM, b = bearing * D2R, p1 = lat * D2R, l1 = lon * D2R;
  const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(b));
  const l2 = l1 + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2));
  return [p2 / D2R, ((l2 / D2R + 540) % 360) - 180];
}

/** Sub-solar point (declination from day of year; mean solar time). */
export function subsolarPoint(tMs: number): { lat: number; lng: number } {
  const d = new Date(tMs);
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  const doy = (tMs - start) / 86_400_000;
  const lat = -23.44 * Math.cos((2 * Math.PI / 365) * (doy + 10));
  const utcH = d.getUTCHours() + d.getUTCMinutes() / 60;
  return { lat, lng: -15 * (utcH - 12) };
}
