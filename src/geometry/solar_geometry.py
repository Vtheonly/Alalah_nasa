"""Vectorized solar/time geometry and spherical pixel geometry.

Contract (readme Sections 4.6, 5.1.1, 5.1.2):
* ``compute_solar_time_and_sza(acq_date, acq_hhmm, lons, lats)`` returns
  ``(solar_hour, solar_date, dn_window_flag, sza_deg)`` with base-60 HHMM
  parsing, local solar time ``tau = (h_UTC + lon/15) mod 24``, solar-date
  rollover across +-180 deg, canonical half-day windows
  (Day = [06:00, 18:00), Night = [18:00, 06:00)) and Spencer's Fourier
  solar declination / equation of time for SZA.
* ``modis_sample_to_geometry`` / ``viirs_sample_to_geometry``: exact spherical
  Earth scan geometry (scan_km, track_km, vza_deg, area_km2).
* ``area_to_vza_deg``: monotone piecewise-linear VZA inversion from pixel area.
"""
from __future__ import annotations

import numpy as np

from ..config import settings as S

_D2R = np.pi / 180.0
_R2D = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Base-60 HHMM parsing (readme 5.1.1: 51 -> 0.85 h)
# ---------------------------------------------------------------------------
def hhmm_to_hours(hhmm) -> np.ndarray | float:
    """``(hhmm // 100) + (hhmm % 100) / 60.0`` (e.g. 51 -> 0.85 hours)."""
    arr = np.asarray(hhmm, dtype=np.int64)
    hours = (arr // 100) + (arr % 100) / 60.0
    if hours.ndim == 0:
        return float(hours)
    return hours


# ---------------------------------------------------------------------------
# Spencer solar position (readme 5.1.1)
# ---------------------------------------------------------------------------
def spencer_solar_position(doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solar declination (rad) and equation of time (minutes) via Spencer's
    standard Fourier approximation with day angle Gamma = 2 pi (doy-1)/365."""
    doy = np.asarray(doy, dtype=np.float64)
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (0.006918
            - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2.0 * gamma) + 0.000907 * np.sin(2.0 * gamma))
    et_min = 229.18 * (0.000075
                       + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2.0 * gamma)
                       - 0.040849 * np.sin(2.0 * gamma))
    return decl, et_min


def solar_zenith_angle(doy, lats, lons, solar_hour):
    """SZA (deg) from Spencer declination + equation of time."""
    decl, et_min = spencer_solar_position(np.asarray(doy, dtype=np.float64))
    phi = np.asarray(lats, dtype=np.float64) * _D2R
    # true solar time -> hour angle (deg)
    omega = 15.0 * (np.asarray(solar_hour, dtype=np.float64) + et_min / 60.0 - 12.0)
    cos_sza = (np.sin(phi) * np.sin(decl)
               + np.cos(phi) * np.cos(decl) * np.cos(omega * _D2R))
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sza))


def compute_solar_time_and_sza(acq_date, acq_hhmm, lons, lats):
    """Return (solar_hour, solar_date, dn_window_flag, sza_deg).

    * ``solar_hour``: local mean solar time in [0, 24).
    * ``solar_date``: UTC date shifted by floor(tau_raw / 24) days.
    * ``dn_window_flag``: canonical latent half-day window
      ('D' if 6.0 <= tau < 18.0 else 'N'), guaranteeing exact 12-hour diurnal
      normalization alignment even at Arctic latitudes.
    * ``sza_deg``: physical solar zenith angle for provenance.
    """
    dates = np.asarray(acq_date, dtype="datetime64[D]")
    h_utc = np.asarray(hhmm_to_hours(acq_hhmm), dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    tau_raw = h_utc + lons / 15.0
    solar_hour = np.mod(tau_raw, 24.0)
    day_shift = np.floor(tau_raw / 24.0).astype(np.int64)
    solar_date = dates + day_shift.astype("timedelta64[D]")

    dn = np.where((solar_hour >= S.DAY_WINDOW[0]) & (solar_hour < S.DAY_WINDOW[1]),
                  "D", "N")

    doy = (dates - dates.astype("datetime64[Y]") +
           np.timedelta64(1, "D")).astype(np.int64) + 1
    sza = solar_zenith_angle(doy, lats, lons, solar_hour)
    return solar_hour, solar_date, dn, sza


def sza_physical_dn_flag(sza_deg: np.ndarray) -> np.ndarray:
    """Physical day/night flag ('D' if SZA < 85 deg) for Product 1 provenance."""
    return np.where(np.asarray(sza_deg) < S.SZA_DAY_THRESHOLD_DEG, "D", "N")


# ---------------------------------------------------------------------------
# Exact spherical scan geometry (readme 5.1.2)
# ---------------------------------------------------------------------------
def _slant_range(alpha_rad: np.ndarray, altitude_km: float) -> np.ndarray:
    """D(alpha) = (R_E + H) cos(alpha) - sqrt(R_E^2 - (R_E+H)^2 sin^2 alpha)."""
    re = S.R_EARTH_KM
    rh = re + altitude_km
    return (rh * np.cos(alpha_rad)
            - np.sqrt(np.maximum(re * re - (rh * rh) * np.sin(alpha_rad) ** 2, 0.0)))


def modis_sample_to_geometry(sample):
    """MODIS ``sample`` (s in [0, 1353]) -> (scan_km, track_km, vza_deg, area_km2).

    alpha(s) = (s - 676.5) * 2 alpha_max / 1353;  theta_VZA =
    arcsin((R_E+H)/R_E sin|alpha|);  track = D(alpha) Delta alpha_0;
    scan = track / cos(theta_VZA);  A = scan * track.
    Yields 1.00 x 1.00 km at nadir and ~4.84 x 2.01 km at swath edge.
    """
    s = np.asarray(sample, dtype=np.float64)
    alpha_max = S.MODIS_HALF_SWATH_DEG * _D2R
    alpha = (s - 676.5) * (2.0 * alpha_max / 1353.0)
    sin_a = np.sin(np.abs(alpha))
    vza = np.arcsin(np.clip((S.R_EARTH_KM + S.MODIS_ALT_KM)
                            / S.R_EARTH_KM * sin_a, -1.0, 1.0))
    d = _slant_range(np.abs(alpha), S.MODIS_ALT_KM)
    track = d * S.MODIS_IFOV_RAD
    scan = track / np.cos(vza)
    area = scan * track
    vza_deg = vza * _R2D
    if scan.ndim == 0:
        return float(scan), float(track), float(vza_deg), float(area)
    return scan, track, vza_deg, area


def viirs_sample_to_geometry(sample):
    """VIIRS I-band ``sample`` (s in [0, 6399]) -> (scan, track, vza, area).

    Maps sample monotonically to |alpha| in [0, 56.06 deg] across the onboard
    3:1 / 2:1 / 1:1 aggregation zones (native pitch p = 28.455 deg / 3200).
    """
    s = np.asarray(sample, dtype=np.float64)
    c = np.abs(s - 3199.5)                       # distance from scan centre
    p = 28.455 / 3200.0                          # native detector pitch (deg)
    c1 = 31.59 / (3.0 * p)                       # end of 3:1 zone (sample units)
    c2 = c1 + 13.09 / (2.0 * p)                  # end of 2:1 zone
    alpha = np.where(
        c <= c1, 3.0 * p * c,
        np.where(c <= c2, 31.59 + 2.0 * p * (c - c1),
                 44.68 + p * (c - c2))) * _D2R
    step = np.where(c <= c1, 3.0, np.where(c <= c2, 2.0, 1.0)) * p * _D2R
    sin_a = np.sin(alpha)
    vza = np.arcsin(np.clip((S.R_EARTH_KM + S.VIIRS_ALT_KM)
                            / S.R_EARTH_KM * sin_a, -1.0, 1.0))
    d = _slant_range(alpha, S.VIIRS_ALT_KM)
    track = d * (S.VIIRS_IFOV_TRACK_KM / S.VIIRS_ALT_KM)
    scan = d * step / np.cos(vza)
    area = scan * track
    vza_deg = vza * _R2D
    if scan.ndim == 0:
        return float(scan), float(track), float(vza_deg), float(area)
    return scan, track, vza_deg, area


def area_to_vza_deg(areas, sensor_family: str):
    """Monotone piecewise-linear VZA inversion from pixel area (readme 5.1.2)."""
    fam = "MODIS" if str(sensor_family).upper().startswith("MOD") else "VIIRS"
    knots_a, knots_v = (S.MODIS_AREA_VZA_KNOTS if fam == "MODIS"
                        else S.VIIRS_AREA_VZA_KNOTS)
    return np.interp(np.asarray(areas, dtype=np.float64),
                     knots_a, knots_v)


def pixel_geometry_from_row(sample_idx, pixarea_km2, scan_km, track_km,
                            sensor_family: str):
    """Resolve (area_km2, vza_deg, scan_km, track_km) for canonical rows.

    Priority per row: MODIS/VIIRS ``sample`` -> exact spherical geometry;
    VIIRS ``Pixarea`` -> direct clip [0.12, 0.75]; ``scan x track`` ->
    A = clip(scan*track, 0.12, 12) + spline VZA inversion.  Handles mixed
    MODIS/VIIRS frames row-wise.
    """
    sf = np.asarray(sensor_family, dtype=object)
    sf = np.array([str(v) for v in sf.ravel()], dtype=object)
    fam = np.where(np.char.startswith(np.char.upper(sf.astype(str)), "MOD"),
                   "MODIS", "VIIRS")
    sample_arr = np.asarray(sample_idx, dtype=np.float64)
    pix_arr = np.asarray(pixarea_km2, dtype=np.float64)
    scan_arr = np.asarray(scan_km, dtype=np.float64)
    track_arr = np.asarray(track_km, dtype=np.float64)
    n = max(len(sample_arr), len(pix_arr), len(scan_arr))
    out_area = np.full(n, np.nan)
    out_vza = np.full(n, np.nan)
    out_scan = np.full(n, np.nan)
    out_track = np.full(n, np.nan)

    has_sample = np.isfinite(sample_arr)
    for family, default_s in (("MODIS", 676.5), ("VIIRS", 3199.5)):
        m = has_sample & (fam == family)
        if not m.any():
            continue
        s = np.where(m, sample_arr, default_s)
        if family == "MODIS":
            sc, tr, vz, ar = modis_sample_to_geometry(s)
        else:
            sc, tr, vz, ar = viirs_sample_to_geometry(s)
        out_area[m] = ar[m]
        out_vza[m] = vz[m]
        out_scan[m] = sc[m]
        out_track[m] = tr[m]

    has_pix = (~has_sample) & (fam == "VIIRS") & np.isfinite(pix_arr)
    if has_pix.any():
        a = np.clip(pix_arr[has_pix], *S.VIIRS_PIXAREA_CLIP)
        out_area[has_pix] = a
        out_vza[has_pix] = area_to_vza_deg(a, "VIIRS")
        out_scan[has_pix] = np.sqrt(a)
        out_track[has_pix] = np.sqrt(a)

    has_st = (~(has_sample | has_pix)) & np.isfinite(scan_arr) & np.isfinite(track_arr)
    if has_st.any():
        a = np.clip(scan_arr[has_st] * track_arr[has_st], *S.AREA_CLIP_KM2)
        out_area[has_st] = a
        out_vza[has_st] = area_to_vza_deg(a, np.where(fam[has_st] == "MODIS",
                                                      "MODIS", "VIIRS"))
        out_scan[has_st] = scan_arr[has_st]
        out_track[has_st] = track_arr[has_st]
    return out_area, out_vza, out_scan, out_track


# ---------------------------------------------------------------------------
# ECEF chords (for cKDTree clustering)
# ---------------------------------------------------------------------------
def latlon_to_ecef(lats, lons, r_km: float = S.R_EARTH_KM) -> np.ndarray:
    """Unit-sphere ECEF coordinates scaled by R_E (chord distance metric)."""
    phi = np.asarray(lats, dtype=np.float64) * _D2R
    lam = np.asarray(lons, dtype=np.float64) * _D2R
    return np.column_stack([r_km * np.cos(phi) * np.cos(lam),
                            r_km * np.cos(phi) * np.sin(lam),
                            r_km * np.sin(phi)])


# ---------------------------------------------------------------------------
# Expected (non-detection) pass geometry and solar hours (readme 5.1.4)
# ---------------------------------------------------------------------------
def p_swath(platform: str, abs_lat: np.ndarray, t_is_post_2022: bool) -> np.ndarray:
    """Swath coverage probability p_swath(|phi|, t)."""
    if str(platform) in S.VIIRS_PLATFORMS:
        return np.full_like(np.asarray(abs_lat, dtype=np.float64), S.PSWATH_VIIRS,
                            dtype=np.float64)
    base = np.clip(S.PSWATH_MODIS_BASE
                   + S.PSWATH_MODIS_SLOPE * np.asarray(abs_lat, dtype=np.float64) / 30.0,
                   S.PSWATH_MODIS_BASE, 1.00)
    if t_is_post_2022:
        base = base * S.PSWATH_MODIS_POST2022
    return base


def expected_solar_hour(platform: str, dn_flag: str, frac_2022_to_end: float = 0.0
                        ) -> float:
    """Expected overpass solar hour ``tau_bar``; Terra/Aqua drift post-2022.

    Terra drifts 10.5 -> 8.75 and Aqua 13.5 -> 15.60 linearly over
    2022-01-01 .. 2026-11-14; Night hours are (+12) mod 24 for Aqua and
    (tau - 12) for Terra when the day hour drops below the night window.
    VIIRS platforms are fixed at 13.5 Day / 1.5 Night.
    """
    f = min(max(float(frac_2022_to_end), 0.0), 1.0)
    if platform == "Terra":
        day = 10.5 + (S.TERRA_DRIFT_END_HOUR - 10.5) * f
        return day if dn_flag == "D" else (day + 12.0) % 24.0
    if platform == "Aqua":
        day = 13.5 + (S.AQUA_DRIFT_END_HOUR - 13.5) * f
        return day if dn_flag == "D" else (day + 12.0) % 24.0
    return 13.5 if dn_flag == "D" else 1.5
