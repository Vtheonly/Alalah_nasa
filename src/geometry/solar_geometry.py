"""Vectorized solar-time / SZA astronomy and spherical pixel geometry.

Contract (readme.md Section 4 & 5.1):
  * ``compute_solar_time_and_sza(acq_date, acq_hhmm, lons, lats)`` returning
    ``(solar_hour, solar_date, dn_window_flag, sza_deg)`` with base-60 HHMM
    parsing, solar-date rollover, the Local-Solar-Time half-day window
    (Day = [06:00, 18:00)), and Spencer's Fourier declination / equation of
    time for the physical Solar Zenith Angle.
  * ``modis_sample_to_geometry(sample)``: exact spherical-Earth scan geometry
    (nadir 1.00 x 1.00 km; edge 4.84 x 2.01 km = 9.72 km^2 at VZA ~65.5 deg).
  * ``viirs_sample_to_geometry(sample)``: monotone zone-aware sample mapping
    across the 3:1 / 2:1 / 1:1 aggregation zones (0/31.59/44.68/56.06 deg)
    with pixel area in [0.141, 0.620] km^2.
  * ``area_to_vza_deg(areas, sensor_family)``: monotone piecewise-linear
    VZA inversion via the spec knot tables.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple, Union

import numpy as np

from src.config import settings as S

# Spencer (1971) Fourier coefficients.
_DECLINATION_COEFFS = (
    (0.0, 0.006918),
    (1, -0.399912),
    (1, 0.070257),  # sin(Gamma)
    (2, -0.006758),
    (2, 0.000907),  # sin(2 Gamma)
)
_EQTIME_COEFFS = (
    0.000075,
    0.001868,
    -0.032077,
    -0.014615,
    -0.040849,
)


def spencer_solar_declination_rad(doy: Union[int, np.ndarray]) -> np.ndarray:
    """Solar declination delta_solar in radians via Spencer's series."""
    gamma = 2.0 * np.pi * (np.asarray(doy, dtype=float) - 1.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.001489 * np.sin(3.0 * gamma)
    )
    return decl


def spencer_equation_of_time_minutes(doy: Union[int, np.ndarray]) -> np.ndarray:
    """Equation of time E_t in minutes via Spencer's series."""
    gamma = 2.0 * np.pi * (np.asarray(doy, dtype=float) - 1.0) / 365.0
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )


def hhmm_to_hours(acq_hhmm: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Base-60 HHMM -> fractional UTC hours (51 -> 0.85)."""
    hhmm = np.asarray(acq_hhmm, dtype=np.int64)
    hours = (hhmm // 100) + (hhmm % 100) / 60.0
    if hours.ndim == 0:
        return float(hours)
    return hours


def compute_solar_time_and_sza(
    acq_date: Union[str, np.ndarray],
    acq_hhmm: Union[int, np.ndarray],
    lons: np.ndarray,
    lats: np.ndarray,
):
    """Local solar time, solar date, half-day window flag and SZA (degrees).

    Returns ``(solar_hour, solar_date, dn_flag, sza_deg)`` where:
      * ``solar_hour`` in [0, 24);
      * ``solar_date`` shifts by floor((h_UTC + lon/15) / 24) days;
      * ``dn_flag`` = 'D' when 6.0 <= solar_hour < 18.0 else 'N'
        (canonical latent half-day window, readme.md 5.1.1 item 3);
      * ``sza_deg`` is the physical Solar Zenith Angle from Spencer
        declination + equation of time (kept for provenance alongside the raw
        sensor daynight flag).
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    scalar = lons.ndim == 0
    if scalar:
        lons, lats = lons[None], lats[None]

    if isinstance(acq_date, str):
        dates = np.array([acq_date] * lons.size, dtype=object)
    else:
        dates = np.asarray(acq_date, dtype=object)
        if dates.size == 1 and lons.size > 1:
            dates = np.repeat(dates, lons.size)

    h_utc = np.atleast_1d(np.asarray(hhmm_to_hours(acq_hhmm), dtype=float))
    if h_utc.size == 1 and lons.size > 1:
        h_utc = np.repeat(h_utc, lons.size)

    tau_raw = h_utc[: lons.size] + lons / 15.0
    solar_hour = np.mod(tau_raw, 24.0)
    day_shift = np.floor(tau_raw / 24.0).astype(int)

    base = np.array(
        [datetime.strptime(str(d)[:10], "%Y-%m-%d") for d in dates[: lons.size]]
    )
    solar_date = np.array(
        [
            (b + timedelta(days=int(s))).strftime("%Y-%m-%d")
            for b, s in zip(base, day_shift)
        ],
        dtype=object,
    )

    dn_flag = np.where(
        (solar_hour >= 6.0) & (solar_hour < 18.0), "D", "N"
    )

    doy = np.array([b.timetuple().tm_yday for b in base], dtype=float)
    decl = spencer_solar_declination_rad(doy)
    et_min = spencer_equation_of_time_minutes(doy)
    tau_true = solar_hour + et_min / 60.0
    omega_deg = 15.0 * (tau_true - 12.0)
    omega = np.deg2rad(omega_deg)
    phi = np.deg2rad(lats)
    cos_sza = np.clip(
        np.sin(phi) * np.sin(decl)
        + np.cos(phi) * np.cos(decl) * np.cos(omega),
        -1.0,
        1.0,
    )
    sza_deg = np.rad2deg(np.arccos(cos_sza))

    if scalar:
        return (
            float(solar_hour[0]),
            str(solar_date[0]),
            str(dn_flag[0]),
            float(sza_deg[0]),
        )
    return solar_hour, solar_date, dn_flag, sza_deg


def sza_dn_flag(sza_deg: np.ndarray) -> np.ndarray:
    """Science-ML style Day/Night from SZA < 85 deg (provenance only)."""
    return np.where(np.asarray(sza_deg) < S.SZA_DAY_NIGHT_THRESHOLD_DEG, "D", "N")


# --------------------------------------------------------------------------- #
# MODIS spherical scan geometry
# --------------------------------------------------------------------------- #
def _modis_scan_angle_rad(sample: np.ndarray) -> np.ndarray:
    """alpha(s) = (s - 676.5) * 2 * alpha_max / 1353, in radians."""
    s = np.asarray(sample, dtype=float)
    two_alpha_max = np.deg2rad(2.0 * S.MODIS_HALF_SWATH_DEG)
    return (s - S.MODIS_CENTER_SAMPLE) * two_alpha_max / 1353.0


def _slant_range_km(alpha_rad: np.ndarray, altitude_km: float) -> np.ndarray:
    """D(alpha) = (R_E + H) cos(alpha) - sqrt(R_E^2 - (R_E+H)^2 sin^2(alpha))."""
    re = S.EARTH_RADIUS_KM
    rph = re + altitude_km
    sin_a = np.sin(alpha_rad)
    return rph * np.cos(alpha_rad) - np.sqrt(
        np.maximum(re * re - (rph * sin_a) ** 2, 0.0)
    )


def modis_sample_to_geometry(sample: Union[int, float, np.ndarray]):
    """Exact MODIS spherical geometry from the 1-km `sample` index.

    Returns ``(scan_km, track_km, vza_deg, area_km2)``; checks: s = 676.5 ->
    1.00 x 1.00 = 1.00 km^2 at VZA 0; s in {0, 1353} -> 4.84 x 2.01 =
    9.72 km^2 at VZA ~65.5 deg (readme.md 5.1.2).
    """
    s = np.asarray(sample, dtype=float)
    alpha = _modis_scan_angle_rad(s)
    vza = np.arcsin(
        (S.EARTH_RADIUS_KM + S.MODIS_ALTITUDE_KM)
        / S.EARTH_RADIUS_KM
        * np.sin(np.abs(alpha))
    )
    d = _slant_range_km(alpha, S.MODIS_ALTITUDE_KM)
    track = d * S.MODIS_IFOV_RAD
    scan = track / np.cos(vza)
    area = scan * track
    scalar = s.ndim == 0
    if scalar:
        return (
            float(scan), float(track), float(np.rad2deg(vza)), float(area)
        )
    return scan, track, np.rad2deg(vza), area


# --------------------------------------------------------------------------- #
# VIIRS I-band geometry (aggregation zones)
# --------------------------------------------------------------------------- #
# Aggregated samples per half-scan inside each zone. Zone angular widths are
# (31.59, 13.09, 11.38) deg and the native along-scan IFOV is chosen so the
# half-scan holds exactly 3200 aggregated samples (total 6400), with the
# nadir 3:1 aggregated pixel at ~0.375 km along scan.
_VIIRS_ZONE_SPAN_DEG = (
    S.VIIRS_AGG_ZONES_DEG[1] - S.VIIRS_AGG_ZONES_DEG[0],
    S.VIIRS_AGG_ZONES_DEG[2] - S.VIIRS_AGG_ZONES_DEG[1],
    S.VIIRS_AGG_ZONES_DEG[3] - S.VIIRS_AGG_ZONES_DEG[2],
)  # (31.59, 13.09, 11.38)
_VIIRS_NATIVE_ANGULAR_IFOV_DEG = 0.008892
_VIIRS_ZONE_SAMPLE_COUNTS = (
    int(round(_VIIRS_ZONE_SPAN_DEG[0] / (3.0 * _VIIRS_NATIVE_ANGULAR_IFOV_DEG))),
    int(round(_VIIRS_ZONE_SPAN_DEG[1] / (2.0 * _VIIRS_NATIVE_ANGULAR_IFOV_DEG))),
    int(round(_VIIRS_ZONE_SPAN_DEG[2] / (1.0 * _VIIRS_NATIVE_ANGULAR_IFOV_DEG))),
)  # (1184, 736, 1280) -> 3200 per half scan, 6400 total
_VIIRS_HALF_SAMPLES = sum(_VIIRS_ZONE_SAMPLE_COUNTS)
_VIIRS_AREA_KNOTS = np.array(
    [k[0] for k in S.VIIRS_AREA_VZA_KNOTS], dtype=float
)
_VIIRS_VZA_KNOTS = np.array(
    [k[1] for k in S.VIIRS_AREA_VZA_KNOTS], dtype=float
)
_MODIS_AREA_KNOTS = np.array(
    [k[0] for k in S.MODIS_AREA_VZA_KNOTS], dtype=float
)
_MODIS_VZA_KNOTS = np.array(
    [k[1] for k in S.MODIS_AREA_VZA_KNOTS], dtype=float
)


def viirs_scan_angle_deg(sample: Union[int, float, np.ndarray]) -> np.ndarray:
    """Monotone sample -> |scan angle| in [0, 56.06] deg across zones.

    The 3:1 / 2:1 / 1:1 aggregation zones occupy (1184, 736, 1280) samples
    per half-scan; within a zone the mapping is linear in sample index.
    """
    s = np.asarray(sample, dtype=float)
    offset = np.abs(s - (_VIIRS_HALF_SAMPLES - 0.5))  # distance from nadir
    offset = np.clip(offset, 0.0, _VIIRS_HALF_SAMPLES)
    angle = np.zeros_like(offset)
    remaining = offset.copy()
    for span_deg, n_samples in zip(
        _VIIRS_ZONE_SPAN_DEG, _VIIRS_ZONE_SAMPLE_COUNTS
    ):
        active = remaining > 0
        take = np.clip(remaining, 0.0, float(n_samples))
        # Each zone traversed contributes its linear angular share.
        angle = angle + np.where(active, take / n_samples * span_deg, 0.0)
        remaining = remaining - n_samples
    return angle


def viirs_sample_to_geometry(sample: Union[int, float, np.ndarray]):
    """VIIRS I-band geometry from the aggregated `sample` index.

    Returns ``(scan_km, track_km, vza_deg, area_km2)`` with area drawn from
    the spec's monotone area<->VZA knot table (0.141 km^2 at nadir up to
    0.620 km^2 at VZA 65 deg; clamped beyond), scan = track = sqrt(A).
    """
    s = np.asarray(sample, dtype=float)
    alpha = np.deg2rad(viirs_scan_angle_deg(s))
    vza = np.arcsin(
        (S.EARTH_RADIUS_KM + S.VIIRS_ALTITUDE_KM)
        / S.EARTH_RADIUS_KM
        * np.sin(alpha)
    )
    vza_deg = np.rad2deg(vza)
    area = np.interp(
        vza_deg, _VIIRS_VZA_KNOTS, _VIIRS_AREA_KNOTS
    )
    side = np.sqrt(area)
    scalar = s.ndim == 0
    if scalar:
        return float(side), float(side), float(vza_deg), float(area)
    return side, side, vza_deg, area


def area_to_vza_deg(
    areas: Union[float, np.ndarray], sensor_family: str
) -> Union[float, np.ndarray]:
    """Monotone piecewise-linear area -> VZA inversion (np.interp knots)."""
    fam = sensor_family.upper()
    if fam == "MODIS":
        vza = np.interp(areas, _MODIS_AREA_KNOTS, _MODIS_VZA_KNOTS)
    elif fam == "VIIRS":
        vza = np.interp(areas, _VIIRS_AREA_KNOTS, _VIIRS_VZA_KNOTS)
    else:
        raise ValueError(f"unknown sensor family: {sensor_family}")
    if np.ndim(vza) == 0:
        return float(vza)
    return vza


def resolve_pixel_geometry(
    sensor_family: str,
    sample_idx=None,
    pixarea_km2=None,
    scan_km=None,
    track_km=None,
):
    """Dispatch to the correct geometry path for one detection.

    Priority (readme.md 5.1.2): ``sample`` (MODIS/Vxx Science feeds) ->
    exact spherical / zone geometry; ``Pixarea`` (Vxx14IMGML) -> direct clip
    [0.12, 0.75] km^2 with sqrt proxy scan/track; ``scan x track``
    (Portolan / FIRMS API) -> clipped product [0.12, 12.0] km^2. VZA is
    recovered via the monotone knot inversion in every non-sample path.

    Returns ``(scan_km, track_km, vza_deg, area_km2)``.
    """
    fam = sensor_family.upper()
    if fam == "VIIRS":
        # VIIRS priority (readme.md 5.1.2): Pixarea (Vxx14IMGML) first, then
        # the zone-aware sample mapping, then scan x track.
        if pixarea_km2 is not None and not (
            np.ndim(pixarea_km2) == 0 and np.isnan(pixarea_km2)
        ):
            area = float(np.clip(pixarea_km2, *S.VIIRS_PIXAREA_CLIP))
            side = float(np.sqrt(area))
            return side, side, area_to_vza_deg(area, fam), area
        if sample_idx is not None and not (
            np.ndim(sample_idx) == 0 and np.isnan(sample_idx)
        ):
            return viirs_sample_to_geometry(sample_idx)
    elif sample_idx is not None and not (
        np.ndim(sample_idx) == 0 and np.isnan(sample_idx)
    ):
        # MODIS priority: exact spherical geometry from the sample index.
        return modis_sample_to_geometry(sample_idx)
    if scan_km is not None and track_km is not None and not (
        np.ndim(scan_km) == 0 and np.isnan(scan_km)
    ):
        area = float(np.clip(scan_km * track_km, *S.SCAN_TRACK_AREA_CLIP))
        return (
            float(scan_km),
            float(track_km),
            area_to_vza_deg(area, fam),
            area,
        )
    # No geometry inputs at all: fall back to family-average pass geometry.
    if fam == "MODIS":
        area = S.MODIS_AVG_AREA_KM2
    else:
        area = S.VIIRS_AVG_AREA_KM2
    side = float(np.sqrt(area))
    return side, side, area_to_vza_deg(area, fam), area


def expected_drift_solar_hour(
    platform: str, day_offset: int, dn_flag: str, t_total_days: int = S.T_DAYS
) -> float:
    """Expected solar hour of platform passes including post-2022 drift.

    Terra drifts 10.5 -> 8.75 and Aqua 13.5 -> 15.60 between 2022-01-01 and
    2026-11-14 (linear); VIIRS platforms stay at 13.5 Day / 1.5 Night. Night
    hours are (+12 mod 24) of the Day hour for MODIS.
    """
    from datetime import date

    drift_start = date(2022, 1, 1)
    base = date(2000, 11, 1)
    drift_day0 = (drift_start - base).days
    drift_span = t_total_days - 1 - drift_day0
    if platform == "Terra":
        nominal, target = S.TERRA_DRIFT_HOURS
    elif platform == "Aqua":
        nominal, target = S.AQUA_DRIFT_HOURS
    else:
        return 13.5 if dn_flag == "D" else 1.5
    if day_offset <= drift_day0:
        hour = nominal
    else:
        frac = min((day_offset - drift_day0) / max(drift_span, 1), 1.0)
        hour = nominal + frac * (target - nominal)
    if dn_flag == "N":
        hour = (hour + 12.0) % 24.0
    return float(hour)


def modis_swath_probability(lat_abs_deg: float, t_index: int) -> float:
    """p_swath for MODIS with post-2022 drift degradation (readme.md 5.1.4)."""
    from datetime import date

    p = float(
        np.clip(0.85 + 0.15 * abs(lat_abs_deg) / 30.0, 0.85, 1.00)
    )
    drift_start = (date(2022, 1, 1) - date(2000, 11, 1)).days
    if t_index >= drift_start:
        p *= S.MODIS_DRIFT_SWATH_FACTOR
    return p
