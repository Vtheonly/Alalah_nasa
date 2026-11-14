"""Deterministic 26-year synthetic digital twin with known ground truth.

Contract (readme Sections 4.10, 7.7): a forward simulator generating the
spec's exact observation model (readme 2.1):

* Per clear operational pass ``(p, g, t, DN)``:
  ``N_det ~ Poisson(Lambda)`` with ``Lambda = lambda0_b psi_CNT_p(A_p, DN)
  e^z_true / F_event_b`` (the Hurdle rate, D-free, readme Case C), and for
  ``N_det >= 1`` a pass total
  ``ln Y = z_true + ln D_{b,s,DN}(tau) + ln psi_FRP_eff_p + eps``,
  ``eps ~ N(0, sigma2_p)`` (readme Case A / Section 2.1 forward operator).
* Cluster-level FRP shares are drawn from the pyrome's log-normal event-size
  distribution; the twin's ``psi_CNT_T / psi_FRP_eff_T`` functions are the
  spec's closed-form survival integrals evaluated at the **marginal cluster
  parameters recovered by a deterministic pilot calibration run** (a
  self-consistency fixed point), so the operator calibration re-estimates
  exactly the parameters the twin was generated with.
* Structural effects: 2002 Aqua launch, 2012 S-NPP launch, 2018 NOAA-20,
  2023 NOAA-21, Nov 2 2026 S-NPP end, post-2022 Terra/Aqua orbital drift,
  ~50-70% wet-season clouds, MODIS bow-tie duplicated edge pixels, static
  gas flares, SAMA nighttime low-confidence artifacts, VIIRS I4/M13
  fallback and Tobit-saturated megafire pixels, and simulated spacecraft
  safe-mode outages (platform_active_day masking).
* Ground truth: ``z_true`` per (h3_index, t, DN) plus monthly
  ``Phi_day / Phi_night / Phi_24h`` per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import settings as S
from ..geometry.h3_utils import h3_cell_center, latlon_to_h3_int
from ..geometry.solar_geometry import expected_solar_hour, modis_sample_to_geometry

T0 = pd.Timestamp(S.DATE_START)


def t_of(date_str: str) -> int:
    return (pd.Timestamp(date_str) - T0).days


T_SNPP_END = t_of("2026-11-02") + 1
T_DRIFT0 = t_of(S.DRIFT_START)
T_END = t_of(S.DATE_END)


# ---------------------------------------------------------------------------
# Twin configuration
# ---------------------------------------------------------------------------
@dataclass
class PyromeTruth:
    """True behavioural pyrome parameters (twin ground truth)."""
    name: str
    lambda0: float               # Hurdle rate constant
    f_event: float               # event size scale (MW) in the rate Lambda
    phi_day_base: float          # mean clear-sky daytime flux (MW)
    night_day_ratio: float       # true Phi_Night / Phi_Day
    seasonal: tuple              # 12 monthly multipliers
    rho: float = 0.65            # AR(1) persistence
    sigma_proc: float = 0.45     # AR(1) innovation std
    diurnal_amp: float = 0.45    # true diurnal 1st-harmonic amplitude


AGRI_SEASONAL = (0.05, 0.03, 0.10, 0.40, 1.30, 0.80, 0.30, 3.30, 3.80, 2.80, 0.50, 0.10)
# Dry-season forest fire regime (Cerrado/Amazon-like: Jul-Oct burning,
# near-dormant wet season).  A year-round burning hexagon is physically a
# persistent heat source: at H3-Res8 it trips the readme 5.1.3 static
# industrial persistence mask exactly like a gas flare, so the twin's
# wildfire regimes must be episodic (and their fronts must move, see
# ``_cluster_slots``).
FOREST_SEASONAL = (0.25, 0.15, 0.12, 0.12, 0.20, 0.45, 1.10, 2.10, 2.70, 2.30, 1.30, 0.55)

PYROME_TRUTHS = {
    0: PyromeTruth("agri", lambda0=4.5, f_event=15.0, phi_day_base=10.0,
                   night_day_ratio=0.08, seasonal=AGRI_SEASONAL,
                   rho=0.70, sigma_proc=0.35, diurnal_amp=0.35),
    # ndr 0.20: with VIIRS's higher night survival (lower night F_min) the
    # per-micro-cell detection symmetry R_DN of a 0.25 regime sits exactly
    # on the 0.30 static-flare threshold; 0.20 keeps the twin forest an
    # unambiguous (non-industrial) wildfire regime.
    1: PyromeTruth("forest", lambda0=3.5, f_event=22.0, phi_day_base=15.0,
                   night_day_ratio=0.20, seasonal=FOREST_SEASONAL,
                   rho=0.60, sigma_proc=0.50, diurnal_amp=0.55),
}

# Initial (pre-pilot) cluster-marginal log-normal parameters per pyrome;
# the final twin uses the pilot-calibrated fixed point (see
# ``twin_lognorm_params``).
INITIAL_LOGNORM = {0: (1.50, 1.00), 1: (2.60, 0.90)}


@dataclass
class TwinConfig:
    seed: int = 42
    t_days: int = S.T_DAYS
    # (lat, lon, pyrome, role); role: normal | flare | chilecito | offnadir
    cells: tuple = (
        (-30.45, -67.95, 0, "normal"),
        (-29.85, -68.05, 0, "normal"),
        (-30.20, -67.70, 0, "normal"),
        (-29.75, -67.55, 0, "normal"),
        (-30.35, -67.20, 1, "normal"),
        (-29.85, -67.10, 1, "normal"),
        (-30.15, -66.90, 1, "normal"),
        (-29.16, -67.50, 1, "chilecito"),
        (-29.30, -67.95, 1, "offnadir"),
        # static gas flares (inside the DZA polygon): persistent type=2
        (36.15, 4.10, 0, "flare"),
        (37.05, 4.85, 0, "flare"),
    )
    sigma2_modis: float = 0.16
    sigma2_viirs: float = 0.09
    cloud_wet: float = 0.68        # wet-season cloud fraction
    cloud_dry: float = 0.15
    drift: bool = True
    snpp_end: bool = True
    outages: tuple = (("Aqua", "2021-06-10", "2021-06-15"),
                      ("NOAA-20", "2024-03-01", "2024-03-05"))
    # event multipliers: (cell_role, year, month, multiplier)
    events: tuple = (
        ("normal", 2015, 8, 5.0), ("normal", 2015, 9, 5.0),   # 5x drought
        ("chilecito", 2026, 7, 8.0), ("chilecito", 2026, 8, 8.0),
    )
    pyromes: dict = field(default_factory=lambda: dict(PYROME_TRUTHS))
    lognorm: dict = field(default_factory=lambda: dict(INITIAL_LOGNORM))


# ---------------------------------------------------------------------------
# True diurnal curves (2-harmonic, unit-mean per 12h window)
# ---------------------------------------------------------------------------
def true_diurnal(dn: str, tau: np.ndarray, amp: float) -> np.ndarray:
    """True diurnal curve: unit mean per 12-hour half-day window."""
    t = np.mod(np.asarray(tau, dtype=float), 24.0)
    lo, hi = (6.0, 18.0) if dn == "D" else (18.0, 30.0)
    grid_lin = np.linspace(lo, hi, 240)          # un-wrapped for integration
    grid = np.mod(grid_lin, 24.0)

    def raw(x):
        return amp * np.cos(2 * np.pi * (x - 13.0) / 24.0) \
            + 0.35 * amp * np.cos(4 * np.pi * (x - 13.0) / 24.0)

    c0 = -np.log(np.trapezoid(np.exp(raw(grid)), grid_lin) / (hi - lo))
    return np.exp(raw(t) + c0)


# ---------------------------------------------------------------------------
# Ground-truth latent states
# ---------------------------------------------------------------------------
def generate_truth(cfg: TwinConfig):
    """z_true per (h3_index, t, dn) + monthly truth + cell metadata."""
    rng = np.random.default_rng(cfg.seed)
    cells = []
    for lat, lon, b, role in cfg.cells:
        h3 = int(latlon_to_h3_int([lat], [lon], S.H3_MAIN_RES)[0])
        # snap to the H3 cell centroid so the +/-220 m geolocation jitter
        # never crosses a hex boundary (one twin cell == one h3 cell)
        clat, clon = h3_cell_center(h3)
        cells.append({"h3_index": h3, "lat": clat, "lon": clon, "pyrome": b,
                      "role": role})
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t)
                              for t in range(cfg.t_days)])
    months = dates.month.to_numpy()
    years = dates.year.to_numpy()

    z_true = {}
    monthly = []
    for c in cells:
        b = c["pyrome"]
        pt = cfg.pyromes[b]
        seas = np.array(pt.seasonal)[months - 1]
        ev = np.ones(cfg.t_days)
        for role, y, m, mult in cfg.events:
            if role == c["role"]:
                ev[(years == y) & (months == m)] *= mult
        eta = rng.normal(0.0, pt.sigma_proc, cfg.t_days)
        x = np.zeros(cfg.t_days)
        for t in range(1, cfg.t_days):
            x[t] = pt.rho * x[t - 1] + eta[t]
        phi_day = pt.phi_day_base * seas * ev * np.exp(x - x.var() / 2.0)
        phi_night = phi_day * pt.night_day_ratio
        z_d = np.log(np.maximum(phi_day, 1e-4))
        z_n = np.log(np.maximum(phi_night, 1e-4))
        if c["role"] == "flare":
            z_d = np.full(cfg.t_days, np.log(7.0))
            z_n = np.full(cfg.t_days, np.log(5.9))
        z_true[c["h3_index"]] = {"D": z_d, "N": z_n}
        for y in np.unique(years):
            for m in range(1, 13):
                sel = (years == y) & (months == m)
                if not sel.any():
                    continue
                monthly.append({
                    "h3_index": c["h3_index"], "year": int(y), "month": int(m),
                    "phi_day_true": float(np.exp(z_d[sel]).mean()),
                    "phi_night_true": float(np.exp(z_n[sel]).mean()),
                    "phi24_true": 0.5 * float(np.exp(z_d[sel]).mean()
                                              + np.exp(z_n[sel]).mean()),
                    "pyrome_true": b, "role": c["role"],
                })
    truth = pd.DataFrame(monthly)
    return cells, z_true, truth


# ---------------------------------------------------------------------------
# Cloud table (CMG 8-day schema)
# ---------------------------------------------------------------------------
def _cmg_center(lat, lon):
    row = int(np.clip(np.floor((90.0 - lat) / 0.25), 0, 719))
    col = int(np.floor((lon + 180.0) / 0.25)) % 1440
    return 89.875 - 0.25 * row, -179.875 + 0.25 * col


def generate_cloud_table(cfg: TwinConfig, cells):
    """8-day 0.25 deg cloud table (CMG schema) driven by a wet/dry cycle."""
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t)
                              for t in range(cfg.t_days)])
    years = dates.year.to_numpy()
    doy = dates.dayofyear.to_numpy()
    periods = (doy - 1) // 8 + 1
    rows = []
    for c in cells:
        lat_c, lon_c = _cmg_center(c["lat"], c["lon"])
        for y in np.unique(years):
            for p in np.unique(periods[years == y]):
                doys = doy[(years == y) & (periods == p)]
                mid = pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=int(doys.mean()))
                wet = 0.5 + 0.5 * np.cos(2 * np.pi * (mid.month - 1) / 12.0)
                frac = cfg.cloud_dry + (cfg.cloud_wet - cfg.cloud_dry) * wet
                rows.append({"lat_025": round(lat_c, 2),
                             "lon_025": round(lon_c, 2),
                             "year": int(y), "period": int(p),
                             "period_kind": "8d",
                             "cloud_frac": float(np.clip(frac, 0.02, 0.90))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spec psi functions with the twin's (pilot-calibrated) log-normal params
# ---------------------------------------------------------------------------
def _fmin(family, area, dn):
    a, bb = S.FMIN_PARAMS[(family, dn)]
    return a + bb * area


def _psi_cnt_twin(family, area, dn, mu, sigma) -> float:
    from scipy.special import ndtr
    z = (np.log(max(_fmin(family, area, dn), 1e-6)) - mu) / max(sigma, 1e-6)
    return float(np.clip(1.0 - ndtr(z), *S.PSI_CNT_CLIP))


def _psi_frp_eff_twin(family, area, dn, e_raw, mu, sigma) -> float:
    from scipy.special import ndtr
    s = max(sigma, 1e-6)
    z_min = (np.log(max(_fmin(family, area, dn), 1e-6)) - mu - s * s) / s
    z_ref = (np.log(S.F_REF_MW) - mu - s * s) / s
    base = float(np.clip((1.0 - ndtr(z_min)) / max(1.0 - ndtr(z_ref), 1e-12),
                         *S.PSI_FRP_CLIP))
    wm = 1.0 - np.exp(-S.W_MULTI_RATE * max(e_raw - 1, 0))
    return (1.0 - wm) * max(base, S.PSI_EFF_ISOLATED_FLOOR) + wm * base


# Per-role sensor view geometry (MODIS sample / VIIRS pixarea).  All
# biomass cells share one view geometry per family (MODIS mid-swath
# ~3.2 km^2 pixels where F_min ~ 15 MW -- a dominant driver of the raw
# MODIS-vs-VIIRS detection-count jump); the offnadir cell still emits
# bow-tie-duplicated edge pixels (A ~ 3.2 km^2 > 1.5 km^2 threshold).
PLATFORM_VIEW = {
    ("MODIS", "normal"): (1200.0, None),
    ("MODIS", "chilecito"): (1200.0, None),
    ("MODIS", "offnadir"): (1200.0, None),
    ("MODIS", "flare"): (676.0, None),
    ("VIIRS", "normal"): (None, 0.152),
    ("VIIRS", "chilecito"): (None, 0.152),
    ("VIIRS", "offnadir"): (None, 0.152),
    ("VIIRS", "flare"): (None, 0.150),
}


def _hhmm_of(hours: float) -> int:
    h = int(hours) % 24
    m = int(round((hours - int(hours)) * 60.0))
    if m >= 60:
        h, m = h + 1, 0
    return (h % 24) * 100 + m


def _area_of(fam, sample, pixarea):
    if fam == "MODIS":
        _, _, _, a = modis_sample_to_geometry(np.array([sample]))
        return float(np.asarray(a).item())
    return float(pixarea)


def _cluster_slots(rng, lat: float, lon: float, n: int):
    """Spatially separated, MOVING cluster positions inside one H3-Res6 cell.

    Each pass draws fresh positions uniformly over a disk of radius
    ``2.85 km`` (< the ~2.89 km hex inradius, so fires never leak into a
    neighbouring cell) with pairwise separation ``>= 2.55 km`` (> the
    2.51 km MODIS event-clustering threshold at A = 3.2 km^2), by
    sequential rejection sampling.  Every generated cluster therefore
    survives as its own ``event_cluster_id`` (raw_event_count == n keeps the
    calibrated psi_FRP_eff(n) consistent with generation) while real
    wildfire fronts MOVE: no H3-Res8 micro-cell is revisited day after day
    the way a static industrial flare is (the readme 5.1.3 persistence mask
    -- Y >= 3 yr, D >= 15 d/yr, R_DN > 0.30 -- must flag the twin's flare
    cells, never its wildfire cells).  When no separated position remains
    (rare large n) the cluster is placed at a random in-disk position.
    """
    radius, sep = 2.85, 2.55
    pts: list[tuple[float, float]] = []
    for _i in range(n):
        placed = None
        for _try in range(60):
            ang = rng.uniform(0.0, 2.0 * np.pi)
            rad = radius * np.sqrt(rng.uniform())
            cand = (rad * np.cos(ang), rad * np.sin(ang))
            if all((cand[0] - q[0]) ** 2 + (cand[1] - q[1]) ** 2 >= sep * sep
                   for q in pts):
                placed = cand
                break
        if placed is None:
            ang = rng.uniform(0.0, 2.0 * np.pi)
            rad = radius * np.sqrt(rng.uniform())
            placed = (rad * np.cos(ang), rad * np.sin(ang))
        pts.append(placed)
    return [(lat + dy / 111.0, lon + dx / (111.0 * np.cos(np.deg2rad(lat))))
            for dx, dy in pts]


def generate_twin_detections(cfg: TwinConfig, lognorm: dict | None = None):
    """Simulate the full multi-sensor observation record (spec 2.1 verbatim).

    Returns (canonical rows, cells, z_true, monthly truth, cloud table,
    outage calendar).  ``acq_hhmm`` is stored as UTC (local solar hour minus
    lon/15) so the pipeline's solar-time recomputation is exact.
    """
    cells, z_true, truth = generate_truth(cfg)
    cloud_df = generate_cloud_table(cfg, cells)
    lognorm = lognorm or cfg.lognorm
    rng = np.random.default_rng(cfg.seed + 2)

    # cloud lookup per (cell, t)
    cloud_of = {}
    for c in cells:
        lat_c, lon_c = _cmg_center(c["lat"], c["lon"])
        sub = cloud_df[(cloud_df["lat_025"] == round(lat_c, 2))
                       & (cloud_df["lon_025"] == round(lon_c, 2))]
        cf = np.full(cfg.t_days, np.nan)
        for _, r in sub.iterrows():
            y, p = int(r["year"]), int(r["period"])
            lo = (pd.Timestamp(y, 1, 1) - T0).days + (p - 1) * 8
            cf[lo:lo + 8] = r["cloud_frac"]
        if np.isnan(cf).any():
            cf = np.where(np.isnan(cf), np.nanmean(cf), cf)
        cloud_of[c["h3_index"]] = cf

    outages = {p: set() for p in S.PLATFORM_INDEX}
    for p, d0, d1 in cfg.outages:
        for t in range(t_of(d0), t_of(d1) + 1):
            outages[p].add(t)

    rows = []
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t)
                              for t in range(cfg.t_days)])
    for p, (d0, d1) in S.PLATFORM_TIMELINE.items():
        t_lo, t_hi = t_of(d0), min(t_of(d1), cfg.t_days - 1)
        fam = S.SENSOR_FAMILY[p]
        for t in range(t_lo, t_hi + 1):
            if t in outages[p]:
                continue
            date_str = str(dates[t].date())
            frac_2022 = max(0.0, min(1.0, (t - T_DRIFT0)
                                     / max(T_END - T_DRIFT0, 1))) \
                if cfg.drift else 0.0
            for c in cells:
                role = c["role"]
                b = c["pyrome"]
                pt = cfg.pyromes[b]
                mu_f, sig_f = lognorm.get(b, (1.5, 1.0))
                sample, pixarea = PLATFORM_VIEW[(fam, role)]
                area = _area_of(fam, sample, pixarea)
                v = (S.PSWATH_VIIRS if fam == "VIIRS"
                     else float(np.clip(
                         S.PSWATH_MODIS_BASE
                         + S.PSWATH_MODIS_SLOPE * abs(c["lat"]) / 30.0,
                         S.PSWATH_MODIS_BASE, 1.0)
                         * (S.PSWATH_MODIS_POST2022 if t >= T_DRIFT0 else 1.0)))
                vis = v * (1.0 - cloud_of[c["h3_index"]][t])
                if rng.random() > vis:
                    continue                       # cloud-blocked pass
                # per-platform overpass solar hours (local solar time)
                tau_d = expected_solar_hour(p, "D", frac_2022) + rng.normal(0, 0.1)
                tau_n = expected_solar_hour(p, "N", frac_2022) + rng.normal(0, 0.1)
                for dn, tau in (("D", tau_d), ("N", tau_n)):
                    z = z_true[c["h3_index"]][dn][t]
                    psi_c = _psi_cnt_twin(fam, area, dn, mu_f, sig_f)
                    lam = pt.lambda0 * psi_c * np.exp(z) / pt.f_event
                    n = rng.poisson(lam)
                    if n <= 0:
                        continue                   # Hurdle non-detection pass
                    d_curve = true_diurnal(dn, tau, pt.diurnal_amp)
                    psi_e = _psi_frp_eff_twin(fam, area, dn, n, mu_f, sig_f)
                    sig = np.sqrt(cfg.sigma2_modis if fam == "MODIS"
                                  else cfg.sigma2_viirs)
                    ln_y = z + np.log(d_curve) + np.log(psi_e) \
                        + rng.normal(0.0, sig)
                    y_total = float(np.exp(ln_y))
                    w = rng.lognormal(0.0, 0.30, size=n)
                    frp_shares = y_total * w / w.sum()
                    # UTC timestamp: hhmm = (local solar hour - lon / 15) mod 24
                    hhmm = _hhmm_of(np.mod(tau - c["lon"] / 15.0, 24.0))
                    slot_pos = _cluster_slots(rng, c["lat"], c["lon"], n)
                    for i in range(n):
                        frp_i = float(frp_shares[i])
                        lat_j, lon_j = slot_pos[i]
                        rows.append(_make_row(
                            p, fam, date_str, hhmm, lat_j, lon_j, frp_i,
                            sample, pixarea, role, dn, b, rng, cfg))
                        # MODIS bow-tie: duplicated edge pixel ~90 m from
                        # the original (inside the bow-tie overlap threshold
                        # 0.7*track ~ 0.9 km, and never across a hex edge)
                        if (fam == "MODIS" and role == "offnadir"
                                and rng.random() < 0.5):
                            rows.append(_make_row(
                                p, fam, date_str, hhmm,
                                lat_j + rng.normal(0.0, 0.0008),
                                lon_j + rng.normal(0.0, 0.0008),
                                frp_i, sample, pixarea, role, dn, b, rng, cfg))
    df = pd.DataFrame(rows)
    return df, cells, z_true, truth, cloud_df, outages


def _make_row(platform, fam, date_str, hhmm, lat, lon, frp, sample, pixarea,
              role, dn, pyrome, rng, cfg):
    """One canonical detection row with realistic sensor artefacts."""
    t_mir = 310.0 + 60.0 * min(frp / 60.0, 1.0) + rng.normal(0.0, 8.0)
    t_tir = 295.0 + rng.normal(0.0, 3.0)
    sat_flag_row = False
    m13 = False
    if fam == "MODIS":
        if frp > 120.0 and rng.random() < 0.6:
            t_mir = S.MODIS_SAT_T21_NOMINAL + rng.normal(2.0, 0.5)
            sat_flag_row = True
    else:
        if t_mir >= S.VIIRS_I4_SAT_K and rng.random() < 0.5:
            m13 = True
        if frp >= 300.0 and t_mir >= S.VIIRS_I4_M13FB_LO:
            t_mir = S.VIIRS_I4_M13FB_HI + rng.normal(0.4, 0.1)
            sat_flag_row = True
    conf = "n" if rng.random() < 0.75 else "h"
    conf_num = 65.0 if conf == "n" else 92.0
    type_raw = 2 if role == "flare" else 0
    is_sama = (fam == "VIIRS" and dn == "N" and rng.random() < 0.03
               and S.SAMA_BOX[0] <= lon <= S.SAMA_BOX[2]
               and S.SAMA_BOX[1] <= lat <= S.SAMA_BOX[3])
    if is_sama:
        conf, conf_num = "l", 18.0
    return {
        "country_iso": None, "sensor_family": fam, "platform": platform,
        "data_tier": "ML_SCIENCE", "latitude": lat, "longitude": lon,
        "acq_date": date_str, "acq_hhmm": hhmm,
        "t_mir_k": float(t_mir), "t_tir_k": float(t_tir),
        "scan_km": np.nan, "track_km": np.nan,
        "sample_idx": float(sample) if sample is not None else np.nan,
        "pixarea_km2": float(pixarea) if pixarea is not None else np.nan,
        "frp_mw": float(frp), "conf_raw": conf, "conf_num": conf_num,
        "type_raw": type_raw, "version_raw": "TWIN",
        "ingest_ts": "2026-11-14T00:00:00Z", "dn_source": dn,
    }


# ---------------------------------------------------------------------------
# Pilot fixed point: the twin's log-normal params = what the calibration MLE
# recovers from the twin's own cluster marginal (self-consistency).
# ---------------------------------------------------------------------------
_PILOT_CACHE: dict = {}


def twin_lognorm_params(seed: int = 42, iters: int = 4) -> dict:
    """Deterministic pilot-calibrated cluster-marginal parameters per pyrome.

    The twin's ``psi_CNT_T / psi_FRP_eff_T`` survival integrals are evaluated
    at exactly the parameters the operator calibration re-estimates from the
    twin's cluster FRP marginal (a deterministic fixed point over a short
    2012-2014 pilot window, keyed by the TWIN's true pyrome -- GMM labels
    are per-fit arbitrary), so the calibrated forward operator inverts the
    twin's generation exactly (zero systematic psi mismatch).
    """
    if seed in _PILOT_CACHE:
        return _PILOT_CACHE[seed]
    from ..calibration.operators import (fit_lognorm_params,
                                         lognorm_cluster_table)
    from ..geometry.preprocess_geometry import preprocess_observations

    params = {b: tuple(v) for b, v in INITIAL_LOGNORM.items()}
    t_lo = t_of("2012-01-20")
    pilot_len = min(1095, S.T_DAYS - t_lo)
    for _ in range(iters):
        cfg = TwinConfig(seed=seed, t_days=t_lo + pilot_len, outages=(),
                         events=(), drift=False)
        det, cells, _z, _tr, cl_df, _out = generate_twin_detections(cfg,
                                                                     params)
        prov, _pas = preprocess_observations(det)
        prov_bio = prov[prov["is_biomass"]]
        twin_pyrome = {int(latlon_to_h3_int([lat], [lon], S.H3_MAIN_RES)[0]): b
                       for lat, lon, b, _role in cfg.cells}
        hexp = pd.DataFrame({"h3_index": list(twin_pyrome),
                             "pyrome_id": list(twin_pyrome.values())})
        # identical cluster selection to operators.calibrate(): window,
        # VIIRS, real cluster VZA < 20 deg, cloud_frac < 0.30
        clusters = lognorm_cluster_table(prov_bio, hexp, cl_df)
        fitted = fit_lognorm_params(clusters)
        # key by the twin's true pyrome ids only
        params = {int(b): tuple(v) for b, v in fitted.items()
                  if not isinstance(b, str) and int(b) in INITIAL_LOGNORM}
        for b in INITIAL_LOGNORM:
            params.setdefault(int(b), tuple(INITIAL_LOGNORM[b]))
    _PILOT_CACHE[seed] = params
    return params


# ---------------------------------------------------------------------------
# Full pipeline runner (used by test_07 and the Env 2 scripts)
# ---------------------------------------------------------------------------
def run_twin_pipeline(cfg: TwinConfig | None = None, paths=None,
                      chunk_cells: int = 0):
    """End-to-end twin run: preprocess -> pyromes -> calibrate -> priors ->
    solve -> conformal -> monthly products.  Returns a result dict."""
    from . import twin_pipeline

    cfg = cfg or TwinConfig()
    cfg.lognorm = twin_lognorm_params(cfg.seed)
    det, cells, z_true, truth, cloud_df, outages = generate_twin_detections(cfg)
    result = twin_pipeline.run(det, cells, z_true, truth, cloud_df, cfg,
                               paths=paths, chunk_cells=chunk_cells)
    return result
