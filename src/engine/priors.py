"""Deterministic Empirical Bayes priors + cloud visibility grids.

Contract (readme Sections 4.8, 5.3.1, 5.1.4): from 2012-2021 VIIRS data
estimate ``(mu_{g,m,DN}, rho_b, sigma_proc,b, lambda0_b, beta_dn,b,
eta_dn,b, Fbar_event,b)``:

* ``Fbar_event,b = max(1.0, median(clear near-nadir VIIRS cluster FRP))``.
* Cell-month baseline prior with shrinkage weight 5.0 toward the
  pyrome-month mean, treating clear zero-detection days as
  ``z_min = ln(0.01)``.
* ``rho_b in [0.25, 0.90]`` and ``sigma_proc,b in [0.25, 1.50]`` via lag-1
  OLS on consecutive active VIIRS cell-days.
* ``lambda0_b in [0.1, 5.0]`` via ``scipy.optimize.minimize_scalar`` matching
  the empirical zero-detection probability on active pyrome months.
* ``(beta_dn,b, eta_dn,b)`` via OLS slope + variance ratio
  (``eta in [0.05, 0.30]``).

Also implements the non-detection cloud visibility model (readme 5.1.4):
``v_{p,g,t} = p_swath,p(|phi|, t) * clip(1 - f_cloud - f_unmapped, 0.10,
0.98)`` with the 2012-2025 climatology fallback for trailing months
(``cloud_source = 'CLIMATOLOGY_FALLBACK'``), plus sparse 3-color neighbor
CSR packing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..config import settings as S
from ..geometry.h3_utils import (build_adjacency, h3_cell_center,
                                 three_color_hex_graph)
from ..geometry.solar_geometry import p_swath

DRIFT_START_T = (pd.Timestamp(S.DRIFT_START) - pd.Timestamp(S.DATE_START)).days


# ---------------------------------------------------------------------------
# Cloud grid lookup (0.25 deg CMG bins, 8-day periods, monthly fallback)
# ---------------------------------------------------------------------------
def cmg_bin_centers(lat, lon):
    """Nearest 0.25 deg CMG cell-center coordinates (UL -179.875, +89.875)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    row = np.clip(np.floor((89.875 - lat) / 0.25), 0, 719).astype(int)
    col = np.floor((lon + 179.875) / 0.25).astype(int) % 1440
    lat_c = 89.875 - 0.25 * row
    lon_c = -179.875 + 0.25 * col
    return lat_c, lon_c


def period_8d(doy: np.ndarray) -> np.ndarray:
    """8-day CMG period index (1..46) from day-of-year."""
    return (np.asarray(doy, dtype=int) - 1) // 8 + 1


def build_cloud_lookup(cloud_df: pd.DataFrame | None):
    """Index the staged cloud table by (lat_025, lon_025, year, period)."""
    if cloud_df is None or not len(cloud_df):
        return {}
    tbl = cloud_df.copy()
    keys = list(zip(tbl["lat_025"].round(2), tbl["lon_025"].round(2),
                    tbl["year"].astype(int), tbl["period"].astype(int)))
    return dict(zip(keys, tbl["cloud_frac"].to_numpy(dtype=float)))


def build_cloud_clim_lookup(clim_df: pd.DataFrame | None):
    """Climatology fallback keyed by (lat_025, lon_025, month)."""
    if clim_df is None or not len(clim_df):
        return {}
    keys = list(zip(clim_df["lat_025"].round(2), clim_df["lon_025"].round(2),
                    clim_df["month"].astype(int)))
    return dict(zip(keys, clim_df["cloud_frac_clim"].to_numpy(dtype=float)))


def cloud_fraction_series(cell_lat, cell_lon, dates, cloud_lookup, clim_lookup):
    """Cloud fraction + source per day for one cell (8-day join, climatology
    fallback for trailing months)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    lat_c, lon_c = cmg_bin_centers(cell_lat, cell_lon)
    doy = dates.dayofyear.to_numpy()
    years = dates.year.to_numpy()
    months = dates.month.to_numpy()
    periods = period_8d(doy)
    out = np.full(len(dates), np.nan)
    source = np.full(len(dates), "NONE", dtype=object)
    for i in range(len(dates)):
        key = (round(float(lat_c), 2), round(float(lon_c), 2),
               int(years[i]), int(periods[i]))
        v = cloud_lookup.get(key)
        if v is None:
            key_m = (round(float(lat_c), 2), round(float(lon_c), 2),
                     int(months[i]))
            v = clim_lookup.get(key_m)
            source[i] = "CLIMATOLOGY_FALLBACK" if v is not None else "MISSING"
        else:
            source[i] = "CMG_8D"
        if v is not None:
            out[i] = float(v)
    # propagate any residual missing values with the cell mean
    if np.isnan(out).any():
        fill = np.nanmean(out) if np.isfinite(np.nanmean(out)) else 0.5
        out = np.where(np.isnan(out), fill, out)
    return out, source


def visibility_matrix(platform: str, cell_lat: float, cloud_frac: np.ndarray,
                      t_index: np.ndarray):
    """v_{p,g,t} = p_swath,p(|phi|, t) * clip(1 - f_cloud - f_unmapped,
    0.10, 0.98) (unmapped fraction 0 when absent)."""
    ps = p_swath(platform, np.full(len(t_index), abs(cell_lat)),
                 bool(np.any(t_index >= DRIFT_START_T)))
    return ps * np.clip(1.0 - cloud_frac, *S.VIS_CLIP)


# ---------------------------------------------------------------------------
# Implied daily states from calibrated pass observations
# ---------------------------------------------------------------------------
def implied_z(pass_table: pd.DataFrame, t_of_date) -> pd.DataFrame:
    """Per (h3_index, t, dn_flag) implied latent log-flux from detections."""
    p = pass_table[pass_table["y_frp_mw"] > 0].copy()
    if p.empty:
        return pd.DataFrame(columns=["h3_index", "t", "dn_flag", "z_imp",
                                     "n_det_platforms"])
    p["t"] = t_of_date(p["solar_date"])
    p["z_imp"] = np.log(p["y_frp_mw"].to_numpy(dtype=float)) \
        - p["mu_gain_det"].to_numpy(dtype=float)
    g = p.groupby(["h3_index", "t", "dn_flag"], as_index=False).agg(
        z_imp=("z_imp", "mean"), n_det_platforms=("platform", "nunique"))
    return g


def daily_state_table(pass_table, active_cells, cloud_lookup, clim_lookup,
                      dates, t_index, platforms):
    """Dense daily (cell, t, dn) detection state for prior estimation.

    Returns a long frame with implied z (or z_min for clear non-detection
    days) and the per-day visibility of the best (clearest) platform.
    """
    zs = implied_z(pass_table, t_of_date_default)
    zmap = {}
    for r in zs.itertuples():
        zmap[(int(r.h3_index), int(r.t), r.dn_flag)] = (r.z_imp,
                                                        r.n_det_platforms)
    # per-platform detection keys (a row is "detected" only if ITS platform
    # reported the pass -- never collapse platforms, INV-02 semantics)
    p = pass_table[pass_table["y_frp_mw"] > 0].copy()
    if len(p):
        p["t"] = t_of_date_default(p["solar_date"])
        det_keys = {(int(h), int(t), d, str(plat))
                    for h, t, d, plat in zip(p["h3_index"], p["t"],
                                             p["dn_flag"], p["platform"])}
    else:
        det_keys = set()
    cell_lats = {}
    for c in active_cells:
        cell_lats[int(c)] = h3_cell_center(int(c))[0]
    rows = []
    dn_flags = ("D", "N")
    for c in active_cells:
        c = int(c)
        lat = cell_lats[c]
        cloud, _src = cloud_fraction_series(lat, h3_cell_center(c)[1], dates,
                                            cloud_lookup, clim_lookup)
        for pi, plat in enumerate(platforms):
            vis = visibility_matrix(plat, lat, cloud, t_index)
            # operational windows only
            t_lo = (pd.Timestamp(S.PLATFORM_TIMELINE[plat][0])
                    - pd.Timestamp(S.DATE_START)).days
            t_hi = (pd.Timestamp(S.PLATFORM_TIMELINE[plat][1])
                    - pd.Timestamp(S.DATE_START)).days
            for t in range(max(0, t_lo), min(len(t_index) - 1, t_hi) + 1):
                for dn in dn_flags:
                    key = (c, t, dn)
                    zval, nplat = zmap.get(key, (S.Z_MIN, 0))
                    detected = 1 if (c, t, dn, plat) in det_keys else 0
                    z_here = zval if detected else zval
                    rows.append((c, t, dn, z_here, vis[t], detected, nplat,
                                 plat))
    return pd.DataFrame(rows, columns=["h3_index", "t", "dn_flag", "z",
                                       "vis", "detected", "n_platforms",
                                       "platform"])


# ---------------------------------------------------------------------------
# Prior estimators
# ---------------------------------------------------------------------------
def estimate_f_event(pass_table: pd.DataFrame) -> dict:
    """Fbar_event,b = max(1.0, median clear near-nadir VIIRS cluster FRP)."""
    p = pass_table[(pass_table["sensor_family"] == "VIIRS")
                   & (pass_table["y_frp_mw"] > 0)
                   & (pass_table["mean_vza_deg"] < 25.0)].copy()
    out = {}
    if p.empty:
        return {0: 1.0}
    p["cluster_frp"] = p["y_frp_mw"] / np.maximum(
        p["raw_event_count"].clip(lower=1), 1)
    for b, g in p.groupby("pyrome_id"):
        out[int(b)] = max(S.F_EVENT_FLOOR_MW,
                          float(np.median(g["cluster_frp"])))
    return out or {0: 1.0}


def estimate_mu_gm(daily: pd.DataFrame, pyrome_of: dict,
                   shrink: float = S.SHRINKAGE_WEIGHT):
    """Cell-month Empirical Bayes shrinkage toward the pyrome-month mean."""
    # Day-level collapse (the spec's "clear non-detection DAYS"): one value
    # per (cell, t, dn) -- the mean implied z if any platform detected,
    # else z_min when at least one clear pass observed no detection.
    # (Per-platform rows would triple-count z_min through the MODIS
    # non-detection rows and crush the prior mean.)
    d = daily[(daily["detected"] == 1) | (daily["vis"] >= 0.5)].copy()
    if d.empty:
        return pd.DataFrame(columns=["h3_index", "month", "dn_flag", "mu",
                                     "n_days", "ybar"]), {}
    day_rows = []
    for (h, t, dn), g in d.groupby(["h3_index", "t", "dn_flag"]):
        if (g["detected"] == 1).any():
            z_day = float(g.loc[g["detected"] == 1, "z"].mean())
        elif (g["vis"] >= 0.5).any():
            z_day = S.Z_MIN          # clear non-detection day (dormant)
        else:
            continue                 # fully cloud-obscured day (uninformative)
        day_rows.append((int(h), int(t), dn, z_day))
    days = pd.DataFrame(day_rows, columns=["h3_index", "t", "dn_flag", "z"])
    dates = pd.to_datetime(pd.Timestamp(S.DATE_START)
                           + pd.to_timedelta(days["t"], unit="D"))
    days["month"] = dates.dt.month
    days["pyrome"] = days["h3_index"].map(lambda c: pyrome_of.get(int(c), 0))
    # ȳ: detected-day cell-month mean (the cell-month log-flux); dormant
    # cells/months (only clear non-detection days) stay at z_min and anchor
    # the pyrome baseline low.
    det_days = days[days["z"] > S.Z_MIN + 1e-9]
    cell_month = det_days.groupby(["h3_index", "month", "dn_flag"],
                                  as_index=False).agg(
        ybar=("z", "mean"), n_days=("z", "size"), pyrome=("pyrome", "first"))
    # dormant (cell, month, dn) groups with clear non-detection days only
    all_days = days.groupby(["h3_index", "month", "dn_flag"], as_index=False).agg(
        pyrome=("pyrome", "first"))
    dormant = all_days.merge(
        cell_month[["h3_index", "month", "dn_flag", "ybar", "n_days"]],
        on=["h3_index", "month", "dn_flag"], how="left")
    dormant = dormant[dormant["ybar"].isna()].copy()
    if len(dormant):
        dormant["ybar"] = S.Z_MIN
        dormant["n_days"] = 0
        cell_month = pd.concat(
            [cell_month, dormant[list(cell_month.columns)]], ignore_index=True)
    pyr_month = cell_month.groupby(["pyrome", "month", "dn_flag"],
                                   as_index=False)["ybar"].mean() \
        .rename(columns={"ybar": "mu_bar"})
    cm = cell_month.merge(pyr_month, on=["pyrome", "month", "dn_flag"],
                          how="left")
    cm["mu_bar"] = cm["mu_bar"].fillna(cm["ybar"])
    cm["mu"] = ((cm["n_days"] * cm["ybar"] + shrink * cm["mu_bar"])
                / (cm["n_days"] + shrink))
    mu_pyr = pyr_month.set_index(["pyrome", "month", "dn_flag"])["mu_bar"] \
        .to_dict()
    return cm[["h3_index", "month", "dn_flag", "mu", "n_days", "ybar",
               "pyrome"]], mu_pyr


def estimate_ar1(pass_table: pd.DataFrame, pyrome_of: dict):
    """rho_b (lag-1 OLS slope) and sigma_proc,b (residual std)."""
    zs = implied_z(pass_table, lambda d: t_of_date_default(d))
    rho: dict = {}
    sig: dict = {}
    if zs.empty:
        return {0: 0.5}, {0: 0.5}
    zs["pyrome"] = zs["h3_index"].map(lambda c: pyrome_of.get(int(c), 0))
    for b, g in zs.groupby("pyrome"):
        g = g.sort_values(["h3_index", "dn_flag", "t"])
        prev_z = g.groupby(["h3_index", "dn_flag"])["t"].transform(lambda s: s - 1)
        prev = g.groupby(["h3_index", "dn_flag"])["z_imp"].shift(1)
        m = (g["t"].to_numpy() == prev_z.to_numpy() + 1) & prev.notna()
        x = prev.to_numpy()[m]
        y = g["z_imp"].to_numpy()[m]
        if len(x) < 8:
            rho[int(b)], sig[int(b)] = 0.5, 0.5
            continue
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        rho[int(b)] = float(np.clip(coef[0], *S.RHO_CLIP))
        sig[int(b)] = float(np.clip(np.sqrt(resid.var(ddof=1))
                                    if len(resid) > 1 else 0.5,
                                    *S.SIGMA_PROC_CLIP))
    return rho or {0: 0.5}, sig or {0: 0.5}


def t_of_date_default(dates):
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return ((d - pd.Timestamp(S.DATE_START)).days).to_numpy(dtype=int)


def estimate_lambda0(daily: pd.DataFrame, pyrome_of: dict,
                     psi_cnt_fn, f_event: dict, mu_map: dict,
                     sigma2_op: float = 0.25,
                     sigma_proc: dict | None = None):
    """lambda0_b in [0.1, 5.0] matching the empirical zero-detection
    probability on clear active passes (minimize_scalar).

    The flux level z used in the rate fit is the detected-day cell-month
    mean implied z (non-detection rows borrow their cell-month detected
    mean -- the z_min-contaminated prior mean would understate e^z and
    saturate the rate).
    """
    d = daily.copy()
    if d.empty:
        return {0: 1.0}
    d["pyrome"] = d["h3_index"].map(lambda c: pyrome_of.get(int(c), 0))
    dates = pd.to_datetime(pd.Timestamp(S.DATE_START)
                           + pd.to_timedelta(d["t"], unit="D"))
    d["month"] = dates.dt.month
    # detected-day cell-month mean implied z
    det_mean = (d[d["detected"] == 1]
                .groupby(["h3_index", "month", "dn_flag"])["z"].mean()
                .to_dict())
    pyr_mean = (d[d["detected"] == 1].groupby(["pyrome", "month", "dn_flag"])["z"]
                .mean().to_dict())
    overall = float(d.loc[d["detected"] == 1, "z"].mean()) if (d["detected"] == 1).any() else S.Z_MIN
    z_fit = []
    for r in d.itertuples():
        key = (int(r.h3_index), int(r.month), r.dn_flag)
        if r.detected == 1:
            z_fit.append(r.z)
        elif key in det_mean:
            z_fit.append(float(det_mean[key]))
        elif (int(r.pyrome), int(r.month), r.dn_flag) in pyr_mean:
            z_fit.append(float(pyr_mean[(int(r.pyrome), int(r.month),
                                         r.dn_flag)]))
        else:
            z_fit.append(overall)
    d["z_fit"] = z_fit
    out = {}
    for b, g in d.groupby("pyrome"):
        # Horvitz-Thompson detection-rate matching: the detected rows with
        # inverse-detection-probability weights 1 / (v (1 - e^{-Lambda}))
        # represent all rows, so matching the empirical detection rate is
        #  sum_i 1 / (v_i (1 - e^{-Lambda_i}))  ==  N_rows.
        n_rows = float(len(g))
        emp_p0 = float(1.0 - g["detected"].mean())
        zf = g["z_fit"].to_numpy(dtype=float)
        vis = np.clip(g["vis"].to_numpy(dtype=float), 0.02, 1.0)
        psi = np.asarray(psi_cnt_fn(b, g), dtype=float)
        fe = max(f_event.get(int(b), 1.0), 1e-6)
        det_mask = g["detected"].to_numpy() == 1
        z_det = zf[det_mask]
        v_det = vis[det_mask]
        psi_det = psi[det_mask]
        if len(z_det) == 0:
            out[int(b)] = S.LAMBDA0_CLIP[0]
            continue

        # Observation-noise (Gauss-Hermite) marginalized HT WEIGHTS: the
        # implied z carries operator + cluster-count noise sigma_op, so the
        # correct Horvitz-Thompson weight for a detected row is
        # E_eps[1 / (v (1 - e^{-lam e^{z_obs - eps}}))]  (the expectation
        # of the weight -- Jensen: NOT the weight at the expected z).
        from scipy.special import roots_hermite
        gh_x, gh_w = roots_hermite(9)
        gh_w = gh_w / np.sqrt(np.pi)
        # total implied-z noise: operator noise + cluster-count noise,
        # proxied by the AR(1) residual std (which includes both)
        sp = (sigma_proc or {}).get(int(b), None)
        sigma_op = float(max(np.sqrt(max(sigma2_op, 1e-4)),
                             float(sp) if sp else 0.0))

        def weight_marg(lam_coef, z_obs, v_obs):
            # E_eps[1 / (v (1 - e^{-lam_coef e^{z_obs - eps}}))]
            tot = np.zeros_like(z_obs)
            for xk, wk in zip(gh_x, gh_w):
                zz = z_obs - sigma_op * np.sqrt(2.0) * xk
                p1 = np.clip(1.0 - np.exp(-lam_coef * np.exp(zz)), 1e-9, 1.0)
                tot += wk / np.maximum(v_obs * p1, 1e-9)
            return tot

        def loss(log_lam):
            lam0 = np.exp(log_lam)
            coef_det = lam0 * psi_det / fe
            w = weight_marg(coef_det, z_det, v_det)
            model_det_rate = len(z_det) / np.sum(w)
            p0_model = 1.0 - model_det_rate
            return (p0_model - emp_p0) ** 2

        res = minimize_scalar(loss, bounds=(np.log(S.LAMBDA0_CLIP[0]),
                                            np.log(S.LAMBDA0_CLIP[1])),
                              method="bounded",
                              options={"xatol": 1e-4})
        cands = [float(np.exp(res.x))]
        grid = np.exp(np.linspace(np.log(S.LAMBDA0_CLIP[0]),
                                  np.log(S.LAMBDA0_CLIP[1]), 41))
        best_loss = np.inf
        best_lam = 1.0
        for lam0 in np.concatenate([grid, cands]):
            l = loss(np.log(lam0))
            if np.isfinite(l) and l < best_loss:
                best_loss = float(l)
                best_lam = float(lam0)
        out[int(b)] = float(np.clip(best_lam, *S.LAMBDA0_CLIP))
    return out or {0: 1.0}


def estimate_dn_coupling(pass_table: pd.DataFrame, pyrome_of: dict,
                         mu_map: dict):
    """beta_dn,b (OLS slope Day ~ Night anomaly) and eta_dn,b (variance
    ratio / explained share, clipped [0.05, 0.30])."""
    zs = implied_z(pass_table, t_of_date_default)
    beta: dict = {}
    eta: dict = {}
    if zs.empty:
        return {0: 0.6}, {0: 0.15}
    zs["pyrome"] = zs["h3_index"].map(lambda c: pyrome_of.get(int(c), 0))
    dates = pd.to_datetime(pd.Timestamp(S.DATE_START)
                           + pd.to_timedelta(zs["t"], unit="D"))
    zs["month"] = dates.dt.month
    zs["mu"] = [mu_map.get((int(c), int(m), dn), S.Z_MIN)
                for c, m, dn in zip(zs["h3_index"], zs["month"], zs["dn_flag"])]
    zs["anom"] = zs["z_imp"] - zs["mu"]
    piv = zs.pivot_table(index=["h3_index", "t", "pyrome"], columns="dn_flag",
                         values="anom")
    if not {"D", "N"} <= set(piv.columns):
        return {0: 0.6}, {0: 0.15}
    piv = piv.dropna(subset=["D", "N"])
    for b, g in piv.groupby(level="pyrome"):
        x = g["N"].to_numpy(dtype=float)
        y = g["D"].to_numpy(dtype=float)
        if len(x) < 8:
            beta[int(b)], eta[int(b)] = 0.6, 0.15
            continue
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1.0 - float(resid.var()) / max(float(y.var()), 1e-12)
        beta[int(b)] = float(np.clip(coef[0], -2.0, 2.0))
        eta[int(b)] = float(np.clip(r2, *S.ETA_DN_CLIP))
    return beta or {0: 0.6}, eta or {0: 0.15}


# ---------------------------------------------------------------------------
# 3-color neighbor CSR packing
# ---------------------------------------------------------------------------
def pack_neighbor_csr(active_cells: np.ndarray):
    """Sparse 3-color neighbor CSR: per cell, active 1-ring neighbor indices.

    Returns (cells, colors, indptr, indices, n_edges).
    """
    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells).ravel())),
                     dtype=np.int64)
    colors, meta = three_color_hex_graph(cells)
    adj, n_edges = build_adjacency(cells)
    indptr = np.zeros(len(cells) + 1, dtype=np.int64)
    for i, nbrs in enumerate(adj):
        indptr[i + 1] = indptr[i] + len(nbrs)
    indices = np.array([j for nbrs in adj for j in nbrs], dtype=np.int64)
    return cells, colors, indptr, indices, n_edges


def estimate_priors(pass_table: pd.DataFrame, hex_pyromes: pd.DataFrame,
                    active_cells: np.ndarray, cloud_df: pd.DataFrame | None,
                    clim_df: pd.DataFrame | None, t_days: int = S.T_DAYS,
                    psi_cnt_fn=None):
    """Full deterministic Empirical Bayes prior estimation.

    ``psi_cnt_fn(b, daily_frame) -> psi array`` supplies the analytical
    survival integral (VIIRS family, mean area) used by the lambda0 fit.
    """
    pyrome_of = dict(zip(hex_pyromes["h3_index"].astype(np.int64),
                         hex_pyromes["pyrome_id"].astype(int)))
    if psi_cnt_fn is None:
        def psi_cnt_fn(b, g):
            return np.full(len(g), 0.5)

    dates = pd.DatetimeIndex([pd.Timestamp(S.DATE_START) + pd.Timedelta(days=t)
                              for t in range(t_days)])
    t_index = np.arange(t_days)
    cloud_lookup = build_cloud_lookup(cloud_df)
    clim_lookup = build_cloud_clim_lookup(clim_df)

    f_event = estimate_f_event(pass_table)
    daily = daily_state_table(pass_table, active_cells, cloud_lookup,
                              clim_lookup, dates, t_index,
                              list(S.PLATFORM_INDEX))
    # restrict prior estimation to the 2012-2021 VIIRS overlap window
    t_lo = (pd.Timestamp(S.CALIB_START) - pd.Timestamp(S.DATE_START)).days
    t_hi = (pd.Timestamp(S.CALIB_END) - pd.Timestamp(S.DATE_START)).days
    daily = daily[(daily["t"] >= t_lo) & (daily["t"] <= t_hi)]
    daily = daily[daily["h3_index"].isin(set(int(c) for c in active_cells))]

    mu_gm, mu_pyr = estimate_mu_gm(daily, pyrome_of)
    mu_map = {(int(r.h3_index), int(r.month), r.dn_flag): float(r.mu)
              for r in mu_gm.itertuples()} if len(mu_gm) else {}
    rho, sigma_proc = estimate_ar1(pass_table, pyrome_of)
    beta_dn, eta_dn = estimate_dn_coupling(pass_table, pyrome_of, mu_map)
    sigma2_op = 0.25
    if pass_table is not None and "sigma2" in getattr(pass_table, "columns", []):
        vals = pass_table["sigma2"].to_numpy(dtype=float)
        if len(vals):
            sigma2_op = float(np.nanmedian(vals))
    lambda0 = estimate_lambda0(daily, pyrome_of, psi_cnt_fn, f_event, mu_map,
                               sigma2_op=sigma2_op, sigma_proc=sigma_proc)
    cells, colors, indptr, indices, n_edges = pack_neighbor_csr(active_cells)

    priors = {
        "f_event": f_event,
        "mu_gm": mu_gm,
        "mu_map": mu_map,
        "mu_pyr": mu_pyr,
        "rho": rho,
        "sigma_proc": sigma_proc,
        "lambda0": lambda0,
        "beta_dn": beta_dn,
        "eta_dn": eta_dn,
        "pyrome_of": pyrome_of,
        "cells": cells,
        "colors": colors,
        "indptr": indptr,
        "indices": indices,
        "n_edges": n_edges,
        "t_days": t_days,
    }
    return priors
