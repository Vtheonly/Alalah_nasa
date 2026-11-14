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
    # bin edges are at 90 - 0.25 k (rows) and -180 + 0.25 k (cols); the
    # CMG cell CENTERS sit half a bin inside (UL center -179.875, +89.875)
    row = np.clip(np.floor((90.0 - lat) / 0.25), 0, 719).astype(int)
    col = np.floor((lon + 180.0) / 0.25).astype(int) % 1440
    lat_c = 89.875 - 0.25 * row
    lon_c = -179.875 + 0.25 * col
    return lat_c, lon_c


def period_8d(doy: np.ndarray) -> np.ndarray:
    """8-day CMG period index (1..46) from day-of-year."""
    return (np.asarray(doy, dtype=int) - 1) // 8 + 1


CLOUD_SOURCE_CODES = {0: "CMG_8D", 1: "CMG_MONTH", 2: "CLIMATOLOGY_FALLBACK",
                      3: "MISSING"}


def _bin_id(lat_c, lon_c) -> np.ndarray:
    """Integer CMG bin id (row * 1440 + col) from 0.25 deg bin centers."""
    lat_c = np.asarray(lat_c, dtype=np.float64)
    lon_c = np.asarray(lon_c, dtype=np.float64)
    row = np.clip(np.rint((89.875 - lat_c) / 0.25), 0, 719).astype(np.int64)
    col = np.rint((lon_c + 179.875) / 0.25).astype(np.int64) % 1440
    return row * 1440 + col


def build_cloud_lookup(cloud_df: pd.DataFrame | None):
    """Index the staged cloud table by (lat_025, lon_025, year, period).

    Only 8-day rows (``period_kind == '8d'``) are indexed here: monthly
    ``Mx14CMQ`` periods 1..12 would otherwise collide with 8-day periods
    1..46.  Monthly rows are indexed by :func:`build_cloud_month_lookup`.
    """
    if cloud_df is None or not len(cloud_df):
        return {}
    tbl = cloud_df
    if "period_kind" in tbl.columns:
        tbl = tbl[tbl["period_kind"].astype(str) == "8d"]
    keys = list(zip(tbl["lat_025"].round(2), tbl["lon_025"].round(2),
                    tbl["year"].astype(int), tbl["period"].astype(int)))
    return dict(zip(keys, tbl["cloud_frac"].to_numpy(dtype=float)))


def build_cloud_month_lookup(cloud_df: pd.DataFrame | None):
    """Monthly ``Mx14CMQ`` fallback rows keyed by (lat, lon, year, month)."""
    if cloud_df is None or not len(cloud_df) or "period_kind" not in cloud_df:
        return {}
    tbl = cloud_df[cloud_df["period_kind"].astype(str) == "month"]
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


def cloud_fraction_matrix(cell_lats, cell_lons, dates, cloud_df, clim_df):
    """Vectorized daily cloud fraction for many cells.

    Fill order per (cell, day): 8-day ``Mx14C8Q`` (``period_kind='8d'``) ->
    monthly ``Mx14CMQ`` (``period_kind='month'``) -> 2012-2025 climatology
    (``CLIMATOLOGY_FALLBACK``) -> cell mean of whatever exists (``MISSING``;
    0.5 if the cell has no cloud information at all).

    Returns ``(cloud (n_cells, T) float64, source (n_cells, T) int8)`` with
    source codes from :data:`CLOUD_SOURCE_CODES`.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    lat_c, lon_c = cmg_bin_centers(cell_lats, cell_lons)
    bins = _bin_id(lat_c, lon_c)
    n, t_len = len(bins), len(dates)
    cloud = np.full((n, t_len), np.nan)
    source = np.full((n, t_len), 3, dtype=np.int8)
    if n == 0 or t_len == 0:
        return cloud, source
    ubins, inv = np.unique(bins, return_inverse=True)
    years = dates.year.to_numpy().astype(np.int64)
    months = dates.month.to_numpy().astype(np.int64)
    p8 = period_8d(dates.dayofyear.to_numpy()).astype(np.int64)

    def _fill(tbl, period_codes, value_col, src_code):
        if tbl is None or not len(tbl):
            return
        tb = _bin_id(tbl["lat_025"].to_numpy(), tbl["lon_025"].to_numpy())
        keep = np.isin(tb, ubins)
        if not keep.any():
            return
        tb = tb[keep]
        codes_tbl = period_codes(tbl[keep])
        ucode, t_code = np.unique(period_codes(None), return_inverse=True)
        pos = np.searchsorted(ucode, codes_tbl)
        ok = (pos < len(ucode)) & (ucode[np.minimum(pos, len(ucode) - 1)]
                                   == codes_tbl)
        mat = np.full((len(ubins), len(ucode)), np.nan)
        bpos = np.searchsorted(ubins, tb)
        vals = tbl[keep][value_col].to_numpy(dtype=np.float64)
        mat[bpos[ok], pos[ok]] = vals[ok]
        cand = mat[inv][:, t_code]                      # (n, T)
        fill = np.isnan(cloud) & np.isfinite(cand)
        cloud[fill] = cand[fill]
        source[fill] = src_code

    if cloud_df is not None and len(cloud_df):
        kind = (cloud_df["period_kind"].astype(str)
                if "period_kind" in cloud_df.columns
                else pd.Series("8d", index=cloud_df.index))
        c8 = cloud_df[kind.to_numpy() == "8d"]
        cm = cloud_df[kind.to_numpy() == "month"]
        _fill(c8, lambda t: (years * 100 + p8) if t is None else
              (t["year"].to_numpy(np.int64) * 100
               + t["period"].to_numpy(np.int64)), "cloud_frac", 0)
        _fill(cm, lambda t: (years * 100 + months) if t is None else
              (t["year"].to_numpy(np.int64) * 100
               + t["period"].to_numpy(np.int64)), "cloud_frac", 1)
    if clim_df is not None and len(clim_df):
        _fill(clim_df, lambda t: months if t is None else
              t["month"].to_numpy(np.int64), "cloud_frac_clim", 2)
    # residual gaps: cell mean (or 0.5 when the cell has no information)
    if np.isnan(cloud).any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_mean = np.nanmean(cloud, axis=1)
        row_mean = np.where(np.isfinite(row_mean), row_mean, 0.5)
        r, c = np.nonzero(np.isnan(cloud))
        cloud[r, c] = row_mean[r]
    return cloud, source


def cloud_fraction_series(cell_lat, cell_lon, dates, cloud_lookup=None,
                          clim_lookup=None, cloud_df=None, clim_df=None):
    """Cloud fraction + source label per day for ONE cell.

    Accepts either the staged frames (``cloud_df`` / ``clim_df``, preferred,
    vectorized) or the legacy dict lookups (8-day keys, climatology keys).
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if cloud_df is not None or clim_df is not None:
        cloud, src = cloud_fraction_matrix([cell_lat], [cell_lon], dates,
                                           cloud_df, clim_df)
        labels = np.array([CLOUD_SOURCE_CODES[int(k)] for k in src[0]],
                          dtype=object)
        return cloud[0], labels
    cloud_lookup = cloud_lookup or {}
    clim_lookup = clim_lookup or {}
    lat_c, lon_c = cmg_bin_centers(cell_lat, cell_lon)
    la, lo = round(float(lat_c), 2), round(float(lon_c), 2)
    years = dates.year.to_numpy()
    months = dates.month.to_numpy()
    periods = period_8d(dates.dayofyear.to_numpy())
    out = np.full(len(dates), np.nan)
    source = np.full(len(dates), "MISSING", dtype=object)
    for i in range(len(dates)):
        v = cloud_lookup.get((la, lo, int(years[i]), int(periods[i])))
        if v is not None:
            out[i], source[i] = float(v), "CMG_8D"
            continue
        v = clim_lookup.get((la, lo, int(months[i])))
        if v is not None:
            out[i], source[i] = float(v), "CLIMATOLOGY_FALLBACK"
    if np.isnan(out).any():
        fill = np.nanmean(out) if np.isfinite(np.nanmean(out)) else 0.5
        out = np.where(np.isnan(out), fill, out)
    return out, source


def visibility_matrix(platform: str, cell_lat: float, cloud_frac: np.ndarray,
                      t_index: np.ndarray):
    """v_{p,g,t} = p_swath,p(|phi|, t) * clip(1 - f_cloud - f_unmapped,
    0.10, 0.98) (unmapped fraction 0 when absent).

    The post-2022 MODIS swath factor (0.96) applies PER DAY (t >=
    2022-01-01), not to the whole series.
    """
    t_index = np.asarray(t_index)
    abs_lat = np.full(len(t_index), abs(float(cell_lat)))
    pre = p_swath(platform, abs_lat, False)
    post = p_swath(platform, abs_lat, True)
    ps = np.where(t_index >= DRIFT_START_T, post, pre)
    return ps * np.clip(1.0 - np.asarray(cloud_frac, dtype=np.float64),
                        *S.VIS_CLIP)


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


def cloud_at_points(lats, lons, dates, cloud_df, clim_df,
                    batch_bins: int = 2000) -> np.ndarray:
    """Daily cloud fraction at arbitrary (lat, lon, date) points (same fill
    order as :func:`cloud_fraction_matrix`), batched over 0.25 deg bins so
    memory stays bounded for millions of points."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    out = np.full(len(lats), 0.5)
    if not len(lats):
        return out
    day = (pd.to_datetime(pd.Series(dates)).to_numpy()
           .astype("datetime64[D]").astype(np.int64))
    lat_c, lon_c = cmg_bin_centers(lats, lons)
    bins = _bin_id(lat_c, lon_c)
    ub, first, inv = np.unique(bins, return_index=True, return_inverse=True)
    d0, d1 = int(day.min()), int(day.max())
    rng = pd.DatetimeIndex((np.arange(d0, d1 + 1)).astype("datetime64[D]"))
    for lo in range(0, len(ub), batch_bins):
        hi = min(lo + batch_bins, len(ub))
        mat, _src = cloud_fraction_matrix(lat_c[first[lo:hi]],
                                          lon_c[first[lo:hi]], rng,
                                          cloud_df, clim_df)
        sel = (inv >= lo) & (inv < hi)
        out[sel] = mat[inv[sel] - lo, day[sel] - d0]
    return out


def cloud_at_cells(cells, dates, cloud_df, clim_df) -> np.ndarray:
    """Daily cloud fraction at H3 cell centres for (cell, date) rows."""
    cells = np.asarray(cells, dtype=np.int64)
    uc, inv = np.unique(cells, return_inverse=True)
    ll = np.array([h3_cell_center(int(c)) for c in uc], dtype=np.float64) \
        if len(uc) else np.zeros((0, 2))
    return cloud_at_points(ll[inv, 0] if len(uc) else [],
                           ll[inv, 1] if len(uc) else [], dates, cloud_df,
                           clim_df)


def _window(t_days: int) -> tuple[int, int]:
    t_lo = (pd.Timestamp(S.CALIB_START) - pd.Timestamp(S.DATE_START)).days
    t_hi = (pd.Timestamp(S.CALIB_END) - pd.Timestamp(S.DATE_START)).days
    return max(0, t_lo), min(t_hi, t_days - 1)


def _window_passes(pass_table: pd.DataFrame, t_days: int) -> pd.DataFrame:
    """INV-03: pass rows inside 2012-01-20 .. 2021-12-31 (and the grid)."""
    if pass_table is None or not len(pass_table):
        return pass_table
    t = t_of_date_default(pass_table["solar_date"])
    lo, hi = _window(t_days)
    return pass_table[(t >= lo) & (t <= hi)]


def operational_days(t_days: int, platform_active=None) -> np.ndarray:
    """(T, P) INV-04 operational calendar (static timeline AND empirical
    platform_active_day when provided)."""
    from .solver import operational_calendar
    return operational_calendar(t_days, platform_active)


def clear_month_matrix(cells, t_days: int, cloud_df, clim_df, op,
                       batch: int = 2000) -> np.ndarray:
    """(n_cells, 12) bool: the calibration-window calendar month contains
    at least one CLEAR operational pass day (max_p op * v >= 0.5) -- the
    "clear non-detection day" evidence for dormant cell-months."""
    lo, hi = _window(t_days)
    cells = np.asarray(cells, dtype=np.int64)
    out = np.zeros((len(cells), 12), dtype=bool)
    if hi < lo or not len(cells):
        return out
    dates = pd.Timestamp(S.DATE_START) + pd.to_timedelta(
        np.arange(lo, hi + 1), unit="D")
    months = dates.month.to_numpy() - 1
    t_idx = np.arange(lo, hi + 1)
    post = t_idx >= DRIFT_START_T
    for b0 in range(0, len(cells), batch):
        cb = cells[b0:b0 + batch]
        ll = np.array([h3_cell_center(int(c)) for c in cb], dtype=np.float64)
        cloud, _ = cloud_fraction_matrix(ll[:, 0], ll[:, 1], dates, cloud_df,
                                         clim_df)
        clear = np.clip(1.0 - cloud, *S.VIS_CLIP)
        best = np.zeros_like(clear)
        for p, pi in S.PLATFORM_INDEX.items():
            opp = op[lo:hi + 1, pi] > 0
            if not opp.any():
                continue
            ps = np.where(post[None, :],
                          p_swath(p, np.abs(ll[:, 0]), True)[:, None],
                          p_swath(p, np.abs(ll[:, 0]), False)[:, None])
            best = np.maximum(best, np.where(opp[None, :], ps * clear, 0.0))
        good = best >= 0.5
        for m in range(12):
            sel = months == m
            if sel.any():
                out[b0:b0 + len(cb), m] = good[:, sel].any(axis=1)
    return out


# ---------------------------------------------------------------------------
# Prior estimators
# ---------------------------------------------------------------------------
def estimate_f_event(pass_table: pd.DataFrame) -> dict:
    """Fbar_event,b = max(1.0, median clear-sky near-nadir VIIRS cluster
    FRP) over the 2012-2021 window (callers pass window rows)."""
    p = pass_table[(pass_table["sensor_family"] == "VIIRS")
                   & (pass_table["y_frp_mw"] > 0)
                   & (pass_table["mean_vza_deg"] < 25.0)].copy()
    if "v_cloud" in p.columns:
        clear = p["v_cloud"] >= S.PSWATH_VIIRS * 0.70     # cloud_frac < 0.30
        if int(clear.sum()) >= 8:
            p = p[clear]
    out = {}
    if p.empty:
        return {0: 1.0}
    p["cluster_frp"] = p["y_frp_mw"] / np.maximum(
        p["raw_event_count"].clip(lower=1), 1)
    for b, g in p.groupby("pyrome_id"):
        out[int(b)] = max(S.F_EVENT_FLOOR_MW,
                          float(np.median(g["cluster_frp"])))
    return out or {0: 1.0}


def estimate_mu_gm(days: pd.DataFrame, pyrome_of: dict,
                   shrink: float = S.SHRINKAGE_WEIGHT):
    """Cell-month Empirical Bayes shrinkage toward the pyrome-month mean.

    ``days``: one row per (h3_index, t, dn_flag) calibration-window DAY with
    ``z`` = mean implied z if any platform detected, else ``z_min`` for a
    clear non-detection day (dormant evidence).  ``ybar`` is the
    detected-day cell-month mean; cell-months with only clear non-detection
    days are dormant (``ybar = z_min``, ``n = 0``) and anchor the pyrome
    baseline low.
    """
    if days is None or days.empty:
        return pd.DataFrame(columns=["h3_index", "month", "dn_flag", "mu",
                                     "n_days", "ybar", "pyrome"]), {}
    days = days.copy()
    if "month" not in days.columns:
        days["month"] = _month_of_t(days["t"])
    days["pyrome"] = days["h3_index"].map(
        lambda c: pyrome_of.get(int(c), 0)).astype(int)
    det_days = days[days["z"] > S.Z_MIN + 1e-9]
    cell_month = det_days.groupby(["h3_index", "month", "dn_flag"],
                                  as_index=False).agg(
        ybar=("z", "mean"), n_days=("z", "size"), pyrome=("pyrome", "first"))
    all_days = days.groupby(["h3_index", "month", "dn_flag"],
                            as_index=False).agg(pyrome=("pyrome", "first"))
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


def build_prior_days(zs_window: pd.DataFrame, cells, clear_cm: np.ndarray
                     ) -> pd.DataFrame:
    """Day table for :func:`estimate_mu_gm`: every detected window day plus
    one representative clear non-detection row for each dormant
    (cell, month, dn) (sufficient statistic: dormant groups only need to
    exist; their ybar is z_min and n = 0)."""
    cells = np.asarray(cells, dtype=np.int64)
    det = zs_window[["h3_index", "t", "dn_flag", "z_imp"]].rename(
        columns={"z_imp": "z"}).copy()
    det["month"] = _month_of_t(det["t"])
    ci, mi = np.nonzero(clear_cm)
    rows = []
    for dn in ("D", "N"):
        rows.append(pd.DataFrame({"h3_index": cells[ci], "t": -1,
                                  "dn_flag": dn, "z": S.Z_MIN,
                                  "month": mi + 1}))
    dormant = pd.concat(rows, ignore_index=True) if rows else det.iloc[:0]
    have = det[["h3_index", "month", "dn_flag"]].drop_duplicates()
    dormant = dormant.merge(have, on=["h3_index", "month", "dn_flag"],
                            how="left", indicator=True)
    dormant = dormant[dormant["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([det, dormant], ignore_index=True)


def estimate_ar1(pass_table: pd.DataFrame, pyrome_of: dict):
    """rho_b (lag-1 OLS slope) and sigma_proc,b (residual std) on
    CONSECUTIVE active cell-days (t == t_prev + 1 within the same cell and
    half-day state; gaps never pair across dormant spells)."""
    zs = implied_z(pass_table, t_of_date_default)
    rho: dict = {}
    sig: dict = {}
    if zs.empty:
        return {0: 0.5}, {0: 0.5}
    zs = zs.sort_values(["h3_index", "dn_flag", "t"]).reset_index(drop=True)
    grp = zs.groupby(["h3_index", "dn_flag"], sort=False)
    t_prev = grp["t"].shift(1)
    z_prev = grp["z_imp"].shift(1)
    m = z_prev.notna() & (zs["t"] == t_prev + 1)
    zs["pyrome"] = zs["h3_index"].map(lambda c: pyrome_of.get(int(c), 0))
    pairs = pd.DataFrame({"pyrome": zs["pyrome"][m], "x": z_prev[m],
                          "y": zs["z_imp"][m]})
    for b in sorted(set(int(v) for v in zs["pyrome"].unique())):
        g = pairs[pairs["pyrome"] == b]
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        if len(x) < 8:
            rho[b], sig[b] = 0.5, 0.5
            continue
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        rho[b] = float(np.clip(coef[0], *S.RHO_CLIP))
        sig[b] = float(np.clip(np.sqrt(resid.var(ddof=1))
                               if len(resid) > 1 else 0.5,
                               *S.SIGMA_PROC_CLIP))
    return rho or {0: 0.5}, sig or {0: 0.5}


def _month_of_t(t) -> np.ndarray:
    """Calendar month (1..12) of solar-day indices t."""
    d = pd.DatetimeIndex(pd.Timestamp(S.DATE_START)
                         + pd.to_timedelta(np.asarray(t, dtype=np.int64),
                                           unit="D"))
    return d.month.to_numpy()


def t_of_date_default(dates):
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return ((d - pd.Timestamp(S.DATE_START)).days).to_numpy(dtype=int)


def estimate_lambda0(det_rows: pd.DataFrame, n_rows_by_pyrome: dict,
                     psi_cnt_fn, f_event: dict, sigma2_op: float = 0.25,
                     sigma_proc: dict | None = None):
    """lambda0_b in [0.1, 5.0] matching the empirical zero-detection
    probability on operational calibration-window passes (minimize_scalar).

    ``det_rows``: detected (cell, t, dn, platform) rows with ``z`` (day-mean
    implied z), ``vis`` (that platform's clear-sky visibility) and
    ``pyrome``; ``n_rows_by_pyrome``: total operational (cell, t, dn,
    platform) rows per pyrome.  Horvitz-Thompson detection-rate matching:
    the detected rows with inverse-detection-probability weights
    ``1 / (v (1 - e^{-Lambda}))`` represent all rows, with the weight
    marginalized over the implied-z noise by Gauss-Hermite quadrature.
    """
    from scipy.special import roots_hermite
    out = {}
    if det_rows is None or det_rows.empty:
        return {0: 1.0}
    gh_x, gh_w = roots_hermite(9)
    gh_w = gh_w / np.sqrt(np.pi)
    for b, g in det_rows.groupby("pyrome"):
        n_rows = float(n_rows_by_pyrome.get(int(b), 0))
        n_det = float(len(g))
        if n_rows <= 0 or n_det == 0:
            out[int(b)] = S.LAMBDA0_CLIP[0]
            continue
        emp_p0 = float(1.0 - min(n_det / n_rows, 1.0))
        z_det = g["z"].to_numpy(dtype=float)
        v_det = np.clip(g["vis"].to_numpy(dtype=float), 0.02, 1.0)
        psi_det = np.asarray(psi_cnt_fn(b, g), dtype=float)
        fe = max(f_event.get(int(b), 1.0), 1e-6)
        sp = (sigma_proc or {}).get(int(b), None)
        sigma_op = float(max(np.sqrt(max(sigma2_op, 1e-4)),
                             float(sp) if sp else 0.0))

        def weight_marg(lam_coef, z_obs, v_obs):
            tot = np.zeros_like(z_obs)
            for xk, wk in zip(gh_x, gh_w):
                zz = z_obs - sigma_op * np.sqrt(2.0) * xk
                p1 = np.clip(1.0 - np.exp(-lam_coef * np.exp(zz)), 1e-9, 1.0)
                tot += wk / np.maximum(v_obs * p1, 1e-9)
            return tot

        def loss(log_lam):
            lam0 = np.exp(log_lam)
            w = weight_marg(lam0 * psi_det / fe, z_det, v_det)
            p0_model = 1.0 - n_det / np.sum(w)
            return (p0_model - emp_p0) ** 2

        res = minimize_scalar(loss, bounds=(np.log(S.LAMBDA0_CLIP[0]),
                                            np.log(S.LAMBDA0_CLIP[1])),
                              method="bounded", options={"xatol": 1e-4})
        grid = np.exp(np.linspace(np.log(S.LAMBDA0_CLIP[0]),
                                  np.log(S.LAMBDA0_CLIP[1]), 41))
        best_loss, best_lam = np.inf, 1.0
        for lam0 in np.concatenate([grid, [float(np.exp(res.x))]]):
            l = loss(np.log(lam0))
            if np.isfinite(l) and l < best_loss:
                best_loss, best_lam = float(l), float(lam0)
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
    zs["month"] = _month_of_t(zs["t"])
    mu_s = pd.Series(mu_map, dtype=float) if mu_map else pd.Series(dtype=float)
    if len(mu_s):
        mu_s.index = pd.MultiIndex.from_tuples(mu_s.index)
        key = pd.MultiIndex.from_arrays([zs["h3_index"].astype(np.int64),
                                         zs["month"].astype(np.int64),
                                         zs["dn_flag"]])
        zs["mu"] = mu_s.reindex(key).fillna(S.Z_MIN).to_numpy()
    else:
        zs["mu"] = S.Z_MIN
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
                    psi_cnt_fn=None, platform_active=None):
    """Full deterministic Empirical Bayes prior estimation.

    EVERY estimator sees only 2012-01-20 .. 2021-12-31 rows (INV-03) and
    only operational passes (INV-04: static timeline AND
    ``platform_active_day``).  Memory is O(#detections + #cells x 12):
    clear non-detection evidence enters through per-(cell, month) clear-day
    indicators and per-pyrome operational row counts, never through a dense
    (cell x platform x day) table.

    ``psi_cnt_fn(b, frame) -> psi array`` supplies the analytical survival
    integral used by the lambda0 fit (``frame`` has platform, dn_flag).
    """
    pyrome_of = dict(zip(hex_pyromes["h3_index"].astype(np.int64),
                         hex_pyromes["pyrome_id"].astype(int)))
    if psi_cnt_fn is None:
        def psi_cnt_fn(b, g):
            return np.full(len(g), 0.5)
    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells))),
                     dtype=np.int64)
    op = operational_days(t_days, platform_active)
    win = _window_passes(pass_table, t_days)
    win = win[win["h3_index"].isin(set(cells.tolist()))]

    f_event = estimate_f_event(win)
    zs = implied_z(win, t_of_date_default)

    # ---- mu_{g,m,DN}: detected days + dormant clear cell-months ----------
    clear_cm = clear_month_matrix(cells, t_days, cloud_df, clim_df, op)
    days = build_prior_days(zs, cells, clear_cm)
    mu_gm, mu_pyr = estimate_mu_gm(days, pyrome_of)
    mu_map = {(int(r.h3_index), int(r.month), r.dn_flag): float(r.mu)
              for r in mu_gm.itertuples()} if len(mu_gm) else {}

    rho, sigma_proc = estimate_ar1(win, pyrome_of)
    beta_dn, eta_dn = estimate_dn_coupling(win, pyrome_of, mu_map)
    sigma2_op = 0.25
    if "sigma2" in getattr(win, "columns", []) and len(win):
        sigma2_op = float(np.nanmedian(win["sigma2"].to_numpy(dtype=float)))

    # ---- lambda0_b: HT matching over operational window passes ----------
    lo, hi = _window(t_days)
    n_op_rows_per_cell = 2.0 * float(op[lo:hi + 1].sum()) if hi >= lo else 0.0
    pyr_cells = pd.Series([pyrome_of.get(int(c), 0) for c in cells])
    n_rows_by_pyrome = {int(b): n * n_op_rows_per_cell
                        for b, n in pyr_cells.value_counts().items()}
    det = win[win["y_frp_mw"] > 0].copy()
    if len(det):
        det["t"] = t_of_date_default(det["solar_date"])
        pidx = det["platform"].map(S.PLATFORM_INDEX).to_numpy()
        det = det[op[det["t"].to_numpy(), pidx.astype(int)] > 0]
        det = det.merge(zs[["h3_index", "t", "dn_flag", "z_imp"]],
                        on=["h3_index", "t", "dn_flag"], how="left")
        det["z"] = det["z_imp"]
        if "v_cloud" in det.columns:
            det["vis"] = det["v_cloud"].to_numpy(dtype=float)
        else:
            from ..calibration.operators import pass_visibility
            det["vis"] = pass_visibility(det, cloud_df, clim_df)
        det["pyrome"] = det["h3_index"].map(
            lambda c: pyrome_of.get(int(c), 0)).astype(int)
    lambda0 = estimate_lambda0(det, n_rows_by_pyrome, psi_cnt_fn, f_event,
                               sigma2_op=sigma2_op, sigma_proc=sigma_proc)
    cells_p, colors, indptr, indices, n_edges = pack_neighbor_csr(cells)

    return {
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
        "cells": cells_p,
        "colors": colors,
        "indptr": indptr,
        "indices": indices,
        "n_edges": n_edges,
        "t_days": t_days,
    }
