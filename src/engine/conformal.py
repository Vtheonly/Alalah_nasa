"""Split Conformal Quantile Rescaling (CQR), diagnostic flags, event indices.

Contract (readme Sections 4.8, 5.4):
* Multiplicative log-space CQR nonconformity scores
  ``s_j = max((ln(q95+1) - ln(q05+1)) / 2, 0.05)`` and
  ``r_j = |ln(Phi_ref+1) - ln(q50+1)| / s_j``; factors
  ``c_hat = Quantile_{ceil((n+1)(1-alpha))/n}({r_j}) in [0.85, 3.5]``
  computed separately for the MODIS-only (2018-2020 VIIRS-masked) and
  multi-sensor (2018-2021 leave-one-VIIRS-platform-out) folds.
* Rescaled intervals ``q05_cal = max(0, exp(ln(q50+1) - c s) - 1)`` and
  ``q95_cal = exp(ln(q50+1) + c s) - 1``.
* ``energy_equiv_events_q50 = phi_q50 / F_event_b``; ``hurdle_adj_events``
  averaging across detecting platforms per half-day (never summing
  SNPP+N20+N21, INV-02) and scaling blind days by ``1 / max(measured_fraction,
  0.25)`` capped at ``5 x raw_clustered_events``.
* ``quality_flag`` (ABSTAIN_LOW_EVIDENCE / CLIMATOLOGY_CLOUD_FALLBACK /
  NOMINAL), ``attribution_flag`` decision tree, ``active_platforms_bitmask``,
  and 2003-2020 climatology indices (hist median, anomaly ratio, z-score,
  percentile).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings as S


# ---------------------------------------------------------------------------
# Split conformal calibration
# ---------------------------------------------------------------------------
def nonconformity_scores(q05, q50, q95, phi_ref):
    """(s_j, r_j) multiplicative log-space CQR scores."""
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    phi_ref = np.asarray(phi_ref, dtype=float)
    s = np.maximum((np.log(q95 + 1.0) - np.log(q05 + 1.0)) / 2.0,
                   S.CONFORMAL_S_FLOOR)
    r = np.abs(np.log(phi_ref + 1.0) - np.log(q50 + 1.0)) / s
    return s, r


def conformal_c_factor(r, alpha: float = 0.10) -> float:
    """Quantile_{ceil((n+1)(1-alpha))/n}({r_j}) clipped to [0.85, 3.5]."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return S.CONFORMAL_C_CLIP[0]
    k = int(np.ceil((len(r) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(r))
    c = float(np.sort(r)[k - 1])
    return float(np.clip(c, *S.CONFORMAL_C_CLIP))


def calibrate_intervals(q05, q50, q95, c_factor, s=None):
    """Rescale intervals: q05_cal = max(0, exp(ln(q50+1) - c s) - 1),
    q95_cal = exp(ln(q50+1) + c s) - 1."""
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    if s is None:
        s, _ = nonconformity_scores(q05, q50, q95, q50)
    q05_cal = np.maximum(0.0, np.exp(np.log(q50 + 1.0) - c_factor * s) - 1.0)
    q95_cal = np.exp(np.log(q50 + 1.0) + c_factor * s) - 1.0
    # enforce ordering: q05_cal <= q50 <= q95_cal
    q05_cal = np.minimum(q05_cal, q50)
    q95_cal = np.maximum(q95_cal, q50)
    return q05_cal, q95_cal


def conformal_multipliers(fold_monthly, phi_ref_by_key, alpha: float = 0.10,
                          group_col: str = "pyrome_id"):
    """Fit per-group c factors from a held-out fold run.

    ``fold_monthly``: monthly rows with (h3_index, year, month, pyrome_id,
    phi_q05_mw, phi_q50_mw, phi_q95_mw).  ``phi_ref_by_key``:
    {(h3_index, year, month): Phi_ref} from the full multi-sensor run.
    """
    out = {}
    if fold_monthly is None or not len(fold_monthly):
        return out
    df = fold_monthly.copy()
    df["phi_ref"] = [phi_ref_by_key.get((int(h), int(y), int(m)), np.nan)
                     for h, y, m in zip(df["h3_index"], df["year"],
                                        df["month"])]
    df = df.dropna(subset=["phi_ref"])
    for b, g in df.groupby(group_col):
        _, r = nonconformity_scores(g["phi_q05_mw"], g["phi_q50_mw"],
                                    g["phi_q95_mw"], g["phi_ref"])
        out[int(b)] = conformal_c_factor(r, alpha)
    return out


def apply_conformal(monthly: pd.DataFrame, c_modis: dict, c_multi: dict,
                    modis_only_mask) -> pd.DataFrame:
    """Apply the era-appropriate conformal multipliers to monthly intervals
    (vectorized).

    ``modis_only_mask``: boolean mask of MODIS-only-era rows (pre-2012-01-20)
    which use ``c_modis``; multi-sensor rows use ``c_multi``.
    """
    out = monthly.copy()
    b = (out["pyrome_id"].to_numpy(dtype=int) if "pyrome_id" in out.columns
         else np.zeros(len(out), dtype=int))
    era = np.asarray(modis_only_mask, dtype=bool)
    cm = pd.Series(c_modis, dtype=float)
    cx = pd.Series(c_multi, dtype=float)
    c_m = cm.reindex(b).fillna(S.CONFORMAL_C_CLIP[0]).to_numpy()
    c_x = cx.reindex(b).fillna(S.CONFORMAL_C_CLIP[0]).to_numpy()
    c = np.where(era, c_m, c_x)
    s, _ = nonconformity_scores(out["phi_q05_mw"], out["phi_q50_mw"],
                                out["phi_q95_mw"], out["phi_q50_mw"])
    q05c, q95c = calibrate_intervals(out["phi_q05_mw"], out["phi_q50_mw"],
                                     out["phi_q95_mw"], c, s)
    out["phi_q05_raw_mw"] = out["phi_q05_mw"].to_numpy()
    out["phi_q95_raw_mw"] = out["phi_q95_mw"].to_numpy()
    out["phi_q05_mw"] = q05c
    out["phi_q95_mw"] = q95c
    out["conformal_c_factor"] = c
    return out


# ---------------------------------------------------------------------------
# Event indices (INV-02: never sum same-orbit platforms)
# ---------------------------------------------------------------------------
def energy_equiv_events(phi_q50: np.ndarray, f_event_b) -> np.ndarray:
    """energy_equiv_events_q50 = phi_q50 / F_event_b."""
    return np.asarray(phi_q50, dtype=float) / np.maximum(
        np.asarray(f_event_b, dtype=float), 1e-6)


def hurdle_adjusted_events(pass_rows, psi_cnt_by_platform, vis_by_platform):
    """E_halfday = mean over detecting platforms of E_raw / (psi_CNT * v).

    ``pass_rows``: list of per-platform dicts {platform, raw_event_count,
    area, dn}.  Never sums SNPP + N20 + N21 (INV-02).
    """
    if not pass_rows:
        return 0.0
    vals = []
    for r in pass_rows:
        psi = psi_cnt_by_platform(r["platform"], r["area"], r["dn"])
        v = max(float(vis_by_platform(r["platform"])), 0.10)
        vals.append(r["raw_event_count"] / max(psi * v, 1e-6))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Diagnostic flags
# ---------------------------------------------------------------------------
def quality_flag_month(cloud_frac_daily: np.ndarray, detected_daily: np.ndarray,
                       cloud_source: str = "CMG_8D") -> str:
    """'ABSTAIN_LOW_EVIDENCE' if >= 5 consecutive days have
    mean_cloud_frac > 0.85 AND zero detections; else
    'CLIMATOLOGY_CLOUD_FALLBACK' if cloud came from climatology; 'NOMINAL'."""
    cloud = np.asarray(cloud_frac_daily, dtype=float)
    det = np.asarray(detected_daily, dtype=bool)
    run = 0
    for c, d in zip(cloud, det):
        if c > S.ABSTAIN_CLOUD_FRAC and not d:
            run += 1
            if run >= S.ABSTAIN_CONSEC_DAYS:
                return "ABSTAIN_LOW_EVIDENCE"
        else:
            run = 0
    if cloud_source == "CLIMATOLOGY_FALLBACK" or \
            cloud_source == "CLIMATOLOGY_CLOUD_FALLBACK":
        return "CLIMATOLOGY_CLOUD_FALLBACK"
    return "NOMINAL"


def attribution_flag(raw_delta: float, q05_cal: float, q95_cal: float,
                     harm_hist_med: float, quality: str) -> str:
    """Readme 5.4.3 decision tree (exact boolean rules)."""
    if quality == "ABSTAIN_LOW_EVIDENCE":
        return "AMBIGUOUS_OVERLAPPING_CI"
    if not np.isfinite(harm_hist_med):
        return "WITHIN_NORMAL_VARIABILITY"
    if q05_cal > harm_hist_med or q95_cal < harm_hist_med:
        return "REAL_FIRE_CHANGE"
    if abs(raw_delta) > 0.50 and q05_cal <= harm_hist_med <= q95_cal:
        return "SENSOR_ARTIFACT_CORRECTED"
    return "WITHIN_NORMAL_VARIABILITY"


def active_platforms_bitmask(platforms) -> int:
    """Bit 0 (1) = Terra, 1 (2) = Aqua, 2 (4) = SNPP, 3 (8) = NOAA-20,
    4 (16) = NOAA-21."""
    mask = 0
    for p in platforms:
        pi = S.PLATFORM_INDEX.get(p)
        if pi is not None:
            mask |= S.PLATFORM_BITMASK[pi]
    return mask


# ---------------------------------------------------------------------------
# Climatology indices (2003-2020)
# ---------------------------------------------------------------------------
def climatology_indices(monthly: pd.DataFrame) -> pd.DataFrame:
    """hist_median_2003_2020, anomaly_ratio, anomaly_zscore, climatology_pctile
    per (h3_index, month) -- vectorized."""
    out = monthly.copy()
    clim = out[(out["year"] >= S.CLIM_START_YEAR)
               & (out["year"] <= S.CLIM_END_YEAR)]
    g = clim.groupby(["h3_index", "month"])["phi_q50_mw"]
    stats = pd.DataFrame({"hist_median_2003_2020": g.median(),
                          "_hist_mean": g.mean(),
                          "_hist_std": g.std(ddof=0)}).reset_index()
    out = out.merge(stats, on=["h3_index", "month"], how="left")
    out["hist_median_2003_2020"] = out["hist_median_2003_2020"].fillna(0.0)
    out["_hist_mean"] = out["_hist_mean"].fillna(out["phi_q50_mw"])
    out["_hist_std"] = out["_hist_std"].fillna(0.0)
    out["anomaly_ratio"] = out["phi_q50_mw"] / np.maximum(
        out["hist_median_2003_2020"], 0.5)
    out["anomaly_zscore"] = np.where(
        out["_hist_std"] > 1e-9,
        (out["phi_q50_mw"] - out["_hist_mean"])
        / out["_hist_std"].where(out["_hist_std"] > 1e-9, 1.0), 0.0)
    # percentile of the value within its cell-month 2003-2020 distribution:
    # 100 * #{hist <= value} / #hist  (50 when no climatology exists)
    hist = clim[["h3_index", "month", "phi_q50_mw"]].sort_values(
        ["h3_index", "month", "phi_q50_mw"])
    key_h = hist["h3_index"].to_numpy(np.int64) * 16 + hist["month"].to_numpy(
        np.int64)
    vals = hist["phi_q50_mw"].to_numpy(dtype=float)
    # composite sortable key: group offset + rank inside the group
    ukey, start, counts = np.unique(key_h, return_index=True,
                                    return_counts=True)
    key_o = out["h3_index"].to_numpy(np.int64) * 16 + out["month"].to_numpy(
        np.int64)
    pos = np.searchsorted(ukey, key_o)
    found = (pos < len(ukey)) & (ukey[np.minimum(pos, len(ukey) - 1)]
                                 == key_o) if len(ukey) else np.zeros(
        len(out), dtype=bool)
    pct = np.full(len(out), 50.0)
    if found.any():
        v = out["phi_q50_mw"].to_numpy(dtype=float)
        # per-row searchsorted inside its group (groups are sorted)
        idx = np.nonzero(found)[0]
        st = start[pos[idx]]
        ct = counts[pos[idx]]
        order = np.argsort(pos[idx], kind="stable")
        res = np.empty(len(idx))
        gpos = pos[idx][order]
        bounds = np.flatnonzero(np.diff(gpos)) + 1
        for seg in np.split(np.arange(len(order)), bounds):
            rows = order[seg]
            s0, c0 = st[rows[0]], ct[rows[0]]
            arr = vals[s0:s0 + c0]
            res[rows] = 100.0 * np.searchsorted(arr, v[idx[rows]],
                                                side="right") / c0
        pct[idx] = res
    out["climatology_pctile"] = pct
    return out.drop(columns=["_hist_mean", "_hist_std"])


# ---------------------------------------------------------------------------
# Held-out conformal folds (readme 5.4.1)
# ---------------------------------------------------------------------------
def run_conformal_folds(solve_fn, monthly_full: pd.DataFrame,
                        hex_pyromes: pd.DataFrame, alpha: float = 0.10):
    """Fit ``c_modis[b]`` and ``c_multi[b]`` from held-out re-solves.

    ``solve_fn(platform_mask, keep_years) -> monthly frame`` re-runs the
    full GMRF solve with a :class:`~src.engine.solver.PlatformMask`.

    * MODIS-only: all VIIRS platforms masked over 2018-01-01..2020-12-31,
      scored on 2018-2020 months against the full multi-sensor reference.
    * Multi-sensor: leave-one-VIIRS-platform-out (S-NPP held out, NOAA-20
      kept) over 2018-01-01..2021-12-31, scored on 2018-2021 months.
    """
    from .solver import PlatformMask
    t0 = pd.Timestamp(S.DATE_START)
    t18 = (pd.Timestamp("2018-01-01") - t0).days
    t20 = (pd.Timestamp("2020-12-31") - t0).days
    t21 = (pd.Timestamp("2021-12-31") - t0).days
    pyr = hex_pyromes[["h3_index", "pyrome_id"]].drop_duplicates("h3_index")
    ref = monthly_full.set_index(["h3_index", "year", "month"])["phi_q50_mw"]
    ref_by_key = ref.to_dict()
    viirs = [S.PLATFORM_INDEX[p] for p in S.VIIRS_PLATFORMS]
    fm = solve_fn(PlatformMask(viirs, t18, t20), (2018, 2020))
    fm = fm.merge(pyr, on="h3_index", how="left")
    fm["pyrome_id"] = fm["pyrome_id"].fillna(0).astype(int)
    c_modis = conformal_multipliers(
        fm[(fm["year"] >= 2018) & (fm["year"] <= 2020)], ref_by_key, alpha)
    fl = solve_fn(PlatformMask([S.PLATFORM_INDEX["SNPP"]], t18, t21),
                  (2018, 2021))
    fl = fl.merge(pyr, on="h3_index", how="left")
    fl["pyrome_id"] = fl["pyrome_id"].fillna(0).astype(int)
    c_multi = conformal_multipliers(
        fl[(fl["year"] >= 2018) & (fl["year"] <= 2021)], ref_by_key, alpha)
    return c_modis, c_multi, fm, fl


# ---------------------------------------------------------------------------
# Monthly diagnostics shared by every pipeline entry point
# ---------------------------------------------------------------------------
def _ym_of(dates) -> tuple[np.ndarray, np.ndarray]:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return d.year.to_numpy(), d.month.to_numpy()


def raw_baselines(pass_obs: pd.DataFrame) -> pd.DataFrame:
    """Raw (unharmonized) per cell-month baselines: counts / FRP / events."""
    cols = ["h3_index", "year", "month", "raw_count_sum", "raw_frp_sum_mw",
            "raw_clustered_events"]
    if pass_obs is None or not len(pass_obs):
        return pd.DataFrame(columns=cols)
    y, m = _ym_of(pass_obs["solar_date"])
    p = pd.DataFrame({"h3_index": pass_obs["h3_index"].to_numpy(np.int64),
                      "year": y, "month": m,
                      "n": pass_obs["n_pixels"].to_numpy(dtype=float)
                      if "n_pixels" in pass_obs else 1.0,
                      "frp": pass_obs["y_frp_mw"].to_numpy(dtype=float),
                      "ev": pass_obs["raw_event_count"].to_numpy(dtype=float)})
    g = p.groupby(["h3_index", "year", "month"], as_index=False).agg(
        raw_count_sum=("n", "sum"), raw_frp_sum_mw=("frp", "sum"),
        raw_clustered_events=("ev", "sum"))
    return g[cols]


def cell_month_evidence(cells, pass_cal: pd.DataFrame, cloud_df, clim_df,
                        op: np.ndarray, neighbors: dict, t_days: int,
                        batch: int = 1000) -> pd.DataFrame:
    """Per (cell, year, month) evidence diagnostics, batched over cells:

    * ``mean_cloud_frac``: mean daily 0.25 deg cloud fraction.
    * ``cloud_fallback``: any day used the 2012-2025 climatology (or no
      cloud information at all).
    * ``abstain``: >= 5 consecutive days with cloud_frac > 0.85 AND zero
      detections in the cell and its 1-ring neighbours (readme 5.4.3).
    * ``measured_fraction``: mean over days of P(>= 1 clear operational
      pass) = 1 - prod_p (1 - op_p v_p), or 1 on detection days.
    * ``active_platforms_bitmask``: OR of the bits of platforms operational
      on any day of the month (INV-04 calendar).
    """
    from .priors import cloud_fraction_matrix
    from ..geometry.h3_utils import h3_cell_center
    cells = np.asarray(cells, dtype=np.int64)
    t0 = pd.Timestamp(S.DATE_START)
    dates = t0 + pd.to_timedelta(np.arange(t_days), unit="D")
    ym = dates.year.to_numpy() * 12 + dates.month.to_numpy() - 1
    uym, mi = np.unique(ym, return_inverse=True)
    n_m = len(uym)
    month_start = np.r_[True, mi[1:] != mi[:-1]]
    cnt = np.bincount(mi, minlength=n_m).astype(float)
    post = np.arange(t_days) >= (pd.Timestamp(S.DRIFT_START) - t0).days
    # detection days per cell (any platform, operational)
    det_t = {}
    if pass_cal is not None and len(pass_cal):
        dp = pass_cal[pass_cal["y_frp_mw"] > 0]
        tt_ = (pd.to_datetime(dp["solar_date"]) - t0).dt.days.to_numpy()
        hh = dp["h3_index"].to_numpy(np.int64)
        ok = (tt_ >= 0) & (tt_ < t_days)
        for h, t in zip(hh[ok], tt_[ok]):
            det_t.setdefault(int(h), set()).add(int(t))
    # bitmask per month (same for every cell)
    bits = np.zeros(n_m, dtype=np.int64)
    for p, pi in S.PLATFORM_INDEX.items():
        on = np.bincount(mi, weights=(op[:, pi] > 0).astype(float),
                         minlength=n_m) > 0
        bits |= np.where(on, S.PLATFORM_BITMASK[pi], 0)
    frames = []
    for b0 in range(0, len(cells), batch):
        cb = cells[b0:b0 + batch]
        n = len(cb)
        ll = np.array([h3_cell_center(int(c)) for c in cb], dtype=np.float64)
        cloud, src = cloud_fraction_matrix(ll[:, 0], ll[:, 1], dates,
                                           cloud_df, clim_df)
        det = np.zeros((n, t_days), dtype=bool)
        det_nb = np.zeros((n, t_days), dtype=bool)
        for i, c in enumerate(cb):
            ts = det_t.get(int(c))
            if ts:
                det[i, list(ts)] = True
            allt = set(ts or ())
            for nb in neighbors.get(int(c), ()):
                allt |= det_t.get(int(nb), set())
            if allt:
                det_nb[i, list(allt)] = True
        clear = np.clip(1.0 - cloud, *S.VIS_CLIP)
        miss_all = np.ones((n, t_days))
        for p, pi in S.PLATFORM_INDEX.items():
            if p in S.VIIRS_PLATFORMS:
                ps = np.full((n, 1), S.PSWATH_VIIRS)
                v = ps * clear
            else:
                base = np.clip(S.PSWATH_MODIS_BASE + S.PSWATH_MODIS_SLOPE
                               * np.abs(ll[:, 0]) / 30.0,
                               S.PSWATH_MODIS_BASE, 1.0)[:, None]
                v = np.where(post[None, :], base * S.PSWATH_MODIS_POST2022,
                             base) * clear
            miss_all *= 1.0 - (op[:, pi] > 0)[None, :] * v
        meas_day = np.where(det, 1.0, 1.0 - miss_all)
        # >= 5 consecutive blind days inside the month (runs reset monthly)
        cond = (cloud > S.ABSTAIN_CLOUD_FRAC) & (~det_nb)
        c_int = cond.astype(np.int64)
        cs = np.cumsum(c_int, axis=1)
        reset = (~cond) | month_start[None, :]
        base = np.where(reset, cs - c_int, 0)
        run = cs - np.maximum.accumulate(base, axis=1)
        run_m = np.zeros((n, n_m))
        for k in range(n):
            np.maximum.at(run_m[k], mi, run[k])
        sum_cloud = np.stack([np.bincount(mi, weights=cloud[k], minlength=n_m)
                              for k in range(n)])
        sum_meas = np.stack([np.bincount(mi, weights=meas_day[k],
                                         minlength=n_m) for k in range(n)])
        fb = np.stack([np.bincount(mi, weights=(src[k] >= 2).astype(float),
                                   minlength=n_m) for k in range(n)]) > 0
        frames.append(pd.DataFrame({
            "h3_index": np.repeat(cb, n_m),
            "year": np.tile(uym // 12, n),
            "month": np.tile(uym % 12 + 1, n),
            "mean_cloud_frac": (sum_cloud / cnt).ravel(),
            "measured_fraction": np.clip(sum_meas / cnt, 0.0, 1.0).ravel(),
            "abstain": (run_m >= S.ABSTAIN_CONSEC_DAYS).ravel(),
            "cloud_fallback": fb.ravel(),
            "active_platforms_bitmask": np.tile(bits, n),
        }))
    if not frames:
        return pd.DataFrame(columns=["h3_index", "year", "month",
                                     "mean_cloud_frac", "measured_fraction",
                                     "abstain", "cloud_fallback",
                                     "active_platforms_bitmask"])
    return pd.concat(frames, ignore_index=True)


def hurdle_events_monthly(pass_cal: pd.DataFrame, lognorm: dict
                          ) -> pd.DataFrame:
    """Sum over detected half-days of the platform-MEAN (never the sum,
    INV-02) of ``E_raw / (psi_CNT(A_p, DN) v_p)`` per cell-month."""
    from ..calibration.operators import psi_cnt
    cols = ["h3_index", "year", "month", "e_sum"]
    if pass_cal is None or not len(pass_cal):
        return pd.DataFrame(columns=cols)
    po = pass_cal[pass_cal["y_frp_mw"] > 0]
    psi = np.empty(len(po))
    fam = po["platform"].map(S.SENSOR_FAMILY).fillna("VIIRS").to_numpy()
    for f in ("MODIS", "VIIRS"):
        m = fam == f
        if m.any():
            psi[m] = psi_cnt(f, po["pyrome_id"].to_numpy()[m],
                             po["mean_area_km2"].to_numpy(dtype=float)[m],
                             po["dn_flag"].to_numpy()[m], lognorm)
    v = (po["v_cloud"].to_numpy(dtype=float) if "v_cloud" in po.columns
         else np.ones(len(po)))
    e = po["raw_event_count"].to_numpy(dtype=float) / np.maximum(
        psi * np.maximum(v, 0.10), 1e-6)
    y, mth = _ym_of(po["solar_date"])
    hd = pd.DataFrame({"h3_index": po["h3_index"].to_numpy(np.int64),
                       "solar_date": po["solar_date"].astype(str).to_numpy(),
                       "dn_flag": po["dn_flag"].to_numpy(), "year": y,
                       "month": mth, "e": e})
    half = hd.groupby(["h3_index", "solar_date", "dn_flag", "year", "month"],
                      as_index=False)["e"].mean()
    return half.groupby(["h3_index", "year", "month"], as_index=False).agg(
        e_sum=("e", "sum"))[cols]


def finalize_monthly(monthly: pd.DataFrame, *, pass_obs: pd.DataFrame,
                     pass_cal: pd.DataFrame, prov: pd.DataFrame,
                     hex_pyromes: pd.DataFrame, priors: dict,
                     artifacts: dict, cloud_df, c_modis: dict, c_multi: dict,
                     op: np.ndarray, t_days: int) -> pd.DataFrame:
    """Turn posterior monthly statistics into the full Product 2/3 frame:
    conformal rescaling (era-specific c), 2003-2020 climatology indices,
    raw baselines, event indices, footprint, evidence diagnostics,
    quality / attribution flags, NRT gap-fill and revision provenance."""
    from ..calibration.operators import predict_burn_area
    m = monthly.copy()
    pyr = hex_pyromes[["h3_index", "pyrome_id"]].drop_duplicates("h3_index")
    if "pyrome_id" in m.columns:
        m = m.drop(columns=["pyrome_id"])
    m = m.merge(pyr, on="h3_index", how="left")
    m["pyrome_id"] = m["pyrome_id"].fillna(0).astype(int)
    era = (pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
           < pd.Timestamp(S.CALIB_START).replace(day=1)).to_numpy()
    m = apply_conformal(m, c_modis, c_multi, era)
    m = climatology_indices(m)
    m = m.merge(raw_baselines(pass_obs), on=["h3_index", "year", "month"],
                how="left")
    for c in ("raw_count_sum", "raw_frp_sum_mw", "raw_clustered_events"):
        m[c] = m[c].fillna(0.0)
    # country
    if prov is not None and len(prov) and "country_iso" in prov.columns:
        cc = prov.groupby("h3_index")["country_iso"].agg(
            lambda v: v.mode().iloc[0] if len(v.mode()) else None)
        m["country_iso"] = m["h3_index"].map(cc)
    else:
        m["country_iso"] = None
    # evidence diagnostics
    cells = np.unique(m["h3_index"].to_numpy(np.int64))
    nb = {}
    cl = np.asarray(priors["cells"], dtype=np.int64)
    for k, c in enumerate(cl):
        nb[int(c)] = cl[priors["indices"][priors["indptr"][k]:
                                          priors["indptr"][k + 1]]].tolist()
    ev = cell_month_evidence(cells, pass_cal, cloud_df,
                             artifacts.get("cloud_climatology"), op, nb,
                             t_days)
    m = m.merge(ev, on=["h3_index", "year", "month"], how="left")
    m["measured_fraction"] = m["measured_fraction"].fillna(0.0)
    m["inferred_fraction"] = (1.0 - m["measured_fraction"]).clip(0.0, 1.0)
    m["mean_cloud_frac"] = m["mean_cloud_frac"].fillna(0.5)
    m["active_platforms_bitmask"] = m["active_platforms_bitmask"].fillna(
        0).astype(int)
    m["quality_flag"] = np.where(
        m["abstain"].fillna(False).astype(bool), "ABSTAIN_LOW_EVIDENCE",
        np.where(m["cloud_fallback"].fillna(True).astype(bool),
                 "CLIMATOLOGY_CLOUD_FALLBACK", "NOMINAL"))
    m = m.drop(columns=["abstain", "cloud_fallback"])
    # event indices
    f_event = priors["f_event"]
    fe = pd.Series(f_event, dtype=float).reindex(m["pyrome_id"]).fillna(
        1.0).to_numpy()
    m["energy_equiv_events_q50"] = energy_equiv_events(m["phi_q50_mw"], fe)
    he = hurdle_events_monthly(pass_cal, artifacts["lognorm_params"])
    m = m.merge(he, on=["h3_index", "year", "month"], how="left")
    m["e_sum"] = m["e_sum"].fillna(0.0)
    m["hurdle_adj_events"] = np.minimum(
        m["e_sum"] / np.maximum(m["measured_fraction"],
                                S.MEASURED_FRACTION_FLOOR),
        S.HURDLE_EVENTS_CAP_FACTOR * m["raw_clustered_events"])
    m = m.drop(columns=["e_sum"])
    # footprint, density, night/day ratio
    hex_area = 36.13
    m["night_day_ratio"] = (m["phi_night_mw"]
                            / m["phi_day_mw"].clip(lower=1e-6)).fillna(0.0)
    m["est_active_area_km2"] = predict_burn_area(
        artifacts.get("area_model"),
        np.log(m["phi_q50_mw"].clip(lower=1e-3) + 0.01),
        m["night_day_ratio"].replace(0.0, 1.0), m["pyrome_id"], hex_area)
    m["phi_flux_mw_km2"] = m["phi_q50_mw"] / hex_area
    m["rel_uncertainty_width"] = ((m["phi_q95_mw"] - m["phi_q05_mw"])
                                  / m["phi_q50_mw"].clip(lower=1e-6))
    # attribution flag (readme 5.4.3 exact rules, vectorized)
    clim = m[(m["year"] >= S.CLIM_START_YEAR) & (m["year"] <= S.CLIM_END_YEAR)]
    raw_med = clim.groupby(["h3_index", "month"])["raw_frp_sum_mw"].median() \
        .rename("raw_hist_med").reset_index()
    m = m.merge(raw_med, on=["h3_index", "month"], how="left")
    rhm = m["raw_hist_med"].fillna(0.0)
    raw_delta = (m["raw_frp_sum_mw"] - rhm) / np.maximum(rhm, 1.0)
    hhm = m["hist_median_2003_2020"]
    real = (m["phi_q05_mw"] > hhm) | (m["phi_q95_mw"] < hhm)
    artifact = (raw_delta.abs() > 0.50) & (m["phi_q05_mw"] <= hhm) & (
        hhm <= m["phi_q95_mw"])
    m["attribution_flag"] = np.where(
        m["quality_flag"] == "ABSTAIN_LOW_EVIDENCE",
        "AMBIGUOUS_OVERLAPPING_CI",
        np.where(real, "REAL_FIRE_CHANGE",
                 np.where(artifact, "SENSOR_ARTIFACT_CORRECTED",
                          "WITHIN_NORMAL_VARIABILITY")))
    m = m.drop(columns=["raw_hist_med"])
    # provenance: NRT gap-fill + revision count
    m["has_nrt_gapfill"] = 0
    m["revision_count"] = 1
    if prov is not None and len(prov) and "data_tier" in prov.columns:
        y, mth = _ym_of(prov["solar_date"] if "solar_date" in prov.columns
                        else prov["acq_date"])
        pv = pd.DataFrame({"h3_index": prov["h3_index"].to_numpy(np.int64),
                           "year": y, "month": mth,
                           "nrt": (prov["data_tier"] == "NRT").to_numpy()})
        if "revision_count" in prov.columns:
            pv["rev"] = prov["revision_count"].fillna(1).to_numpy()
        else:
            pv["rev"] = 1
        g = pv.groupby(["h3_index", "year", "month"], as_index=False).agg(
            nrt=("nrt", "max"), rev=("rev", "max"))
        m = m.merge(g, on=["h3_index", "year", "month"], how="left")
        m["has_nrt_gapfill"] = m["nrt"].fillna(False).astype(int)
        m["revision_count"] = m["rev"].fillna(1).astype(int)
        m = m.drop(columns=["nrt", "rev"])
    return m


# ---------------------------------------------------------------------------
# Single Environment-2 harmonization path
# ---------------------------------------------------------------------------
def harmonize_and_calibrate(pass_cal, pass_obs, prov, hex_pyromes, priors, artifacts,
              cloud_df, platform_active, t_days: int = S.T_DAYS,
              batch_size: int = S.BATCH_SIZE, paths=None,
              keep_states: bool = True, workers_per_gpu: int = 1):
    """THE single Environment-2 harmonization path (shared by the twin and
    ``scripts/run_env2_gpu_solve.py``): main GMRF solve -> held-out
    conformal folds -> finalized monthly Product 2/3 frame + sparse daily.
    """
    from .solver import solve_all_chunks
    active_cells = np.array(sorted(set(int(c) for c in
                                       pass_cal["h3_index"].unique())),
                            dtype=np.int64)
    common = dict(t_days=t_days, platform_active=platform_active,
                  batch_size=batch_size, workers_per_gpu=workers_per_gpu)
    res = solve_all_chunks(pass_cal, active_cells, hex_pyromes, priors,
                               artifacts, cloud_df, paths=paths,
                               keep_states=keep_states, **common)

    def solve_fold(mask, years):
        fold = solve_all_chunks(pass_cal, active_cells, hex_pyromes,
                                    priors, artifacts, cloud_df,
                                    platform_mask=mask, daily_mode="none",
                                    keep_years=years, keep_states=False,
                                    **common)
        return fold["monthly"]

    c_modis, c_multi, fold_m, fold_l = run_conformal_folds(
        solve_fold, res["monthly"], hex_pyromes)
    monthly = finalize_monthly(
        res["monthly"], pass_obs=pass_obs, pass_cal=pass_cal, prov=prov,
        hex_pyromes=hex_pyromes, priors=priors, artifacts=artifacts,
        cloud_df=cloud_df, c_modis=c_modis, c_multi=c_multi, op=res["op"],
        t_days=t_days)
    return {"res": res, "monthly": monthly, "daily": res["daily"],
            "c_modis": c_modis, "c_multi": c_multi, "fold_modis": fold_m,
            "fold_loo": fold_l}
