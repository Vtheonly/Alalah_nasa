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
    """Apply the era-appropriate conformal multipliers to monthly intervals.

    ``modis_only_mask``: boolean Series marking MODIS-only-era rows
    (pre-2012-01-20) which use ``c_modis``; multi-sensor rows use ``c_multi``.
    """
    out = monthly.copy()
    q05c = np.empty(len(out))
    q95c = np.empty(len(out))
    c_used = np.empty(len(out))
    for i, row in enumerate(out.itertuples()):
        b = int(getattr(row, "pyrome_id", 0))
        c = c_modis.get(b, 0.85) if bool(modis_only_mask.iloc[i]) \
            else c_multi.get(b, 0.85)
        s, _ = nonconformity_scores(row.phi_q05_mw, row.phi_q50_mw,
                                    row.phi_q95_mw, row.phi_q50_mw)
        a, bnd = calibrate_intervals(row.phi_q05_mw, row.phi_q50_mw,
                                     row.phi_q95_mw, c, s)
        q05c[i] = a
        q95c[i] = bnd
        c_used[i] = c
    out["phi_q05_mw"] = q05c
    out["phi_q95_mw"] = q95c
    out["conformal_c_factor"] = c_used
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
    per (h3_index, month)."""
    out = monthly.copy()
    clim = (out[(out["year"] >= S.CLIM_START_YEAR)
                & (out["year"] <= S.CLIM_END_YEAR)]
            .groupby(["h3_index", "month"])["phi_q50_mw"])
    med = clim.median().rename("hist_median_2003_2020")
    mu = clim.mean().rename("_hist_mean")
    sd = clim.std(ddof=0).rename("_hist_std")
    out = out.merge(med, on=["h3_index", "month"], how="left")
    out = out.merge(mu, on=["h3_index", "month"], how="left")
    out = out.merge(sd, on=["h3_index", "month"], how="left")
    out["hist_median_2003_2020"] = out["hist_median_2003_2020"].fillna(0.0)
    out["_hist_mean"] = out["_hist_mean"].fillna(out["phi_q50_mw"])
    out["_hist_std"] = out["_hist_std"].fillna(0.0)
    out["anomaly_ratio"] = out["phi_q50_mw"] / np.maximum(
        out["hist_median_2003_2020"], 0.5)
    out["anomaly_zscore"] = np.where(
        out["_hist_std"] > 1e-9,
        (out["phi_q50_mw"] - out["_hist_mean"]) / out["_hist_std"], 0.0)
    # percentile of the current month within the 2003-2020 distribution
    pct = []
    hist = {}
    sub = out[(out["year"] >= S.CLIM_START_YEAR)
              & (out["year"] <= S.CLIM_END_YEAR)]
    for (h, m), g in sub.groupby(["h3_index", "month"]):
        hist[(int(h), int(m))] = np.sort(g["phi_q50_mw"].to_numpy())
    for r in out.itertuples():
        arr = hist.get((int(r.h3_index), int(r.month)))
        if arr is None or len(arr) == 0:
            pct.append(50.0)
        else:
            pct.append(100.0 * float(np.mean(arr <= r.phi_q50_mw)))
    out["climatology_pctile"] = pct
    return out.drop(columns=["_hist_mean", "_hist_std"])
