"""Scientific validation diagnostics (readme Sections 4.10, 7.9).

* Chow structural-break test and Pettitt non-parametric change point on the
  raw vs harmonized yearly series (2012 S-NPP transition must break raw
  p < 0.001 and pass harmonized p > 0.05).
* 2019-2021 VIIRS-masked ablation: log-RMSE, empirical CRPS, 90% coverage,
  Spearman rho, Top-3 Peak Month Match vs the raw MODIS baseline.
* MCD64CMQ burned-area correlation comparison (harmonized FRE vs raw counts).
* Residual VZA / solar-time slope checks.
* Spotlight reports (Chilecito ARG 2026, Kabylie & Hassi Messaoud DZA,
  Nov 2 2026 S-NPP end).
* Saves ``diagnostics.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Structural break tests
# ---------------------------------------------------------------------------
def chow_test(y: np.ndarray, break_idx: int) -> dict:
    """Chow test for a mean break at ``break_idx`` (F statistic + p-value)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 8 or not (3 <= break_idx <= n - 3):
        return {"F": float("nan"), "p": float("nan")}
    rss = float(np.sum((y - y.mean()) ** 2))
    a, b = y[:break_idx], y[break_idx:]
    rss1 = float(np.sum((a - a.mean()) ** 2))
    rss2 = float(np.sum((b - b.mean()) ** 2))
    k = 2.0
    f = ((rss - rss1 - rss2) / k) / ((rss1 + rss2) / (n - 2 * k))
    p = float(1.0 - stats.f.cdf(f, k, n - 2 * k))
    return {"F": float(f), "p": p}


def pettitt_test(y: np.ndarray) -> dict:
    """Pettitt non-parametric change point: max |U_t| + approximate p."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return {"K": 0.0, "p": 1.0, "idx": 0}
    u = np.zeros(n)
    for t in range(n):
        u[t] = 2.0 * np.sum(y[:t + 1] <= y[t]) - (t + 1) \
            if t > 0 else 1.0
    # standard Pettitt U_t = 2 * sum_{i<=t} sum_{j>t} sign(y_i - y_j)
    ut = np.zeros(n)
    for t in range(1, n):
        ut[t] = ut[t - 1] + 2.0 * np.sum(np.sign(y[t] - y[:t]))
    k_idx = int(np.argmax(np.abs(ut)))
    k_t = float(abs(ut[k_idx]))
    p = 2.0 * np.exp(-6.0 * k_t ** 2 / (n ** 3 + n ** 2))
    return {"K": k_t, "p": float(min(p, 1.0)), "idx": k_idx}


def structural_break_suite(yearly: pd.DataFrame) -> dict:
    """Chow + Pettitt on raw counts and harmonized flux around 2012."""
    out = {}
    y = yearly.reset_index()
    years = y["year"].to_numpy()
    i12 = int(np.searchsorted(years, 2012))
    for col, tag in (("raw_count", "raw"), ("phi", "harm")):
        if col not in y.columns:
            continue
        series = y[col].to_numpy(dtype=float)
        chow = chow_test(series, max(3, min(i12, len(series) - 3)))
        pet = pettitt_test(series)
        out[f"chow_{tag}_2012"] = chow
        out[f"pettitt_{tag}_2012"] = pet
    return out


# ---------------------------------------------------------------------------
# Ablation metrics (2019-2021 VIIRS-masked)
# ---------------------------------------------------------------------------
def crps_interval(q05, q50, q95, obs) -> float:
    """Empirical CRPS approximated from the (q05, q50, q95) quantile triple."""
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    obs = np.asarray(obs, dtype=float)
    # quantile (pinball) loss CRPS approximation over the 3 quantile levels
    levels = np.array([0.05, 0.50, 0.95])
    qs = np.stack([q05, q50, q95], axis=1)
    crps = np.zeros(len(obs))
    for j, tau in enumerate(levels):
        d = obs - qs[:, j]
        crps += 2.0 * (np.maximum(tau * d, (tau - 1.0) * d)) / len(levels)
    return float(np.mean(crps))


def ablation_metrics(pred_q05, pred_q50, pred_q95, truth) -> dict:
    """log-RMSE, CRPS, 90% coverage, Spearman rho, vs raw baseline."""
    q05 = np.asarray(pred_q05, dtype=float)
    q50 = np.asarray(pred_q50, dtype=float)
    q95 = np.asarray(pred_q95, dtype=float)
    t = np.asarray(truth, dtype=float)
    log_rmse = float(np.sqrt(np.mean((np.log(q50 + 1.0)
                                      - np.log(t + 1.0)) ** 2)))
    cov = float(np.mean((q05 <= t) & (t <= q95)))
    rho = float(stats.spearmanr(q50, t).statistic)
    return {"log_rmse": log_rmse, "crps": crps_interval(q05, q50, q95, t),
            "coverage_90": cov, "spearman_rho": rho, "n": int(len(t))}


def top3_peak_month_match(pred_monthly: pd.DataFrame,
                          true_monthly: pd.DataFrame) -> float:
    """Fraction of cells whose Top-3 peak months match the truth's."""
    pred = pred_monthly.groupby("h3_index").apply(
        lambda g: set(g.nlargest(3, "phi_q50_mw")["month"]), include_groups=False)
    true = true_monthly.groupby("h3_index").apply(
        lambda g: set(g.nlargest(3, "phi24_true")["month"]), include_groups=False)
    hits, tot = 0, 0
    for c in true.index:
        if c in pred.index:
            hits += len(pred[c] & true[c])
            tot += 3
    return float(hits / max(tot, 1))


# ---------------------------------------------------------------------------
# Full diagnostics orchestrator
# ---------------------------------------------------------------------------
def run_diagnostics(monthly: pd.DataFrame, truth: pd.DataFrame,
                    burned_area: pd.DataFrame | None = None,
                    pass_obs: pd.DataFrame | None = None,
                    pass_cal: pd.DataFrame | None = None) -> dict:
    """Compute the full diagnostics dict (saved as diagnostics.json)."""
    bio = monthly[monthly["role"].isin(["normal", "chilecito", "offnadir"])] \
        if "role" in monthly.columns else monthly
    m = bio.merge(truth, on=["h3_index", "year", "month"], how="inner")
    yearly = m.groupby("year").agg(raw_count=("raw_count_sum", "sum"),
                                   phi=("phi_q50_mw", "sum"))
    diag = {"structural_breaks": structural_break_suite(yearly)}

    # ablation window metrics (harmonized vs truth)
    abl = m[(m["year"] >= 2019) & (m["year"] <= 2021)]
    diag["ablation_2019_2021"] = ablation_metrics(
        abl["phi_q05_mw"], abl["phi_q50_mw"], abl["phi_q95_mw"],
        abl["phi24_true"])
    # raw MODIS baseline for the same window (raw counts scaled by the
    # best constant factor fitted on 2012-2018)
    base = m[(m["year"] >= 2012) & (m["year"] <= 2018)]
    if len(base):
        k = float(np.sum(base["phi24_true"]) / np.sum(base["raw_count_sum"]))
    else:
        k = 1.0
    diag["ablation_2019_2021"]["raw_baseline"] = ablation_metrics(
        abl["raw_count_sum"] * k, abl["raw_count_sum"] * k,
        abl["raw_count_sum"] * k, abl["phi24_true"])
    diag["top3_peak_month_match"] = top3_peak_month_match(
        m.rename(columns={"month": "month"})[["h3_index", "month",
                                              "phi_q50_mw"]],
        m[["h3_index", "month", "phi24_true"]])

    # MCD64CMQ burned-area correlations
    if burned_area is not None and len(burned_area):
        ba = burned_area.copy()
        ba["lat_025"] = ba["lat_025"].round(2)
        ba["lon_025"] = ba["lon_025"].round(2)
        from ..geometry.h3_utils import h3_cell_center
        cmg = pd.DataFrame(
            [dict(h3_index=int(c), lat_025=round(h3_cell_center(int(c))[0], 2),
                   lon_025=round(h3_cell_center(int(c))[1], 2))
             for c in bio["h3_index"].unique()])
        mm = m.merge(cmg, on="h3_index", how="left").merge(
            ba, on=["lat_025", "lon_025", "year", "month"], how="inner")
        if len(mm):
            r_harm = float(np.corrcoef(np.log1p(mm["fre_tj_q50"]),
                                       np.log1p(mm["burned_area_km2"]))[0, 1])
            r_raw = float(np.corrcoef(np.log1p(mm["raw_count_sum"]),
                                      np.log1p(mm["burned_area_km2"]))[0, 1])
            diag["mcd64cmq_corr"] = {"harmonized_fre": r_harm,
                                     "raw_counts": r_raw}
    # residual VZA / solar-time slopes (implied-z residual vs pass geometry)
    po_src = pass_cal if (pass_cal is not None and len(pass_cal)
                          and "mu_gain_det" in pass_cal.columns) else pass_obs
    if po_src is not None and len(po_src) and "mu_gain_det" in po_src.columns:
        po = po_src[po_src["y_frp_mw"] > 0]
        if len(po):
            r_vza = float(stats.linregress(po["mean_vza_deg"],
                                           po["mu_gain_det"]).slope)
            diag["residual_slopes"] = {"mu_gain_vs_vza": r_vza}
    # spotlight reports
    diag["spotlights"] = spotlight_reports(m)
    return diag


def spotlight_reports(m: pd.DataFrame) -> dict:
    """Chilecito ARG 2026, Kabylie DZA, Hassi Messaoud, Nov 2026 S-NPP end."""
    out = {}
    from ..config import settings as S
    from ..geometry.h3_utils import h3_cell_center
    for sp in S.SPOTLIGHT_PRESETS:
        sub = m[m["role"] == sp.key] if "role" in m.columns else pd.DataFrame()
        if not len(sub):
            cells = [c for c in m["h3_index"].unique()
                     if (lambda lc: sp.bbox[0] <= lc[1] <= sp.bbox[2]
                         and sp.bbox[1] <= lc[0] <= sp.bbox[3])(
                             h3_cell_center(int(c)))]
            sub = m[m["h3_index"].isin(cells)]
        if len(sub):
            out[sp.key] = {
                "median_ratio": float((sub["phi_q50_mw"]
                                       / sub["phi24_true"]).median()),
                "n_months": int(len(sub)),
            }
    n26 = m[m["year"] == 2026]
    if len(n26):
        by_month = n26.groupby("month").agg(
            raw=("raw_count_sum", "sum"), phi=("phi_q50_mw", "sum"),
            phi_t=("phi24_true", "sum"))
        if 10 in by_month.index and 11 in by_month.index:
            out["snpp_end_nov2026"] = {
                "raw_oct_nov_pct": float(100 * (by_month.loc[11, "raw"]
                                                 / by_month.loc[10, "raw"] - 1)),
                "harm_oct_nov_pct": float(100 * (by_month.loc[11, "phi"]
                                                  / by_month.loc[10, "phi"] - 1)),
                "true_oct_nov_pct": float(100 * (by_month.loc[11, "phi_t"]
                                                  / by_month.loc[10, "phi_t"] - 1)),
            }
    return out


def save_diagnostics(diag: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(diag, indent=2, default=float))
