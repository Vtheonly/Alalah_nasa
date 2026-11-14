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
    """Pettitt (1979) non-parametric single change-point test.

    ``U_t = sum_{i<=t} sum_{j>t} sign(y_j - y_i)`` for t = 1..n-1 (computed
    in O(n log n) from mid-ranks: ``U_t = t (n + 1) - 2 sum_{i<=t} r_i``),
    ``K = max_t |U_t|``, approximate ``p = 2 exp(-6 K^2 / (n^3 + n^2))``.
    ``idx`` is the first index of the post-change segment (the change
    occurs between ``idx - 1`` and ``idx``).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return {"K": 0.0, "p": 1.0, "idx": 0}
    r = stats.rankdata(y)
    t = np.arange(1, n)
    u = t * (n + 1) - 2.0 * np.cumsum(r)[:-1]
    k_pos = int(np.argmax(np.abs(u)))
    k_t = float(abs(u[k_pos]))
    p = 2.0 * np.exp(-6.0 * k_t ** 2 / (n ** 3 + n ** 2))
    return {"K": k_t, "p": float(min(p, 1.0)), "idx": int(t[k_pos])}


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
def raw_modis_monthly(pass_obs: pd.DataFrame) -> pd.DataFrame:
    """Unharmonized raw MODIS (Terra + Aqua) FRP sum per cell-month."""
    po = pass_obs[(pass_obs["platform"].isin(["Terra", "Aqua"]))
                  & (pass_obs["y_frp_mw"] > 0)]
    d = pd.DatetimeIndex(pd.to_datetime(po["solar_date"]))
    f = pd.DataFrame({"h3_index": po["h3_index"].to_numpy(np.int64),
                      "year": d.year, "month": d.month,
                      "raw_modis_frp": po["y_frp_mw"].to_numpy(dtype=float)})
    return f.groupby(["h3_index", "year", "month"], as_index=False).sum()


def viirs_masked_ablation(fold_monthly: pd.DataFrame, reference: pd.DataFrame,
                          pass_obs: pd.DataFrame, ref_col: str,
                          c_modis: dict | None = None) -> dict:
    """2019-2021 VIIRS-masked ablation (readme 5.4 / 7.9).

    ``fold_monthly``: posterior monthly rows of the re-solve with ALL VIIRS
    platforms masked over 2019-01-01..2021-12-31; ``reference``: per
    cell-month ``ref_col`` (twin truth, or the full multi-sensor product on
    real data).  The fold intervals are conformal-rescaled with the
    MODIS-only factors ``c_modis``.  Raw MODIS baseline: Terra+Aqua raw FRP
    scaled by the constant fitted on 2012-2018 against the reference.
    """
    from ..engine import conformal as conf
    f = fold_monthly[(fold_monthly["year"] >= 2019)
                     & (fold_monthly["year"] <= 2021)].copy()
    if "pyrome_id" not in f.columns:
        f["pyrome_id"] = 0
    if c_modis:
        f = conf.apply_conformal(f, c_modis, c_modis,
                                 np.ones(len(f), dtype=bool))
    ref = reference[["h3_index", "year", "month", ref_col]]
    m = f.merge(ref, on=["h3_index", "year", "month"], how="inner")
    out = ablation_metrics(m["phi_q05_mw"], m["phi_q50_mw"], m["phi_q95_mw"],
                           m[ref_col])
    raw = raw_modis_monthly(pass_obs)
    allref = reference.merge(raw, on=["h3_index", "year", "month"],
                             how="left").fillna({"raw_modis_frp": 0.0})
    base = allref[(allref["year"] >= 2012) & (allref["year"] <= 2018)]
    k = float(base[ref_col].sum() / max(base["raw_modis_frp"].sum(), 1e-9))
    mb = m.merge(raw, on=["h3_index", "year", "month"], how="left").fillna(
        {"raw_modis_frp": 0.0})
    rb = mb["raw_modis_frp"].to_numpy() * k
    out["raw_baseline"] = ablation_metrics(rb, rb, rb, mb[ref_col])
    out["raw_baseline"]["scale_k"] = k
    return out


def mcd64cmq_correlation(monthly: pd.DataFrame,
                         burned_area: pd.DataFrame) -> dict | None:
    """Harmonized monthly FRE vs raw counts correlation with independent
    MCD64CMQ burned area, joined on the EXACT 0.25 deg CMG bin centres
    (x.125 / x.375 / x.625 / x.875) of each H3 cell centre."""
    from ..engine.priors import cmg_bin_centers
    from ..geometry.h3_utils import h3_cell_center
    if burned_area is None or not len(burned_area):
        return None
    cells = np.unique(monthly["h3_index"].to_numpy(np.int64))
    ll = np.array([h3_cell_center(int(c)) for c in cells])
    lat_c, lon_c = cmg_bin_centers(ll[:, 0], ll[:, 1])
    cmg = pd.DataFrame({"h3_index": cells, "lat_025": np.round(lat_c, 3),
                        "lon_025": np.round(lon_c, 3)})
    ba = burned_area.copy()
    ba["lat_025"] = ba["lat_025"].astype(float).round(3)
    ba["lon_025"] = ba["lon_025"].astype(float).round(3)
    # aggregate H3 cells to their CMG bin (both sides on the same support)
    mm = monthly.merge(cmg, on="h3_index", how="left")
    agg = mm.groupby(["lat_025", "lon_025", "year", "month"],
                     as_index=False).agg(fre=("fre_tj_q50", "sum"),
                                         raw=("raw_count_sum", "sum"))
    j = agg.merge(ba, on=["lat_025", "lon_025", "year", "month"], how="inner")
    j = j[np.isfinite(j["burned_area_km2"])]
    if len(j) < 3:
        return {"n": int(len(j))}
    return {"harmonized_fre": float(stats.pearsonr(
                np.log1p(j["fre"]), np.log1p(j["burned_area_km2"]))[0]),
            "raw_counts": float(stats.pearsonr(
                np.log1p(j["raw"]), np.log1p(j["burned_area_km2"]))[0]),
            "n": int(len(j))}


def residual_slopes(pass_cal: pd.DataFrame, daily: pd.DataFrame) -> dict | None:
    """Post-harmonization residual slopes: ``r = z_imp - z_hat`` with
    ``z_imp = ln Y - mu_gain_det`` (each detected pass) and ``z_hat`` the
    reconstructed half-day state (``ln phi_day_mw`` / ``ln phi_night_mw`` of
    the daily product -- detection days are always in the sparse daily
    set), regressed on pass VZA and on solar hour.  A harmonized record has
    ~0 slopes (no residual view-angle or overpass-time bias)."""
    if pass_cal is None or daily is None or not len(daily):
        return None
    po = pass_cal[pass_cal["y_frp_mw"] > 0].copy()
    po["t"] = (pd.to_datetime(po["solar_date"])
               - pd.Timestamp("2000-11-01")).dt.days
    d = daily[["h3_index", "t", "phi_day_mw", "phi_night_mw"]]
    j = po.merge(d, on=["h3_index", "t"], how="inner")
    if len(j) < 10:
        return None
    z_hat = np.log(np.where(j["dn_flag"] == "D", j["phi_day_mw"],
                            j["phi_night_mw"]).astype(float))
    z_imp = np.log(j["y_frp_mw"].to_numpy(dtype=float)) \
        - j["mu_gain_det"].to_numpy(dtype=float)
    r = z_imp - z_hat
    out = {"n": int(len(j)), "mean_residual": float(np.mean(r))}
    for col, key in (("mean_vza_deg", "vza"), ("mean_solar_hour",
                                                "solar_hour")):
        lr = stats.linregress(j[col].to_numpy(dtype=float), r)
        out[f"slope_per_{key}"] = float(lr.slope)
        out[f"p_{key}"] = float(lr.pvalue)
    for fam in ("MODIS", "VIIRS"):
        sel = (j["sensor_family"] == fam).to_numpy()
        if sel.sum() >= 10:
            out[f"slope_per_vza_{fam}"] = float(stats.linregress(
                j.loc[sel, "mean_vza_deg"].to_numpy(dtype=float),
                r[sel]).slope)
    return out


def run_diagnostics(monthly: pd.DataFrame, truth: pd.DataFrame | None = None,
                    burned_area: pd.DataFrame | None = None,
                    pass_obs: pd.DataFrame | None = None,
                    pass_cal: pd.DataFrame | None = None,
                    ablation_fold: pd.DataFrame | None = None,
                    daily: pd.DataFrame | None = None,
                    c_modis: dict | None = None) -> dict:
    """Compute the full diagnostics dict (saved as diagnostics.json).

    ``truth`` (synthetic twin) supplies ``phi24_true``; on real data the
    reference for the VIIRS-masked ablation is the full multi-sensor
    harmonized product (``phi_q50_mw``) -- never the model compared with
    itself on the same run.
    """
    bio = monthly[monthly["role"].isin(["normal", "chilecito", "offnadir"])] \
        if "role" in monthly.columns else monthly
    if truth is not None:
        m = bio.merge(truth, on=["h3_index", "year", "month"], how="inner",
                      suffixes=("", "_t"))
        if "role" in m.columns:
            m = m[m["role"].isin(["normal", "chilecito", "offnadir"])]
        ref_col = "phi24_true"
        reference = m[["h3_index", "year", "month", ref_col]]
    else:
        m = bio.copy()
        ref_col = "phi_ref_full"
        reference = m[["h3_index", "year", "month", "phi_q50_mw"]].rename(
            columns={"phi_q50_mw": ref_col})
    yearly = m.groupby("year").agg(raw_count=("raw_count_sum", "sum"),
                                   phi=("phi_q50_mw", "sum"))
    diag = {"structural_breaks": structural_break_suite(yearly),
            "reference": "synthetic_truth" if truth is not None
            else "full_multisensor_product"}
    if ablation_fold is not None and pass_obs is not None and len(
            ablation_fold):
        diag["ablation_2019_2021"] = viirs_masked_ablation(
            ablation_fold, reference, pass_obs, ref_col, c_modis)
    if truth is not None:
        diag["top3_peak_month_match"] = top3_peak_month_match(
            m[["h3_index", "month", "phi_q50_mw"]],
            m[["h3_index", "month", "phi24_true"]])
    mc = mcd64cmq_correlation(bio, burned_area)
    if mc is not None:
        diag["mcd64cmq_corr"] = mc
    rs = residual_slopes(pass_cal, daily)
    if rs is not None:
        diag["residual_slopes"] = rs
    diag["spotlights"] = spotlight_reports(m, ref_col if truth is not None
                                           else None)
    return diag


def spotlight_reports(m: pd.DataFrame, ref_col: str | None = None) -> dict:
    """Chilecito ARG 2026, Kabylie DZA, Hassi Messaoud, Alaska, Black
    Summer, Kalimantan + the Nov 2, 2026 S-NPP end (Oct -> Nov 2026)."""
    out = {}
    from ..config import settings as S
    from ..geometry.h3_utils import h3_cell_center
    centers = {int(c): h3_cell_center(int(c)) for c in m["h3_index"].unique()}
    for sp in S.SPOTLIGHT_PRESETS:
        sub = m[m["role"] == sp.key] if "role" in m.columns else m.iloc[:0]
        if not len(sub):
            cells = [c for c, (lat, lon) in centers.items()
                     if sp.bbox[0] <= lon <= sp.bbox[2]
                     and sp.bbox[1] <= lat <= sp.bbox[3]]
            sub = m[m["h3_index"].isin(cells)]
        if not len(sub):
            continue
        rep = {"n_cells": int(sub["h3_index"].nunique()),
               "n_months": int(len(sub)),
               "max_anomaly_ratio": float(sub["anomaly_ratio"].max())
               if "anomaly_ratio" in sub else None}
        if ref_col is not None and ref_col in sub.columns:
            rep["median_ratio_vs_truth"] = float(
                (sub["phi_q50_mw"] / sub[ref_col]).median())
        if sp.key == "chilecito_arg_2026":
            jja = sub[(sub["year"] == 2026) & sub["month"].isin([7, 8])]
            rep["jul_aug_2026_phi_q50_mw"] = float(jja["phi_q50_mw"].sum())
            rep["jul_aug_2026_raw_count"] = float(jja["raw_count_sum"].sum())
        out[sp.key] = rep
    n26 = m[m["year"] == 2026]
    if len(n26):
        agg = {"raw": ("raw_count_sum", "sum"), "phi": ("phi_q50_mw", "sum")}
        if ref_col is not None and ref_col in n26.columns:
            agg["phi_t"] = (ref_col, "sum")
        by_month = n26.groupby("month").agg(**agg)
        if 10 in by_month.index and 11 in by_month.index:
            rep = {"raw_oct_nov_pct": float(100 * (by_month.loc[11, "raw"]
                                                    / max(by_month.loc[10, "raw"], 1e-9) - 1)),
                   "harm_oct_nov_pct": float(100 * (by_month.loc[11, "phi"]
                                                     / max(by_month.loc[10, "phi"], 1e-9) - 1))}
            if "phi_t" in by_month.columns:
                rep["true_oct_nov_pct"] = float(100 * (
                    by_month.loc[11, "phi_t"] / by_month.loc[10, "phi_t"] - 1))
            out["snpp_end_nov2026"] = rep
    return out


def save_diagnostics(diag: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(diag, indent=2, default=float))
