"""Twin pipeline orchestration: detections -> full harmonization products.

Runs the complete Environment-2 chain on synthetic-twin detections:
normalize/clip -> preprocess -> pyromes -> operator calibration -> priors ->
GMRF solve -> conformal calibration -> monthly/daily products, and computes
raw (unharmonized) baselines plus the VIIRS-masked / leave-one-out conformal
folds required by readme 5.4.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings as S
from ..engine import conformal as conf
from ..engine import solver as slv
from ..geometry.preprocess_geometry import preprocess_observations
from ..ingest.normalize_dedup import (clip_to_countries,
                                      compute_platform_active_day,
                                      load_country_polygons,
                                      science_tier_dominance_dedup)
from . import synthetic_twin as twin

T0 = pd.Timestamp(S.DATE_START)


def _t_of(dates):
    return ((pd.DatetimeIndex(pd.to_datetime(dates)) - T0).days).to_numpy()


def _monthly_from_z(z_map, cells, t_days):
    """True monthly means directly from the MAP latent states (reference)."""
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t) for t in range(t_days)])
    months = dates.month.to_numpy()
    years = dates.year.to_numpy()
    rows = []
    for i, c in enumerate(cells):
        for y in np.unique(years):
            for m in range(1, 13):
                sel = (years == y) & (months == m)
                if not sel.any():
                    continue
                phi_d = float(np.exp(z_map[i, sel, 0]).mean())
                phi_n = float(np.exp(z_map[i, sel, 1]).mean())
                rows.append({"h3_index": int(c), "year": int(y), "month": int(m),
                             "phi_ref": 0.5 * (phi_d + phi_n)})
    return pd.DataFrame(rows)


def _raw_baselines(pass_obs, t_days):
    """Raw (unharmonized) per cell-month baselines: counts / FRP / events."""
    p = pass_obs.copy()
    p["t"] = _t_of(p["solar_date"])
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t) for t in p["t"]])
    p["year"] = dates.year.to_numpy()
    p["month"] = dates.month.to_numpy()
    g = p.groupby(["h3_index", "year", "month"], as_index=False).agg(
        raw_count_sum=("n_pixels", "sum"),
        raw_frp_sum_mw=("y_frp_mw", "sum"),
        raw_clustered_events=("raw_event_count", "sum"))
    return g


def _platform_mask_fn(masked_platforms, t_lo, t_hi):
    def mask(pi, t):
        if pi not in masked_platforms:
            return True
        return not (t_lo <= t <= t_hi)
    return mask


def run(det, cells, z_true, truth, cloud_df, cfg, paths=None,
        chunk_cells: int = 0, sweeps: int = S.SOLVER_SWEEPS):
    """Full pipeline on twin detections.  Returns a result dict."""
    from ..calibration import operators as ops
    from ..calibration.pyromes import fit_pyromes
    from ..engine import priors as pr

    t_days = cfg.t_days
    # ---- normalize ---------------------------------------------------------
    if paths is not None:
        polygons = load_country_polygons(paths.country_polygons_parquet)
        det = clip_to_countries(det, polygons)
    det = science_tier_dominance_dedup(det)
    platform_active = compute_platform_active_day(det)

    # ---- preprocess ---------------------------------------------------------
    prov, pass_obs = preprocess_observations(det)
    prov_bio = prov[prov["is_biomass"]].copy()

    # ---- pyromes --------------------------------------------------------------
    hex_pyromes, _gmm, _scaler = fit_pyromes(prov_bio)

    # ---- operator calibration ---------------------------------------------------
    artifacts, pass_cal = ops.calibrate(prov_bio, pass_obs, hex_pyromes,
                                        cloud_df)

    # ---- priors -------------------------------------------------------------------
    def psi_cnt_fn(b, g):
        """Per-row psi_CNT (platform-aware: MODIS at A_avg 2.35, VIIRS 0.23)."""
        n = len(g)
        fams = [S.SENSOR_FAMILY.get(str(p), "VIIRS")
                for p in (g["platform"].to_numpy()
                          if "platform" in g.columns
                          else np.full(n, "SNPP", dtype=object))]
        dns = g["dn_flag"].to_numpy()
        out = np.empty(n)
        for fam in ("MODIS", "VIIRS"):
            m = np.array([f == fam for f in fams])
            if m.any():
                out[m] = ops.psi_cnt(fam, b, S.PASS_GEOMETRY_AVG[fam][0],
                                     dns[m], artifacts["lognorm_params"])
        return out
    active_cells = sorted(set(int(c) for c in pass_cal["h3_index"].unique()))
    priors = pr.estimate_priors(pass_cal, hex_pyromes, active_cells, cloud_df,
                                artifacts.get("cloud_climatology"),
                                t_days=t_days, psi_cnt_fn=psi_cnt_fn)

    # ---- main solve --------------------------------------------------------------
    res = slv.solve_all_chunks(pass_cal, active_cells, hex_pyromes, priors,
                               artifacts, cloud_df, t_days=t_days,
                               platform_active=platform_active)
    monthly_state, daily_state = slv.aggregate_trajectories(
        res["cells"], t_days,
        {(int(c), s): None for c in res["cells"] for s in (0, 1)}) \
        if False else (res["monthly"], res["daily"])
    monthly, daily = slv.combine_states_to_products(monthly_state, daily_state)
    monthly = monthly.merge(
        hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index", how="left")
    monthly["pyrome_id"] = monthly["pyrome_id"].fillna(0).astype(int)

    # ---- conformal folds -----------------------------------------------------------
    t_2018 = (pd.Timestamp("2018-01-01") - T0).days
    t_2021 = (pd.Timestamp("2021-12-31") - T0).days
    viirs_idx = [S.PLATFORM_INDEX[p] for p in S.VIIRS_PLATFORMS]
    fold_modis = slv.solve_all_chunks(
        pass_cal, active_cells, hex_pyromes, priors, artifacts, cloud_df,
        t_days=t_days, platform_active=platform_active,
        platform_mask=_platform_mask_fn(set(viirs_idx), t_2018, t_2021))
    fm, _fd = slv.combine_states_to_products(fold_modis["monthly"],
                                             fold_modis["daily"])
    fm = fm.merge(hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index",
                  how="left")
    fm["pyrome_id"] = fm["pyrome_id"].fillna(0).astype(int)
    phi_ref_by_key = {(int(h), int(y), int(m)): v for h, y, m, v in
                      zip(monthly["h3_index"], monthly["year"],
                          monthly["month"], monthly["phi_q50_mw"])}
    c_modis = conf.conformal_multipliers(
        fm[(fm["year"] >= 2018) & (fm["year"] <= 2020)], phi_ref_by_key)

    # leave-one-VIIRS-platform-out folds (SNPP held out 2018-2021)
    snpp_idx = S.PLATFORM_INDEX["SNPP"]
    fold_loo = slv.solve_all_chunks(
        pass_cal, active_cells, hex_pyromes, priors, artifacts, cloud_df,
        t_days=t_days, platform_active=platform_active,
        platform_mask=_platform_mask_fn({snpp_idx}, t_2018, t_2021))
    fl, _fd2 = slv.combine_states_to_products(fold_loo["monthly"],
                                              fold_loo["daily"])
    fl = fl.merge(hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index",
                  how="left")
    fl["pyrome_id"] = fl["pyrome_id"].fillna(0).astype(int)
    c_multi = conf.conformal_multipliers(
        fl[(fl["year"] >= 2018) & (fl["year"] <= 2021)], phi_ref_by_key)

    # ---- apply conformal rescaling ---------------------------------------------------
    modis_only_era = monthly["year"] < 2012
    monthly = conf.apply_conformal(monthly, c_modis, c_multi, modis_only_era)
    monthly = conf.climatology_indices(monthly)

    # ---- raw baselines + event indices -------------------------------------------------
    raw = _raw_baselines(pass_obs, t_days)
    monthly = monthly.merge(raw, on=["h3_index", "year", "month"], how="left")
    monthly[["raw_count_sum", "raw_frp_sum_mw", "raw_clustered_events"]] = \
        monthly[["raw_count_sum", "raw_frp_sum_mw",
                 "raw_clustered_events"]].fillna(0.0)
    f_event = priors["f_event"]
    monthly["energy_equiv_events_q50"] = conf.energy_equiv_events(
        monthly["phi_q50_mw"],
        [f_event.get(int(b), 1.0) for b in monthly["pyrome_id"]])

    z_map = res["z_map"]
    phi_ref = _monthly_from_z(z_map, res["cells"], t_days)
    result = {
        "prov": prov, "prov_bio": prov_bio, "pass_obs": pass_obs,
        "pass_cal": pass_cal, "hex_pyromes": hex_pyromes,
        "artifacts": artifacts, "priors": priors, "platform_active":
            platform_active,
        "z_map": z_map, "var": res["var"], "cells": res["cells"],
        "monthly": monthly, "daily": daily,
        "monthly_state": monthly_state, "truth": truth, "z_true": z_true,
        "cloud_df": cloud_df, "cfg": cfg,
        "c_modis": c_modis, "c_multi": c_multi,
        "phi_ref": phi_ref, "solver_results": res["results"],
    }
    return result


def twin_assertion_metrics(res: dict) -> dict:
    """All readme test_07 assertion metrics from a twin pipeline result."""
    m = res["monthly"].merge(res["truth"], on=["h3_index", "year", "month"],
                             suffixes=("", "_t"))
    roles = set(res["cfg"].cells and {r[3] for r in res["cfg"].cells} or set())
    bio = m[m["role"].isin(["normal", "chilecito", "offnadir"])] \
        if "role" in m.columns else m
    yearly = bio.groupby("year").agg(
        raw_count=("raw_count_sum", "sum"), phi=("phi_q50_mw", "sum"),
        phi_true=("phi24_true", "sum"))
    out = {}
    # 2012 S-NPP step (raw jump vs harmonized step)
    out["raw_jump_2012_pct"] = 100.0 * (
        yearly.loc[2013:2014, "raw_count"].mean()
        / yearly.loc[2010:2011, "raw_count"].mean() - 1.0)
    out["harm_step_2012_pct"] = 100.0 * (
        yearly.loc[2013:2014, "phi"].mean()
        / yearly.loc[2010:2011, "phi"].mean() - 1.0)
    out["true_step_2012_pct"] = 100.0 * (
        yearly.loc[2013:2014, "phi_true"].mean()
        / yearly.loc[2010:2011, "phi_true"].mean() - 1.0)
    # 3-satellite inflation (2019-2023 vs 2013-2017)
    out["raw_3sat_pct"] = 100.0 * (
        yearly.loc[2019:2023, "raw_count"].mean()
        / yearly.loc[2013:2017, "raw_count"].mean() - 1.0)
    out["harm_3sat_pct"] = 100.0 * (
        yearly.loc[2019:2023, "phi"].mean()
        / yearly.loc[2013:2017, "phi"].mean() - 1.0)
    out["true_3sat_pct"] = 100.0 * (
        yearly.loc[2019:2023, "phi_true"].mean()
        / yearly.loc[2013:2017, "phi_true"].mean() - 1.0)
    # Nov 2026 S-NPP cutoff (Oct -> Nov 2026)
    n26 = bio[bio["year"] == 2026].groupby("month").agg(
        raw=("raw_count_sum", "sum"), phi=("phi_q50_mw", "sum"),
        phi_true=("phi24_true", "sum"))
    if 10 in n26.index and 11 in n26.index:
        out["raw_nov2026_pct"] = 100.0 * (n26.loc[11, "raw"] / n26.loc[10, "raw"] - 1)
        out["harm_nov2026_pct"] = 100.0 * (n26.loc[11, "phi"] / n26.loc[10, "phi"] - 1)
        out["true_nov2026_pct"] = 100.0 * (n26.loc[11, "phi_true"]
                                           / n26.loc[10, "phi_true"] - 1)
    # post-2022 drift era level (harmonized vs true annual ratio)
    drift = bio[(bio["year"] >= 2023) & (bio["year"] <= 2026)]
    out["harm_drift_ratio"] = float(
        (drift["phi_q50_mw"] / drift["phi24_true"]).median())
    # 5x drought spike (2015 vs 2014)
    spike_true = yearly.loc[2015, "phi_true"] / yearly.loc[2014, "phi_true"]
    spike_harm = yearly.loc[2015, "phi"] / yearly.loc[2014, "phi"]
    out["spike_true_x"] = float(spike_true)
    out["spike_harm_x"] = float(spike_harm)
    out["spike_recovery"] = float(spike_harm / spike_true)
    # monthly correlation + level + night/day ratio
    out["log_corr"] = float(np.corrcoef(np.log(bio["phi_q50_mw"] + 1),
                                        np.log(bio["phi24_true"] + 1))[0, 1])
    out["median_ratio"] = float((bio["phi_q50_mw"]
                                 / bio["phi24_true"]).median())
    ndr_hat = bio.groupby("h3_index").apply(
        lambda g: g["phi_night_mw"].mean() / max(g["phi_day_mw"].mean(), 1e-9),
        include_groups=False)
    ndr_true = bio.groupby("h3_index").apply(
        lambda g: g["phi_night_true"].mean()
        / max(g["phi_day_true"].mean(), 1e-9), include_groups=False)
    ndr_err = (ndr_hat / ndr_true - 1.0).abs()
    out["ndr_max_abs_err_pct"] = float(100 * ndr_err.max())
    out["ndr_never_collapsed"] = bool((ndr_hat > 0.02).all()
                                      and (ndr_hat < 0.95).all())
    # per-pyrome level bias
    out["pyrome_median_ratio"] = {
        int(k): float(v) for k, v in
        bio.groupby("pyrome_true").apply(
            lambda g: (g["phi_q50_mw"] / g["phi24_true"]).median(),
            include_groups=False).items()}
    return out
