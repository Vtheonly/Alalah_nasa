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
    """Monthly means directly from the MAP latent states (reference)."""
    dates = T0 + pd.to_timedelta(np.arange(t_days), unit="D")
    ym = dates.year.to_numpy() * 12 + dates.month.to_numpy() - 1
    uym, mi = np.unique(ym, return_inverse=True)
    cnt = np.bincount(mi, minlength=len(uym))
    phi24 = 0.5 * (np.exp(z_map[:, :, 0]) + np.exp(z_map[:, :, 1]))
    rows = []
    for i, c in enumerate(cells):
        mm = np.bincount(mi, weights=phi24[i], minlength=len(uym)) / cnt
        rows.append(pd.DataFrame({"h3_index": int(c), "year": uym // 12,
                                  "month": uym % 12 + 1, "phi_ref": mm}))
    return pd.concat(rows, ignore_index=True)


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

    # ---- pyromes + operator calibration (2012-2021 freeze) ----------------
    hex_pyromes, _gmm, _scaler = fit_pyromes(prov_bio)
    artifacts, pass_cal = ops.calibrate(prov_bio, pass_obs, hex_pyromes,
                                        cloud_df)

    # ---- priors -------------------------------------------------------------
    psi_cnt_fn = ops.psi_cnt_row_fn(artifacts)
    active_cells = sorted(set(int(c) for c in pass_cal["h3_index"].unique()))
    priors = pr.estimate_priors(pass_cal, hex_pyromes, active_cells, cloud_df,
                                artifacts.get("cloud_climatology"),
                                t_days=t_days, psi_cnt_fn=psi_cnt_fn,
                                platform_active=platform_active)

    # ---- solve + conformal + finalize ---------------------------------------
    h = conf.harmonize_and_calibrate(pass_cal, pass_obs, prov, hex_pyromes, priors, artifacts,
                  cloud_df, platform_active, t_days=t_days,
                  batch_size=chunk_cells or S.BATCH_SIZE)
    res = h["res"]
    monthly = h["monthly"]
    ablation = viirs_masked_ablation_fold(
        pass_cal, active_cells, hex_pyromes, priors, artifacts, cloud_df,
        platform_active, t_days, chunk_cells or S.BATCH_SIZE)
    phi_ref = _monthly_from_z(res["z_map"], res["cells"], t_days)
    return {
        "prov": prov, "prov_bio": prov_bio, "pass_obs": pass_obs,
        "pass_cal": pass_cal, "hex_pyromes": hex_pyromes,
        "artifacts": artifacts, "priors": priors,
        "platform_active": platform_active,
        "z_map": res["z_map"], "var": res["var"], "cells": res["cells"],
        "monthly": monthly, "daily": h["daily"],
        "monthly_state": res["monthly"], "truth": truth, "z_true": z_true,
        "cloud_df": cloud_df, "cfg": cfg,
        "c_modis": h["c_modis"], "c_multi": h["c_multi"],
        "fold_modis": h["fold_modis"], "fold_loo": h["fold_loo"],
        "phi_ref": phi_ref, "solver_results": res["results"],
        "op": res["op"], "ablation_fold": ablation,
    }


def viirs_masked_ablation_fold(pass_cal, active_cells, hex_pyromes, priors,
                               artifacts, cloud_df, platform_active,
                               t_days: int, batch_size: int = S.BATCH_SIZE,
                               workers_per_gpu: int = 1):
    """Re-solve with ALL VIIRS platforms masked over 2019-01-01..2021-12-31
    (readme 2019-2021 VIIRS-masked ablation); 2019-2021 monthly rows."""
    viirs = [S.PLATFORM_INDEX[p] for p in S.VIIRS_PLATFORMS]
    t19 = (pd.Timestamp("2019-01-01") - T0).days
    t21 = (pd.Timestamp("2021-12-31") - T0).days
    fold = slv.solve_all_chunks(
        pass_cal, active_cells, hex_pyromes, priors, artifacts, cloud_df,
        t_days=t_days, platform_active=platform_active,
        platform_mask=slv.PlatformMask(viirs, t19, t21), daily_mode="none",
        keep_years=(2019, 2021), keep_states=False, batch_size=batch_size,
        workers_per_gpu=workers_per_gpu)
    m = fold["monthly"].merge(hex_pyromes[["h3_index", "pyrome_id"]]
                              .drop_duplicates("h3_index"),
                              on="h3_index", how="left")
    m["pyrome_id"] = m["pyrome_id"].fillna(0).astype(int)
    return m


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
