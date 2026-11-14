"""Deterministic 26-year synthetic digital twin with known ground truth.

Contract (readme.md Sections 4 & 7): a forward simulator producing the true
latent states ``(Phi_day_true, Phi_night_true)`` and simulated noisy,
censored, bow-tie-duplicated, cloud-blocked, drifting, S-NPP-terminating
observations across all five spacecraft, so the full harmonization engine
can be falsified against known truth (structural continuity, night/day
ratio, drift correction, drought-spike preservation, Nov-2026 cutoff).

The twin's forward operators use the SAME closed-form functions as the
production calibration (``operators.normalize_diurnal_coeffs``,
``psi_frp_eff``, ...), so the solver consumes twin observations with the
twin's known operator parameters — an exact end-to-end inversion test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.calibration import operators as op
from src.config import settings as S
from src.engine import solver as sv
from src.geometry import h3_utils
from src.geometry.solar_geometry import expected_drift_solar_hour

# Twin ground-truth operator parameters.
TWIN_DIURNAL_COEFFS = np.array([0.6, -0.4, 0.2, 0.1])
TWIN_LN_GAMMA_MODIS = -0.25
TWIN_LOGNORM = (0.7, 1.0)          # (mu_F, sigma_F) cluster FRP distribution
TWIN_F_EVENT_MW = 5.0
TWIN_LAMBDA0 = 5.0
TWIN_RHO = 0.70
TWIN_SIGMA_PROC = 1.40
TWIN_BETA_DN = 0.50
TWIN_ETA_DN = 0.15
TWIN_NIGHT_DAY_RATIO = 0.30
TWIN_SIGMA = {"MODIS": 0.45, "VIIRS": 0.30}
TWIN_MODIS_SAT_FRP_MW = 60.0       # Channel-21 saturation ceiling (FRP equivalent)
TWIN_DORMANT_MW = 1.0  # background scale (log-flux base ~ 0)


def generate_twin(
    n_cells: int = 5,
    T: int = S.T_DAYS,
    seed: int = 42,
    cloud_mean: float = 0.50,
    blind_window: Optional[Tuple[int, int]] = None,
    drought_cell: int = 0,
) -> Dict[str, object]:
    """Build the digital twin; returns everything the solver + tests need."""
    rng = np.random.default_rng(seed)

    # --- spatial support: a compact H3-Res6 patch ---------------------------
    center = int(h3_utils.latlon_to_h3_int(-29.16, -67.50, S.H3_MAIN_RES)[0])
    cells = np.array(
        sorted({center, *h3_utils.h3_neighbors(center)}), dtype=np.int64
    )[:n_cells]
    while len(cells) < n_cells:
        ring = set()
        for c in cells:
            ring |= set(h3_utils.h3_neighbors(int(c)))
        cells = np.array(sorted(ring), dtype=np.int64)[:n_cells]
    latlon = np.array(
        [h3_utils.h3_center_latlon(int(c)) for c in cells]
    )
    B = len(cells)

    colors, adjacency = h3_utils.three_color_hex_graph(cells)
    nb_idx, nb_ptr = [], [0]
    for i in range(B):
        nb_idx.extend(adjacency[i])
        nb_ptr.append(len(nb_idx))

    dates = pd.date_range(S.DATE_START, periods=T, freq="D")
    month_ids, month_keys = sv.month_grid(T)
    doy = dates.dayofyear.to_numpy()

    # --- ground-truth latent states -----------------------------------------
    base_level = rng.uniform(-0.6, 0.6, size=B)        # mean log-flux
    season_amp = rng.uniform(0.4, 0.9, size=B)
    season_phase = rng.uniform(0, 2 * np.pi, size=B)
    z_day = np.empty((B, T))
    ar_noise = np.zeros((B, T))
    for i in range(B):
        w = rng.standard_normal(T)
        ar = np.zeros(T)
        for t in range(1, T):
            ar[t] = TWIN_RHO * ar[t - 1] + np.sqrt(
                1.0 - TWIN_RHO**2
            ) * TWIN_SIGMA_PROC * w[t]
        ar_noise[i] = ar
    seasonal = season_amp[:, None] * np.sin(
        2 * np.pi * (doy[None, :] - 152.0) / 365.0 + season_phase[:, None]
    )
    z_day = (
        base_level[:, None] + seasonal + ar_noise
        + np.log(TWIN_DORMANT_MW)
    )
    # Fire episodes: marked Poisson in the dry season. Each event ramps up
    # over ~2 days (fire growth), holds a plateau of absolute flux
    # uniform in ln(8 .. 150) MW for 4-9 days, then decays over ~2 days —
    # dynamics compatible with the AR(1) process prior. Episodes form
    # spatial clusters (fire complexes hitting 1-3 adjacent cells with a
    # 0-2 day lag), so the spatial prior reinforces real events instead of
    # fighting isolated single-cell spikes.
    episode_mask = np.zeros((B, T), dtype=bool)
    for i in range(B):
        t = 0
        while t < T:
            dry = 0.35 + 0.65 * max(
                0.0, np.sin(2 * np.pi * (doy[t] - 152.0) / 365.0)
            )
            if rng.random() < 0.025 * dry:
                dur = int(rng.integers(6, 13))
                level = rng.uniform(np.log(12.0), np.log(160.0))
                shape = np.concatenate(
                    [
                        np.array([0.55, 1.0]),
                        np.ones(dur),
                        np.linspace(1.0, 0.0, 3)[:-1],
                    ]
                )
                _apply_episode(z_day, episode_mask, i, t, level, shape)
                # spread to 1-2 neighbours with a short lag
                n_spread = int(rng.integers(0, 3))
                for nb in rng.choice(
                    adjacency[i], size=min(n_spread, len(adjacency[i])),
                    replace=False,
                ) if n_spread and len(adjacency[i]) else []:
                    lag = int(rng.integers(0, 3))
                    lvl = level + rng.normal(0.0, 0.25)
                    _apply_episode(
                        z_day, episode_mask, nb, t + lag, lvl, shape
                    )
                t += len(shape)
            else:
                t += 1
    # 5x drought anomaly for the spotlight cell: a sustained elevated burning
    # level (>= 50 MW day flux) from 2019-06 through 2020-02 — a Black
    # Summer style episode carrying far more than 5x the typical monthly flux.
    drought_lo = (pd.Timestamp("2019-06-01") - pd.Timestamp(S.DATE_START)).days
    drought_hi = (pd.Timestamp("2020-02-29") - pd.Timestamp(S.DATE_START)).days
    z_day[drought_cell, drought_lo : drought_hi + 1] = np.maximum(
        z_day[drought_cell, drought_lo : drought_hi + 1], np.log(50.0)
    )
    z_day = np.clip(z_day, np.log(TWIN_DORMANT_MW), S.Z_CLIP_HI)
    z_night = z_day + np.log(TWIN_NIGHT_DAY_RATIO)
    phi_day_true = np.exp(z_day)
    phi_night_true = np.exp(z_night)

    # --- cloud fields (50% mean + optional blind window) --------------------
    cloud = rng.uniform(0.0, 2.0 * cloud_mean, size=(B, T))
    cloud = np.clip(cloud, 0.0, 0.98)
    if blind_window is not None:
        cloud[:, blind_window[0] : blind_window[1] + 1] = 0.97

    # --- platform simulation -------------------------------------------------
    calendar = sv.build_platform_calendar(None, T)
    diurnal_norm = {
        (b, s, dn): op.normalize_diurnal_coeffs(TWIN_DIURNAL_COEFFS)
        for b in range(S.PYROME_K)
        for s in op.SEASONS
        for dn in ("D", "N")
    }
    lognorm = np.tile(np.array(TWIN_LOGNORM), (S.PYROME_K, 1))
    psi_cnt_table = sv.build_psi_cnt_table(lognorm, sv.DEFAULT_AVG_AREAS)
    k_table = sv.build_k_table(
        np.full(S.PYROME_K, TWIN_LAMBDA0),
        np.full(S.PYROME_K, TWIN_F_EVENT_MW),
        psi_cnt_table,
    )

    vis = {"D": np.zeros((B, T, S.N_PLATFORMS)),
           "N": np.zeros((B, T, S.N_PLATFORMS))}
    rows: List[dict] = []
    raw_pixel_counts = np.zeros((B, T), dtype=np.int64)
    raw_frp = np.zeros((B, T))
    for p_idx, platform in enumerate(S.PLATFORMS):
        family = S.PLATFORM_TIMELINE[platform].sensor_family
        sigma_obs = TWIN_SIGMA[family]
        op_mask = calendar[p_idx] > 0
        for dn in ("D", "N"):
            dn_i = 0 if dn == "D" else 1
            z_true = z_day if dn == "D" else z_night
            # Expected solar hour with post-2022 MODIS drift.
            taus = np.array(
                [
                    expected_drift_solar_hour(platform, t, dn, t_total_days=T)
                    for t in range(T)
                ]
            )
            ln_d = np.log(
                np.array(
                    [
                        op.diurnal_value(
                            diurnal_norm[(0, "JJA", dn)], float(taus[t]), dn
                        )
                        for t in range(T)
                    ]
                )
            )
            ln_gamma = TWIN_LN_GAMMA_MODIS if family == "MODIS" else 0.0
            # clear-sky visibility per (cell, t, platform)
            for b in range(B):
                lat = abs(latlon[b][0])
                if family == "VIIRS":
                    p_swath = 0.99
                else:
                    from src.geometry.solar_geometry import (
                        modis_swath_probability,
                    )

                    p_swath = np.array(
                        [modis_swath_probability(lat, t) for t in range(T)]
                    )
                v = p_swath * np.clip(1.0 - cloud[b] - 0.03, 0.10, 0.98)
                vis[dn][b, :, p_idx] = v
                k = k_table[0, p_idx, dn_i]
                lam = np.clip(k * phi_day_true[b] * (
                    1.0 if dn == "D" else TWIN_NIGHT_DAY_RATIO
                ), *S.LAMBDA_CLIP)
                detect_prob = v * (1.0 - np.exp(-lam))
                detected = (rng.uniform(size=T) < detect_prob) & op_mask
                for t in np.flatnonzero(detected):
                    e_raw = 1 + int(rng.poisson(1.5))
                    fmin = op.fmin_threshold(
                        sv.DEFAULT_AVG_AREAS[p_idx, dn_i], family, dn
                    )
                    ln_psi = np.log(
                        op.psi_frp_eff(
                            fmin, TWIN_LOGNORM[0], TWIN_LOGNORM[1], e_raw
                        )
                    )
                    latent = (
                        z_true[b, t] + ln_d[t] + ln_gamma + ln_psi
                    )
                    ln_y = latent + sigma_obs * rng.standard_normal()
                    sat_flag = 0
                    if family == "MODIS" and ln_y > np.log(
                        TWIN_MODIS_SAT_FRP_MW
                    ):
                        ln_y = np.log(TWIN_MODIS_SAT_FRP_MW) + abs(
                            sigma_obs * rng.standard_normal() * 0.05
                        )
                        sat_flag = 1
                    y = float(np.exp(ln_y))
                    n_pix = (
                        1 + rng.poisson(1.0) if family == "MODIS"
                        else 3 + rng.poisson(4)
                    )
                    # MODIS bow-tie duplication off-nadir doubles raw pixels
                    # (each weighted 0.5 downstream, conserving FRP).
                    if family == "MODIS" and rng.random() < 0.35:
                        n_pix *= 2
                    conf = float(rng.uniform(45.0, 99.0))
                    i4 = 1 if (
                        family == "VIIRS"
                        and rng.random() < 0.12
                        and y > 0
                    ) else 0
                    obs_sigma = sigma_obs * (
                        S.I4_M13_SIGMA_INFLATION if i4 else 1.0
                    )
                    rows.append(
                        {
                            "cell_pos": b,
                            "h3_index": int(cells[b]),
                            "t_index": int(t),
                            "solar_date": dates[t].strftime("%Y-%m-%d"),
                            "platform": platform,
                            "sensor_family": family,
                            "dn_flag": dn,
                            "y_frp_mw": y,
                            "sat_flag": sat_flag,
                            "mu_gain": float(ln_d[t] + ln_gamma + ln_psi),
                            "obs_sigma": obs_sigma,
                            "raw_event_count": e_raw,
                            "n_pixels": int(n_pix),
                            "mean_solar_hour": float(taus[t]),
                            "mean_vza_deg": 30.0,
                            "mean_area_km2": sv.DEFAULT_AVG_AREAS[p_idx, dn_i],
                            "mean_conf_num": conf,
                            "has_i4_m13_fallback": i4,
                            "pyrome_id": 0,
                        }
                    )
                    raw_pixel_counts[b, t] += n_pix
                    raw_frp[b, t] += y

    pass_df = pd.DataFrame(rows)

    # --- prior means from the 2012-2021 true climatology --------------------
    mu_day = np.zeros((B, T))
    mu_night = np.zeros((B, T))
    calib_lo = (pd.Timestamp(S.CALIB_START) - pd.Timestamp(S.DATE_START)).days
    calib_hi = (pd.Timestamp(S.CALIB_END) - pd.Timestamp(S.DATE_START)).days
    for b in range(B):
        for m in range(int(month_ids.max()) + 1):
            sel = month_ids == m
            in_calib = sel & (np.arange(T) >= calib_lo) & (
                np.arange(T) <= calib_hi
            )
            use = in_calib if in_calib.any() else sel
            mu_day[b, sel] = float(z_day[b, use].mean())
            mu_night[b, sel] = float(z_night[b, use].mean())

    priors = {
        "f_event": np.full(S.PYROME_K, TWIN_F_EVENT_MW),
        "rho": np.full(S.PYROME_K, TWIN_RHO),
        "sigma_proc": np.full(S.PYROME_K, TWIN_SIGMA_PROC),
        "lambda0": np.full(S.PYROME_K, TWIN_LAMBDA0),
        "beta_dn": np.full(S.PYROME_K, TWIN_BETA_DN),
        "eta_dn": np.full(S.PYROME_K, TWIN_ETA_DN),
        "pyrome": np.zeros(B, dtype=np.int64),
        "mu_day": mu_day,
        "mu_night": mu_night,
    }

    # monthly raw aggregates (for the raw-jump falsification tests)
    n_months = int(month_ids.max()) + 1
    raw_monthly_counts = np.zeros((n_months,))
    raw_monthly_frp = np.zeros((n_months,))
    for m in range(n_months):
        sel = month_ids == m
        raw_monthly_counts[m] = raw_pixel_counts[:, sel].sum()
        raw_monthly_frp[m] = raw_frp[:, sel].sum()

    return {
        "cells": cells,
        "latlon": latlon,
        "colors": colors,
        "neighbor_indices": np.array(nb_idx, dtype=np.int64),
        "neighbor_indptr": np.array(nb_ptr, dtype=np.int64),
        "phi_day_true": phi_day_true,
        "phi_night_true": phi_night_true,
        "z_day_true": z_day,
        "z_night_true": z_night,
        "pass_df": pass_df,
        "vis_matrix": vis,
        "platform_calendar": calendar,
        "k_table": k_table,
        "psi_cnt_table": psi_cnt_table,
        "lognorm_params": lognorm,
        "priors": priors,
        "month_ids": month_ids,
        "month_keys": month_keys,
        "dates": dates,
        "raw_monthly_counts": raw_monthly_counts,
        "raw_monthly_frp": raw_monthly_frp,
        "cloud": cloud,
        "diurnal_norm": diurnal_norm,
    }


def build_chunk_problem(
    twin: Dict[str, object],
    cell_positions: List[int],
    device,
    mask_platforms: Optional[Tuple[str, ...]] = None,
    mask_window: Optional[Tuple[int, int]] = None,
) -> sv.ChunkProblem:
    """Materialize one ChunkProblem from the twin (optionally VIIRS-masked).

    ``mask_platforms`` removes those platforms' observations (ablation
    folds); ``mask_window`` restricts the masking to a (t_lo, t_hi) range.
    """
    import torch

    df = twin["pass_df"]
    if mask_platforms:
        sel = ~df["platform"].isin(mask_platforms)
        if mask_window is not None:
            inside = (
                (df["t_index"] >= mask_window[0])
                & (df["t_index"] <= mask_window[1])
            )
            sel = ~(~sel & inside) | ~inside
            keep = df[inside & df["platform"].isin(mask_platforms)]
            sel = ~(df["platform"].isin(mask_platforms) & inside)
        df = df[sel]
    B_all = len(twin["cells"])
    T = twin["pass_df"]["t_index"].max() + 1
    if len(df):
        T = max(T, int(df["t_index"].max()) + 1)
    T = len(twin["month_ids"])
    cell_pos_map = {c: i for i, c in enumerate(cell_positions)}
    obs = sv.build_observation_tensors(
        df, cell_pos_map, len(cell_positions), T,
        twin["platform_calendar"], twin["k_table"],
        twin["priors"]["pyrome"], twin["vis_matrix"], device,
    )
    priors = twin["priors"]
    colors = twin["colors"][cell_positions]
    # remap neighbor positions to the chunk-local frame
    pos_map = {int(c): i for i, c in enumerate(cell_positions)}
    nb_idx, nb_ptr = [], [0]
    for i, c in enumerate(cell_positions):
        lo, hi = twin["neighbor_indptr"][c], twin["neighbor_indptr"][c + 1]
        for nb in twin["neighbor_indices"][lo:hi]:
            if int(nb) in pos_map:
                nb_idx.append(pos_map[int(nb)])
        nb_ptr.append(len(nb_idx))
    return sv.ChunkProblem(
        cell_ids=np.asarray(cell_positions),
        colors=colors,
        rho=np.full(len(cell_positions), TWIN_RHO),
        omega=np.full(
            len(cell_positions), 1.0 / TWIN_SIGMA_PROC**2
        ),
        eta_dn=np.full(len(cell_positions), TWIN_ETA_DN),
        beta_dn=np.full(len(cell_positions), TWIN_BETA_DN),
        mu_day=priors["mu_day"][cell_positions],
        mu_night=priors["mu_night"][cell_positions],
        month_ids=twin["month_ids"],
        obs=obs,
        neighbor_indices=np.array(nb_idx, dtype=np.int64),
        neighbor_indptr=np.array(nb_ptr, dtype=np.int64),
    )


def _apply_episode(
    z_day: np.ndarray, mask: np.ndarray, cell: int, t0: int,
    level: float, shape: np.ndarray,
) -> None:
    """Blend one ramp-hold-decay episode into cell ``cell`` starting ``t0``."""
    T = z_day.shape[1]
    start = max(t0, 0)
    end = min(t0 + len(shape), T)
    if end <= start:
        return
    s = shape[start - t0 : end - t0]
    z_day[cell, start:end] = np.maximum(
        z_day[cell, start:end],
        (1.0 - s) * z_day[cell, start:end] + s * level,
    )
    mask[cell, start:end] = True
