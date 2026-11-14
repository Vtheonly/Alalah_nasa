"""Deterministic Empirical-Bayes priors from 2012-2021 VIIRS data.

Contract (readme.md Sections 4 & 5.3.1):
  * ``F_event_b`` = max(1.0, median clear-sky near-nadir VIIRS cluster FRP).
  * Cell-month baseline ``mu_{g,m,DN}`` = empirical-Bayes shrinkage of the
    cell-month mean log-flux (clear zero-detection days treated as z_min)
    toward the pyrome-month mean with weight 5.0.
  * AR(1) persistence ``rho_b in [0.25, 0.90]`` and innovation std
    ``sigma_proc,b in [0.25, 1.50]`` via lag-1 OLS on consecutive active
    VIIRS cell-days.
  * Hurdle rate ``lambda0_b in [0.1, 5.0]`` via ``minimize_scalar`` matching
    the empirical zero-detection probability on active pyrome months.
  * Day-Night coupling ``(beta_dn,b, eta_dn,b)`` via OLS slope and variance
    ratio (``eta in [0.05, 0.30]``).
  * Sparse 3-color neighbor CSR indices for the spatial prior.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.config import settings as S


# --------------------------------------------------------------------------- #
# Sparse neighbor CSR packing
# --------------------------------------------------------------------------- #
def build_neighbor_csr(
    adjacency: Dict[int, List[int]], n_cells: int
) -> Dict[str, np.ndarray]:
    """Pack an adjacency map into CSR arrays (indices, indptr, degree)."""
    indptr = [0]
    indices: List[int] = []
    for i in range(n_cells):
        indices.extend(sorted(adjacency.get(i, [])))
        indptr.append(len(indices))
    return {
        "indices": np.asarray(indices, dtype=np.int64),
        "indptr": np.asarray(indptr, dtype=np.int64),
        "degree": np.diff(np.asarray(indptr)).astype(np.int64),
    }


# --------------------------------------------------------------------------- #
# Per-pyrome estimators
# --------------------------------------------------------------------------- #
def estimate_f_event(
    viirs_clusters_frp: pd.Series, pyrome_ids: np.ndarray
) -> np.ndarray:
    """Median clear near-nadir VIIRS cluster FRP per pyrome, floored at 1 MW."""
    out = np.full(S.PYROME_K, S.F_EVENT_MIN_MW)
    df = pd.DataFrame(
        {"frp": np.asarray(viirs_clusters_frp, dtype=float),
         "pyrome": pyrome_ids}
    )
    med = df.groupby("pyrome")["frp"].median()
    for b, val in med.items():
        out[int(b)] = max(S.F_EVENT_MIN_MW, float(val))
    return out


def estimate_ar1(
    series_by_cell: Dict[int, np.ndarray]
) -> Tuple[float, float]:
    """Lagged-1 OLS slope and residual std across concatenated cell series."""
    xs, ys = [], []
    for arr in series_by_cell.values():
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= 2:
            xs.append(arr[:-1])
            ys.append(arr[1:])
    if not xs:
        return float(np.clip(0.6, *S.AR1_RHO_RANGE)), 0.7
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    slope = float(np.cov(x, y)[0, 1] / max(np.var(x), 1e-12))
    resid = y - slope * x
    sigma = float(np.std(resid))
    rho = float(np.clip(slope, *S.AR1_RHO_RANGE))
    sig = float(np.clip(sigma, *S.SIGMA_PROC_RANGE))
    return rho, sig


def estimate_lambda0(
    empirical_zero_prob: float, psi_cnt_mean: float, f_event: float,
    mean_flux_mw: float,
) -> float:
    """Scalar lambda0 matching the empirical zero-detection probability.

    Solves ``exp(-lambda0 * psi_cnt * Phi / F_event) = p_zero`` for
    ``lambda0`` on active pyrome months via ``minimize_scalar`` inside
    [0.1, 5.0].
    """
    target = float(np.clip(empirical_zero_prob, 1e-6, 1.0 - 1e-9))

    def discrepancy(lam: float) -> float:
        rate = lam * psi_cnt_mean * max(mean_flux_mw, 1e-6) / max(f_event, 1e-6)
        pred = float(np.exp(-np.clip(rate, 0.0, 700.0)))
        return (pred - target) ** 2

    res = minimize_scalar(
        discrepancy,
        bounds=(S.HURDLE_LAMBDA0_RANGE[0], S.HURDLE_LAMBDA0_RANGE[1]),
        method="bounded",
    )
    lam = float(res.x)
    return float(np.clip(lam, *S.HURDLE_LAMBDA0_RANGE))


def estimate_day_night_coupling(
    day_anomalies: np.ndarray, night_anomalies: np.ndarray
) -> Tuple[float, float]:
    """OLS slope beta_dn and variance ratio eta_dn in [0.05, 0.30].

    ``beta_dn`` is the OLS slope of night anomalies on day anomalies;
    ``eta_dn`` is the ratio of residual variance to predictor variance,
    clipped into the spec range.
    """
    d = np.asarray(day_anomalies, dtype=float)
    n = np.asarray(night_anomalies, dtype=float)
    if len(d) < 3 or len(n) < 3 or len(d) != len(n):
        return 0.5, float(np.clip(0.15, *S.ETA_DN_RANGE))
    var_d = float(np.var(d))
    beta = float(np.cov(n, d)[0, 1] / max(var_d, 1e-12))
    resid = n - beta * d
    eta = float(np.var(resid) / max(var_d, 1e-12))
    eta = float(np.clip(eta, *S.ETA_DN_RANGE))
    return beta, eta


# --------------------------------------------------------------------------- #
# Cell-month empirical-Bayes baseline
# --------------------------------------------------------------------------- #
def estimate_cell_month_baseline(
    cell_logflux: Dict[int, np.ndarray],
    cell_month: Dict[int, np.ndarray],
    cell_dn: Dict[int, np.ndarray],
    cell_pyrome: Dict[int, int],
    pyrome_month_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrinkage baseline mu_{g, m, DN} for every cell.

    ``cell_logflux`` values already encode clear non-detection days as
    ``z_min = ln(0.01)`` (readme.md 5.3.1). Returns two ``(n_cells, 12, 2)``
    arrays (Day, Night) of shrunk cell-month means.
    """
    n_cells = len(cell_logflux)
    mu_day = np.zeros((n_cells, 12))
    mu_night = np.zeros((n_cells, 12))
    for g in range(n_cells):
        lf = np.asarray(cell_logflux[g], dtype=float)
        mo = np.asarray(cell_month[g], dtype=int)
        dn = np.asarray(cell_dn[g], dtype=int)
        b = int(cell_pyrome.get(g, 0))
        for m in range(12):
            for dn_idx, out in ((1, mu_day), (0, mu_night)):
                sel = (mo == m) & (dn == dn_idx)
                n_gm = int(sel.sum())
                if n_gm > 0:
                    ybar = float(lf[sel].mean())
                else:
                    ybar = float(pyrome_month_mean[b, m, dn_idx])
                mu_bar = float(pyrome_month_mean[b, m, dn_idx])
                out[g, m] = (
                    n_gm * ybar + S.EB_SHRINKAGE_WEIGHT * mu_bar
                ) / (n_gm + S.EB_SHRINKAGE_WEIGHT)
    return mu_day, mu_night


def pyrome_month_means(
    logflux: np.ndarray,
    months: np.ndarray,
    dns: np.ndarray,
    pyromes: np.ndarray,
) -> np.ndarray:
    """Pyrome x month x DN mean log-flux (fallback 0 = dormant-ish)."""
    out = np.zeros((S.PYROME_K, 12, 2))
    counts = np.zeros((S.PYROME_K, 12, 2))
    for lf, m, dn, b in zip(logflux, months, dns, pyromes):
        out[int(b), int(m), int(dn)] += lf
        counts[int(b), int(m), int(dn)] += 1
    out /= np.maximum(counts, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Full prior bundle
# --------------------------------------------------------------------------- #
def estimate_priors(
    pass_obs: pd.DataFrame,
    hex_pyromes: pd.DataFrame,
    adjacency: Dict[int, List[int]],
    n_cells: int,
) -> Dict[str, object]:
    """Estimate every Empirical-Bayes prior from calibrated pass observations.

    ``pass_obs`` must carry columns ``cell_pos, t_index, dn_flag, y_frp_mw,
    sensor_family, pyrome_id, mu_gain, obs_sigma, k_miss, vis, is_op_pass``.
    """
    viirs = pass_obs[
        (pass_obs["sensor_family"] == "VIIRS") & (pass_obs["y_frp_mw"] > 0)
    ]
    pyrome_of_cell = dict(
        zip(hex_pyromes["cell_pos"], hex_pyromes["pyrome_id"])
    )

    # F_event per pyrome from clear near-nadir VIIRS cluster FRP.
    f_event = estimate_f_event(
        viirs["y_frp_mw"], viirs["pyrome_id"].to_numpy()
    )

    # AR(1) per pyrome on consecutive active VIIRS cell-days.
    rho = np.full(S.PYROME_K, 0.6)
    sigma_proc = np.full(S.PYROME_K, 0.7)
    for b in range(S.PYROME_K):
        sub = viirs[viirs["pyrome_id"] == b]
        series: Dict[int, np.ndarray] = {}
        for cell, part in sub.groupby("cell_pos"):
            ts = np.sort(part["t_index"].to_numpy())
            lf = part.sort_values("t_index")[
                "y_frp_mw"
            ].to_numpy(dtype=float)
            lf = np.log(np.maximum(lf, 1e-6))
            keep = np.concatenate(([True], np.diff(ts) == 1))
            series[int(cell)] = lf[keep]
        rho[b], sigma_proc[b] = estimate_ar1(series)

    # lambda0 per pyrome matching empirical zero-detection probability.
    lambda0 = np.zeros(S.PYROME_K)
    for b in range(S.PYROME_K):
        sub = pass_obs[pass_obs["pyrome_id"] == b]
        op = sub[sub["is_op_pass"] > 0]
        if len(op):
            p_zero = float((op["y_frp_mw"] <= 0).mean())
        else:
            p_zero = 0.5
        mean_flux = float(
            np.exp(
                np.log(
                    np.maximum(
                        op.loc[op["y_frp_mw"] > 0, "y_frp_mw"]
                    ).mean()
                )
            )
        ) if (op["y_frp_mw"] > 0).any() else 1.0
        psi_mean = 0.35
        lambda0[b] = estimate_lambda0(p_zero, psi_mean, f_event[b],
                                      mean_flux)

    # Day-Night coupling per pyrome on anomalies vs the cell mean.
    beta_dn = np.full(S.PYROME_K, 0.5)
    eta_dn = np.full(S.PYROME_K, 0.15)
    for b in range(S.PYROME_K):
        sub = viirs[viirs["pyrome_id"] == b]
        if len(sub) == 0:
            continue
        day = sub[sub["dn_flag"] == "D"]
        night = sub[sub["dn_flag"] == "N"]
        by_key_day = day.groupby("cell_pos")["y_frp_mw"].mean()
        by_key_night = night.groupby("cell_pos")["y_frp_mw"].mean()
        common = sorted(set(by_key_day.index) & set(by_key_night.index))
        if len(common) >= 3:
            d = np.log(
                np.maximum(by_key_day.loc[common].to_numpy(), 1e-6)
            )
            n = np.log(
                np.maximum(by_key_night.loc[common].to_numpy(), 1e-6)
            )
            beta_dn[b], eta_dn[b] = estimate_day_night_coupling(d, n)

    csr = build_neighbor_csr(adjacency, n_cells)
    return {
        "f_event": f_event,
        "rho": rho,
        "sigma_proc": sigma_proc,
        "lambda0": lambda0,
        "beta_dn": beta_dn,
        "eta_dn": eta_dn,
        "neighbor_csr": csr,
        "n_cells": n_cells,
        "pyrome_of_cell": pyrome_of_cell,
    }
