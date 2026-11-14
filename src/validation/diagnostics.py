"""Scientific validation diagnostics (readme.md Sections 4 & 5.4).

* Chow and Pettitt structural-break tests on raw vs harmonized series.
* 2019-2021 VIIRS-masked ablation metrics: CRPS, log-RMSE, 90% coverage,
  Spearman rho, Top-3 Peak Month Match.
* MCD64CMQ burned-area correlation comparison (harmonized FRE vs raw counts).
* Residual VZA / solar-time slope checks.
* Spotlight audit reports (Chilecito ARG 2026, Kabylie & Hassi Messaoud DZA,
  Nov 2, 2026 S-NPP end).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import settings as S


# --------------------------------------------------------------------------- #
# Structural break tests
# --------------------------------------------------------------------------- #
def chow_test(series: np.ndarray, break_index: int) -> Dict[str, float]:
    """Chow F-test for a mean break at ``break_index`` (2-segment OLS)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    k = 1  # one regressor (constant)
    if not (0 < break_index < n - 2):
        return {"F": float("nan"), "p_value": float("nan")}
    # restricted: single mean
    ssr_r = float(((x - x.mean()) ** 2).sum())
    seg1, seg2 = x[:break_index], x[break_index:]
    ssr_u = float(
        ((seg1 - seg1.mean()) ** 2).sum()
        + ((seg2 - seg2.mean()) ** 2).sum()
    )
    dof = n - 2 * k
    if dof <= 0 or ssr_u <= 0:
        return {"F": float("nan"), "p_value": float("nan")}
    f_stat = ((ssr_r - ssr_u) / k) / (ssr_u / dof)
    p_value = float(stats.f.sf(f_stat, k, dof))
    return {"F": float(f_stat), "p_value": p_value}


def pettitt_test(series: np.ndarray) -> Dict[str, float]:
    """Pettitt non-parametric change-point test."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 4:
        return {"K": 0.0, "p_value": 1.0, "change_index": 0}
    ranks = stats.rankdata(x)
    U = 2.0 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1.0)
    k_idx = int(np.argmax(np.abs(U)))
    K = float(abs(U[k_idx]))
    p = float(2.0 * np.exp((-6.0 * K**2) / (n**3 + n**2)))
    return {"K": K, "p_value": min(p, 1.0), "change_index": k_idx}


# --------------------------------------------------------------------------- #
# Ablation metrics
# --------------------------------------------------------------------------- #
def crps_from_quantiles(
    obs: np.ndarray, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray
) -> float:
    """Quantile-based CRPS: 2 x mean pinball loss over a dense quantile grid
    interpolated from the (q05, q50, q95) triplet."""
    obs = np.asarray(obs, dtype=float)
    levels = np.linspace(0.02, 0.98, 25)
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    qs = np.column_stack([q05, q50, q95])  # levels 0.05, 0.5, 0.95
    knot_levels = np.array([0.05, 0.50, 0.95])
    dense = np.stack(
        [np.interp(levels, knot_levels, qs[i]) for i in range(len(obs))],
        axis=0,
    )
    pinball = np.maximum(
        levels[None, :] * (obs[:, None] - dense),
        (levels[None, :] - 1.0) * (obs[:, None] - dense),
    )
    return float(2.0 * pinball.mean())


def crps_point(obs: np.ndarray, pred: np.ndarray) -> float:
    """CRPS of a deterministic forecast equals the MAE."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(obs - pred)))


def log_rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(
        np.sqrt(
            np.mean(
                (np.log(np.maximum(obs, 1e-6)) - np.log(np.maximum(pred, 1e-6)))
                ** 2
            )
        )
    )


def coverage_90(obs, q05, q95) -> float:
    obs = np.asarray(obs, dtype=float)
    return float(
        np.mean(
            (obs >= np.asarray(q05, dtype=float))
            & (obs <= np.asarray(q95, dtype=float))
        )
    )


def spearman_rho(a, b) -> float:
    return float(stats.spearmanr(np.asarray(a), np.asarray(b)).statistic)


def top3_peak_month_match(
    harmonized: np.ndarray, reference: np.ndarray
) -> int:
    """Overlap size between the top-3 months of each series."""
    h = np.argsort(np.asarray(harmonized))[::-1][:3]
    r = np.argsort(np.asarray(reference))[::-1][:3]
    return int(len(set(h.tolist()) & set(r.tolist())))


# --------------------------------------------------------------------------- #
# Cross-validation vs independent burned area
# --------------------------------------------------------------------------- #
def burned_area_correlation(
    harmonized_fre: np.ndarray,
    raw_counts: np.ndarray,
    burned_area: np.ndarray,
) -> Dict[str, float]:
    """Harmonized monthly FRE should correlate more strongly with
    independent MCD64CMQ burned area than unharmonized raw counts."""
    return {
        "harmonized_fre_corr": spearman_rho(harmonized_fre, burned_area),
        "raw_counts_corr": spearman_rho(raw_counts, burned_area),
    }


# --------------------------------------------------------------------------- #
# Residual slopes (VZA / solar-time confounding checks)
# --------------------------------------------------------------------------- #
def residual_slope(
    residuals: np.ndarray, covariate: np.ndarray
) -> Dict[str, float]:
    """OLS slope of residuals vs a covariate + significance."""
    residuals = np.asarray(residuals, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    if len(residuals) < 4:
        return {"slope": 0.0, "p_value": 1.0}
    result = stats.linregress(covariate, residuals)
    return {"slope": float(result.slope), "p_value": float(result.pvalue)}


# --------------------------------------------------------------------------- #
# Spotlight reports
# --------------------------------------------------------------------------- #
def spotlight_report(
    name: str,
    bbox: Tuple[float, float, float, float],
    cell_latlon: np.ndarray,
    monthly_q50: np.ndarray,
    month_keys: List[Tuple[int, int]],
) -> Dict[str, object]:
    """Aggregate harmonized activity inside a spotlight bounding box."""
    lat = cell_latlon[:, 0]
    lon = cell_latlon[:, 1]
    inside = (
        (lon >= bbox[0]) & (lon <= bbox[2])
        & (lat >= bbox[1]) & (lat <= bbox[3])
    )
    series = (
        np.nanmean(monthly_q50[inside], axis=0)
        if inside.any() else np.zeros(len(month_keys))
    )
    keys = [f"{y}-{m:02d}" for (y, m) in month_keys]
    peak_i = int(np.nanargmax(series)) if np.isfinite(series).any() else 0
    return {
        "name": name,
        "bbox": list(bbox),
        "n_cells": int(inside.sum()),
        "peak_month": keys[peak_i],
        "peak_phi_q50_mw": float(series[peak_i]),
        "mean_phi_q50_mw": float(np.nanmean(series)),
    }


def run_diagnostics(
    raw_monthly: np.ndarray,
    harmonized_monthly: np.ndarray,
    month_keys: List[Tuple[int, int]],
    ablation: Optional[Dict[str, np.ndarray]] = None,
    truth_monthly: Optional[np.ndarray] = None,
    burned_area: Optional[np.ndarray] = None,
    cell_latlon: Optional[np.ndarray] = None,
    monthly_q50_by_cell: Optional[np.ndarray] = None,
    residuals: Optional[np.ndarray] = None,
    vza: Optional[np.ndarray] = None,
    solar_hour: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Full validation suite -> JSON-serializable diagnostics dict."""
    keys = [f"{y}-{m:02d}" for (y, m) in month_keys]
    t_break_2012 = next(
        i for i, k in enumerate(keys) if k >= "2012-01"
    )
    raw_break = chow_test(raw_monthly, t_break_2012)
    harm_break = chow_test(harmonized_monthly, t_break_2012)
    raw_pettitt = pettitt_test(raw_monthly)
    harm_pettitt = pettitt_test(harmonized_monthly)

    diag: Dict[str, object] = {
        "chow_2012": {
            "raw": raw_break,
            "harmonized": harm_break,
            "raw_p_below_0_001": bool(raw_break["p_value"] < 0.001),
            "harmonized_p_above_0_05": bool(harm_break["p_value"] > 0.05),
        },
        "pettitt": {
            "raw": raw_pettitt,
            "harmonized": harm_pettitt,
        },
    }
    if ablation is not None and truth_monthly is not None:
        diag["ablation_2019_2021"] = ablation
    if burned_area is not None:
        diag["burned_area"] = burned_area
    if cell_latlon is not None and monthly_q50_by_cell is not None:
        diag["spotlights"] = {
            name: spotlight_report(
                name, preset["bbox"], cell_latlon, monthly_q50_by_cell,
                month_keys,
            )
            for name, preset in S.SPOTLIGHT_PRESETS.items()
        }
    if residuals is not None and vza is not None:
        diag["residual_vza_slope"] = residual_slope(residuals, vza)
    if residuals is not None and solar_hour is not None:
        diag["residual_solar_time_slope"] = residual_slope(
            residuals, solar_hour
        )
    return diag


def save_diagnostics(diag: Dict[str, object], out_json) -> None:
    from pathlib import Path

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)
