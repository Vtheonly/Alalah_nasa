"""Split Conformal Quantile Rescaling (CQR), flags, and calendar indices.

Contract (readme.md Sections 4 & 5.4):
  * Multiplicative log-space CQR multipliers from held-out 2018-2020
    VIIRS-masked folds (MODIS-only) and 2018-2021 leave-one-VIIRS-platform-out
    folds (multi-sensor), guaranteeing non-negative calibrated intervals
    ``0 <= q05_cal <= q50 <= q95_cal`` with multipliers in [0.85, 3.5].
  * ``ABSTAIN_LOW_EVIDENCE`` when >= 5 consecutive days in the month have
    mean cloud fraction > 0.85 AND zero detections across the cell and its
    1-ring neighbours; ``CLIMATOLOGY_CLOUD_FALLBACK`` when the cloud value
    came from the 2012-2025 climatology; else ``NOMINAL``.
  * ``attribution_flag`` boolean rules (REAL_FIRE_CHANGE /
    SENSOR_ARTIFACT_CORRECTED / AMBIGUOUS_OVERLAPPING_CI /
    WITHIN_NORMAL_VARIABILITY).
  * Hurdle-adjusted event counts that MEAN across detecting same-orbit
    platforms (INV-02: never sum SNPP + N20 + N21) and
    ``energy_equiv_events_q50 = phi_q50 / F_event_b``.
  * 2003-2020 climatology indices (hist median, anomaly ratio / z-score /
    percentile).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import settings as S


# --------------------------------------------------------------------------- #
# Split conformal quantile rescaling
# --------------------------------------------------------------------------- #
def conformal_quantile(
    scores: np.ndarray,
    alpha: float = S.CONFORMAL_ALPHA,
    clip: Tuple[float, float] = S.CONFORMAL_C_RANGE,
) -> float:
    """Finite-sample conformal order statistic clipped to [0.85, 3.5].

    ``k = ceil((n + 1) * (1 - alpha))`` (1-indexed); the multiplier is the
    k-th smallest nonconformity score.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float(clip[0])
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    ordered = np.sort(scores)
    return float(np.clip(ordered[k - 1], *clip))


def interval_half_widths(
    q05: np.ndarray, q95: np.ndarray
) -> np.ndarray:
    """s_j = max((ln(q95+1) - ln(q05+1)) / 2, 0.05)."""
    q05 = np.asarray(q05, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    return np.maximum(
        (np.log(q95 + 1.0) - np.log(np.maximum(q05, 0.0) + 1.0)) / 2.0,
        S.CONFORMAL_MIN_SCORE,
    )


def nonconformity_scores(
    ref: np.ndarray, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(s_j, r_j)`` score pairs for CQR calibration."""
    s = interval_half_widths(q05, q95)
    ref = np.asarray(ref, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    r = np.abs(np.log(ref + 1.0) - np.log(np.maximum(q50, 0.0) + 1.0)) / s
    return s, r


def compute_cqr_multiplier(
    ref: np.ndarray,
    q05_raw: np.ndarray,
    q50_raw: np.ndarray,
    q95_raw: np.ndarray,
    alpha: float = S.CONFORMAL_ALPHA,
) -> float:
    """One CQR multiplier for a fold of held-out reference tiles."""
    _, r = nonconformity_scores(ref, q05_raw, q50_raw, q95_raw)
    return conformal_quantile(r, alpha=alpha)


def rescale_intervals(
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiplicative log-space rescaling around the median.

    ``q05_cal = max(0, exp(ln(q50+1) - c*s) - 1)``,
    ``q95_cal = exp(ln(q50+1) + c*s) - 1``.
    """
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    s = interval_half_widths(q05, q95)
    ln_mid = np.log(np.maximum(q50, 0.0) + 1.0)
    q05_cal = np.maximum(0.0, np.exp(ln_mid - c * s) - 1.0)
    q95_cal = np.maximum(0.0, np.exp(ln_mid + c * s) - 1.0)
    return q05_cal, q95_cal


def empirical_coverage(
    ref: np.ndarray, q05_cal: np.ndarray, q95_cal: np.ndarray
) -> float:
    ref = np.asarray(ref, dtype=float)
    return float(
        np.mean((ref >= np.asarray(q05_cal)) & (ref <= np.asarray(q95_cal)))
    )


# --------------------------------------------------------------------------- #
# Diagnostic flags
# --------------------------------------------------------------------------- #
def evaluate_quality_flags(
    daily_cloud_frac: np.ndarray,
    daily_has_detection: np.ndarray,
    month_day_mask: np.ndarray,
    cloud_source: str = "CMG",
) -> str:
    """Quality flag for one cell-month.

    ``ABSTAIN_LOW_EVIDENCE`` iff >= 5 consecutive days inside the month
    have cloud fraction > 0.85 AND zero detections across the cell and its
    1-ring neighbours (``daily_has_detection`` is the cell+ring OR).
    """
    cloud = np.asarray(daily_cloud_frac, dtype=float)
    det = np.asarray(daily_has_detection, dtype=bool)
    in_month = np.asarray(month_day_mask, dtype=bool)
    run = 0
    for t in range(len(cloud)):
        if not in_month[t]:
            run = 0
            continue
        if cloud[t] > S.ABSTAIN_CLOUD_FRAC and not det[t]:
            run += 1
            if run >= S.ABSTAIN_CONSECUTIVE_DAYS:
                return "ABSTAIN_LOW_EVIDENCE"
        else:
            run = 0
    if cloud_source == S.CLOUD_CLIMATOLOGY_FLAG:
        return "CLIMATOLOGY_CLOUD_FALLBACK"
    return "NOMINAL"


def attribution_flag(
    quality_flag: str,
    q05_cal: float,
    q95_cal: float,
    harm_hist_med: float,
    raw_frp_sum: float,
    raw_hist_med: float,
) -> str:
    """Exact attribution boolean rules (readme.md 5.4 item 3)."""
    if quality_flag == "ABSTAIN_LOW_EVIDENCE":
        return "AMBIGUOUS_OVERLAPPING_CI"
    delta_raw = (raw_frp_sum - raw_hist_med) / max(raw_hist_med, 1.0)
    if q05_cal > harm_hist_med or q95_cal < harm_hist_med:
        return "REAL_FIRE_CHANGE"
    if abs(delta_raw) > 0.50 and q05_cal <= harm_hist_med <= q95_cal:
        return "SENSOR_ARTIFACT_CORRECTED"
    return "WITHIN_NORMAL_VARIABILITY"


# --------------------------------------------------------------------------- #
# Event count products (INV-02 compliant)
# --------------------------------------------------------------------------- #
def energy_equiv_events_q50(
    phi_q50_mw: float, f_event_mw: float
) -> float:
    """Energy-equivalent event count = phi_q50 / F_event_b."""
    return float(phi_q50_mw) / max(float(f_event_mw), 1e-9)


def hurdle_adj_events(
    pass_rows: pd.DataFrame,
    psi_cnt_by_pass: np.ndarray,
    vis_by_pass: np.ndarray,
    measured_fraction: float,
    raw_clustered_events: float,
) -> float:
    """Hurdle-adjusted cluster events for one cell-month.

    Within each detected half-day the raw event count is MEANed across the
    detecting platforms (never summing SNPP + N20 + N21, INV-02), each
    divided by ``psi_CNT * v``; the monthly total scales unobserved blind
    days by ``1 / max(measured_fraction, 0.25)`` capped at
    ``5 * raw_clustered_events``.
    """
    if len(pass_rows) == 0:
        return 0.0
    halfday = pass_rows[
        ["h3_index", "solar_date", "dn_flag"]
    ].drop_duplicates().index
    total = 0.0
    for _, hd in pass_rows.groupby(
        ["h3_index", "solar_date", "dn_flag"], sort=True
    ):
        e_vals = (
            hd["raw_event_count"].to_numpy(dtype=float)
            / np.maximum(
                np.asarray(psi_cnt_by_pass, dtype=float)
                * np.maximum(np.asarray(vis_by_pass, dtype=float), 0.05),
                1e-6,
            )
        )
        total += float(np.mean(e_vals))
    scale = 1.0 / max(float(measured_fraction), 0.25)
    capped = min(
        total * scale, 5.0 * max(float(raw_clustered_events), 0.0)
    )
    return float(max(capped, 0.0))


# --------------------------------------------------------------------------- #
# 2003-2020 climatology calendar indices
# --------------------------------------------------------------------------- #
def climatology_indices(
    phi_q50_series: pd.Series,
    year: int,
    month: int,
) -> Dict[str, float]:
    """Historical median / anomaly ratio / z-score / percentile for one
    cell-month against the 2003-2020 same-calendar-month climatology."""
    clim = phi_q50_series[
        (phi_q50_series.index.get_level_values("year").astype(int)
         >= S.CLIM_START_YEAR)
        & (phi_q50_series.index.get_level_values("year").astype(int)
           <= S.CLIM_END_YEAR)
        & (phi_q50_series.index.get_level_values("month").astype(int) == month)
    ]
    current = float(
        phi_q50_series.get((year, month), np.nan)
    )
    hist_median = float(clim.median()) if len(clim) else 0.0
    anomaly_ratio = current / max(hist_median, 0.5)
    if len(clim) >= 2 and np.isfinite(current):
        sd = float(clim.std()) if len(clim) >= 2 else 0.0
        anomaly_zscore = (
            (current - float(clim.mean())) / sd if sd > 1e-9 else 0.0
        )
        climatology_pctile = float(
            (clim < current).mean()
        )
    else:
        anomaly_zscore = 0.0
        climatology_pctile = 0.5
    return {
        "hist_median_2003_2020": hist_median,
        "anomaly_ratio": anomaly_ratio,
        "anomaly_zscore": anomaly_zscore,
        "climatology_pctile": climatology_pctile,
    }
