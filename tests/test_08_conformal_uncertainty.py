"""test_08: split-conformal coverage, flags, and fixed-lag assimilation.

12 tests (readme.md Section 7 item 8): split-conformal coverage inside
[1 - alpha - 0.03, 1 - alpha + 0.05] for alpha in {0.05, 0.10, 0.20};
q05 <= q50 <= q95 non-negativity; 2005 uncertainty strictly wider than
2024; cloudy uncertainty wider than clear months; ABSTAIN_LOW_EVIDENCE
trigger; and the 14-day fixed-lag update matching the full batch (< 1e-3)
across an active fire transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import settings as S
from src.engine import conformal as cf
from src.engine import solver as sv
from src.validation.synthetic_twin import TWIN_F_EVENT_MW

torch.set_default_dtype(torch.float64)


def _gaussian_fold(alpha, seed, n_cal=3000, n_test=5000):
    """Calibrate CQR on a half-normal (non-negative flux-like) fold."""
    rng = np.random.default_rng(seed)
    cal = np.abs(rng.normal(size=n_cal))
    test = np.abs(rng.normal(size=n_test))
    q05_c = float(np.quantile(cal, 0.05))
    q50_c = float(np.quantile(cal, 0.50))
    q95_c = float(np.quantile(cal, 0.95))
    c = cf.compute_cqr_multiplier(
        cal, np.full(n_cal, q05_c), np.full(n_cal, q50_c),
        np.full(n_cal, q95_c), alpha=alpha,
    )
    q05_t = np.full(n_test, q05_c)
    q50_t = np.full(n_test, q50_c)
    q95_t = np.full(n_test, q95_c)
    q05_cal, q95_cal = cf.rescale_intervals(q05_t, q50_t, q95_t, c)
    return cf.empirical_coverage(test, q05_cal, q95_cal), c


# --------------------------------------------------------------------------- #
# Conformal machinery
# --------------------------------------------------------------------------- #

def test_08_conformal_order_statistic():
    # Scores inside the multiplier clip range [0.85, 3.5].
    scores = np.linspace(0.90, 3.40, 100)  # sorted, 100 values
    # alpha = 0.10: k = ceil(101 * 0.9) = 91 -> 91st smallest.
    assert cf.conformal_quantile(scores, alpha=0.10) == pytest.approx(
        scores[90]
    )
    # alpha = 0.20: k = ceil(101 * 0.8) = 81.
    assert cf.conformal_quantile(scores, alpha=0.20) == pytest.approx(
        scores[80]
    )
    # n = 10, alpha = 0.10: k = ceil(11 * 0.9) = 10 -> clipped to n.
    small = np.linspace(1.0, 2.0, 10)
    assert cf.conformal_quantile(small, alpha=0.10) == pytest.approx(
        small[9]
    )


def test_08_conformal_multiplier_clip():
    huge = np.full(50, 100.0)
    assert cf.conformal_quantile(huge) == S.CONFORMAL_C_RANGE[1]
    tiny = np.full(50, 0.01)
    assert cf.conformal_quantile(tiny) == S.CONFORMAL_C_RANGE[0]


def test_08_coverage_alpha_05():
    coverage, c = _gaussian_fold(0.05, seed=11)
    assert 0.92 <= coverage <= 1.00, (coverage, c)


def test_08_coverage_alpha_10():
    coverage, c = _gaussian_fold(0.10, seed=12)
    assert 0.87 <= coverage <= 0.95, (coverage, c)


def test_08_coverage_alpha_20():
    coverage, c = _gaussian_fold(0.20, seed=13)
    assert 0.77 <= coverage <= 0.85, (coverage, c)


def test_08_rescaled_intervals_ordered_and_nonnegative():
    q50 = np.array([0.0, 1.0, 5.0, 100.0])
    q05 = np.array([0.0, 0.5, 2.0, 40.0])
    q95 = np.array([0.2, 2.0, 12.0, 250.0])
    for c in (0.85, 1.0, 2.0, 3.5):
        q05_cal, q95_cal = cf.rescale_intervals(q05, q50, q95, c)
        assert np.all(q05_cal >= 0.0)
        assert np.all(q05_cal <= q50 + 1e-12)
        assert np.all(q50 <= q95_cal + 1e-12)
        # Wider multipliers widen the interval monotonically.
    q05_a, q95_a = cf.rescale_intervals(q05, q50, q95, 0.85)
    q05_b, q95_b = cf.rescale_intervals(q05, q50, q95, 2.0)
    assert np.all(q05_b <= q05_a) and np.all(q95_b >= q95_a)


def test_08_modis_only_multiplier_wider_than_multi_sensor():
    """MODIS-only tiles (wide intervals + per-tile bias) require a larger
    CQR multiplier than multi-sensor tiles (tight, unbiased)."""
    rng = np.random.default_rng(7)
    ref = np.abs(rng.normal(5.0, 1.5, 1200))
    # MODIS-only: wide raw intervals and a random per-tile multiplicative
    # bias (unharmonized sensor artifact).
    bias = np.exp(rng.normal(-0.35, 0.35, 1200))
    q50_m = ref * bias
    q05_m = q50_m * 0.45
    q95_m = q50_m * 2.4
    # multi-sensor: tight and well-centred intervals.
    q05_s = ref * 0.85
    q50_s = ref * 1.0
    q95_s = ref * 1.15
    c_modis = cf.compute_cqr_multiplier(ref, q05_m, q50_m, q95_m)
    c_multi = cf.compute_cqr_multiplier(ref, q05_s, q50_s, q95_s)
    assert c_modis > c_multi, (c_modis, c_multi)
    assert c_modis > 1.0  # genuine widening beyond the raw interval
    assert S.CONFORMAL_C_RANGE[0] <= c_modis <= S.CONFORMAL_C_RANGE[1]


# --------------------------------------------------------------------------- #
# Diagnostic flags & event products
# --------------------------------------------------------------------------- #

def test_08_abstain_low_evidence_trigger():
    # 10-day month, 8 consecutive heavily-clouded no-detection days.
    cloud = np.array([0.2, 0.3, 0.9, 0.95, 0.97, 0.99, 0.96, 0.94, 0.92, 0.3])
    det = np.zeros(10, dtype=bool)
    det[0] = True
    month_mask = np.ones(10, dtype=bool)
    flag = cf.evaluate_quality_flags(cloud, det, month_mask)
    assert flag == "ABSTAIN_LOW_EVIDENCE"
    # A detection inside the blind run breaks the streak.
    det2 = det.copy()
    det2[5] = True
    assert cf.evaluate_quality_flags(cloud, det2, month_mask) == "NOMINAL"
    # Cloud below 0.85 never abstains.
    cloud3 = np.full(10, 0.84)
    assert cf.evaluate_quality_flags(cloud3, np.zeros(10, bool), month_mask) \
        == "NOMINAL"
    # Climatology-flagged cloud month.
    assert cf.evaluate_quality_flags(
        np.full(10, 0.3), np.zeros(10, bool), month_mask,
        cloud_source=S.CLOUD_CLIMATOLOGY_FLAG,
    ) == "CLIMATOLOGY_CLOUD_FALLBACK"


def test_08_attribution_flag_rules():
    # ABSTAIN dominates.
    assert cf.attribution_flag(
        "ABSTAIN_LOW_EVIDENCE", 1.0, 2.0, 1.5, 10.0, 5.0
    ) == "AMBIGUOUS_OVERLAPPING_CI"
    # CI entirely above the historical median -> real change.
    assert cf.attribution_flag(
        "NOMINAL", 3.0, 6.0, 1.5, 10.0, 5.0
    ) == "REAL_FIRE_CHANGE"
    # CI entirely below -> real change (downward).
    assert cf.attribution_flag(
        "NOMINAL", 0.1, 0.5, 1.5, 0.2, 5.0
    ) == "REAL_FIRE_CHANGE"
    # Raw jumped > 50% but harmonized CI spans the median -> sensor artifact.
    assert cf.attribution_flag(
        "NOMINAL", 1.0, 2.0, 1.5, 12.0, 5.0
    ) == "SENSOR_ARTIFACT_CORRECTED"
    # Everything within variability.
    assert cf.attribution_flag(
        "NOMINAL", 1.0, 2.0, 1.5, 6.0, 5.0
    ) == "WITHIN_NORMAL_VARIABILITY"


def test_08_event_counts_respect_inv02():
    # energy-equivalent events: phi / F_event
    assert cf.energy_equiv_events_q50(50.0, TWIN_F_EVENT_MW) == 10.0
    # hurdle-adjusted events MEAN across same-orbit platforms (never sum):
    # SNPP + N20 + N21 each detect 2 clusters, psi = 1, v = 1 -> E = 2.
    rows = pd.DataFrame(
        {
            "h3_index": [1, 1, 1],
            "solar_date": ["2020-08-05"] * 3,
            "dn_flag": ["D"] * 3,
            "platform": ["SNPP", "NOAA-20", "NOAA-21"],
            "raw_event_count": [2, 2, 2],
        }
    )
    e = cf.hurdle_adj_events(
        rows, np.ones(3), np.ones(3),
        measured_fraction=1.0, raw_clustered_events=2.0,
    )
    assert e == pytest.approx(2.0)  # mean(2,2,2) = 2, NOT 6
    # Cloud-scaled blind-day inflation, capped at 5x raw events.
    e_blind = cf.hurdle_adj_events(
        rows, np.ones(3), np.full(3, 0.4),
        measured_fraction=0.4, raw_clustered_events=2.0,
    )
    assert e_blind <= 5.0 * 2.0


# --------------------------------------------------------------------------- #
# Uncertainty ordering + fixed-lag assimilation
# --------------------------------------------------------------------------- #

def test_08_twin_2005_wider_than_2024_after_conformal(twin, twin_solved):
    idx = {
        f"{y}-{m:02d}": i for i, (y, m) in enumerate(twin["month_keys"])
    }
    q05, q50, q95 = (
        twin_solved["monthly_q05"], twin_solved["monthly_q50"],
        twin_solved["monthly_q95"],
    )
    # MODIS-only era multiplier exceeds the multi-sensor one (twin eras).
    ids_modis = [i for k, i in idx.items()
                 if 2004 <= int(k[:4]) <= 2006]
    ids_multi = [i for k, i in idx.items()
                 if 2023 <= int(k[:4]) <= 2025]
    ref_m = np.nanmean(q50[:, ids_modis], axis=0)
    c_modis = cf.compute_cqr_multiplier(
        ref_m, np.nanmean(q05[:, ids_modis], axis=0), ref_m,
        np.nanmean(q95[:, ids_modis], axis=0),
    )
    ref_s = np.nanmean(q50[:, ids_multi], axis=0)
    c_multi = cf.compute_cqr_multiplier(
        ref_s, np.nanmean(q05[:, ids_multi], axis=0), ref_s,
        np.nanmean(q95[:, ids_multi], axis=0),
    )
    ids05 = [i for k, i in idx.items() if k.startswith("2005")]
    ids24 = [i for k, i in idx.items() if k.startswith("2024")]
    q05_c05, q95_c05 = cf.rescale_intervals(
        q05[:, ids05], q50[:, ids05], q95[:, ids05], c_modis
    )
    q05_c24, q95_c24 = cf.rescale_intervals(
        q05[:, ids24], q50[:, ids24], q95[:, ids24], c_multi
    )
    w05 = float(np.nanmean(q95_c05 - q05_c05))
    w24 = float(np.nanmean(q95_c24 - q05_c24))
    assert w05 > w24  # 2005 (MODIS-only) strictly wider than 2024


def _small_problem(T=120, seed=3):
    """3-cell, T-day problem with a clear active-fire transition at the end."""
    from src.engine.solver import ChunkProblem

    rng = np.random.default_rng(seed)
    B, P = 3, S.N_PLATFORMS
    z_true = np.zeros((B, T))
    z_true[:, T - 40:] = np.log(25.0)  # late fire transition
    obs = {}
    for dn in ("D", "N"):
        y = torch.zeros(B, T, P)
        ln_y = torch.from_numpy(z_true)
        for p in range(P):
            y[:, :, p] = torch.exp(
                ln_y + 0.25 * torch.from_numpy(
                    rng.standard_normal((B, T))
                )
            )
        obs[dn] = {
            "y": y,
            "sat": torch.zeros(B, T, P, dtype=torch.uint8),
            "mu_gain": torch.zeros(B, T, P),
            "inv_sig2": torch.full((B, T, P), 1.0 / 0.09),
            "k_miss": torch.full((B, T, P), 0.4),
            "vis": torch.full((B, T, P), 0.9),
            "is_op": torch.ones(B, T, P, dtype=torch.uint8),
        }
    return ChunkProblem(
        cell_ids=np.arange(B), colors=np.array([0, 1, 2]),
        rho=np.full(B, 0.7), omega=np.full(B, 1.0 / 0.7**2),
        eta_dn=np.full(B, 0.12), beta_dn=np.full(B, 0.5),
        mu_day=np.zeros((B, T)), mu_night=np.zeros((B, T)),
        month_ids=np.zeros(T, dtype=np.int64), obs=obs,
        neighbor_indices=np.array([1, 0, 2, 1]),
        neighbor_indptr=np.array([0, 1, 3, 4]),
    )


def test_08_fixed_lag_matches_full_batch():
    """14-day fixed-lag update anchored at t_start-1 reproduces the full
    batch MAP on the trailing window (< 1e-3) across the fire transition."""
    prob = _small_problem()
    full = sv.solve_chunk(prob, sweeps=8, newton_steps=20, sample=False)
    lag = sv.run_fixed_lag_update(
        lambda cells, device: prob,
        {"z_day": full["z_day"], "z_night": full["z_night"]},
        n_cells=3, window_days=14, T=prob.obs["D"]["y"].shape[1],
        sweeps=8, newton_steps=20,
    )
    t_start = lag["t_start"]
    for key, lag_key in (("z_day", "z_day"), ("z_night", "z_night")):
        diff = np.max(
            np.abs(full[key][:, t_start:] - lag[lag_key][:, t_start:])
        )
        assert diff < 1e-3, (key, diff)
