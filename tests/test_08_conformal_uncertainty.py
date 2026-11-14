"""test_08 -- Conformal uncertainty tests (12 tests).

Readme Section 7.8: split-conformal coverage inside [1-alpha-0.03,
1-alpha+0.05] for alpha in {0.05, 0.10, 0.20}; q05 <= q50 <= q95
non-negativity; 2005 uncertainty strictly wider than 2024; cloudy
uncertainty wider than clear months; ABSTAIN_LOW_EVIDENCE trigger; 14-day
fixed-lag update matching the full batch across an active fire transition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.engine import conformal as conf


def _synthetic_quantiles(n=4000, seed=7):
    """Synthetic (q05, q50, q95, truth): q50 carries id noise vs truth."""
    rng = np.random.default_rng(seed)
    truth = np.exp(rng.normal(1.0, 0.8, n))
    sigma = np.exp(rng.normal(-1.2, 0.5, n))
    err = sigma * rng.standard_normal(n)
    q50 = np.maximum(truth * np.exp(err * 0.3), 1e-6)
    return (np.maximum(q50 - 1.645 * sigma, 0.0), q50,
            q50 + 1.645 * sigma, truth)


# ---------------------------------------------------------------- coverage
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_08_split_conformal_coverage(alpha):
    """Split conformal: fit c on the calibration half, evaluate coverage on
    the held-out half -- lands inside [1-alpha-0.03, 1-alpha+0.05]."""
    q05, q50, q95, truth = _synthetic_quantiles(6000)
    half = 3000
    s_cal, r_cal = conf.nonconformity_scores(q05[:half], q50[:half],
                                             q95[:half], truth[:half])
    c = conf.conformal_c_factor(r_cal, alpha)
    s_eval, _ = conf.nonconformity_scores(q05[half:], q50[half:],
                                          q95[half:], q50[half:])
    q05c, q95c = conf.calibrate_intervals(q05[half:], q50[half:],
                                          q95[half:], c, s_eval)
    cover = float(np.mean((q05c <= truth[half:])
                          & (truth[half:] <= q95c)))
    assert 1.0 - alpha - 0.03 <= cover <= 1.0 - alpha + 0.05, \
        f"alpha={alpha}: coverage {cover}"


def test_08_quantile_ordering_nonnegativity():
    q05, q50, q95, truth = _synthetic_quantiles(500, 3)
    c = 2.0
    q05c, q95c = conf.calibrate_intervals(q05, q50, q95, c)
    assert (q05c <= q50).all()
    assert (q50 <= q95c).all()
    assert (q05c >= 0.0).all()
    # exact formula: q05_cal = max(0, exp(ln(q50+1) - c s) - 1) and
    # q95_cal = exp(ln(q50+1) + c s) - 1, with ordering enforcement
    s, _ = conf.nonconformity_scores(q05, q50, q95, q50)
    expect05 = np.maximum(np.minimum(np.maximum(
        0.0, np.exp(np.log(q50 + 1.0) - c * s) - 1.0), q50), 0.0)
    assert np.allclose(q05c, expect05, rtol=1e-12)
    expect95 = np.maximum(np.exp(np.log(q50 + 1.0) + c * s) - 1.0, q50)
    assert np.allclose(q95c, expect95, rtol=1e-12)


def test_08_c_factor_clipped_and_quantile_rule():
    r = np.full(100, 1.7)
    r[0] = 0.01
    c = conf.conformal_c_factor(r, 0.10)
    # k = ceil((n+1)(1-alpha)) = ceil(90.9) = 91 -> 91st of 100 sorted = 1.7
    assert c == pytest.approx(1.7, abs=1e-12)
    r2 = np.full(100, 99.0)
    assert conf.conformal_c_factor(r2, 0.10) == S.CONFORMAL_C_CLIP[1]
    r3 = np.full(100, 0.1)
    assert conf.conformal_c_factor(r3, 0.10) == S.CONFORMAL_C_CLIP[0]


def test_08_scores_floor():
    q05 = q50 = q95 = np.array([0.0, 0.0])
    truth = np.array([0.0, 0.0])
    s, r = conf.nonconformity_scores(q05, q50, q95, truth)
    assert (s == S.CONFORMAL_S_FLOOR).all()
    assert np.isfinite(r).all()


# ---------------------------------------------------------------- era width
def test_08_2005_uncertainty_wider_than_2024(full_twin):
    """Pre-2012 MODIS-only intervals are strictly wider than 2024."""
    m = full_twin["monthly"]
    w = (m["phi_q95_mw"] - m["phi_q05_mw"]) / m["phi_q50_mw"].clip(lower=1e-6)
    w2005 = w[m["year"] == 2005].median()
    w2024 = w[m["year"] == 2024].median()
    assert w2005 > w2024


def test_08_cloudy_month_wider_than_clear(full_twin):
    """Wet-season (cloudy) months carry wider intervals than dry months."""
    m = full_twin["monthly"]
    m = m.assign(relw=(m["phi_q95_mw"] - m["phi_q05_mw"])
                 / m["phi_q50_mw"].clip(lower=1e-6))
    wet = m[m["month"].isin([1, 2, 3])]["relw"].median()
    dry = m[m["month"].isin([8, 9, 10])]["relw"].median()
    assert wet > dry


def test_08_abstain_low_evidence_trigger():
    cloud = np.full(31, 0.5)
    det = np.zeros(31, dtype=bool)
    det[10] = True                      # break the run
    assert conf.quality_flag_month(cloud, det) == "NOMINAL"
    cloud2 = np.full(31, 0.9)
    det2 = np.zeros(31, dtype=bool)
    assert conf.quality_flag_month(cloud2, det2) == "ABSTAIN_LOW_EVIDENCE"
    # 4 consecutive days is not enough (>= 5 required)
    cloud3 = np.full(31, 0.9)
    det3 = np.zeros(31, dtype=bool)
    det3[4] = True
    assert conf.quality_flag_month(cloud3, det3) == "ABSTAIN_LOW_EVIDENCE"
    cloud4 = np.full(31, 0.9)
    det4 = np.zeros(31, dtype=bool)
    det4[[4, 9, 14, 19]].fill(False)
    for i in (4, 9, 14, 19):
        det4[i] = True
    assert conf.quality_flag_month(cloud4, det4) == "ABSTAIN_LOW_EVIDENCE"


def test_08_quality_flag_climatology_fallback():
    cloud = np.full(31, 0.2)
    det = np.zeros(31, dtype=bool)
    assert conf.quality_flag_month(cloud, det, "CLIMATOLOGY_CLOUD_FALLBACK") \
        == "CLIMATOLOGY_CLOUD_FALLBACK"
    assert conf.quality_flag_month(cloud, det, "CMG_8D") == "NOMINAL"


# ---------------------------------------------------------------- attribution
def test_08_attribution_flag_decision_tree():
    # abstain dominates
    assert conf.attribution_flag(99.0, 0.0, 1.0, 0.5,
                                 "ABSTAIN_LOW_EVIDENCE") \
        == "AMBIGUOUS_OVERLAPPING_CI"
    # interval excludes the historical median -> real change
    assert conf.attribution_flag(0.0, 2.0, 3.0, 1.0, "NOMINAL") \
        == "REAL_FIRE_CHANGE"
    assert conf.attribution_flag(0.0, 0.1, 0.9, 1.0, "NOMINAL") \
        == "REAL_FIRE_CHANGE"
    # raw moved > 50% but harmonized CI spans the median -> sensor artifact
    assert conf.attribution_flag(0.8, 0.5, 1.5, 1.0, "NOMINAL") \
        == "SENSOR_ARTIFACT_CORRECTED"
    # nothing special
    assert conf.attribution_flag(0.1, 0.5, 1.5, 1.0, "NOMINAL") \
        == "WITHIN_NORMAL_VARIABILITY"


def test_08_bitmask_encoding():
    assert conf.active_platforms_bitmask(["Terra"]) == 1
    assert conf.active_platforms_bitmask(["Aqua"]) == 2
    assert conf.active_platforms_bitmask(["SNPP"]) == 4
    assert conf.active_platforms_bitmask(["NOAA-20"]) == 8
    assert conf.active_platforms_bitmask(["NOAA-21"]) == 16
    assert conf.active_platforms_bitmask(["SNPP", "NOAA-20", "NOAA-21"]) == 28
    assert conf.active_platforms_bitmask(list(["Terra", "Aqua", "SNPP",
                                               "NOAA-20", "NOAA-21"])) == 31


# ---------------------------------------------------------------- fixed lag
def test_08_fixed_lag_update_matches_full_batch(full_twin):
    """run_fixed_lag_update(window_days=14) anchored on the saved state
    tracks the full-batch MAP (< 0.75 log-units) across active
    transitions, and the anchor row is held exactly."""
    from src.engine.solver import GMRFSolver
    solver = full_twin["solver_results"][0]["solver"]
    z_full = solver.z.clone()
    solver2 = full_twin["solver_results"][0]["solver"]
    # re-create from the same data through a fresh posterior-less solver
    from src.engine import solver as slv
    data = solver2.d
    fresh = GMRFSolver(data)
    fresh.z = z_full.clone()               # saved MAP state
    z_lag = fresh.solve_fixed_lag(window_days=14)
    w = min(14, data.t_days)
    diff = float((z_lag[:, -w:, :] - z_full[:, -w:, :]).abs().max())
    assert diff < 0.75
    t_start = data.t_days - w
    assert torch_equal(z_lag[:, t_start - 1, :], z_full[:, t_start - 1, :])


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def test_08_energy_equiv_events_formula():
    phi = np.array([10.0, 20.0])
    f_event = np.array([5.0, 4.0])
    e = conf.energy_equiv_events(phi, f_event)
    assert np.allclose(e, [2.0, 5.0])


def test_08_hurdle_adjusted_events_never_sums_platforms():
    """E_halfday averages across detecting platforms (INV-02)."""
    rows = [dict(platform="SNPP", raw_event_count=3, area=0.23, dn="D"),
            dict(platform="NOAA-20", raw_event_count=3, area=0.23, dn="D"),
            dict(platform="NOAA-21", raw_event_count=3, area=0.23, dn="D")]
    psi = lambda p, a, dn: 0.9
    vis = lambda p: 0.8
    e = conf.hurdle_adjusted_events(rows, psi, vis)
    # mean of 3/(0.9*0.8) across the three platforms -- NOT the sum
    assert e == pytest.approx(3.0 / (0.9 * 0.8), rel=1e-12)
    assert e < 3.0 / (0.9 * 0.8) + 1e-9
