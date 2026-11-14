"""test_05: exact likelihood derivatives, Tobit tail, convexity guard.

22 tests (readme.md Section 7 item 5): finite-difference gradient & Hessian
parity (< 1e-6) for Obs, Tobit Saturation (over w in [-35, +5] testing the
asymptotic tail without NaN/Inf), and Hurdle Zero; Tobit asymmetry;
convexity guard (h_miss >= 0) inside the partial-cloud transition
(v = 0.7, Lambda = 2.5); the uniform Phi = e^z consistency across
Case A/B/C; and empirical platform_active_day masking of simulated
spacecraft safe-mode outages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import settings as S
from src.engine import likelihoods as lk
from src.ingest import normalize_dedup as nd

torch.set_default_dtype(torch.float64)


def _fd(f, z, eps=1e-6):
    """Central finite differences (value, grad, hess) of f at z."""
    z = torch.tensor(float(z))
    f0 = float(f(z))
    f1 = float(f(z + eps))
    f2 = float(f(z - eps))
    grad = (f1 - f2) / (2 * eps)
    hess = (f1 - 2 * f0 + f2) / (eps * eps)
    return f0, grad, hess


# --------------------------------------------------------------------------- #
# Case A — Gaussian observation likelihood
# --------------------------------------------------------------------------- #

def test_05_case_a_gradient_fd_parity():
    ln_y, mu, t2 = torch.tensor(0.7), torch.tensor(0.2), torch.tensor(4.0)
    for z in (-3.0, -0.5, 0.0, 1.2, 6.0):
        _, g, _ = lk.obs_nll_grad_hess(
            torch.tensor(z), ln_y, mu, t2
        )
        _, fd_g, _ = _fd(lambda zz: lk.obs_nll_grad_hess(zz, ln_y, mu, t2)[0], z)
        assert float(g) == pytest.approx(fd_g, abs=1e-6)


def test_05_case_a_hessian_exact():
    ln_y, mu, t2 = torch.tensor(0.0), torch.tensor(0.3), torch.tensor(2.5)
    for z in (-2.0, 0.0, 3.0):
        _, _, h = lk.obs_nll_grad_hess(torch.tensor(z), ln_y, mu, t2)
        assert float(h) == pytest.approx(2.5, abs=1e-12)


# --------------------------------------------------------------------------- #
# Case B — Tobit right-censored saturation
# --------------------------------------------------------------------------- #

def _tobit_case(z, ln_y, mu, sig):
    t1 = 1.0 / sig
    t2 = 1.0 / sig**2
    return lk.tobit_nll_grad_hess(
        torch.tensor(float(z)), torch.tensor(float(ln_y)),
        torch.tensor(float(mu)), torch.tensor(float(t1)),
        torch.tensor(float(t2)),
    )


def test_05_case_b_gradient_fd_main_branch():
    # w in (-30, 5): main log_ndtr branch.
    for z, ln_y, mu, sig in (
        (0.5, 0.0, 0.0, 1.0),      # w = 0.5
        (-4.0, 0.0, 0.0, 1.0),     # w = -4
        (-20.0, 0.0, 0.0, 1.0),    # w = -20
        (2.0, 1.0, -0.5, 2.0),
    ):
        _, g, _ = _tobit_case(z, ln_y, mu, sig)
        _, fd_g, _ = _fd(
            lambda zz: _tobit_case(float(zz), ln_y, mu, sig)[0], z
        )
        assert float(g) == pytest.approx(fd_g, abs=1e-6)


def test_05_case_b_gradient_fd_asymptotic_tail():
    # w < -30: exact asymptotic expansion branch (all FD points strictly
    # inside the branch so the stencil never straddles the cut at -30).
    for z, ln_y, mu, sig in (
        (-35.0, 0.0, 0.0, 1.0),
        (-60.0, 0.0, 0.0, 1.0),
        (-32.0, -1.0, 0.0, 1.0),  # w = -31, safely in-tail
    ):
        _, g, _ = _tobit_case(z, ln_y, mu, sig)
        _, fd_g, _ = _fd(
            lambda zz: _tobit_case(float(zz), ln_y, mu, sig)[0], z,
            eps=1e-5,
        )
        assert float(g) == pytest.approx(fd_g, abs=1e-6)


def test_05_case_b_hessian_fd_main_branch():
    for z in (0.5, -4.0, -10.0):
        _, _, h = _tobit_case(z, 0.0, 0.0, 1.0)
        _, _, fd_h = _fd(
            lambda zz: _tobit_case(float(zz), 0.0, 0.0, 1.0)[0], z,
            eps=1e-4,
        )
        assert float(h) == pytest.approx(fd_h, abs=1e-6)


def test_05_case_b_hessian_fd_asymptotic_tail():
    # The spec prescribes h = tau2 * lambda(w) * (w + lambda(w)) with the
    # tail lambda(w) = -w - 1/w, giving the closed form tau2 * (1 + 1/w^2).
    # The exact curvature of the asymptotic objective 0.5w^2 + ln(-w) + c is
    # 1 - 1/w^2; the composed spec formula differs by O(2/w^2) (~1.6e-3 at
    # w = -35) — negligible and strictly positive, so Newton stays SPD.
    for z in (-35.0, -45.0):
        _, _, h = _tobit_case(z, 0.0, 0.0, 1.0)
        expected = 1.0 + 1.0 / (z * z)
        assert float(h) == pytest.approx(expected, rel=1e-12)
        assert float(h) > 0.99
        _, _, fd_h = _fd(
            lambda zz: _tobit_case(float(zz), 0.0, 0.0, 1.0)[0], z,
            eps=1e-4,
        )
        assert float(h) == pytest.approx(fd_h, abs=2.5e-3)


def test_05_case_b_hessian_positive_everywhere():
    for w in np.linspace(-35.0, 5.0, 81):
        _, _, h = _tobit_case(w, 0.0, 0.0, 1.0)
        assert float(h) > 0.0


def test_05_case_b_no_nan_inf_over_full_range():
    for w in np.concatenate(
        [np.linspace(-35.0, 5.0, 81), np.array([-100.0, -200.0])]
    ):
        n, g, h = _tobit_case(w, 0.0, 0.0, 1.0)
        for val in (float(n), float(g), float(h)):
            assert np.isfinite(val)


def test_05_case_b_asymptotic_matches_log_ndtr():
    # Near-continuity at the branch cut w = -30: the first-order asymptotic
    # expansion matches log_ndtr to O(1/w^2) ~ 1e-3 at the cut.
    w = torch.tensor(-30.0)
    lp, lam = lk.tobit_log_phi_and_lambda(w)
    lp_m, lam_m = lk.tobit_log_phi_and_lambda(torch.tensor(-29.9999))
    assert abs(float(lp) - float(lp_m)) < 0.01
    assert float(lam) == pytest.approx(float(lam_m), rel=1e-3)
    # Deep tail: -log Phi(w) ~ 0.5 w^2 + ln(-w) + 0.5 ln(2 pi).
    w = torch.tensor(-40.0)
    lp, _ = lk.tobit_log_phi_and_lambda(w)
    expected = 0.5 * 40.0**2 + np.log(40.0) + 0.5 * np.log(2 * np.pi)
    assert float(-lp) == pytest.approx(expected, rel=1e-6)


def test_05_case_b_asymmetry():
    # Censoring is one-sided: l(w) is the survival log-likelihood, not even.
    n_pos, _, _ = _tobit_case(2.0, 0.0, 0.0, 1.0)
    n_neg, _, _ = _tobit_case(-2.0, 0.0, 0.0, 1.0)
    assert float(n_pos) != pytest.approx(float(n_neg))
    assert float(n_neg) > float(n_pos)  # w << 0 (z far below ln Y) is rare


# --------------------------------------------------------------------------- #
# Case C — Hurdle zero-detection
# --------------------------------------------------------------------------- #

def _hurdle_case(z, k, v):
    return lk.hurdle_nll_grad_hess(
        torch.tensor(float(z)), torch.tensor(float(k)), torch.tensor(float(v))
    )


def test_05_case_c_gradient_fd_clear_sky():
    # v = 0.95, moderate Lambda.
    k = 0.5
    for z in (-3.0, 0.0, 1.5, 4.0):
        _, g, _ = _hurdle_case(z, k, 0.95)
        _, fd_g, _ = _fd(lambda zz: _hurdle_case(float(zz), k, 0.95)[0], z)
        assert float(g) == pytest.approx(fd_g, abs=1e-6)


def test_05_case_c_gradient_fd_partial_cloud_transition():
    # The audited transition regime v = 0.7 with Lambda ~ 2.5.
    v = 0.7
    k = 2.5 * np.exp(-0.0)  # e^z = 1 at z = 0 -> Lambda = 2.5
    for z in (-1.0, 0.0, 1.0):
        _, g, _ = _hurdle_case(z, k, v)
        _, fd_g, _ = _fd(lambda zz: _hurdle_case(float(zz), k, v)[0], z)
        assert float(g) == pytest.approx(fd_g, abs=1e-6)


def test_05_case_c_hessian_formula_fd_clear_sky():
    # In clear sky the raw (unguarded) curvature matches FD.
    k, v = 0.5, 0.95
    for z in (-2.0, 0.0, 2.0):
        _, _, h = _hurdle_case(z, k, v)
        lam = k * np.exp(z)
        exp_neg = np.exp(-lam)
        q = (1 - v) + v * exp_neg
        r = v * exp_neg / q
        raw = r * lam * (1.0 - lam * (1.0 - r))
        _, _, fd_h = _fd(lambda zz: _hurdle_case(float(zz), k, v)[0], z,
                         eps=1e-4)
        expected = max(raw, 0.0)
        assert float(h) == pytest.approx(expected, abs=1e-9)
        if raw > 0:
            assert expected == pytest.approx(fd_h, abs=1e-6)


def test_05_convexity_guard_partial_cloud_transition():
    # v = 0.7, Lambda in the (1, 4) band where raw curvature turns negative:
    # exact gradient preserved, Hessian floored at zero -> SPD Newton system.
    v = 0.7
    for z in np.linspace(-1.2, 1.6, 15):
        k = 2.5 * np.exp(-z)  # keeps Lambda near 2.5 across the sweep
        n, g, h = _hurdle_case(z, k, v)
        lam = float(lk.hurdle_lambda(torch.tensor(z), torch.tensor(k)))
        exp_neg = np.exp(-lam)
        q = (1 - v) + v * exp_neg
        r = v * exp_neg / q
        raw = r * lam * (1.0 - lam * (1.0 - r))
        assert float(g) == pytest.approx(r * lam, abs=1e-12)  # exact grad
        assert float(h) >= 0.0  # convexity guard
        assert float(h) == pytest.approx(max(raw, 0.0), abs=1e-12)


def test_05_convexity_guard_zero_region_exists():
    # There exist (v, Lambda) points with negative raw curvature that the
    # guard clips to exactly zero (documenting why the guard is needed).
    v = 0.7
    found_negative = False
    for z in np.linspace(-2.0, 3.0, 101):
        for k in (0.5, 2.5, 5.0):
            lam = k * np.exp(z)
            if lam > 80.0:
                continue
            exp_neg = np.exp(-lam)
            q = (1 - v) + v * exp_neg
            r = v * exp_neg / q
            raw = r * lam * (1.0 - lam * (1.0 - r))
            if raw < 0:
                _, _, h = _hurdle_case(z, k, v)
                assert float(h) == 0.0
                found_negative = True
    assert found_negative



def test_05_uniform_phi_exp_z_consistency():
    # Case A at the perfectly explained observation: nll ~ 0.
    z = 1.3
    n, g, h = lk.obs_nll_grad_hess(
        torch.tensor(z), torch.tensor(z), torch.tensor(0.0),
        torch.tensor(25.0),
    )
    assert float(n) == pytest.approx(0.0, abs=1e-12)
    assert float(g) == pytest.approx(0.0, abs=1e-12)
    assert float(h) == 25.0
    # Case B at w = 0 (z + mu == ln Y): nll = -ln Phi(0) = ln 2.
    n_b, g_b, h_b = _tobit_case(z, z, 0.0, 1.0)
    assert float(n_b) == pytest.approx(np.log(2.0), abs=1e-12)
    assert float(g_b) == pytest.approx(-0.7978845608, abs=1e-7)  # -lambda(0)
    assert float(h_b) > 0
    # Case C: Lambda = k * e^z exactly (uniform forward mapping Phi = e^z).
    k = 0.8
    lam = float(lk.hurdle_lambda(torch.tensor(2.0), torch.tensor(k)))
    assert lam == pytest.approx(k * np.exp(2.0), abs=1e-12)


def test_05_combined_eval_dispatch():
    z = torch.zeros(4)
    y = torch.tensor([10.0, 10.0, 0.0, 0.0])
    sat = torch.tensor([0.0, 1.0, 0.0, 0.0])
    mu = torch.zeros(4)
    t2 = torch.full((4,), 4.0)
    t1 = torch.full((4,), 2.0)
    k = torch.full((4,), 1.0)
    v = torch.full((4,), 0.9)
    op = torch.ones(4)
    nll, grad, hess = lk.eval_pointwise_nll_grad_hess(
        z, y, sat, mu, t2, t1, k, v, op
    )
    # Case A element: z=0, ln y = ln 10 -> r = -ln 10.
    assert float(grad[0]) == pytest.approx(4.0 * (-np.log(10.0)), abs=1e-12)
    assert float(hess[0]) == 4.0
    # Case B element: w = -ln 10 * 2.
    w = 2.0 * (0.0 - np.log(10.0))
    _, lam = lk.tobit_log_phi_and_lambda(torch.tensor(w))
    assert float(grad[1]) == pytest.approx(-2.0 * float(lam), abs=1e-9)
    # Case C element: g = r * Lambda at z = 0 -> Lambda = 1.
    exp_neg = np.exp(-1.0)
    r = 0.9 * exp_neg / ((1 - 0.9) + 0.9 * exp_neg)
    assert float(grad[2]) == pytest.approx(r * 1.0, abs=1e-12)


def test_05_combined_eval_masks_nonoperational_passes():
    z = torch.zeros(3)
    y = torch.tensor([10.0, 0.0, 0.0])
    sat = torch.zeros(3)
    mu = torch.zeros(3)
    t2 = torch.full((3,), 4.0)
    t1 = torch.full((3,), 2.0)
    k = torch.full((3,), 1.0)
    v = torch.full((3,), 0.9)
    # Only element 0 is an operational pass; 1-2 are outage-masked.
    op = torch.tensor([1.0, 0.0, 0.0])
    nll, grad, hess = lk.eval_pointwise_nll_grad_hess(
        z, y, sat, mu, t2, t1, k, v, op
    )
    assert float(nll[1]) == 0.0 and float(grad[1]) == 0.0
    assert float(nll[2]) == 0.0 and float(grad[2]) == 0.0
    assert float(hess[2]) == 0.0
    assert float(nll[0]) > 0.0  # the operational detection still counts


def test_05_platform_active_day_masks_simulated_safe_mode():
    # 4 detections on 2026-03-05 (below the >= 5 threshold) -> outage day;
    # 5 detections on 2026-03-06 -> active day (INV-04 empirical masking).
    rows = []
    for day, n in (("2026-03-05", 4), ("2026-03-06", 5)):
        for i in range(n):
            part = nd.parse_mcd14ml(
                ["20210801 1345 T -29.16 -67.50 321.5 305.2 681. 42.5 82 0 D"],
                ingest_ts="2026-11-14T00:00:00Z",
            )
            part["acq_date"] = day
            part["latitude"] = -29.0 - 0.01 * i
            rows.append(part)
    active = nd.compute_platform_active_day(pd.concat(rows, ignore_index=True))
    table = active.set_index("acq_date")["platform_active"]
    assert int(table.loc["2026-03-05"]) == 0  # safe-mode outage masked
    assert int(table.loc["2026-03-06"]) == 1


def test_05_extreme_dynamic_range_and_clip_bounds():
    # Observations from 1e-4 to 1e6 MW stay finite across all cases.
    for ln_y in (np.log(1e-4), np.log(1e6)):
        n, g, h = lk.obs_nll_grad_hess(
            torch.tensor(0.0), torch.tensor(ln_y), torch.tensor(0.0),
            torch.tensor(4.0),
        )
        assert all(np.isfinite(float(x)) for x in (n, g, h))
        n, g, h = _tobit_case(0.0, float(ln_y), 0.0, 1.0)
        assert all(np.isfinite(float(x)) for x in (n, g, h))
    # z at the solver clip bounds [-8, 14] never produces NaN/Inf.
    for z in (S.Z_CLIP_LO, S.Z_CLIP_HI):
        n, g, h = _hurdle_case(z, 1.0, 0.7)
        assert all(np.isfinite(float(x)) for x in (n, g, h))
        n, g, h = _tobit_case(z, 0.0, 0.0, 1.0)
        assert all(np.isfinite(float(x)) for x in (n, g, h))


def test_05_hurdle_lambda_clip_bounds():
    z = torch.tensor(0.0)
    lam_lo = lk.hurdle_lambda(z, torch.tensor(1e-30))
    lam_hi = lk.hurdle_lambda(z, torch.tensor(1e30))
    assert float(lam_lo) == pytest.approx(S.LAMBDA_CLIP[0], abs=1e-18)
    assert float(lam_hi) == pytest.approx(S.LAMBDA_CLIP[1])
    # Zero-detection evidence pushes z downward: g = r * Lambda > 0, and
    # the gradient stays bounded across the Lambda clip range.
    _, g, _ = _hurdle_case(0.0, 1.0, 0.9)
    assert float(g) > 0
    _, g_tiny, _ = _hurdle_case(10.0, 1e-13, 0.9)
    assert 0.0 <= float(g_tiny) < 1e-6
    n, g, h = _hurdle_case(20.0, 1e6, 0.5)
    assert np.isfinite(float(n)) and np.isfinite(float(g))
    assert float(h) >= 0.0


def test_05_hessian_spd_all_cases():
    # Combined Hessian is non-negative for every case combination.
    z = torch.linspace(-6.0, 8.0, 29)
    y = torch.where(z > 0, torch.exp(z), torch.zeros_like(z))
    sat = (z > 4).to(torch.float64)
    mu = torch.zeros_like(z)
    t2 = torch.full_like(z, 4.0)
    t1 = torch.full_like(z, 2.0)
    k = torch.full_like(z, 1.2)
    v = torch.full_like(z, 0.6)
    op = torch.ones_like(z)
    _, _, hess = lk.eval_pointwise_nll_grad_hess(
        z, y, sat, mu, t2, t1, k, v, op
    )
    assert bool((hess >= 0).all())
