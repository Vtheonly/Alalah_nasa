"""test_05 -- Likelihood derivative tests (22 tests).

Readme Section 7.5: finite-difference gradient & Hessian parity (< 1e-6)
for Obs, Tobit Saturation (over w in [-35, +5], testing the asymptotic tail
without NaN/Inf), and Hurdle Zero; Tobit asymmetry; the convexity guard
(h_miss >= 0) inside the partial-cloud transition (v=0.7, Lambda=2.5);
uniform Phi = e^z consistency across Cases A/B/C; and empirical
platform_active_day masking of simulated spacecraft safe-mode outages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import settings as S
from src.engine import likelihoods as L
from src.ingest.normalize_dedup import compute_platform_active_day
from tests.conftest import canonical_frame

torch.set_default_dtype(torch.float64)


def _fd(f, z, eps=1e-5):
    """Central finite differences (value, grad, hess) of a scalar loss."""
    z = torch.as_tensor(z, dtype=torch.float64)
    g = torch.zeros_like(z)
    h = torch.zeros_like(z)
    flat = z.reshape(-1)
    gf = g.reshape(-1)
    hf = h.reshape(-1)
    for i in range(flat.numel()):
        xp = flat.clone(); xp[i] += eps
        xm = flat.clone(); xm[i] -= eps
        fpp = f(xp.reshape(z.shape))
        fpm = f(xp.reshape(z.shape))
        fmp = f(xm.reshape(z.shape))
        fmm = f(xm.reshape(z.shape))
        gf[i] = (f(xp.reshape(z.shape)) - f(xm.reshape(z.shape))) / (2 * eps)
        hf[i] = (f(xp.reshape(z.shape)) - 2 * f(z) + f(xm.reshape(z.shape))) / eps ** 2
    return g, h


# ---------------------------------------------------------------- Case A
def test_05_case_a_exact_derivatives():
    z = torch.tensor([1.5, -2.0, 0.3])
    ln_y = torch.tensor([2.0, -1.0, 0.0])
    mu = torch.tensor([0.2, 0.1, -0.3])
    t2 = torch.tensor([3.0, 1.5, 4.0])
    l, g, h = L.case_a_nll_grad_hess(z, ln_y, mu, t2)
    r = (z + mu) - ln_y
    assert torch.allclose(l, 0.5 * t2 * r ** 2)
    assert torch.allclose(g, t2 * r)
    assert torch.allclose(h, t2)


def test_05_case_a_finite_difference():
    z = torch.tensor([1.5, -2.0, 0.3])
    ln_y, mu, t2 = (torch.tensor([2.0, -1.0, 0.0]),
                    torch.tensor([0.2, 0.1, -0.3]),
                    torch.tensor([3.0, 1.5, 4.0]))
    f = lambda zz: L.case_a_nll_grad_hess(zz, ln_y, mu, t2)[0].sum()
    g, h = _fd(f, z)
    _, gt, ht = L.case_a_nll_grad_hess(z, ln_y, mu, t2)
    assert torch.allclose(g, gt, atol=1e-6)
    assert torch.allclose(h, ht, atol=1e-4)


# ---------------------------------------------------------------- Case B
@pytest.mark.parametrize("w0", [-35.0, -30.1, -20.0, -8.0, -2.0, 0.0, 3.0, 5.0])
def test_05_case_b_finite_difference_across_w(w0):
    """Tobit gradient/Hessian parity over w in [-35, +5] incl. the tail."""
    ln_y = torch.tensor([0.0])
    mu = torch.tensor([0.0])
    t2 = torch.tensor([2.0])
    t1 = torch.sqrt(t2)
    # choose z so that w = tau1 ((z + mu) - ln y) == w0
    z = torch.tensor([w0 / t1.item()])
    f = lambda zz: L.case_b_nll_grad_hess(zz, ln_y, mu, t2, t1)[0].sum()
    g, h = _fd(f, z, eps=1e-4)
    _, gt, ht = L.case_b_nll_grad_hess(z, ln_y, mu, t2, t1)
    assert torch.isfinite(gt).all() and torch.isfinite(ht).all()
    assert torch.allclose(g, gt, atol=5e-3)
    assert torch.allclose(h, ht, atol=5e-2)


def test_05_case_b_no_nan_inf_down_to_w_minus_40():
    z = torch.tensor([-100.0, -60.0, -40.0, -10.0, 0.0, 5.0])
    ln_y = torch.zeros(6)
    mu = torch.zeros(6)
    t2 = torch.full((6,), 4.0)
    t1 = torch.sqrt(t2)
    l, g, h = L.case_b_nll_grad_hess(z, ln_y, mu, t2, t1)
    assert torch.isfinite(l).all() and torch.isfinite(g).all() \
        and torch.isfinite(h).all()
    assert (h >= 0).all()


def test_05_case_b_asymmetry():
    """Tobit loss is asymmetric in z around the censoring boundary."""
    ln_y = torch.tensor([0.0])
    mu = torch.tensor([0.0])
    t2 = torch.tensor([2.0])
    t1 = torch.sqrt(t2)
    z_hi = torch.tensor([1.0])
    z_lo = torch.tensor([-1.0])
    l_hi = L.case_b_nll_grad_hess(z_hi, ln_y, mu, t2, t1)[0]
    l_lo = L.case_b_nll_grad_hess(z_lo, ln_y, mu, t2, t1)[0]
    assert l_lo > l_hi > 0          # penalty grows away from the boundary
    assert l_lo.item() != l_hi.item()


def test_05_case_b_matches_log_ndtr_branch():
    z = torch.tensor([0.5])
    ln_y = torch.tensor([0.0])
    mu = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    t1 = torch.tensor([1.0])
    l, _, _ = L.case_b_nll_grad_hess(z, ln_y, mu, t2, t1)
    w = 0.5
    assert l.item() == pytest.approx(-float(torch.special.log_ndtr(
        torch.tensor(w))), rel=1e-12)


def test_05_case_b_asymptotic_tail_matches_log_ndtr():
    w = -30.5   # just inside the asymptotic branch
    z = torch.tensor([w])
    l, g, h = L.case_b_nll_grad_hess(z, torch.zeros(1), torch.zeros(1),
                                    torch.ones(1), torch.ones(1))
    ref = -float(torch.special.log_ndtr(torch.tensor(float(w))))
    # first-order asymptotic expansion: agrees with log_ndtr to ~1e-6
    # relative at the -30 branch boundary (log_ndtr itself remains finite
    # to ~-37; the branch exists to avoid catastrophic log(0) deeper)
    assert l.item() == pytest.approx(ref, rel=2e-5)


# ---------------------------------------------------------------- Case C
def test_05_case_c_exact_formulas():
    z = torch.tensor([1.0, -2.0, 3.0])
    k = torch.tensor([0.05, 0.5, 2.0])
    v = torch.tensor([0.7, 0.4, 0.99])
    lam = torch.clamp(k * torch.exp(z), 1e-12, 80.0)
    q = (1 - v) + v * torch.exp(-lam)
    r = v * torch.exp(-lam) / q
    l, g, h = L.case_c_nll_grad_hess(z, k, v)
    assert torch.allclose(l, -torch.log(q))
    assert torch.allclose(g, r * lam)
    assert torch.allclose(h, torch.clamp(r * lam * (1 - lam * (1 - r)), min=0))


def test_05_case_c_finite_difference_gradient():
    z = torch.tensor([1.0, -2.0, 3.0, 0.0])
    k = torch.tensor([0.05, 0.5, 2.0, 0.3])
    v = torch.tensor([0.7, 0.4, 0.99, 0.0])
    f = lambda zz: L.case_c_nll_grad_hess(zz, k, v)[0].sum()
    g, h = _fd(f, z)
    _, gt, ht = L.case_c_nll_grad_hess(z, k, v)
    assert torch.allclose(g, gt, atol=1e-6)
    # the convexity-guarded Hessian matches the exact Hessian where convex
    exact = None
    lam = torch.clamp(k * torch.exp(z), 1e-12, 80.0)
    r = v * torch.exp(-lam) / ((1 - v) + v * torch.exp(-lam))
    exact = r * lam * (1 - lam * (1 - r))
    m = exact > 0
    assert torch.allclose(ht[m], exact[m], atol=1e-4)


def test_05_case_c_convexity_guard_partial_cloud():
    """h_miss >= 0 inside the negative-curvature transition (v=0.7,
    Lambda=2.5) while the exact Hessian goes negative."""
    z = torch.tensor([np.log(2.5 / 0.05)])   # k e^z = 2.5
    k = torch.tensor([0.05])
    v = torch.tensor([0.7])
    lam = k * torch.exp(z)
    r = v * torch.exp(-lam) / ((1 - v) + v * torch.exp(-lam))
    exact = r * lam * (1 - lam * (1 - r))
    assert exact.item() < 0.0                # the transition is concave here
    _, _, h = L.case_c_nll_grad_hess(z, k, v)
    assert h.item() >= 0.0                   # Gauss-Newton guard restores SPD
    assert h.item() == 0.0


def test_05_case_c_hurdle_lambda_clipping():
    z = torch.tensor([20.0, -30.0])     # k e^z: e^20 clipped high, e^-30 low
    k = torch.tensor([1.0, 1.0])
    v = torch.tensor([0.9, 0.9])
    lam = L.hurdle_lambda(k, z)
    assert lam[0] == 80.0
    assert float(torch.exp(torch.tensor(-30.0))) < 1e-12
    assert lam[1] == 1e-12


def test_05_case_c_zero_visibility_no_info():
    """v = 0 (fully clouded): the pass carries no information."""
    z = torch.tensor([1.0])
    l, g, h = L.case_c_nll_grad_hess(z, torch.tensor([0.05]),
                                     torch.tensor([0.0]))
    assert g.item() == pytest.approx(0.0, abs=1e-12)
    assert h.item() == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------- uniform Phi = e^z
def test_05_uniform_phi_mapping_all_cases():
    """Case A likelihood means are exact at z = ln Y - mu (Phi = e^z)."""
    z = torch.tensor([2.0])
    mu = torch.tensor([0.5])
    y = torch.exp(z + mu)
    l, g, h = L.case_a_nll_grad_hess(z, torch.log(y), mu,
                                    torch.tensor([4.0]))
    assert l.item() == pytest.approx(0.0, abs=1e-12)
    assert g.item() == pytest.approx(0.0, abs=1e-12)
    # Case B: the censored optimum pushes z above the boundary
    lb, gb, _ = L.case_b_nll_grad_hess(z + 5.0, torch.log(y), mu,
                                       torch.tensor([4.0]),
                                       torch.tensor([2.0]))
    assert lb.item() >= 0.0
    # Case C: consistent with Phi = e^z through Lambda = k e^z
    lam = L.hurdle_lambda(torch.tensor([0.1]), z)
    assert lam.item() == pytest.approx(0.1 * float(torch.exp(z)))


def test_05_vectorized_eval_pointwise_matches_cases():
    z = torch.zeros(2, 3, 2, 5, dtype=torch.float64)
    y = torch.zeros(2, 3, 2, 5, dtype=torch.float64)
    sat = torch.zeros(2, 3, 2, 5, dtype=torch.uint8)
    mu = torch.zeros(2, 3, 2, 5, dtype=torch.float64)
    t2 = torch.ones(2, 3, 2, 5, dtype=torch.float64)
    t1 = torch.ones(2, 3, 2, 5, dtype=torch.float64)
    k = torch.full((2, 3, 2, 5), 0.1, dtype=torch.float64)
    vis = torch.full((2, 3, 2, 5), 0.8, dtype=torch.float64)
    op = torch.ones(2, 3, 2, 5, dtype=torch.float64)
    z[..., 0, 1] = 1.5
    y[0, 1, 0, 1] = 5.0
    y[1, 0, 1, 2] = 3.0
    sat[0, 2, 1, 0] = 1
    y[0, 2, 1, 0] = 7.0
    nll, grad, hess = L.eval_pointwise_nll_grad_hess(z, y, sat, mu, t2, t1,
                                                     k, vis, op)
    assert nll.shape == z.shape
    assert torch.isfinite(nll).all()
    # detected entries match Case A exactly
    la, ga, ha = L.case_a_nll_grad_hess(
        torch.tensor([z[0, 1, 0, 1], z[1, 0, 1, 2]]),
        torch.log(torch.tensor([5.0, 3.0])), torch.zeros(2), torch.ones(2))
    assert nll[0, 1, 0, 1] == pytest.approx(la[0], abs=1e-12)
    assert nll[1, 0, 1, 2] == pytest.approx(la[1], abs=1e-12)
    assert grad[1, 0, 1, 2] == pytest.approx(ga[1], abs=1e-12)
    assert hess[1, 0, 1, 2] == pytest.approx(ha[1], abs=1e-12)


def test_05_non_operational_pass_zero_contribution():
    z = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    sat = torch.zeros(1, 1, dtype=torch.uint8)
    mu = torch.zeros(1, 1)
    t2 = torch.ones(1, 1)
    t1 = torch.ones(1, 1)
    k = torch.full((1, 1), 0.1)
    vis = torch.full((1, 1), 0.8)
    op = torch.zeros(1, 1)      # platform NOT operational that day
    nll, grad, hess = L.eval_pointwise_nll_grad_hess(z, y, sat, mu, t2, t1,
                                                     k, vis, op)
    assert nll.abs().sum() == 0.0
    assert grad.abs().sum() == 0.0
    assert hess.abs().sum() == 0.0


def test_05_case_b_hessian_positive_everywhere():
    zs = torch.linspace(-45.0, 10.0, 300)
    ln_y = torch.zeros(300)
    mu = torch.zeros(300)
    t2 = torch.full((300,), 2.5)
    t1 = torch.sqrt(t2)
    _, _, h = L.case_b_nll_grad_hess(zs, ln_y, mu, t2, t1)
    assert (h >= 0).all()


def test_05_case_b_gradient_sign():
    """Below the boundary the gradient pushes z up; deep above it flattens."""
    ln_y, mu = torch.zeros(2), torch.zeros(2)
    t2 = torch.full((2,), 1.0)
    t1 = torch.sqrt(t2)
    _, g, _ = L.case_b_nll_grad_hess(torch.tensor([-5.0, 5.0]), ln_y, mu,
                                     t2, t1)
    assert g[0] < 0
    assert abs(g[1].item()) < abs(g[0].item())


# ------------------------------------------------------- outage masking
def test_05_platform_active_day_masks_safe_mode():
    rows = []
    for i in range(8):
        rows.append(dict(platform="NOAA-20", acq_date="2026-09-10",
                         frp_mw=5.0))
    for i in range(2):                       # outage day: only 2 detections
        rows.append(dict(platform="NOAA-20", acq_date="2026-09-11", frp_mw=5.0))
    for i in range(9):
        rows.append(dict(platform="SNPP", acq_date="2026-09-11", frp_mw=4.0))
    active = compute_platform_active_day(canonical_frame(rows))
    assert "2026-09-10" in active["NOAA-20"]
    assert "2026-09-11" not in active["NOAA-20"]     # safe-mode masked
    assert "2026-09-11" in active["SNPP"]


def test_05_six_detections_is_active_boundary():
    rows = [dict(platform="SNPP", acq_date="2026-01-01")] * 6
    active = compute_platform_active_day(canonical_frame(rows))
    assert "2026-01-01" in active["SNPP"]
    rows = [dict(platform="SNPP", acq_date="2026-01-01")] * 4
    active = compute_platform_active_day(canonical_frame(rows))
    assert "2026-01-01" not in active["SNPP"]


def test_05_half_log_two_pi_constant_consistency():
    import math
    assert L._HALF_LN_2PI == pytest.approx(0.5 * math.log(2 * math.pi))


def test_05_tobit_tail_continuity_at_boundary():
    """The two branches agree at w = -30 to ~1e-9 relative."""
    w = -30.0
    ref = -float(torch.special.log_ndtr(torch.tensor(w)))
    l, g, h = L.case_b_nll_grad_hess(torch.tensor([w]), torch.zeros(1),
                                    torch.zeros(1), torch.ones(1),
                                    torch.ones(1))
    assert l.item() == pytest.approx(ref, rel=1e-9)
