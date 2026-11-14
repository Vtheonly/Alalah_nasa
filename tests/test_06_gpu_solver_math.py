"""test_06: float64 Thomas/Takahashi/Cholesky solver mathematics.

20 tests (readme.md Section 7 item 6): Thomas solver vs torch.linalg.solve
(< 1e-10); Takahashi diagonal & lag-1 vs torch.linalg.inv(H) (< 1e-10);
empirical covariance of 10,000 joint Cholesky draws L^T u = D^{-1/2} eps
matching H^{-1}; Armijo strict objective descent; 3-color spatial cloud-gap
recovery; CUDA load_inline kernel bit-parity (< 1e-10, hardware-gated on
CPU-only machines); dynamic range 1e-4 .. 1e6 MW stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import settings as S
from src.engine import cuda_kernels
from src.engine import solver as sv
from src.engine import thomas_takahashi as tt

torch.set_default_dtype(torch.float64)


def _spd_tridiagonal(B, T, seed=0, rho=0.7, omega=2.0):
    rng = np.random.default_rng(seed)
    a = np.tile((1 + rho**2) * omega + rng.normal(0, 0.01, (B, T)), (1, 1))
    a[:, 0] = omega
    a[:, -1] = omega
    b = np.full((B, T - 1), -rho * omega)
    return (
        torch.from_numpy(a), torch.from_numpy(b),
        torch.from_numpy(rng.normal(size=(B, T))),
    )


# --------------------------------------------------------------------------- #
# Thomas solver
# --------------------------------------------------------------------------- #

def test_06_thomas_vs_linalg_solve():
    a, b, g = _spd_tridiagonal(4, 120)
    x = tt.thomas_solve(a, b, g)
    x_ref = tt.solve_tridiagonal_reference(a, b, g)
    assert torch.max(torch.abs(x - x_ref)).item() < 1e-10


def test_06_thomas_batched_consistency():
    a, b, g = _spd_tridiagonal(3, 80, seed=5)
    _, _, x_batch = tt.thomas_factor_and_solve(a, b, g)
    for i in range(3):
        _, _, x_i = tt.thomas_factor_and_solve(
            a[i : i + 1], b[i : i + 1], g[i : i + 1]
        )
        assert torch.max(torch.abs(x_batch[i] - x_i[0])).item() < 1e-12


def test_06_thomas_pivot_floor_no_nan():
    # Pivots that would go negative are floored at 1e-7: the factorization
    # completes with finite d >= floor and a bounded solve on a short system.
    T = 8
    a = torch.full((1, T), 1e-5)
    a[:, 0] = 1.0
    a[:, -1] = 1.0
    b = torch.full((1, T - 1), -0.4)
    g = torch.full((1, T), 1e-3)
    d, l, x = tt.thomas_factor_and_solve(a, b, g)
    assert bool(torch.isfinite(d).all() and torch.isfinite(x).all())
    assert float(d.min()) >= S.PIVOT_FLOOR - 1e-15
    # The unfloored pivot would be negative on step 1: floor provably active.
    assert float(d[0, 1]) == S.PIVOT_FLOOR


def test_06_thomas_extreme_dynamic_range():
    # Right-hand sides spanning 1e-4 .. 1e6 MW stay exact vs dense solve.
    a, b, _ = _spd_tridiagonal(1, 60)
    g = torch.tensor(
        [[1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6] * 10], dtype=torch.float64
    )
    x = tt.thomas_solve(a, b, g)
    x_ref = tt.solve_tridiagonal_reference(a, b, g)
    rel = torch.max(torch.abs(x - x_ref) / torch.clamp(torch.abs(x_ref), min=1e-30))
    assert float(rel) < 1e-8


# --------------------------------------------------------------------------- #
# Takahashi marginals
# --------------------------------------------------------------------------- #

def test_06_takahashi_diag_vs_inverse():
    a, b, _ = _spd_tridiagonal(3, 100, seed=1)
    d, l, _ = tt.thomas_factor_and_solve(a, b, torch.zeros(3, 100))
    var, cov1 = tt.takahashi_marginals(d, l)
    inv = tt.inverse_tridiagonal_reference(a, b)
    idx = torch.arange(100)
    assert torch.max(torch.abs(var - inv[:, idx, idx])).item() < 1e-10


def test_06_takahashi_lag1_vs_inverse():
    a, b, _ = _spd_tridiagonal(3, 100, seed=2)
    d, l, _ = tt.thomas_factor_and_solve(a, b, torch.zeros(3, 100))
    _, cov1 = tt.takahashi_marginals(d, l)
    inv = tt.inverse_tridiagonal_reference(a, b)
    idx = torch.arange(99)
    assert torch.max(torch.abs(cov1 - inv[:, idx, idx + 1])).item() < 1e-10


def test_06_joint_cholesky_covariance_matches_inverse():
    # 10,000 draws: empirical diagonal & lag-1 covariance match H^{-1}
    # within the Monte-Carlo noise floor (rel ~ sqrt(2/M) ~ 1.4%).
    a, b, _ = _spd_tridiagonal(2, 40, seed=3)
    d, l, _ = tt.thomas_factor_and_solve(a, b, torch.zeros(2, 40))
    M = 10000
    eps = torch.randn(2, 40, M)
    u = tt.joint_cholesky_sample(d, l, eps)
    emp_var = (u * u).mean(dim=2)
    var, cov1 = tt.takahashi_marginals(d, l)
    rel_err = (
        (emp_var - var).abs() / var.clamp(min=1e-12)
    ).max().item()
    assert rel_err < 0.08
    emp_cov1 = (u[:, :-1, :] * u[:, 1:, :]).mean(dim=2)
    rel_err1 = (
        (emp_cov1 - cov1).abs() / cov1.abs().clamp(min=1e-12)
    ).max().item()
    assert rel_err1 < 0.12
    # Cross-covariance beyond lag-1 vanishes for the tridiagonal H.
    emp_lag2 = (u[:, :-2, :] * u[:, 2:, :]).mean(dim=2)
    inv = tt.inverse_tridiagonal_reference(a, b)
    idx = torch.arange(38)
    true_lag2 = inv[:, idx, idx + 2]
    assert (
        (emp_lag2 - true_lag2).abs().max().item()
        < 0.05 * float(var.mean())
    )


def test_06_joint_cholesky_mean_zero():
    a, b, _ = _spd_tridiagonal(2, 30, seed=4)
    d, l, _ = tt.thomas_factor_and_solve(a, b, torch.zeros(2, 30))
    u = tt.joint_cholesky_sample(d, l, torch.randn(2, 30, 5000))
    assert float(u.mean().abs()) < 0.05


# --------------------------------------------------------------------------- #
# Prior building blocks
# --------------------------------------------------------------------------- #

def test_06_prior_diagonal_pattern():
    rho = torch.tensor([0.6])
    omega = torch.tensor([3.0])
    a = sv.prior_diag(rho, omega, 5)
    expected = [
        3.0, 3.0 * (1 + 0.36), 3.0 * (1 + 0.36), 3.0 * (1 + 0.36), 3.0
    ]
    assert torch.allclose(a[0], torch.tensor(expected))


def test_06_prior_grad_matches_quad_fd():
    rho = torch.tensor([0.7])
    omega = torch.tensor([2.5])
    delta = torch.randn(1, 40)
    quad = sv.prior_quad(delta, rho, omega)
    g = sv.prior_grad(delta, rho, omega)
    eps = 1e-6
    for t in (0, 1, 20, 38, 39):
        d1 = delta.clone(); d1[0, t] += eps
        d2 = delta.clone(); d2[0, t] -= eps
        fd = float(sv.prior_quad(d1, rho, omega) - sv.prior_quad(d2, rho, omega)) / (2 * eps)
        assert fd == pytest.approx(float(g[0, t]), abs=1e-5)


def test_06_prior_anchor_boundary_term():
    # Fixed-lag anchor replaces (1-rho^2) d0^2 with
    # (d0 - rho*anchor)^2 minus the anchor-constant (dropped: it cancels in
    # Armijo differences; readme.md 5.3.4 item 7).
    rho = torch.tensor([0.5])
    omega = torch.tensor([2.0])
    delta = torch.randn(1, 10)
    anchor = torch.tensor([0.3])
    q_anchor = sv.prior_quad(delta, rho, omega, anchor_delta=anchor)
    diff = delta[:, 1:] - rho[:, None] * delta[:, :-1]
    expected = 0.5 * omega * (
        delta[:, 0] ** 2 - 2.0 * rho * anchor * delta[:, 0]
        + (diff * diff).sum(dim=1)
    )
    assert torch.allclose(q_anchor, expected)


# --------------------------------------------------------------------------- #
# Armijo / Newton behaviour
# --------------------------------------------------------------------------- #

def _tiny_block_problem(T=60, seed=6):
    rng = np.random.default_rng(seed)
    B, P = 2, S.N_PLATFORMS
    z_true = torch.zeros(B, T)
    obs = {
        "y": torch.zeros(B, T, P),
        "sat": torch.zeros(B, T, P, dtype=torch.uint8),
        "mu_gain": torch.zeros(B, T, P),
        "inv_sig2": torch.ones(B, T, P),
        "k_miss": torch.full((B, T, P), 0.3),
        "vis": torch.full((B, T, P), 0.9),
        "is_op": torch.ones(B, T, P, dtype=torch.uint8),
    }
    y = torch.exp(z_true.unsqueeze(-1) + 0.2 * torch.randn(B, T, P))
    obs["y"] = y
    mu = torch.zeros(B, T)
    return obs, mu, z_true


def test_06_armijo_strict_objective_descent():
    obs, mu, _ = _tiny_block_problem()
    from src.engine.solver import solve_block_newton

    z0 = torch.zeros(2, 60)
    rho = torch.tensor([0.7, 0.6])
    omega = torch.tensor([2.0, 2.5])
    eta = torch.tensor([0.15, 0.15])
    beta = torch.tensor([0.5, 0.5])
    nb_sum = torch.zeros(2, 60)
    nb_deg = torch.zeros(2)
    delta_opp = torch.zeros(2, 60)

    def objective(zz):
        from src.engine import likelihoods as lk

        inv_sig = torch.sqrt(obs["inv_sig2"])
        nll, _, _ = lk.eval_pointwise_nll_grad_hess(
            zz.unsqueeze(-1).expand(2, 60, S.N_PLATFORMS),
            obs["y"], obs["sat"].to(torch.float64), obs["mu_gain"],
            obs["inv_sig2"], inv_sig, obs["k_miss"], obs["vis"],
            obs["is_op"].to(torch.float64),
        )
        d = zz - mu
        return float(nll.sum() + (sv.prior_quad(d, rho, omega)).sum())

    j_before = objective(z0)
    z_out, _ = solve_block_newton(
        z0.clone(), mu, rho, omega, eta, beta, obs, nb_sum, nb_deg,
        delta_opp, newton_steps=8,
    )
    j_after = objective(z_out)
    assert j_after <= j_before  # strict descent (monotone Armijo)
    assert j_after < j_before - 1e-6


def test_06_newton_recovers_synthetic_state():
    obs, mu, z_true = _tiny_block_problem(seed=7)
    from src.engine.solver import solve_block_newton

    rho = torch.tensor([0.7, 0.7])
    omega = torch.tensor([2.0, 2.0])
    eta = torch.tensor([0.1, 0.1])
    beta = torch.tensor([0.5, 0.5])
    z_out, _ = solve_block_newton(
        torch.zeros(2, 60), mu, rho, omega, eta, beta, obs,
        torch.zeros(2, 60), torch.zeros(2), torch.zeros(2, 60),
        newton_steps=15,
    )
    # Observations dominate the weak prior: recovered near truth.
    assert float(torch.sqrt(((z_out - z_true) ** 2).mean())) < 0.35


def test_06_spatial_cloud_gap_recovery():
    """A fully obscured cell is reconstructed through its 1-ring neighbors.

    Three adjacent cells (linear chain); the middle cell has NO detections
    and heavy cloud (v ~ 0.1); its neighbors both burn at 30 MW. The
    spatial coupling (alpha_s = 0.10) propagates the signal, and the MAP of
    the obscured cell must be substantially closer to the truth than the
    dormant prior baseline.
    """
    from src.engine.solver import ChunkProblem, solve_chunk

    B, T, P = 3, 40, S.N_PLATFORMS
    z_true = torch.zeros(B, T)
    z_true[0] = np.log(30.0)
    z_true[2] = np.log(30.0)
    z_true[1] = np.log(30.0)  # truly burning as well
    obs = {}
    for dn in ("D", "N"):
        y = torch.zeros(B, T, P)
        mu_gain = torch.zeros(B, T, P)
        vis = torch.full((B, T, P), 0.9)
        vis[1] = 0.10  # middle cell nearly blind
        y[0, :, 2] = 30.0  # SNPP observes neighbors
        y[2, :, 3] = 30.0
        obs[dn] = {
            "y": y, "sat": torch.zeros(B, T, P, dtype=torch.uint8),
            "mu_gain": mu_gain,
            "inv_sig2": torch.full((B, T, P), 1.0 / 0.09),
            "k_miss": torch.full((B, T, P), 0.3), "vis": vis,
            "is_op": torch.ones(B, T, P, dtype=torch.uint8),
        }
    prob = ChunkProblem(
        cell_ids=np.arange(3), colors=np.array([0, 1, 0]),
        rho=np.full(3, 0.7), omega=np.full(3, 1.0),
        eta_dn=np.full(3, 0.1), beta_dn=np.full(3, 0.5),
        mu_day=np.zeros((3, T)), mu_night=np.zeros((3, T)),
        month_ids=np.zeros(T, dtype=np.int64), obs=obs,
        neighbor_indices=np.array([1, 0, 2, 1]),
        neighbor_indptr=np.array([0, 1, 3, 4]),
    )
    res = solve_chunk(prob, sweeps=6, newton_steps=15, sample=False)
    z_mid = res["z_day"][1]
    # Spatial prior pulls the blinded cell far above the dormant floor
    # toward the burning neighbors (recovery, not collapse).
    assert float(z_mid.mean()) > np.log(0.01) + 1.0
    assert float(z_mid.mean()) > 1.0  # e^z >= ~3 MW via neighbor coupling


# --------------------------------------------------------------------------- #
# Chunk VRAM & calendars
# --------------------------------------------------------------------------- #

def test_06_chunk_vram_under_6gb():
    est = sv.estimate_chunk_vram_bytes(
        batch_size=S.BATCH_SIZE, T=S.T_DAYS, n_platforms=S.N_PLATFORMS
    )
    assert est == 5_698_392_000  # exact analytic plan
    assert est < 6 * 1024**3


def test_06_chunk_vram_scales_with_batch():
    e1 = sv.estimate_chunk_vram_bytes(batch_size=500)
    e2 = sv.estimate_chunk_vram_bytes(batch_size=1000)
    # Dense + state memory scale linearly; the sampling sub-batch is
    # constant in B, so the total ratio sits just under 2.
    assert e2 > e1
    assert 1.8 < e2 / e1 <= 2.0
    e1s = sv.estimate_chunk_vram_bytes(batch_size=500, mc_samples=0)
    e2s = sv.estimate_chunk_vram_bytes(batch_size=1000, mc_samples=0)
    assert e2s / e1s == pytest.approx(2.0, rel=1e-12)


def test_06_month_grid_span():
    month_ids, keys = sv.month_grid(S.T_DAYS)
    assert len(keys) == 313  # 2000-11 .. 2026-11
    assert keys[0] == (2000, 11)
    assert keys[-1] == (2026, 11)
    assert int(month_ids.max()) == 312


def test_06_platform_calendar_boundaries():
    cal = sv.build_platform_calendar(None, S.T_DAYS)
    d0 = pd.Timestamp(S.DATE_START)
    t_snpp_end = (pd.Timestamp("2026-11-02") - d0).days
    p_snpp = S.PLATFORM_INDEX["SNPP"]
    assert int(cal[p_snpp, t_snpp_end]) == 1
    assert int(cal[p_snpp, t_snpp_end + 1]) == 0  # SNPP closes after 11-02
    p_n21 = S.PLATFORM_INDEX["NOAA-21"]
    assert int(cal[p_n21, S.T_DAYS - 1]) == 1  # through 2026-11-14
    p_terra = S.PLATFORM_INDEX["Terra"]
    assert int(cal[p_terra, 0]) == 1  # from 2000-11-01


def test_06_platform_active_day_outage_masked_in_calendar():
    active = pd.DataFrame(
        {
            "platform": ["Terra", "Terra"],
            "acq_date": ["2010-06-01", "2010-06-02"],
            "platform_active": [0, 1],
        }
    )
    cal = sv.build_platform_calendar(active, S.T_DAYS)
    d0 = pd.Timestamp(S.DATE_START)
    t1 = (pd.Timestamp("2010-06-01") - d0).days
    p = S.PLATFORM_INDEX["Terra"]
    assert int(cal[p, t1]) == 0  # safe-mode outage masked
    assert int(cal[p, t1 + 1]) == 1


# --------------------------------------------------------------------------- #
# CUDA kernel parity (hardware-gated)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware not available"
)
def test_06_cuda_kernel_bit_parity():
    a, b, g = _spd_tridiagonal(4, 200, seed=9)
    d_cpu, l_cpu, x_cpu = tt.thomas_factor_and_solve(a, b, g)
    dev = torch.device("cuda:0")
    a_d, b_d, g_d = a.to(dev), b.to(dev), g.to(dev)
    assert cuda_kernels.cuda_available()
    d_gpu, l_gpu, x_gpu = cuda_kernels.thomas_solve(a_d, b_d, g_d)
    assert torch.max(torch.abs(x_gpu.cpu() - x_cpu)).item() < 1e-10
    v_cpu, c_cpu = tt.takahashi_marginals(d_cpu, l_cpu)
    v_gpu, c_gpu = cuda_kernels.takahashi(d_gpu, l_gpu)
    assert torch.max(torch.abs(v_gpu.cpu() - v_cpu)).item() < 1e-10
    eps = torch.randn(4, 200, 8)
    u_cpu = tt.joint_cholesky_sample(d_cpu, l_cpu, eps)
    u_gpu = cuda_kernels.cholesky_sample(d_gpu, l_gpu, eps.to(dev))
    assert torch.max(torch.abs(u_gpu.cpu() - u_cpu)).item() < 1e-10
