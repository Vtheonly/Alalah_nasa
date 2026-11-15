"""test_06 -- GPU solver math tests (20 tests).

Readme Section 7.6: float64 Thomas solver vs torch.linalg.solve (< 1e-10);
Takahashi diagonal & lag-1 vs torch.linalg.inv(H) (< 1e-10); empirical
covariance of 10,000 joint Cholesky draws L^T u = D^{-1/2} eps matching
H^{-1}; Armijo strict objective descent; 3-color spatial cloud-gap
recovery; CUDA load_inline kernel bit-parity (< 1e-10, hardware-gated);
dynamic range 1e-4..1e6 MW stability.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import settings as S
from src.engine import cuda_kernels, thomas_takahashi as tt

torch.set_default_dtype(torch.float64)


def _spd_tridiag(bsz, t_len, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 2.0, (bsz, t_len))
    b = rng.uniform(-0.4, 0.4, (bsz, t_len - 1))
    # make diagonally dominant (SPD)
    a[:, 1:-1] = np.abs(b[:, :-1]) + np.abs(b[:, 1:]) + rng.uniform(0.1, 0.5, (bsz, t_len - 2))
    a[:, 0] = np.abs(b[:, 0]) + rng.uniform(0.1, 0.5, bsz)
    a[:, -1] = np.abs(b[:, -1]) + rng.uniform(0.1, 0.5, bsz)
    return torch.tensor(a), torch.tensor(b)


def _dense(a, b):
    bsz, t_len = a.shape
    H = torch.diag_embed(a)
    idx = torch.arange(t_len - 1)
    H[:, idx, idx + 1] = b
    H[:, idx + 1, idx] = b
    return H


# ---------------------------------------------------------------- Thomas
def test_06_thomas_vs_linalg_solve():
    a, b = _spd_tridiag(4, 60, 1)
    g = torch.randn(4, 60)
    dz, l, d = tt.thomas_factor_and_solve(a, b, g)
    H = _dense(a, b)
    ref = torch.linalg.solve(H, g)
    assert torch.max(torch.abs(dz - ref)) < 1e-10


def test_06_thomas_numpy_parity_with_torch():
    a, b = _spd_tridiag(3, 50, 2)
    g = torch.randn(3, 50)
    dz_t, l_t, d_t = tt.thomas_factor_and_solve(a, b, g)
    dz_n, l_n, d_n = tt.thomas_factor_and_solve_numpy(a.numpy(), b.numpy(),
                                                      g.numpy())
    assert np.max(np.abs(dz_n - dz_t.numpy())) < 1e-12
    assert np.max(np.abs(d_n - d_t.numpy())) < 1e-12


def test_06_thomas_factor_floor():
    a = torch.tensor([[1.0, 0.0, 1.0]])     # near-singular interior
    b = torch.tensor([[0.9, 0.9]])
    l, d = tt.thomas_factor(a, b)
    assert torch.isfinite(d).all()
    assert (d >= 1e-7).all()


def test_06_thomas_cuda_dispatch_matches_cpu():
    a, b = _spd_tridiag(3, 40, 3)
    g = torch.randn(3, 40)
    dz, _, _ = tt.thomas_factor_and_solve(a, b, g)
    dz2, _, _ = cuda_kernels.thomas_solve_batched(a, b, g, use_cuda=False)
    assert torch.allclose(dz, dz2, atol=1e-14)


def test_06_thomas_dynamic_range():
    """Stable across fluxes 1e-4 .. 1e6 MW (z in [-9, 14])."""
    a = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    b = torch.tensor([[-0.5, -0.5, -0.5]])
    g = torch.tensor([[1e-4, 1.0, 1e3, 1e6]])
    dz, _, _ = tt.thomas_factor_and_solve(a, b, g)
    H = _dense(a, b)
    ref = torch.linalg.solve(H, g)
    assert torch.max(torch.abs(dz - ref) / ref.abs().clamp(min=1e-30)) < 1e-10


# ---------------------------------------------------------------- Takahashi
def test_06_takahashi_vs_inv():
    a, b = _spd_tridiag(3, 50, 4)
    l, d = tt.thomas_factor(a, b)
    sig_d, sig_l = tt.takahashi_marginals(l, d)
    H = _dense(a, b)
    inv = torch.linalg.inv(H)
    assert torch.max(torch.abs(sig_d - inv.diagonal(dim1=1, dim2=2))) < 1e-10
    off = inv[:, torch.arange(49), torch.arange(1, 50)]
    assert torch.max(torch.abs(sig_l - off)) < 1e-10


def test_06_takahashi_cuda_dispatch():
    a, b = _spd_tridiag(3, 40, 5)
    l, d = tt.thomas_factor(a, b)
    sd_c, sl_c = tt.takahashi_marginals(l, d)
    sd_k, sl_k = cuda_kernels.takahashi_batched(l, d, use_cuda=False)
    assert torch.allclose(sd_c, sd_k, atol=1e-14)
    assert torch.allclose(sl_c, sl_k, atol=1e-14)


def test_06_ar1_tridiag_structure():
    omega = torch.tensor([2.0, 3.0])
    rho = torch.tensor([0.6, 0.8])
    a, b = tt.build_ar1_tridiag(omega, rho, 5)
    assert a.shape == (2, 5) and b.shape == (2, 4)
    assert torch.allclose(a[:, 0], omega)
    assert torch.allclose(a[:, -1], omega)
    interior = ((1.0 + rho ** 2) * omega).unsqueeze(1)
    assert torch.allclose(a[:, 1:-1], interior.expand(2, 3))
    assert torch.allclose(b, (-rho * omega).unsqueeze(1).expand(2, 4))


# ---------------------------------------------------------------- Sampling
def test_06_joint_cholesky_covariance_matches_inv():
    """Empirical covariance of 10,000 draws matches H^{-1}."""
    a, b = _spd_tridiag(2, 40, 6)
    l, d = tt.thomas_factor(a, b)
    H = _dense(a, b)
    inv = torch.linalg.inv(H)
    torch.manual_seed(0)
    eps = torch.randn(2, 40, 10_000)
    u = tt.joint_cholesky_sample(l, d, eps)
    # covariance along T for each batch row
    for i in range(2):
        cov = torch.cov(u[i])          # (T, T): samples along dim 1
        assert cov.shape == inv[i].shape
        assert torch.max(torch.abs(cov - inv[i]) / inv[i].abs().max()) < 0.30
        assert torch.allclose(cov.diagonal(), inv[i].diagonal(), rtol=0.15)


def test_06_joint_cholesky_mean_zero():
    a, b = _spd_tridiag(2, 30, 7)
    l, d = tt.thomas_factor(a, b)
    torch.manual_seed(1)
    eps = torch.randn(2, 30, 5_000)
    u = tt.joint_cholesky_sample(l, d, eps)
    assert u.mean(dim=2).abs().max() < 0.05


def test_06_cholesky_cuda_dispatch():
    a, b = _spd_tridiag(2, 30, 8)
    l, d = tt.thomas_factor(a, b)
    torch.manual_seed(2)
    eps = torch.randn(2, 30, 50)
    u1 = tt.joint_cholesky_sample(l, d, eps)
    u2 = cuda_kernels.cholesky_sample_batched(l, d, eps, use_cuda=False)
    assert torch.allclose(u1, u2, atol=1e-13)


@pytest.mark.cuda
def test_06_cuda_load_inline_bit_parity():
    """CUDA kernel parity (< 1e-10) vs the PyTorch CPU reference."""
    assert cuda_kernels.cuda_available(), "nvcc/CUDA extension unavailable"
    a, b = _spd_tridiag(4, 512, 9)
    g = torch.randn(4, 512)
    dz_cpu, l_cpu, d_cpu = tt.thomas_factor_and_solve(a, b, g)
    dz_gpu, l_gpu, d_gpu = cuda_kernels.thomas_solve_batched(a, b, g,
                                                             use_cuda=True)
    assert torch.max(torch.abs(dz_cpu - dz_gpu)) < 1e-10
    sd_c, sl_c = tt.takahashi_marginals(l_cpu, d_cpu)
    sd_g, sl_g = cuda_kernels.takahashi_batched(l_cpu, d_cpu, use_cuda=True)
    assert torch.max(torch.abs(sd_c - sd_g)) < 1e-10
    torch.manual_seed(3)
    eps = torch.randn(4, 512, 16)
    u_c = tt.joint_cholesky_sample(l_cpu, d_cpu, eps)
    u_g = cuda_kernels.cholesky_sample_batched(l_cpu, d_cpu, eps,
                                               use_cuda=True)
    assert torch.max(torch.abs(u_c - u_g)) < 1e-10


@pytest.mark.cuda
def test_06_cuda_single_launch_shape():
    assert cuda_kernels.cuda_available()
    a, b = _spd_tridiag(1001, 97, 10)      # non-power-of-two batch
    g = torch.randn(1001, 97)
    dz, _, _ = cuda_kernels.thomas_solve_batched(a, b, g, use_cuda=True)
    H = _dense(a, b)
    ref = torch.linalg.solve(H, g)
    assert torch.max(torch.abs(dz - ref)) < 1e-10


@pytest.mark.cuda
def test_06_cuda_multi_device_residency():
    """Every visible GPU (cuda:0..3 on a 4x L4 node) runs the kernels on
    its own device: outputs stay on the input device (no .cpu() round
    trip, no cross-device mixing) and match the CPU reference."""
    assert cuda_kernels.cuda_available(), cuda_kernels.compile_error()
    a, b = _spd_tridiag(37, 129, 11)
    g = torch.randn(37, 129)
    dz_ref, l_ref, d_ref = tt.thomas_factor_and_solve(a, b, g)
    sd_ref, _ = tt.takahashi_marginals(l_ref, d_ref)
    torch.manual_seed(5)
    eps = torch.randn(37, 129, 8)
    u_ref = tt.joint_cholesky_sample(l_ref, d_ref, eps)
    for k in range(torch.cuda.device_count()):
        dev = torch.device("cuda", k)
        dz, l, d = cuda_kernels.thomas_solve_batched(
            a.to(dev), b.to(dev), g.to(dev), use_cuda=True)
        assert dz.device == dev and l.device == dev and d.device == dev
        l2, d2 = cuda_kernels.thomas_factor_batched(a.to(dev), b.to(dev),
                                                    use_cuda=True)
        assert torch.equal(l2, l) and torch.equal(d2, d)
        sd, _ = cuda_kernels.takahashi_batched(l, d, use_cuda=True)
        u = cuda_kernels.cholesky_sample_batched(l, d, eps.to(dev),
                                                 use_cuda=True)
        assert sd.device == dev and u.device == dev
        assert torch.max(torch.abs(dz.cpu() - dz_ref)) < 1e-10
        assert torch.max(torch.abs(sd.cpu() - sd_ref)) < 1e-10
        assert torch.max(torch.abs(u.cpu() - u_ref)) < 1e-10


def _nvcc_toolkit_present() -> bool:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:  # noqa: BLE001
        return False
    import os
    return bool(CUDA_HOME) and os.path.exists(
        os.path.join(CUDA_HOME, "bin", "nvcc"))


@pytest.mark.skipif(not _nvcc_toolkit_present(),
                    reason="nvcc / CUDA toolkit not installed (toolchain-gated)")
def test_06_cuda_extension_compiles_with_nvcc():
    """The load_inline extension BUILDS (nvcc + host C++; no GPU needed):
    bindings come from functions=[...] + C++ prototypes, with no duplicate
    PYBIND11_MODULE in the CUDA translation unit."""
    assert "PYBIND11_MODULE" not in cuda_kernels._CUDA_SOURCE
    assert "CUDAGuard" in cuda_kernels._CUDA_SOURCE
    mod = cuda_kernels.build_module()
    for fn in ("thomas_solve_cuda", "thomas_factor_cuda", "takahashi_cuda",
               "cholesky_sample_cuda"):
        assert hasattr(mod, fn)


# ---------------------------------------------------------------- Armijo
def test_06_armijo_strict_descent():
    from src.engine.solver import GMRFSolver, SolverData
    data = _tiny_solver_data()
    solver = GMRFSolver(data)
    j0 = float(solver.objective(solver.z))
    history = []
    orig = solver._newton_block

    def traced(color, s, t_lo=0):
        pre = float(solver.objective(solver.z))
        orig(color, s, t_lo)
        post = float(solver.objective(solver.z))
        history.append((pre, post))

    solver._newton_block = traced
    solver.solve(sweeps=1)
    assert len(history) > 0
    for pre, post in history:
        assert post <= pre + 1e-6      # strict objective descent every block


def _tiny_solver_data():
    from src.engine.solver import SolverData
    rng = np.random.default_rng(11)
    B, T, P = 4, 30, 5
    y = np.zeros((B, T, P, 2))
    y[:, ::3, 2, 0] = np.exp(rng.normal(1.0, 0.3, (B, T // 3 + 1)))[:, :T // 3 * 0 + len(range(0, T, 3))]
    y[:, 1::4, 0, 1] = 1.0
    mu = np.zeros((B, T, P, 2))
    tau2 = np.full((B, T, P, 2), 1.0 / 0.25)
    k = np.full((B, P, 2), 0.2)
    vis = np.full((B, T, P), 0.7)
    op = np.ones((T, P))
    op[:5, :] = 0.0
    mu_prior = np.full((B, T, 2), 0.5)
    return SolverData(
        cells=np.arange(B), colors=torch.tensor([0, 1, 2, 0]),
        y=torch.tensor(y), sat=torch.zeros((B, T, P, 2), dtype=torch.uint8),
        mu=torch.tensor(mu), tau2=torch.tensor(tau2),
        k_miss=torch.tensor(k), vis=torch.tensor(vis), op=torch.tensor(op),
        mu_prior=torch.tensor(mu_prior), omega=torch.full((B,), 2.0),
        rho=torch.full((B,), 0.6), eta_dn=torch.full((B,), 0.1),
        beta_dn=torch.full((B,), 0.5), n_nbrs=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        edge_i=torch.tensor([0]), edge_j=torch.tensor([3]),
        t_days=T, pyrome_of_cell={i: 0 for i in range(B)})


def test_06_three_color_spatial_cloud_gap_recovery():
    """A cell with NO observations recovers its neighbours' state through
    the spatial 1-ring coupling (3-color Gauss-Seidel)."""
    from src.engine.solver import GMRFSolver
    data = _tiny_solver_data()
    # cell 2 has no observations at all
    data.y[2] = 0.0
    data.mu_prior[2] = 0.5
    solver = GMRFSolver(data)
    # give cell 2 a chain to the observed cells via edges 1-2, 2-3
    data.edge_i = torch.tensor([0, 1, 2])
    data.edge_j = torch.tensor([3, 2, 3])
    data.n_nbrs = torch.tensor([1.0, 1.0, 2.0, 2.0])
    solver = GMRFSolver(data)
    solver.solve(sweeps=S.SOLVER_SWEEPS)
    z = solver.z.numpy()
    observed_mask = (data.y.numpy() > 0).any(axis=(2, 3))
    # observed cells have nonzero deviations; cell 2 pulled toward them
    assert observed_mask[0].any() and observed_mask[1].any()
    assert np.abs(z[2].mean() - z[1].mean()) < 2.0


def test_06_newton_block_delta_exact():
    from src.engine.solver import GMRFSolver
    data = _tiny_solver_data()
    solver = GMRFSolver(data)
    torch.manual_seed(4)
    idx = torch.tensor([0, 3])
    z0 = solver.z.clone()
    z1 = z0.clone()
    z1[idx, :, 0] += torch.randn(len(idx), data.t_days) * 0.2  # block-only move
    delta = solver._block_delta_j(z0, z1, idx, 0)
    true = float(solver.objective(z1)) - float(solver.objective(z0))
    assert delta == pytest.approx(true, rel=1e-9, abs=1e-6)


def test_06_fixed_lag_update_matches_full_batch():
    """14-day fixed-lag re-solve anchored at t_start-1 matches the full
    batch (< 1e-3) across an active fire transition."""
    from src.engine.solver import GMRFSolver
    data = _tiny_solver_data()
    # active fire transition inside the trailing 14-day window
    T = data.t_days
    data.y[:, T - 9:, 2, 0] = 40.0
    data.y[:, T - 9:, 2, 1] = 12.0
    solver = GMRFSolver(data)
    z_full = solver.solve(sweeps=S.SOLVER_SWEEPS).clone()
    solver2 = GMRFSolver(data)
    solver2.z = z_full.clone()          # saved state from the full batch
    z2 = solver2.solve_fixed_lag(window_days=14)
    w = min(14, T)
    diff = (z2[:, -w:, :] - z_full[:, -w:, :]).abs().max()
    assert float(diff) < 1e-3
    # the transition is real (states jump inside the window) ...
    assert float((z_full[:, -3:, 0].mean() - z_full[:, T - 14:T - 10, 0]
                  .mean()).abs()) > 1.0
    # ... and the anchor row (t_start - 1) and everything before it is
    # untouched by the windowed re-solve
    t_start = T - w
    assert torch.equal(z2[:, :t_start, :], z_full[:, :t_start, :])


def test_06_fixed_lag_assimilates_new_window_data():
    """Starting from a state solved WITHOUT the trailing detections, the
    windowed update assimilates them and lands on the exact conditional
    MAP of the window (verified by its vanishing windowed gradient)."""
    from src.engine.solver import GMRFSolver
    data = _tiny_solver_data()
    T = data.t_days
    y_new = data.y.clone()
    y_new[:, T - 9:, 2, 0] = 40.0
    old = GMRFSolver(data)
    z_old = old.solve(sweeps=S.SOLVER_SWEEPS).clone()
    data.y = y_new
    upd = GMRFSolver(data)
    upd.z = z_old.clone()
    z_upd = upd.solve_fixed_lag(window_days=14).clone()
    t_start = T - 14
    assert torch.equal(z_upd[:, :t_start], z_old[:, :t_start])
    assert float(z_upd[:, -3:, 0].mean()) > float(z_old[:, -3:, 0].mean()) + 1.0
    # windowed stationarity: block gradients vanish on [t_start, T-1]
    for color in (0, 1, 2):
        for s in (0, 1):
            built = upd._block_system(color, s, t_start)
            if built is None:
                continue
            _idx, _a, _b, g = built
            assert float(g.abs().max()) < 1e-3


def test_06_solver_z_bounds_respected():
    from src.engine.solver import GMRFSolver
    data = _tiny_solver_data()
    data.y[:] = 0.0
    data.mu_prior[:] = 13.9
    solver = GMRFSolver(data)
    z = solver.solve(sweeps=2)
    assert (z <= 14.0 + 1e-9).all()


def test_06_vram_estimate_under_6gb():
    from src.engine.solver import estimate_chunk_vram_bytes
    est = estimate_chunk_vram_bytes(batch_size=S.BATCH_SIZE, t_days=S.T_DAYS,
                                    n_platforms=S.N_PLATFORMS,
                                    m_samples=S.MC_SAMPLES, sub_batch=32)
    assert est < 6 * 1024 ** 3
    # the spec's dense per-state observation plane (B=1000, T=9510, P=5)
    dense = S.BATCH_SIZE * S.T_DAYS * S.N_PLATFORMS * 8
    assert dense * 8 < 6 * 1024 ** 3


def test_06_thomas_batch_and_t_shapes():
    a, b = _spd_tridiag(7, 11, 12)
    g = torch.randn(7, 11)
    dz, l, d = tt.thomas_factor_and_solve(a, b, g)
    assert dz.shape == (7, 11) and l.shape == (7, 10) and d.shape == (7, 11)
    eps = torch.randn(7, 11, 3)
    u = tt.joint_cholesky_sample(l, d, eps)
    assert u.shape == (7, 11, 3)


# ---------------------------------------------------------------------------
# Resumable chunk results (Kaggle: retry after an out-of-memory kill)
# ---------------------------------------------------------------------------
def test_06_solve_resume_skips_finished_chunks_and_is_identical(short_twin,
                                                                tmp_path,
                                                                monkeypatch):
    import numpy as np
    import pandas as pd
    from src.engine import solver as slv
    tw = short_twin
    cells = np.array(sorted(set(int(c) for c in tw["pass_cal"]["h3_index"])))[:6]
    args = (tw["pass_cal"], cells, tw["hex_pyromes"], tw["priors"],
            tw["artifacts"], tw["cloud_df"])
    kw = dict(t_days=tw["cfg"].t_days, platform_active=tw["platform_active"],
              batch_size=2, keep_states=False, n_devices=0)
    ref = slv.solve_all_chunks(*args, **kw)               # no resume dir
    root = tmp_path / "resume"
    monkeypatch.setenv("FIRE_HARMONIZER_RESUME_DIR", str(root))
    calls = []
    real = slv._chunk_job
    monkeypatch.setattr(slv, "_chunk_job",
                        lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
    first = slv.solve_all_chunks(*args, **kw)
    assert sorted(calls) == [0, 1, 2]
    solve_dirs = list(root.glob("solve_*"))
    assert len(solve_dirs) == 1
    assert len(list(solve_dirs[0].glob("chunk_*.pkl"))) == 3
    assert not list(solve_dirs[0].glob("*.part"))
    # "crash": chunk 1 lost -> only chunk 1 is recomputed
    (solve_dirs[0] / "chunk_000001.pkl").unlink()
    calls.clear()
    again = slv.solve_all_chunks(*args, **kw)
    assert calls == [1]
    key = ["h3_index", "year", "month"]
    for out in (first, again):
        pd.testing.assert_frame_equal(
            ref["monthly"].sort_values(key).reset_index(drop=True),
            out["monthly"].sort_values(key).reset_index(drop=True))
        assert "z_map" not in out["results"][0]       # states dropped
    # different batching -> different fingerprint, never mixed
    slv.solve_all_chunks(*args, **dict(kw, batch_size=3))
    assert len(list(root.glob("solve_*"))) == 2


def test_06_gpu_worker_distributes_only_remaining_chunks(tmp_path, monkeypatch):
    """A retry with FEWER workers must still cover every unfinished chunk
    exactly once (workers split the remaining list, not ci % n_workers)."""
    import pickle
    from src.engine import solver as slv
    done_before = {0, 3, 4}
    for ci in done_before:
        slv._dump_atomic({"ci": ci}, slv._chunk_path(str(tmp_path), ci))
    todo = [ci for ci in range(10) if ci not in done_before]
    job = tmp_path / "job.pkl"
    job.write_bytes(pickle.dumps({"chunks": [[ci] for ci in range(10)],
                                  "shared": {}, "todo": todo,
                                  "opts": {"keep_states": False,
                                           "checkpoint_dir": None}}))
    solved = []
    monkeypatch.setattr(slv, "_chunk_job",
                        lambda shared, ci, chunk, dev, use_cuda, opts:
                        (solved.append(ci), {"ci": ci, "z_map": 1})[1])
    monkeypatch.setattr(slv.torch.cuda, "set_device", lambda d: None)
    monkeypatch.setattr(slv.torch.cuda, "empty_cache", lambda: None)
    for rank in range(3):                      # 3 workers instead of e.g. 8
        slv._gpu_worker(rank, 3, 4, str(job), str(tmp_path))
    assert sorted(solved) == todo
    for ci in range(10):
        res = pickle.loads(open(slv._chunk_path(str(tmp_path), ci), "rb").read())
        assert res["ci"] == ci and "z_map" not in res
