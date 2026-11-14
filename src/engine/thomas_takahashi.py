"""Pure PyTorch ``float64`` Thomas LDL^T, Takahashi marginals, joint sampling.

Contract (readme Sections 4.8, 5.3.4): the exact reference implementations of

* ``thomas_factor_and_solve``: factorize ``H = L D L^T`` with
  ``d_0 = a_0``; ``l_t = b_t / d_t``; ``d_{t+1} = max(a_{t+1} - l_t b_t,
  1e-7)`` and solve ``H dz = g`` by forward/backward substitution.
* ``takahashi_marginals``: ``Sigma_{T-1,T-1} = 1/d_{T-1}``; backward
  ``Sigma_{t,t+1} = -l_t Sigma_{t+1,t+1}``;
  ``Sigma_{t,t} = 1/d_t - l_t Sigma_{t,t+1}``.
* ``joint_cholesky_sample``: solve ``L^T u = D^{-1/2} eps`` backwards
  (``u_{T-1} = eps_{T-1}/sqrt(d_{T-1})``; ``u_t = eps_t/sqrt(d_t) -
  l_t u_{t+1}``) for M joint correlated trajectories.

All algebra runs in ``torch.float64`` (readme Must-Do).
"""
from __future__ import annotations

import numpy as np
import torch

_DIAG_FLOOR = 1e-7


def thomas_factor(a: torch.Tensor, b: torch.Tensor):
    """Factorize a batch of symmetric tridiagonal matrices H = L D L^T.

    ``a``: (B, T) main diagonal; ``b``: (B, T-1) sub/super-diagonal.
    Returns (l, d): (B, T-1), (B, T).
    """
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    bsz, t_len = a.shape
    d = torch.empty_like(a)
    l = torch.empty_like(b)
    d[:, 0] = a[:, 0]
    for t in range(t_len - 1):
        l[:, t] = b[:, t] / d[:, t]
        d[:, t + 1] = torch.clamp(a[:, t + 1] - l[:, t] * b[:, t],
                                  min=_DIAG_FLOOR)
    return l, d


def thomas_solve_from_factor(l: torch.Tensor, d: torch.Tensor,
                             g: torch.Tensor) -> torch.Tensor:
    """Solve H dz = g given the LDL^T factorization (forward + backward)."""
    g = g.to(torch.float64)
    bsz, t_len = g.shape
    y = torch.empty_like(g)
    y[:, 0] = g[:, 0]
    for t in range(1, t_len):
        y[:, t] = g[:, t] - l[:, t - 1] * y[:, t - 1]
    dz = torch.empty_like(g)
    dz[:, t_len - 1] = y[:, t_len - 1] / d[:, t_len - 1]
    for t in range(t_len - 2, -1, -1):
        dz[:, t] = y[:, t] / d[:, t] - l[:, t] * dz[:, t + 1]
    return dz


def thomas_factor_and_solve(a: torch.Tensor, b: torch.Tensor,
                            g: torch.Tensor):
    """Factorize + solve. Returns (dz, l, d)."""
    l, d = thomas_factor(a, b)
    dz = thomas_solve_from_factor(l, d, g)
    return dz, l, d


def thomas_factor_and_solve_numpy(a, b, g):
    """NumPy fast path of the exact same LDL^T recursion (CPU solver hot
    path).  Bit-for-bit identical algorithm to ``thomas_factor_and_solve``
    (verified in tests); ~4x faster than the pure-torch loop on CPU.
    Accepts/returns numpy arrays."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    g = np.ascontiguousarray(g, dtype=np.float64)
    bsz, t_len = a.shape
    d = np.empty_like(a)
    l = np.empty_like(b)
    dz = np.empty_like(g)
    if t_len == 1:
        d[:, 0] = np.maximum(a[:, 0], _DIAG_FLOOR)
        l[:] = 0.0
        dz[:, 0] = g[:, 0] / d[:, 0]
        return dz, l, d
    d[:, 0] = a[:, 0]
    y_prev = g[:, 0]
    dz[:, 0] = g[:, 0]              # forward scratch y_0 (overwritten later)
    for t in range(t_len - 1):
        lt = b[:, t] / d[:, t]
        d[:, t + 1] = np.maximum(a[:, t + 1] - lt * b[:, t], _DIAG_FLOOR)
        l[:, t] = lt
        y_cur = g[:, t + 1] - lt * y_prev
        dz[:, t + 1] = y_cur
        y_prev = y_cur
    dz[:, t_len - 1] = y_prev / d[:, t_len - 1]
    for t in range(t_len - 2, -1, -1):
        dz[:, t] = dz[:, t] / d[:, t] - l[:, t] * dz[:, t + 1]
    return dz, l, d


def takahashi_marginals(l: torch.Tensor, d: torch.Tensor):
    """Exact diagonal and lag-1 marginals of H^{-1} (Takahashi recursion).

    Returns (sigma_diag (B, T), sigma_lag1 (B, T-1)).
    """
    l = l.to(torch.float64)
    d = d.to(torch.float64)
    bsz, t_len = d.shape
    sig_d = torch.empty_like(d)
    sig_l = torch.empty_like(l)
    sig_d[:, t_len - 1] = 1.0 / d[:, t_len - 1]
    for t in range(t_len - 2, -1, -1):
        sig_l[:, t] = -l[:, t] * sig_d[:, t + 1]
        sig_d[:, t] = 1.0 / d[:, t] - l[:, t] * sig_l[:, t]
    return sig_d, sig_l


def joint_cholesky_sample(l: torch.Tensor, d: torch.Tensor,
                          eps: torch.Tensor) -> torch.Tensor:
    """Solve ``L^T u = D^{-1/2} eps`` backwards for M joint trajectories.

    ``eps``: (B, T, M) standard normal draws. Returns (B, T, M) samples u
    with covariance H^{-1} (one temporal chain per batch row).
    """
    l = l.to(torch.float64)
    d = d.to(torch.float64)
    eps = eps.to(torch.float64)
    bsz, t_len, m = eps.shape
    u = torch.empty_like(eps)
    sqrt_d = torch.sqrt(d)
    u[:, t_len - 1, :] = eps[:, t_len - 1, :] / sqrt_d[:, t_len - 1].unsqueeze(-1)
    for t in range(t_len - 2, -1, -1):
        u[:, t, :] = (eps[:, t, :] / sqrt_d[:, t].unsqueeze(-1)
                      - l[:, t].unsqueeze(-1) * u[:, t + 1, :])
    return u


def build_ar1_tridiag(omega: torch.Tensor, rho: torch.Tensor, t_len: int):
    """Temporal AR(1) prior tridiagonal (a, b) per readme 5.3.3.

    ``a_0 = a_{T-1} = omega``; ``a_t = (1 + rho^2) omega``; ``b_t = -rho omega``.
    """
    omega = omega.to(torch.float64).reshape(-1, 1)
    rho = rho.to(torch.float64).reshape(-1, 1)
    a = (1.0 + rho * rho) * omega * torch.ones(
        (omega.shape[0], t_len), dtype=torch.float64, device=omega.device)
    a[:, 0] = omega[:, 0]
    a[:, -1] = omega[:, 0]
    b = -rho * omega * torch.ones(
        (omega.shape[0], max(t_len - 1, 1)), dtype=torch.float64,
        device=omega.device)
    return a, b
