"""Pure PyTorch float64 Thomas LDL^T, Takahashi marginals, joint sampling.

Contract (readme.md Sections 4 & 5.3.4):
  * ``thomas_factor_and_solve(a, b, g)`` — batched tridiagonal factorization
    ``H = L D L^T`` (pivot floor 1e-7) and forward/backward substitution.
  * ``takahashi_marginals(d, l)`` — exact posterior diagonal and lag-1
    covariance recursions.
  * ``joint_cholesky_sample(d, l, eps)`` — exact joint trajectories via
    ``L^T u = D^{-1/2} eps`` backwards recursion.

All recurrences are batched over the leading cell axis and compiled with
``torch.jit.script`` (still pure PyTorch float64; identical algorithm), with
a transparent eager fallback when scripting is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.config import settings as S

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Eager (also TorchScript-compatible) reference implementations
# --------------------------------------------------------------------------- #
def thomas_factor_solve_eager(
    a: Tensor, b: Tensor, g: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Factor the batched tridiagonal system and solve H x = g.

    ``a``: (B, T) main diagonal; ``b``: (B, T-1) sub-diagonal; ``g``: (B, T)
    right-hand side. Returns ``(d, l, x)`` with pivots floored at 1e-7.
    """
    B, T = a.shape
    d = torch.empty_like(a)
    l = torch.empty_like(b)
    d[:, 0] = a[:, 0]
    for t in range(1, T):
        l[:, t - 1] = b[:, t - 1] / d[:, t - 1]
        d[:, t] = torch.clamp(a[:, t] - l[:, t - 1] * b[:, t - 1],
                              min=S.PIVOT_FLOOR)
    y = torch.empty_like(g)
    y[:, 0] = g[:, 0]
    for t in range(1, T):
        y[:, t] = g[:, t] - l[:, t - 1] * y[:, t - 1]
    x = torch.empty_like(g)
    x[:, T - 1] = y[:, T - 1] / d[:, T - 1]
    for t in range(T - 2, -1, -1):
        x[:, t] = y[:, t] / d[:, t] - l[:, t] * x[:, t + 1]
    return d, l, x


def takahashi_marginals_eager(d: Tensor, l: Tensor) -> Tuple[Tensor, Tensor]:
    """Exact diagonal and lag-1 posterior covariances of H = L D L^T.

    ``Sigma_{T-1,T-1} = 1/d_{T-1}``; going backwards
    ``Sigma_{t,t+1} = -l_t Sigma_{t+1,t+1}`` and
    ``Sigma_{t,t} = 1/d_t - l_t Sigma_{t,t+1}``.
    """
    B, T = d.shape
    var = torch.empty_like(d)
    cov1 = torch.empty_like(l)  # (B, T-1): cov(z_t, z_{t+1})
    var[:, T - 1] = 1.0 / d[:, T - 1]
    for t in range(T - 2, -1, -1):
        cov1[:, t] = -l[:, t] * var[:, t + 1]
        var[:, t] = 1.0 / d[:, t] - l[:, t] * cov1[:, t]
    return var, cov1


def joint_cholesky_sample_eager(d: Tensor, l: Tensor, eps: Tensor) -> Tensor:
    """Solve L^T u = D^{-1/2} eps backwards; cov(u) = H^{-1}.

    ``eps``: (B, T, M) standard normal draws; returns ``u`` of the same
    shape with exact joint posterior covariance.
    """
    B, T, M = eps.shape
    u = torch.empty_like(eps)
    u[:, T - 1, :] = eps[:, T - 1, :] / torch.sqrt(d[:, T - 1]).unsqueeze(-1)
    for t in range(T - 2, -1, -1):
        u[:, t, :] = (
            eps[:, t, :] / torch.sqrt(d[:, t]).unsqueeze(-1)
            - l[:, t].unsqueeze(-1) * u[:, t + 1, :]
        )
    return u


# --------------------------------------------------------------------------- #
# TorchScript acceleration with transparent fallback
# --------------------------------------------------------------------------- #
_script_thomas = None
_script_takahashi = None
_script_sample = None
try:  # pragma: no cover - availability depends on the torch build
    _script_thomas = torch.jit.script(thomas_factor_solve_eager)
    _script_takahashi = torch.jit.script(takahashi_marginals_eager)
    _script_sample = torch.jit.script(joint_cholesky_sample_eager)
except Exception:  # noqa: BLE001 - fall back to eager loops
    _script_thomas = None
    _script_takahashi = None
    _script_sample = None


def thomas_factor_and_solve(
    a: Tensor, b: Tensor, g: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    fn = _script_thomas if _script_thomas is not None else thomas_factor_solve_eager
    return fn(a, b, g)


def takahashi_marginals(d: Tensor, l: Tensor) -> Tuple[Tensor, Tensor]:
    fn = _script_takahashi if _script_takahashi is not None else takahashi_marginals_eager
    return fn(d, l)


def joint_cholesky_sample(d: Tensor, l: Tensor, eps: Tensor) -> Tensor:
    fn = _script_sample if _script_sample is not None else joint_cholesky_sample_eager
    return fn(d, l, eps)


# --------------------------------------------------------------------------- #
# Convenience wrappers used by the solver and tests
# --------------------------------------------------------------------------- #
def thomas_solve(a: Tensor, b: Tensor, g: Tensor) -> Tensor:
    """Solve H x = g only (factorization discarded)."""
    _, _, x = thomas_factor_and_solve(a, b, g)
    return x


def solve_tridiagonal_reference(
    a: Tensor, b: Tensor, g: Tensor
) -> Tensor:
    """Dense ``torch.linalg.solve`` reference for verification tests."""
    B, T = a.shape
    H = torch.zeros(B, T, T, dtype=torch.float64, device=a.device)
    idx = torch.arange(T, device=a.device)
    H[:, idx, idx] = a
    H[:, idx[1:], idx[:-1]] = b
    H[:, idx[:-1], idx[1:]] = b
    return torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)


def inverse_tridiagonal_reference(a: Tensor, b: Tensor) -> Tensor:
    """Dense ``torch.linalg.inv(H)`` reference for Takahashi verification."""
    B, T = a.shape
    H = torch.zeros(B, T, T, dtype=torch.float64, device=a.device)
    idx = torch.arange(T, device=a.device)
    H[:, idx, idx] = a
    H[:, idx[1:], idx[:-1]] = b
    H[:, idx[:-1], idx[1:]] = b
    return torch.linalg.inv(H)
