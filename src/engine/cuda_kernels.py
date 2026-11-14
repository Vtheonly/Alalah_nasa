"""Single-launch CUDA C++ float64 Thomas/Takahashi/Cholesky kernels.

Contract (readme.md Sections 4 & 5.3.4 item 5 / audit item 8):
  * ``thomas_solve_cuda``, ``takahashi_cuda``, ``cholesky_sample_cuda``
    compiled lazily via ``torch.utils.cpp_extension.load_inline`` as ONE
    thread-per-cell kernel each (eliminating the 9,510 sequential kernel
    launches of the naive per-day approach, ~50x speedup).
  * Dispatch falls back transparently to the pure-PyTorch
    ``thomas_takahashi.py`` CPU reference when CUDA or nvcc is unavailable,
    so CPU-only test environments run the identical math.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from src.engine import thomas_takahashi as tt

logger = logging.getLogger("fire_harmonizer.engine.cuda_kernels")

_cuda_module = None
_compilation_attempted = False

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Thread-per-cell Thomas factorization + solve: one launch for all cells.
// a: (B, T) main diagonal; b: (B, T-1) sub-diagonal; g: (B, T) RHS.
// Outputs: d (B, T), l (B, T-1), x (B, T).
// ---------------------------------------------------------------------------
__global__ void thomas_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ g,
    double* __restrict__ d,
    double* __restrict__ l,
    double* __restrict__ x,
    int B, int T, double pivot_floor) {

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= B) return;
  const double* a_c = a + (long)cell * T;
  const double* b_c = b + (long)cell * (T - 1);
  const double* g_c = g + (long)cell * T;
  double* d_c = d + (long)cell * T;
  double* l_c = l + (long)cell * (T - 1);
  double* x_c = x + (long)cell * T;

  d_c[0] = a_c[0];
  for (int t = 1; t < T; ++t) {
    l_c[t - 1] = b_c[t - 1] / d_c[t - 1];
    double next = a_c[t] - l_c[t - 1] * b_c[t - 1];
    d_c[t] = next < pivot_floor ? pivot_floor : next;
  }
  // forward substitution (in-place on x as workspace)
  x_c[0] = g_c[0];
  for (int t = 1; t < T; ++t) {
    x_c[t] = g_c[t] - l_c[t - 1] * x_c[t - 1];
  }
  // backward substitution
  x_c[T - 1] = x_c[T - 1] / d_c[T - 1];
  for (int t = T - 2; t >= 0; --t) {
    x_c[t] = x_c[t] / d_c[t] - l_c[t] * x_c[t + 1];
  }
}

// ---------------------------------------------------------------------------
// Thread-per-cell Takahashi marginals: var (B, T) and cov1 (B, T-1).
// ---------------------------------------------------------------------------
__global__ void takahashi_kernel(
    const double* __restrict__ d,
    const double* __restrict__ l,
    double* __restrict__ var,
    double* __restrict__ cov1,
    int B, int T) {

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= B) return;
  const double* d_c = d + (long)cell * T;
  const double* l_c = l + (long)cell * (T - 1);
  double* v_c = var + (long)cell * T;
  double* c_c = cov1 + (long)cell * (T - 1);

  v_c[T - 1] = 1.0 / d_c[T - 1];
  for (int t = T - 2; t >= 0; --t) {
    c_c[t] = -l_c[t] * v_c[t + 1];
    v_c[t] = 1.0 / d_c[t] - l_c[t] * c_c[t];
  }
}

// ---------------------------------------------------------------------------
// Thread-per-cell exact joint Cholesky sampling: eps (B, T, M) -> u.
// ---------------------------------------------------------------------------
__global__ void cholesky_sample_kernel(
    const double* __restrict__ d,
    const double* __restrict__ l,
    const double* __restrict__ eps,
    double* __restrict__ u,
    int B, int T, int M) {

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= B) return;
  const double* d_c = d + (long)cell * T;
  const double* l_c = l + (long)cell * (T - 1);
  const double* e_c = eps + (long)cell * T * M;
  double* u_c = u + (long)cell * T * M;

  for (int m = 0; m < M; ++m) {
    u_c[(long)(T - 1) * M + m] = e_c[(long)(T - 1) * M + m] / sqrt(d_c[T - 1]);
  }
  for (int t = T - 2; t >= 0; --t) {
    double inv_sqrt_d = 1.0 / sqrt(d_c[t]);
    double lt = l_c[t];
    for (int m = 0; m < M; ++m) {
      u_c[(long)t * M + m] =
          e_c[(long)t * M + m] * inv_sqrt_d - lt * u_c[(long)(t + 1) * M + m];
    }
  }
}

std::vector<torch::Tensor> thomas_solve_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor g, double pivot_floor) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && g.is_cuda(),
              "thomas_solve_cuda requires CUDA tensors");
  const auto B = a.size(0);
  const auto T = a.size(1);
  auto d = torch::empty_like(a);
  auto l = torch::empty_like(b);
  auto x = torch::empty_like(g);
  const int threads = 128;
  const int blocks = (B + threads - 1) / threads;
  thomas_kernel<<<blocks, threads>>>(
      a.data_ptr<double>(), b.data_ptr<double>(), g.data_ptr<double>(),
      d.data_ptr<double>(), l.data_ptr<double>(), x.data_ptr<double>(),
      (int)B, (int)T, pivot_floor);
  return {d, l, x};
}

std::vector<torch::Tensor> takahashi_cuda(
    torch::Tensor d, torch::Tensor l) {
  TORCH_CHECK(d.is_cuda() && l.is_cuda(), "takahashi_cuda requires CUDA tensors");
  const auto B = d.size(0);
  const auto T = d.size(1);
  auto var = torch::empty_like(d);
  auto cov1 = torch::empty_like(l);
  const int threads = 128;
  const int blocks = (B + threads - 1) / threads;
  takahashi_kernel<<<blocks, threads>>>(
      d.data_ptr<double>(), l.data_ptr<double>(),
      var.data_ptr<double>(), cov1.data_ptr<double>(), (int)B, (int)T);
  return {var, cov1};
}

torch::Tensor cholesky_sample_cuda(
    torch::Tensor d, torch::Tensor l, torch::Tensor eps) {
  TORCH_CHECK(d.is_cuda() && l.is_cuda() && eps.is_cuda(),
              "cholesky_sample_cuda requires CUDA tensors");
  const auto B = d.size(0);
  const auto T = d.size(1);
  const auto M = eps.size(2);
  auto u = torch::empty_like(eps);
  const int threads = 64;
  const int blocks = (B + threads - 1) / threads;
  cholesky_sample_kernel<<<blocks, threads>>>(
      d.data_ptr<double>(), l.data_ptr<double>(), eps.data_ptr<double>(),
      u.data_ptr<double>(), (int)B, (int)T, (int)M);
  return u;
}
"""

_CPP_SOURCE = r"""
std::vector<torch::Tensor> thomas_solve_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor g, double pivot_floor);
std::vector<torch::Tensor> takahashi_cuda(torch::Tensor d, torch::Tensor l);
torch::Tensor cholesky_sample_cuda(
    torch::Tensor d, torch::Tensor l, torch::Tensor eps);
"""


def _get_cuda_module():
    """Lazily compile the inline extension; None when CUDA/nvcc is absent."""
    global _cuda_module, _compilation_attempted
    if _compilation_attempted:
        return _cuda_module
    _compilation_attempted = True
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable -> pure-PyTorch float64 reference path")
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        _cuda_module = load_inline(
            name="fire_harmonizer_cuda_kernels",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=[
                "thomas_solve_cuda",
                "takahashi_cuda",
                "cholesky_sample_cuda",
            ],
            verbose=False,
        )
        logger.info("CUDA kernels compiled via load_inline")
    except Exception as exc:  # noqa: BLE001
        logger.warning("CUDA kernel compilation failed: %s", exc)
        _cuda_module = None
    return _cuda_module


def cuda_available() -> bool:
    return _get_cuda_module() is not None


def thomas_solve(
    a: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
    pivot_floor: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thomas factor+solve on CUDA when available, CPU reference otherwise."""
    from src.config import settings as S

    mod = _get_cuda_module()
    if mod is not None and a.is_cuda:
        d, l, x = mod.thomas_solve_cuda(a, b, g, float(S.PIVOT_FLOOR))
        return d, l, x
    return tt.thomas_factor_and_solve(a, b, g)


def takahashi(
    d: torch.Tensor, l: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Takahashi marginals on CUDA when available, CPU reference otherwise."""
    mod = _get_cuda_module()
    if mod is not None and d.is_cuda:
        var, cov1 = mod.takahashi_cuda(d, l)
        return var, cov1
    return tt.takahashi_marginals(d, l)


def cholesky_sample(
    d: torch.Tensor, l: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Exact joint Cholesky sampling (CUDA or CPU reference)."""
    mod = _get_cuda_module()
    if mod is not None and d.is_cuda:
        return mod.cholesky_sample_cuda(d, l, eps)
    return tt.joint_cholesky_sample(d, l, eps)
