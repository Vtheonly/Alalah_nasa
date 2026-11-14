"""Single-launch CUDA C++ ``float64`` Thomas / Takahashi / Cholesky kernels.

Contract (readme Sections 4.8, 5.3.4.5): a thread-per-cell CUDA kernel via
``torch.utils.cpp_extension.load_inline`` eliminating the 9,510-iteration
per-``t`` kernel-launch bottleneck of the pure-PyTorch loop, automatically
falling back to ``thomas_takahashi.py`` (the CPU reference) whenever CUDA or
``nvcc`` is unavailable.  The CUDA path is verified bit-for-bit (< 1e-10)
against the CPU reference in ``tests/test_06_gpu_solver_math.py``.
"""
from __future__ import annotations

import threading

import torch

from .thomas_takahashi import (joint_cholesky_sample as _cpu_sample,
                               takahashi_marginals as _cpu_takahashi,
                               thomas_factor_and_solve as _cpu_thomas)

_DIAG_FLOOR = 1e-7
_LOCK = threading.Lock()
_MODULE = None
_COMPILE_ERROR = None

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define DIAG_FLOOR 1e-7

// ---------------------------------------------------------------------------
// Thread-per-cell batched Thomas LDL^T factor + solve.
// a: (B, T) main diag; b: (B, T-1) off diag; g: (B, T) rhs.
// outputs l: (B, T-1), d: (B, T), dz: (B, T). One thread per cell row.
// ---------------------------------------------------------------------------
__global__ void thomas_solve_kernel(const double* __restrict__ a,
                                     const double* __restrict__ b,
                                     const double* __restrict__ g,
                                     double* __restrict__ l,
                                     double* __restrict__ d,
                                     double* __restrict__ dz,
                                     int64_t B, int64_t T) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const double* arow = a + row * T;
    const double* brow = b + row * (T - 1);
    const double* grow = g + row * T;
    double* lrow = l + row * (T - 1);
    double* drow = d + row * T;
    double* zrow = dz + row * T;

    drow[0] = arow[0];
    for (int64_t t = 0; t < T - 1; ++t) {
        lrow[t] = brow[t] / drow[t];
        double dn = arow[t + 1] - lrow[t] * brow[t];
        drow[t + 1] = dn > DIAG_FLOOR ? dn : DIAG_FLOOR;
    }
    // forward substitution (L y = g)
    zrow[0] = grow[0];                      // reuse dz as scratch y
    for (int64_t t = 1; t < T; ++t) {
        zrow[t] = grow[t] - lrow[t - 1] * zrow[t - 1];
    }
    // backward substitution (D L^T dz = y)
    zrow[T - 1] = zrow[T - 1] / drow[T - 1];
    for (int64_t t = T - 2; t >= 0; --t) {
        zrow[t] = zrow[t] / drow[t] - lrow[t] * zrow[t + 1];
    }
}

// ---------------------------------------------------------------------------
// Thread-per-cell Takahashi marginals: sigma_diag (B, T), sigma_lag1 (B, T-1)
// ---------------------------------------------------------------------------
__global__ void takahashi_kernel(const double* __restrict__ l,
                                 const double* __restrict__ d,
                                 double* __restrict__ sig_d,
                                 double* __restrict__ sig_l,
                                 int64_t B, int64_t T) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const double* lrow = l + row * (T - 1);
    const double* drow = d + row * T;
    double* sdrow = sig_d + row * T;
    double* slrow = sig_l + row * (T - 1);
    sdrow[T - 1] = 1.0 / drow[T - 1];
    for (int64_t t = T - 2; t >= 0; --t) {
        slrow[t] = -lrow[t] * sdrow[t + 1];
        sdrow[t] = 1.0 / drow[t] - lrow[t] * slrow[t];
    }
}

// ---------------------------------------------------------------------------
// Thread-per-cell joint Cholesky sampling: solve L^T u = D^{-1/2} eps
// eps: (B, T, M) -> u: (B, T, M)
// ---------------------------------------------------------------------------
__global__ void cholesky_sample_kernel(const double* __restrict__ l,
                                       const double* __restrict__ d,
                                       const double* __restrict__ eps,
                                       double* __restrict__ u,
                                       int64_t B, int64_t T, int64_t M) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const double* lrow = l + row * (T - 1);
    const double* drow = d + row * T;
    const double* erow = eps + row * T * M;
    double* urow = u + row * T * M;
    for (int64_t m = 0; m < M; ++m) {
        urow[(T - 1) * M + m] = erow[(T - 1) * M + m] / sqrt(drow[T - 1]);
    }
    for (int64_t t = T - 2; t >= 0; --t) {
        double inv_sqrt_d = 1.0 / sqrt(drow[t]);
        for (int64_t m = 0; m < M; ++m) {
            urow[t * M + m] = erow[t * M + m] * inv_sqrt_d
                              - lrow[t] * urow[(t + 1) * M + m];
        }
    }
}

std::vector<torch::Tensor> thomas_solve_cuda(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor g) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && g.is_cuda(), "inputs must be CUDA");
    auto a_c = a.contiguous().to(torch::kDouble);
    auto b_c = b.contiguous().to(torch::kDouble);
    auto g_c = g.contiguous().to(torch::kDouble);
    int64_t B = a_c.size(0);
    int64_t T = a_c.size(1);
    auto l = torch::empty({B, std::max<int64_t>(T - 1, 1)}, a_c.options());
    auto d = torch::empty_like(a_c);
    auto dz = torch::empty_like(g_c);
    if (B > 0 && T >= 2) {
        int threads = 128;
        int blocks = (int)((B + threads - 1) / threads);
        thomas_solve_kernel<<<blocks, threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
            a_c.data_ptr<double>(), b_c.data_ptr<double>(),
            g_c.data_ptr<double>(), l.data_ptr<double>(), d.data_ptr<double>(),
            dz.data_ptr<double>(), B, T);
    }
    return {dz, l, d};
}

std::vector<torch::Tensor> takahashi_cuda(torch::Tensor l, torch::Tensor d) {
    TORCH_CHECK(l.is_cuda() && d.is_cuda(), "inputs must be CUDA");
    auto l_c = l.contiguous().to(torch::kDouble);
    auto d_c = d.contiguous().to(torch::kDouble);
    int64_t B = d_c.size(0);
    int64_t T = d_c.size(1);
    auto sig_d = torch::empty_like(d_c);
    auto sig_l = torch::empty_like(l_c);
    if (B > 0 && T >= 2) {
        int threads = 128;
        int blocks = (int)((B + threads - 1) / threads);
        takahashi_kernel<<<blocks, threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
            l_c.data_ptr<double>(), d_c.data_ptr<double>(),
            sig_d.data_ptr<double>(), sig_l.data_ptr<double>(), B, T);
    }
    return {sig_d, sig_l};
}

torch::Tensor cholesky_sample_cuda(torch::Tensor l, torch::Tensor d,
                                   torch::Tensor eps) {
    TORCH_CHECK(l.is_cuda() && d.is_cuda() && eps.is_cuda(),
                "inputs must be CUDA");
    auto l_c = l.contiguous().to(torch::kDouble);
    auto d_c = d.contiguous().to(torch::kDouble);
    auto eps_c = eps.contiguous().to(torch::kDouble);
    int64_t B = eps_c.size(0);
    int64_t T = eps_c.size(1);
    int64_t M = eps_c.size(2);
    auto u = torch::empty_like(eps_c);
    if (B > 0 && T >= 2 && M > 0) {
        int threads = 64;
        int blocks = (int)((B + threads - 1) / threads);
        cholesky_sample_kernel<<<blocks, threads, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
            l_c.data_ptr<double>(), d_c.data_ptr<double>(),
            eps_c.data_ptr<double>(), u.data_ptr<double>(), B, T, M);
    }
    return u;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("thomas_solve_cuda", &thomas_solve_cuda, "batched Thomas LDL^T solve");
    m.def("takahashi_cuda", &takahashi_cuda, "batched Takahashi marginals");
    m.def("cholesky_sample_cuda", &cholesky_sample_cuda,
          "batched joint Cholesky sampling");
}
"""

_CPP_SOURCE = ""


def _try_compile():
    """Lazily compile the inline extension (once per process)."""
    global _MODULE, _COMPILE_ERROR
    if _MODULE is not None or _COMPILE_ERROR is not None:
        return
    if not torch.cuda.is_available():
        _COMPILE_ERROR = RuntimeError("CUDA not available")
        return
    try:
        from torch.utils.cpp_extension import load_inline
        _MODULE = load_inline(
            name="fire_harmonizer_cuda_kernels",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["thomas_solve_cuda", "takahashi_cuda",
                       "cholesky_sample_cuda"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - CPU fallback per contract
        _COMPILE_ERROR = exc


def cuda_available() -> bool:
    with _LOCK:
        _try_compile()
    return _MODULE is not None


def thomas_solve_batched(a, b, g, use_cuda: bool | None = None):
    """Batched Thomas solve with transparent CUDA/CPU dispatch.

    Returns (dz, l, d); all ``torch.float64``.
    """
    want_cuda = use_cuda if use_cuda is not None else cuda_available()
    if want_cuda:
        with _LOCK:
            _try_compile()
        if _MODULE is not None:
            a64 = a.to("cuda", torch.float64)
            b64 = b.to("cuda", torch.float64)
            g64 = g.to("cuda", torch.float64)
            dz, l, d = _MODULE.thomas_solve_cuda(a64, b64, g64)
            return dz.cpu(), l.cpu(), d.cpu()
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    g64 = g.to(torch.float64)
    return _cpu_thomas(a64, b64, g64)


def takahashi_batched(l, d, use_cuda: bool | None = None):
    """Batched Takahashi marginals with transparent dispatch."""
    want_cuda = use_cuda if use_cuda is not None else cuda_available()
    if want_cuda:
        with _LOCK:
            _try_compile()
        if _MODULE is not None:
            l64 = l.to("cuda", torch.float64)
            d64 = d.to("cuda", torch.float64)
            sig_d, sig_l = _MODULE.takahashi_cuda(l64, d64)
            return sig_d.cpu(), sig_l.cpu()
    return _cpu_takahashi(l.to(torch.float64), d.to(torch.float64))


def cholesky_sample_batched(l, d, eps, use_cuda: bool | None = None):
    """Batched joint Cholesky sampling with transparent dispatch."""
    want_cuda = use_cuda if use_cuda is not None else cuda_available()
    if want_cuda:
        with _LOCK:
            _try_compile()
        if _MODULE is not None:
            l64 = l.to("cuda", torch.float64)
            d64 = d.to("cuda", torch.float64)
            eps64 = eps.to("cuda", torch.float64)
            u = _MODULE.cholesky_sample_cuda(l64, d64, eps64)
            return u.cpu()
    return _cpu_sample(l.to(torch.float64), d.to(torch.float64),
                       eps.to(torch.float64))


def reset_module() -> None:
    """Reset the compiled-module cache (tests)."""
    global _MODULE, _COMPILE_ERROR
    with _LOCK:
        _MODULE = None
        _COMPILE_ERROR = None
