"""Multi-GPU float64 GMRF MAP-Laplace solver (readme.md Sections 4 & 5.3.3-4).

Pipeline per chunk of ``batch_size = 1000`` active cells:
  1. Build dense per-regime ``(B, T=9510, P=5)`` observation + non-detection
     tensors (Day and Night blocks each own their tensors because a polar
     platform can observe both half-day windows on one solar date).
  2. Run ``SOLVER_SWEEPS`` Gauss-Seidel sweeps over the H3 3-coloring x
     Day/Night blocks; each block performs up to ``NEWTON_STEPS`` damped
     Newton iterations with Armijo backtracking on the exact tridiagonal
     system (Thomas LDL^T, pivot floor 1e-7, diagonal floor 1e-6).
  3. Takahashi exact posterior marginals + ``M=200`` exact joint Cholesky
     trajectories (soft-clipped to z in [-8, 14] before exp), aggregated to
     monthly/daily quantiles (q05, q50, q95, mean).
  4. ``run_fixed_lag_update(window_days=14)`` re-solves the trailing window
     anchored at ``t_start - 1`` via the AR(1) prior to the saved state.

All algebra runs in ``torch.float64``; CUDA kernels accelerate transparently
when available, with the identical pure-PyTorch reference on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import settings as S
from src.engine import cuda_kernels
from src.engine import likelihoods as lk

logger = logging.getLogger("fire_harmonizer.engine.solver")

Tensor = torch.Tensor
_DN = ("D", "N")
_DN_INDEX = {"D": 0, "N": 1}


# --------------------------------------------------------------------------- #
# Time grid helpers
# --------------------------------------------------------------------------- #
def month_grid(T: int, start: str = S.DATE_START):
    """Month id (0..n_months-1) and (year, month) keys for every day t."""
    d0 = pd.Timestamp(start)
    month_ids = np.empty(T, dtype=np.int64)
    keys: List[Tuple[int, int]] = []
    cur = d0
    prev_key = None
    mid = -1
    for t in range(T):
        key = (cur.year, cur.month)
        if key != prev_key:
            mid += 1
            keys.append(key)
            prev_key = key
        month_ids[t] = mid
        cur += pd.Timedelta(days=1)
    return month_ids, keys


def build_platform_calendar(
    platform_active_day: Optional[pd.DataFrame], T: int
) -> np.ndarray:
    """(P, T) int8: operational AND empirically-active per platform/day."""
    cal = np.zeros((S.N_PLATFORMS, T), dtype=np.int8)
    d0 = pd.Timestamp(S.DATE_START)
    for name, win in S.PLATFORM_TIMELINE.items():
        p = S.PLATFORM_INDEX[name]
        first = (pd.Timestamp(win.first_date) - d0).days
        last = (pd.Timestamp(win.last_date) - d0).days
        lo, hi = max(first, 0), min(last, T - 1)
        if hi >= lo:
            cal[p, lo : hi + 1] = 1
    if platform_active_day is not None and len(platform_active_day):
        for row in platform_active_day.itertuples():
            p = S.PLATFORM_INDEX.get(row.platform)
            t = (pd.Timestamp(row.acq_date) - d0).days
            if p is not None and 0 <= t < T and not row.platform_active:
                cal[p, t] = 0
    return cal


def estimate_chunk_vram_bytes(
    batch_size: int = S.BATCH_SIZE,
    T: int = S.T_DAYS,
    n_platforms: int = S.N_PLATFORMS,
    mc_samples: int = S.MC_SAMPLES,
    sample_subbatch: int = 16,
) -> int:
    """Analytic peak-memory plan for one solver chunk (bytes).

    Per Day/Night regime: ``y, mu_gain, inv_sig2, k_miss, vis`` float64
    (5 x 8 B) plus ``sat, is_op`` bool (2 x 1 B) over (B, T, P); plus
    (B, T) state/workspace arrays for both regimes; plus one sampling
    sub-batch ``(B_sub, T, M)`` pair. ``inv_sig = sqrt(inv_sig2)`` is
    computed on demand inside block solves instead of being stored.
    """
    per_regime_elems = batch_size * T * n_platforms
    dense = 2 * (5 * 8 + 2 * 1) * per_regime_elems
    states = 16 * batch_size * T * 8
    sampling = 2 * sample_subbatch * T * mc_samples * 8
    return int(dense + states + sampling)


# --------------------------------------------------------------------------- #
# Hurdle rate tables
# --------------------------------------------------------------------------- #
def build_psi_cnt_table(
    lognorm_params: np.ndarray, avg_areas: np.ndarray
) -> np.ndarray:
    """(K, P, 2) psi_CNT from per-platform average unseen-pass areas."""
    from src.calibration.operators import psi_cnt as psi_cnt_fn

    table = np.zeros((S.PYROME_K, S.N_PLATFORMS, 2))
    for p, name in enumerate(S.PLATFORMS):
        family = S.PLATFORM_TIMELINE[name].sensor_family
        for dn_i, dn in enumerate(_DN):
            area = avg_areas[p, dn_i]
            alpha, beta = S.FMIN_PARAMS[(family, dn)]
            fmin = alpha + beta * area
            for b in range(S.PYROME_K):
                mu_f, sig_f = lognorm_params[b]
                table[b, p, dn_i] = psi_cnt_fn(fmin, mu_f, sig_f)
    return table


def build_k_table(
    lambda0: np.ndarray, f_event: np.ndarray, psi_cnt_table: np.ndarray
) -> np.ndarray:
    """(K, P, 2) Hurdle rate k = lambda0 * psi_CNT / F_event."""
    return lambda0[:, None, None] * psi_cnt_table / f_event[:, None, None]


DEFAULT_AVG_AREAS = np.array(
    [
        [S.MODIS_AVG_AREA_KM2, S.MODIS_AVG_AREA_KM2],  # Terra
        [S.MODIS_AVG_AREA_KM2, S.MODIS_AVG_AREA_KM2],  # Aqua
        [S.VIIRS_AVG_AREA_KM2, S.VIIRS_AVG_AREA_KM2],  # SNPP
        [S.VIIRS_AVG_AREA_KM2, S.VIIRS_AVG_AREA_KM2],  # NOAA-20
        [S.VIIRS_AVG_AREA_KM2, S.VIIRS_AVG_AREA_KM2],  # NOAA-21
    ]
)


# --------------------------------------------------------------------------- #
# Dense per-regime observation tensors
# --------------------------------------------------------------------------- #
def build_observation_tensors(
    pass_df: pd.DataFrame,
    cell_pos_map: Dict[int, int],
    B: int,
    T: int,
    platform_calendar: np.ndarray,
    k_table: np.ndarray,
    pyrome: np.ndarray,
    vis_matrix: Optional[Dict[str, np.ndarray]] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, Tensor]]:
    """Materialize the per-regime dense (B, T, P) observation tensors.

    * Detected pass rows scatter Case A/B parameters (y, sat, mu_gain,
      1/sigma^2).
    * Every other operational healthy pass carries Case C parameters
      (k_miss, vis).
    * Non-operational passes (retired platforms, safe-mode outages) carry
      ``is_op = 0`` and contribute exactly zero likelihood mass (INV-04).
    """
    P = S.N_PLATFORMS
    tensors: Dict[str, Dict[str, Tensor]] = {}
    op_np = platform_calendar.astype(np.float64)  # (P, T)
    for dn in _DN:
        dn_i = _DN_INDEX[dn]
        tensors[dn] = {
            "y": torch.zeros(B, T, P, dtype=torch.float64, device=device),
            "sat": torch.zeros(B, T, P, dtype=torch.uint8, device=device),
            "mu_gain": torch.zeros(B, T, P, dtype=torch.float64,
                                   device=device),
            "inv_sig2": torch.ones(B, T, P, dtype=torch.float64,
                                   device=device),
            "k_miss": torch.zeros(B, T, P, dtype=torch.float64,
                                  device=device),
            "vis": torch.zeros(B, T, P, dtype=torch.float64, device=device),
            "is_op": torch.zeros(B, T, P, dtype=torch.uint8, device=device),
        }
        k_bp = k_table[pyrome][:, :, dn_i]  # (B, P)
        tensors[dn]["k_miss"].copy_(
            torch.from_numpy(
                np.broadcast_to(k_bp[:, None, :], (B, T, P)).copy()
            ).to(device)
        )
        if vis_matrix is not None and dn in vis_matrix:
            tensors[dn]["vis"].copy_(
                torch.from_numpy(vis_matrix[dn]).to(device)
            )
        tensors[dn]["is_op"].copy_(
            torch.from_numpy(
                np.broadcast_to(op_np.T[None, :, :], (B, T, P)).copy()
            ).to(torch.uint8)
        )

    if len(pass_df):
        for row in pass_df.itertuples():
            b = cell_pos_map.get(int(row.cell_pos))
            if b is None:
                continue
            t = int(row.t_index)
            p = S.PLATFORM_INDEX.get(row.platform)
            if p is None or not (0 <= t < T):
                continue
            reg = tensors[getattr(row, "dn_flag", "D")]
            reg["y"][b, t, p] = float(row.y_frp_mw)
            reg["sat"][b, t, p] = int(row.sat_flag)
            reg["mu_gain"][b, t, p] = float(row.mu_gain)
            sigma = max(float(row.obs_sigma), 1e-6)
            reg["inv_sig2"][b, t, p] = 1.0 / sigma**2
            reg["is_op"][b, t, p] = 1
    return tensors


# --------------------------------------------------------------------------- #
# Chunk problem container
# --------------------------------------------------------------------------- #
@dataclass
class ChunkProblem:
    """Everything one chunk solve needs."""

    cell_ids: np.ndarray                 # (B,) global cell positions
    colors: np.ndarray                   # (B,) in {0,1,2}
    rho: np.ndarray                      # (B,)
    omega: np.ndarray                    # (B,) = 1 / sigma_proc^2
    eta_dn: np.ndarray                   # (B,)
    beta_dn: np.ndarray                  # (B,)
    mu_day: np.ndarray                   # (B, T)
    mu_night: np.ndarray                 # (B, T)
    month_ids: np.ndarray                # (T,)
    obs: Dict[str, Dict[str, Tensor]]    # per-dn dense tensors
    neighbor_indices: np.ndarray         # flattened neighbor positions
    neighbor_indptr: np.ndarray          # (B+1,)

    def to(self, device: torch.device) -> "ChunkProblem":
        def mv(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        out = dict(self.__dict__)
        out["obs"] = {
            dn: {k: mv(v) for k, v in reg.items()}
            for dn, reg in self.obs.items()
        }
        return ChunkProblem(**out)


# --------------------------------------------------------------------------- #
# Prior building blocks (AR(1) tridiagonal)
# --------------------------------------------------------------------------- #
def prior_diag(rho: Tensor, omega: Tensor, T: int) -> Tensor:
    """(B, T) AR(1) prior main diagonal: omega, (1+rho^2) omega, omega."""
    B = rho.shape[0]
    a = (1.0 + rho * rho)[:, None] * omega[:, None].expand(B, T)
    a = a.clone()
    a[:, 0] = omega
    a[:, T - 1] = omega
    return a


def prior_grad(delta: Tensor, rho: Tensor, omega: Tensor) -> Tensor:
    """omega * A_AR1 delta (boundary-correct)."""
    B, T = delta.shape
    gl = torch.zeros_like(delta)
    gl[:, 0] = delta[:, 0] - rho * delta[:, 1]
    gl[:, T - 1] = delta[:, T - 1] - rho * delta[:, T - 2]
    if T > 2:
        gl[:, 1:-1] = (
            (1.0 + rho * rho)[:, None] * delta[:, 1:-1]
            - rho[:, None] * delta[:, :-2]
            - rho[:, None] * delta[:, 2:]
        )
    return omega[:, None] * gl


def prior_quad(delta: Tensor, rho: Tensor, omega: Tensor,
               anchor_delta: Optional[Tensor] = None) -> Tensor:
    """omega/2 * [(1-rho^2) d0^2 + sum (dt - rho d(t-1))^2].

    With a fixed-lag anchor the leading term becomes
    ``(d0 - rho * anchor)^2`` (constants dropped; Armijo differences only).
    """
    diff = delta[:, 1:] - rho[:, None] * delta[:, :-1]
    if anchor_delta is None:
        lead = (1.0 - rho * rho) * delta[:, 0] * delta[:, 0]
    else:
        lead = delta[:, 0] * delta[:, 0] \
            - 2.0 * rho * anchor_delta * delta[:, 0]
    q = lead + (diff * diff).sum(dim=1)
    return 0.5 * omega * q


# --------------------------------------------------------------------------- #
# Block Newton solve with Armijo backtracking
# --------------------------------------------------------------------------- #
def solve_block_newton(
    z: Tensor,
    mu: Tensor,
    rho: Tensor,
    omega: Tensor,
    eta_dn: Tensor,
    beta_dn: Tensor,
    obs: Dict[str, Tensor],
    neighbor_sum: Tensor,
    neighbor_deg: Tensor,
    delta_opp: Tensor,
    anchor_delta: Optional[Tensor] = None,
    newton_steps: int = S.NEWTON_STEPS,
) -> Tuple[Tensor, float]:
    """Damped Newton solve of one (color, dn) tridiagonal block.

    Returns ``(z_new, converged_flag)``; the spatial and Day-Night coupling
    terms hold the out-of-block values fixed (Gauss-Seidel). The spatial
    objective drops neighbor-constant terms (they cancel in Armijo
    differences); the linear and quadratic z_i parts are exact.
    """
    B, T = z.shape
    P = S.N_PLATFORMS
    alpha_s = S.ALPHA_SPATIAL

    a_spatial = alpha_s * omega * neighbor_deg             # (B,)
    a_dn = eta_dn * omega                                  # (B,)
    y = obs["y"].to(torch.float64)
    sat = obs["sat"].to(torch.float64)
    is_op = obs["is_op"].to(torch.float64)
    mu_gain = obs["mu_gain"]
    inv_sig2 = obs["inv_sig2"]
    inv_sig = torch.sqrt(inv_sig2)
    k_miss = obs["k_miss"]
    vis = obs["vis"]

    def objective(zz: Tensor) -> Tensor:
        d = zz - mu
        nll, _, _ = lk.eval_pointwise_nll_grad_hess(
            zz.unsqueeze(-1).expand(B, T, P), y, sat, mu_gain, inv_sig2,
            inv_sig, k_miss, vis, is_op,
        )
        j_obs = nll.sum(dim=2).sum(dim=1)
        quad = prior_quad(d, rho, omega, anchor_delta)
        # Spatial: alpha/2 * (deg * z^2 - 2 z * nb_sum) per day (constants
        # independent of z cancel in Armijo comparisons).
        sp = 0.5 * alpha_s * omega * (
            (neighbor_deg[:, None] * zz * zz - 2.0 * zz * neighbor_sum)
            .sum(dim=1)
        )
        dd = (
            eta_dn * omega * 0.5
            * ((d - beta_dn[:, None] * delta_opp) ** 2).sum(dim=1)
        )
        return (j_obs + quad + sp + dd).sum()

    j_old = objective(z)
    for _ in range(newton_steps):
        nll, obs_g, obs_h = lk.eval_pointwise_nll_grad_hess(
            z.unsqueeze(-1).expand(B, T, P), y, sat, mu_gain, inv_sig2,
            inv_sig, k_miss, vis, is_op,
        )
        obs_g_sum = obs_g.sum(dim=2)
        obs_h_sum = obs_h.sum(dim=2)

        a = prior_diag(rho, omega, T) + obs_h_sum
        if anchor_delta is not None:
            a[:, 0] = (1.0 + rho * rho) * omega
        a = a + a_spatial[:, None] + a_dn[:, None]
        a = torch.clamp(a, min=S.DIAG_FLOOR)
        b = (-rho[:, None] * omega[:, None]).expand(B, T - 1).contiguous()

        delta = z - mu
        g = prior_grad(delta, rho, omega) + obs_g_sum
        if anchor_delta is not None:
            g[:, 0] = g[:, 0] + omega * rho * rho * delta[:, 0] \
                - omega * rho * anchor_delta
        g = g + alpha_s * omega[:, None] * (
            neighbor_deg[:, None] * z - neighbor_sum
        )
        g = g + (eta_dn * omega)[:, None] * (
            delta - beta_dn[:, None] * delta_opp
        )

        d, l, x = cuda_kernels.thomas_solve(a, b, g)
        directional = float((g * x).sum())
        if directional <= 0.0:
            break
        accepted = False
        alpha_used = 0.0
        for alpha in S.ARMIJO_STEPS:
            z_new = torch.clamp(z - alpha * x, S.Z_CLIP_LO, S.Z_CLIP_HI)
            j_new = objective(z_new)
            if float(j_new) <= j_old - S.ARMIJO_C1 * alpha * directional:
                z = z_new
                j_old = j_new
                accepted = True
                alpha_used = alpha
                break
        if not accepted:
            break
        if float(torch.max(torch.abs(alpha_used * x))) < S.NEWTON_TOL:
            break
    return z, 1.0


# --------------------------------------------------------------------------- #
# Chunk solve: Gauss-Seidel, marginals, joint sampling
# --------------------------------------------------------------------------- #
def solve_chunk(
    problem: ChunkProblem,
    device: torch.device = torch.device("cpu"),
    sweeps: int = S.SOLVER_SWEEPS,
    newton_steps: int = S.NEWTON_STEPS,
    mc_samples: int = S.MC_SAMPLES,
    sample_subbatch: int = 16,
    sample: bool = True,
    seed: int = S.GMM_RANDOM_STATE,
    t_start: int = 0,
    anchor_state: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """Solve one chunk of cells; returns states, variances and aggregates.

    ``t_start > 0`` restricts the solve to ``[t_start, T-1]`` with the AR(1)
    prior anchored to the saved state at ``t_start - 1`` (fixed-lag path).
    """
    pr = problem.to(device)
    B = len(pr.cell_ids)
    T_full = pr.obs["D"]["y"].shape[1]
    T_win = T_full - t_start

    mu_all = {"D": pr.mu_day, "N": pr.mu_night}
    rho = torch.from_numpy(pr.rho.astype(np.float64)).to(device)
    omega = torch.from_numpy(pr.omega.astype(np.float64)).to(device)
    eta = torch.from_numpy(pr.eta_dn.astype(np.float64)).to(device)
    beta = torch.from_numpy(pr.beta_dn.astype(np.float64)).to(device)

    z_all = {
        "D": torch.zeros(B, T_full, dtype=torch.float64, device=device),
        "N": torch.zeros(B, T_full, dtype=torch.float64, device=device),
    }
    anchor_delta: Dict[str, Optional[Tensor]] = {"D": None, "N": None}
    if anchor_state is not None and t_start > 0:
        for dn in _DN:
            key = "z_day" if dn == "D" else "z_night"
            z_all[dn][:, :t_start] = torch.from_numpy(
                anchor_state[key][:, :t_start]
            ).to(device)
            anchor_delta[dn] = (
                torch.from_numpy(
                    anchor_state[key][:, t_start - 1].astype(np.float64)
                ).to(device)
                - mu_all[dn][:, t_start - 1]
            )
    else:
        # Warm start at the prior (climatological) mean.
        for dn in _DN:
            z_all[dn].copy_(
                torch.from_numpy(mu_all[dn].astype(np.float64)).to(device)
            )

    nb_idx = torch.from_numpy(pr.neighbor_indices.astype(np.int64)).to(device)
    nb_ptr = torch.from_numpy(pr.neighbor_indptr.astype(np.int64)).to(device)
    deg = (nb_ptr[1:] - nb_ptr[:-1]).to(torch.float64)

    def neighbor_sum_for(z_dn: Tensor) -> Tensor:
        if int(deg.sum()) == 0:
            return torch.zeros_like(z_dn)
        gathered = z_dn[nb_idx]
        out = torch.zeros_like(z_dn)
        out.index_add_(
            0,
            torch.repeat_interleave(
                torch.arange(B, device=device), (nb_ptr[1:] - nb_ptr[:-1])
            ),
            gathered,
        )
        return out

    # --- Gauss-Seidel sweeps over 3-coloring x Day/Night --------------------
    for _ in range(sweeps):
        for color in (0, 1, 2):
            cells = np.flatnonzero(pr.colors == color)
            if len(cells) == 0:
                continue
            cells_t = torch.from_numpy(cells).to(device)
            for dn in _DN:
                opp = "N" if dn == "D" else "D"
                z = z_all[dn][cells_t, t_start:].clone()
                mu = mu_all[dn][cells_t, t_start:]
                nb_sum = neighbor_sum_for(z_all[dn].clone())[
                    cells_t, t_start:
                ]
                nb_deg = deg[cells_t]
                delta_opp = (
                    z_all[opp][cells_t, t_start:] - mu_all[opp][cells_t, t_start:]
                )
                obs_block = {
                    k: v[cells_t, t_start:]
                    for k, v in pr.obs[dn].items()
                }
                z_new, _ = solve_block_newton(
                    z, mu, rho[cells_t], omega[cells_t], eta[cells_t],
                    beta[cells_t], obs_block, nb_sum, nb_deg, delta_opp,
                    anchor_delta=anchor_delta[dn],
                    newton_steps=newton_steps,
                )
                z_all[dn][cells_t, t_start:] = z_new

    # --- Takahashi exact posterior marginals --------------------------------
    var = {
        "D": torch.zeros(B, T_full, dtype=torch.float64, device=device),
        "N": torch.zeros(B, T_full, dtype=torch.float64, device=device),
    }
    for dn in _DN:
        opp = "N" if dn == "D" else "D"
        for color in (0, 1, 2):
            cells = np.flatnonzero(pr.colors == color)
            if len(cells) == 0:
                continue
            cells_t = torch.from_numpy(cells).to(device)
            a = _block_hessian(
                z_all[dn][cells_t], mu_all[dn][cells_t], rho[cells_t],
                omega[cells_t], eta[cells_t], pr.obs[dn], cells_t, deg,
                device,
            )
            b = (-rho[cells_t][:, None] * omega[cells_t][:, None]).expand(
                len(cells), T_full - 1
            ).contiguous()
            d, l, _ = cuda_kernels.thomas_solve(a, b, torch.zeros_like(a))
            v, _ = cuda_kernels.takahashi(d, l)
            var[dn][cells_t] = v

    results: Dict[str, object] = {
        "cell_ids": pr.cell_ids,
        "z_day": z_all["D"].detach().cpu().numpy(),
        "z_night": z_all["N"].detach().cpu().numpy(),
        "var_day": var["D"].detach().cpu().numpy(),
        "var_night": var["N"].detach().cpu().numpy(),
        "t_start": t_start,
    }

    if sample:
        g = torch.Generator(device="cpu").manual_seed(seed)
        month_ids = pr.month_ids
        n_months = int(month_ids.max()) + 1 if T_full else 0
        monthly_q = {q: np.full((B, n_months), np.nan)
                     for q in ("q05", "q50", "q95", "mean")}
        daily_q = {q: np.full((B, T_full), np.nan)
                   for q in ("q05", "q50", "q95")}
        for lo in range(0, B, sample_subbatch):
            hi = min(lo + sample_subbatch, B)
            u = {}
            for dn in _DN:
                eps = torch.randn(
                    hi - lo, T_full, mc_samples, dtype=torch.float64,
                    generator=g,
                ).to(device)
                a = _block_hessian(
                    z_all[dn][lo:hi], mu_all[dn][lo:hi], rho[lo:hi],
                    omega[lo:hi], eta[lo:hi], pr.obs[dn],
                    torch.arange(lo, hi, device=device), deg, device,
                )
                b = (-rho[lo:hi][:, None] * omega[lo:hi][:, None]).expand(
                    hi - lo, T_full - 1
                ).contiguous()
                d, l, _ = cuda_kernels.thomas_solve(
                    a, b, torch.zeros_like(a)
                )
                u[dn] = cuda_kernels.cholesky_sample(d, l, eps)
            z_d = z_all["D"][lo:hi]
            z_n = z_all["N"][lo:hi]
            phi_d = torch.exp(
                torch.clamp(
                    z_d.unsqueeze(-1) + u["D"], S.Z_CLIP_LO, S.Z_CLIP_HI
                )
            )
            phi_n = torch.exp(
                torch.clamp(
                    z_n.unsqueeze(-1) + u["N"], S.Z_CLIP_LO, S.Z_CLIP_HI
                )
            )
            phi24 = 0.5 * (phi_d + phi_n)  # (Bs, T, M)
            q05, q50, q95 = torch.quantile(
                phi24,
                torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64),
                dim=2,
            )
            daily_q["q05"][lo:hi] = q05.cpu().numpy()
            daily_q["q50"][lo:hi] = q50.cpu().numpy()
            daily_q["q95"][lo:hi] = q95.cpu().numpy()
            mid = torch.from_numpy(month_ids).to(device)
            sums = torch.zeros(
                hi - lo, n_months, mc_samples,
                dtype=torch.float64, device=device,
            )
            sums.index_add_(1, mid, phi24)
            counts = torch.bincount(mid, minlength=n_months).to(
                device
            ).to(torch.float64)
            monthly = sums / counts.clamp(min=1.0)[None, :, None]
            mq05, mq50, mq95 = torch.quantile(
                monthly,
                torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64),
                dim=2,
            )
            monthly_q["q05"][lo:hi] = mq05.cpu().numpy()
            monthly_q["q50"][lo:hi] = mq50.cpu().numpy()
            monthly_q["q95"][lo:hi] = mq95.cpu().numpy()
            monthly_q["mean"][lo:hi] = monthly.mean(dim=2).cpu().numpy()
        results.update(
            {
                "monthly_q05": monthly_q["q05"],
                "monthly_q50": monthly_q["q50"],
                "monthly_q95": monthly_q["q95"],
                "monthly_mean": monthly_q["mean"],
                "daily_q05": daily_q["q05"],
                "daily_q50": daily_q["q50"],
                "daily_q95": daily_q["q95"],
            }
        )
    return results


def _block_hessian(
    z: Tensor, mu: Tensor, rho: Tensor, omega: Tensor, eta: Tensor,
    obs: Dict[str, Tensor], cells: Tensor, deg: Tensor, device,
) -> Tensor:
    """Block tridiagonal Hessian at z (for Takahashi / sampling)."""
    Bc, T = z.shape
    P = S.N_PLATFORMS
    y = obs["y"][cells].to(torch.float64)
    sat = obs["sat"][cells].to(torch.float64)
    mu_gain = obs["mu_gain"][cells]
    inv_sig2 = obs["inv_sig2"][cells]
    inv_sig = torch.sqrt(inv_sig2)
    k_miss = obs["k_miss"][cells]
    vis = obs["vis"][cells]
    is_op = obs["is_op"][cells].to(torch.float64)
    _, _, obs_h = lk.eval_pointwise_nll_grad_hess(
        z.unsqueeze(-1).expand(Bc, T, P), y, sat, mu_gain, inv_sig2,
        inv_sig, k_miss, vis, is_op,
    )
    a = prior_diag(rho, omega, T) + obs_h.sum(dim=2)
    a = a + (S.ALPHA_SPATIAL * omega * deg[cells])[:, None]
    a = a + (eta * omega)[:, None]
    return torch.clamp(a, min=S.DIAG_FLOOR)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def solve_all_chunks(
    build_chunk: Callable[[List[int], torch.device], ChunkProblem],
    n_cells: int,
    batch_size: int = S.BATCH_SIZE,
    device_count: Optional[int] = None,
    **chunk_kwargs,
) -> Dict[str, object]:
    """Distribute active cells across devices in ``batch_size`` chunks.

    ``build_chunk(cell_positions, device)`` materializes one ChunkProblem.
    With >1 CUDA devices, chunks round-robin across GPUs via
    ``torch.multiprocessing`` (spawn); single-device and CPU environments
    solve sequentially with the identical math.
    """
    if device_count is None:
        device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
    all_cells = np.arange(n_cells)
    chunks = [
        all_cells[i : i + batch_size] for i in range(0, n_cells, batch_size)
    ]
    results: List[Optional[Dict[str, object]]] = [None] * len(chunks)

    if device_count > 1:
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")

        def worker(rank: int, out_q):
            torch.cuda.set_device(rank)
            dev = torch.device(f"cuda:{rank}")
            for ci in range(rank, len(chunks), device_count):
                prob = build_chunk(list(map(int, chunks[ci])), dev)
                out_q.put((ci, solve_chunk(prob, device=dev, **chunk_kwargs)))

        out_q = ctx.Queue()
        procs = [
            ctx.Process(target=worker, args=(r, out_q))
            for r in range(device_count)
        ]
        for p in procs:
            p.start()
        for _ in range(len(chunks)):
            ci, res = out_q.get()
            results[ci] = res
        for p in procs:
            p.join()
    else:
        dev = (
            torch.device("cuda:0") if device_count == 1
            else torch.device("cpu")
        )
        for ci, cells in enumerate(chunks):
            prob = build_chunk(list(map(int, cells)), dev)
            results[ci] = solve_chunk(prob, device=dev, **chunk_kwargs)

    return _merge_chunk_results([r for r in results if r is not None])


def _merge_chunk_results(
    results: List[Dict[str, object]]
) -> Dict[str, object]:
    merged: Dict[str, object] = {"chunks": results}
    for key in (
        "z_day", "z_night", "var_day", "var_night", "monthly_q05",
        "monthly_q50", "monthly_q95", "monthly_mean", "daily_q05",
        "daily_q50", "daily_q95",
    ):
        if all(key in r for r in results):
            merged[key] = np.concatenate(
                [np.asarray(r[key]) for r in results], axis=0
            )
    merged["cell_ids"] = np.concatenate(
        [np.asarray(r["cell_ids"]) for r in results]
    )
    return merged


def run_fixed_lag_update(
    build_chunk: Callable[[List[int], torch.device], ChunkProblem],
    saved_state: Dict[str, np.ndarray],
    n_cells: int,
    window_days: int = 14,
    T: int = S.T_DAYS,
    device: Optional[torch.device] = None,
    **chunk_kwargs,
) -> Dict[str, object]:
    """Fixed-lag rolling assimilation over the trailing ``window_days``.

    Re-solves ``[t_start, T-1]`` where ``t_start = T - window_days``,
    anchoring ``t_start - 1`` via the AR(1) prior to the saved MAP state so
    the trailing window matches the full-batch solution on overlapping days
    (verified in test_08). obs_time / ingest_ts / revision_count
    bookkeeping is preserved by the products layer.
    """
    dev = device or (
        torch.device("cuda:0") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    t_start = max(T - window_days, 0)
    anchor = {
        "z_day": np.asarray(saved_state["z_day"], dtype=np.float64),
        "z_night": np.asarray(saved_state["z_night"], dtype=np.float64),
    }
    prob = build_chunk(list(range(n_cells)), dev)
    return solve_chunk(
        prob, device=dev, t_start=t_start, anchor_state=anchor,
        sample=False, **chunk_kwargs,
    )
