"""Multi-GPU float64 GMRF MAP-Laplace solver (readme Sections 4.8, 5.3.3-5.3.4).

* Partitions active cells into chunks of ``batch_size = 1000`` (``< 6 GB``
  VRAM per GPU) and distributes them over ``0..device_count()-1`` with
  ``torch.multiprocessing`` (one worker process per GPU, optionally several
  per GPU); runs in-process on a single GPU or on CPU.
* Builds dense ``(B, T, P=5, 2)`` observation tensors per chunk with fully
  vectorized NumPy (``y = 0`` encodes Hurdle non-detections; the ``op``
  ``(T, P)`` mask carries the INV-04 calendar = static timeline AND
  empirical ``platform_active_day``).
* Runs ``SOLVER_SWEEPS = 6`` outer sweeps over the H3 **3-coloring**
  (``color in {0,1,2}``) x alternating ``('D','N')``; per block up to
  ``NEWTON_STEPS = 15`` damped Newton-Raphson iterations with Armijo
  backtracking (``alpha in {1, .5, .25, .125, .0625}``,
  ``J(z_new) <= J(z) - 1e-4 alpha sum(g dz)``, early exit
  ``||alpha dz||_inf < 1e-4``), solved via the fused ``float64`` Thomas
  ``LDL^T`` (single-launch CUDA kernel on GPU, NumPy recursion on CPU).
  Every Newton quantity is BLOCK-LOCAL (only the block's rows are
  evaluated) and the Armijo test uses the exact block-local objective delta.
* Posterior: exact Takahashi marginal variances and ``M = 200`` joint
  trajectories ``L^T u = D^{-1/2} eps`` per half-day chain; per trajectory
  the fused 24h flux ``Phi^(m) = 0.5 (Phi_Day^(m) + Phi_Night^(m))`` is formed
  FIRST and the quantiles ``q05/q50/q95/mean`` are taken over the fused
  trajectories (readme 5.3.4.5), monthly (``Phi`` monthly mean, ``FRE``
  monthly sum ``0.0864 sum_t Phi``) and daily.  Daily rows are emitted
  SPARSELY inside the chunk (``phi_q50 > 0.25`` OR within +-3 days of a
  detection in that cell OR inside a spotlight ROI) -- never dense
  ``N x 9510``.
* ``run_fixed_lag_update(window_days=14)`` re-solves ``[t_start, T-1]`` with
  ``z[t_start-1]`` held fixed (AR(1) anchor): the windowed system is the
  exact conditional of the full system (interior diagonal at ``t_start``,
  coupling to the fixed anchor carried in the gradient).

Objective (exact; gradients/Hessians below are its exact derivatives):

``J = sum_obs ell(z) + sum_{g,s} 0.5 w_g [(1-rho^2) d_0^2 + sum_t (d_t - rho d_{t-1})^2]
      + sum_s sum_{(i,j) in E} 0.5 w_ij (z_i - z_j)^2
      + sum_g 0.5 eta w_g [(d_D - beta d_N)^2 + (d_N - beta d_D)^2]``

with ``w_g = 1/sigma_proc,b^2`` and symmetric edge weight
``w_ij = alpha_s (w_i + w_j)/2`` (equal to the spec's ``alpha_s w_p`` within a
pyrome, so the spatial gradient is ``alpha_s w (|N| z - sum z')`` and the
diagonal increment ``alpha_s w |N|``).  Only intra-chunk edges enter, and
the spatial diagonal counts exactly those edges (no phantom out-of-chunk
neighbours).
"""
from __future__ import annotations

import math
import logging
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import torch

from ..config import settings as S
from . import cuda_kernels, likelihoods
from . import thomas_takahashi as tt
from .priors import DRIFT_START_T, cloud_fraction_matrix

logger = logging.getLogger("fire_harmonizer.engine.solver")

DN_STATES = ("D", "N")
T0 = pd.Timestamp(S.DATE_START)


# ---------------------------------------------------------------------------
# Platform masks (picklable; used by conformal folds)
# ---------------------------------------------------------------------------
class PlatformMask:
    """Mask platforms ``platform_indices`` on days ``t_lo..t_hi`` (inclusive).

    Picklable (multi-GPU workers) and callable as ``mask(p_index, t) ->
    bool`` (True = keep).
    """

    def __init__(self, platform_indices, t_lo: int, t_hi: int):
        self.platform_indices = tuple(sorted(int(p) for p in platform_indices))
        self.t_lo = int(t_lo)
        self.t_hi = int(t_hi)

    def __call__(self, pi: int, t: int) -> bool:
        return not (int(pi) in self.platform_indices
                    and self.t_lo <= int(t) <= self.t_hi)

    def apply(self, op: np.ndarray) -> np.ndarray:
        op = op.copy()
        lo = max(self.t_lo, 0)
        hi = min(self.t_hi, op.shape[0] - 1)
        if hi >= lo:
            for pi in self.platform_indices:
                op[lo:hi + 1, pi] = 0.0
        return op


def _apply_platform_mask(op: np.ndarray, platform_mask) -> np.ndarray:
    if platform_mask is None:
        return op
    if hasattr(platform_mask, "apply"):
        return platform_mask.apply(op)
    out = op.copy()
    t_len, n_p = op.shape
    for pi in range(n_p):
        keep = np.fromiter((bool(platform_mask(pi, t)) for t in range(t_len)),
                           dtype=bool, count=t_len)
        out[~keep, pi] = 0.0
    return out


# ---------------------------------------------------------------------------
# Observation tensor construction
# ---------------------------------------------------------------------------
class SolverData:
    """Dense per-chunk solver tensors (torch.float64)."""

    def __init__(self, cells, colors, y, sat, mu, tau2, k_miss, vis, op,
                 mu_prior, omega, rho, eta_dn, beta_dn, n_nbrs, edge_i, edge_j,
                 t_days, pyrome_of_cell, in_spotlight=None, cloud=None,
                 cloud_source=None):
        self.cells = cells
        self.colors = colors
        self.y = y                    # (B, T, P, 2)
        self.sat = sat                # (B, T, P, 2) uint8
        self.mu = mu                  # (B, T, P, 2)
        self.tau2 = tau2              # (B, T, P, 2)
        self.k_miss = k_miss          # (B, P, 2)
        self.vis = vis                # (B, T, P)
        self.op = op                  # (T, P)
        self.mu_prior = mu_prior      # (B, T, 2)
        self.omega = omega            # (B,)
        self.rho = rho                # (B,)
        self.eta_dn = eta_dn          # (B,)
        self.beta_dn = beta_dn        # (B,)
        self.n_nbrs = n_nbrs          # (B,) intra-chunk neighbour count
        self.edge_i = edge_i          # (E,) chunk-local indices
        self.edge_j = edge_j
        self.t_days = t_days
        self.pyrome_of_cell = pyrome_of_cell
        self.in_spotlight = in_spotlight      # (B,) bool numpy (optional)
        self.cloud = cloud                    # (B, T) numpy (optional)
        self.cloud_source = cloud_source      # (B, T) int8 numpy (optional)

    HOST_ONLY = ("y", "sat", "mu", "tau2")

    def to(self, device, skip=()):
        for name in ("y", "sat", "mu", "tau2", "k_miss", "vis", "op",
                     "mu_prior", "omega", "rho", "eta_dn", "beta_dn",
                     "n_nbrs", "edge_i", "edge_j", "colors"):
            if name in skip:
                continue
            setattr(self, name, getattr(self, name).to(device))
        return self


def operational_calendar(t_days: int, platform_active=None,
                         platform_mask=None) -> np.ndarray:
    """INV-04 ``op[t, p]``: static timeline AND empirical active days AND
    the optional fold mask.  Fully vectorized."""
    P = S.N_PLATFORMS
    op = np.zeros((t_days, P), dtype=np.float64)
    for p, (d0, d1) in S.PLATFORM_TIMELINE.items():
        pi = S.PLATFORM_INDEX[p]
        lo = max(0, (pd.Timestamp(d0) - T0).days)
        hi = min(t_days - 1, (pd.Timestamp(d1) - T0).days)
        if hi >= lo:
            op[lo:hi + 1, pi] = 1.0
    if platform_active is not None:
        for p, days in platform_active.items():
            pi = S.PLATFORM_INDEX.get(p)
            if pi is None:
                continue
            days = list(days)
            active = np.zeros(t_days, dtype=bool)
            if len(days):
                t_act = (pd.DatetimeIndex(pd.to_datetime(
                    [str(d)[:10] for d in days])) - T0).days.to_numpy()
                t_act = t_act[(t_act >= 0) & (t_act < t_days)]
                active[t_act] = True
            op[~active, pi] = 0.0
    return _apply_platform_mask(op, platform_mask)


def _cell_centers(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from ..geometry.h3_utils import h3_cell_center
    if len(cells) == 0:
        return np.zeros(0), np.zeros(0)
    ll = np.array([h3_cell_center(int(c)) for c in cells], dtype=np.float64)
    return ll[:, 0], ll[:, 1]


def spotlight_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Cells whose centre lies inside any SPOTLIGHT_PRESETS bbox."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    out = np.zeros(len(lats), dtype=bool)
    for sp in S.SPOTLIGHT_PRESETS:
        out |= ((lons >= sp.bbox[0]) & (lons <= sp.bbox[2])
                & (lats >= sp.bbox[1]) & (lats <= sp.bbox[3]))
    return out


def build_observation_tensors(pass_table, active_cells, hex_pyromes, priors,
                              artifacts, cloud_df, t_days: int = S.T_DAYS,
                              platform_active=None,
                              platform_mask=None) -> SolverData:
    """Dense (B, T, P, 2) observation + non-detection tensors for one chunk.

    ``platform_mask`` (``PlatformMask`` or callable ``(p_index, t) -> keep``)
    optionally masks platforms (VIIRS-masked ablation, leave-one-VIIRS-
    platform-out conformal folds).
    """
    from ..calibration import operators as ops

    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells))),
                     dtype=np.int64)
    B = len(cells)
    P = S.N_PLATFORMS
    platforms = list(S.PLATFORM_INDEX)
    hp = hex_pyromes[["h3_index", "pyrome_id"]].drop_duplicates("h3_index")
    pyr_s = pd.Series(hp["pyrome_id"].astype(int).to_numpy(),
                      index=hp["h3_index"].astype(np.int64).to_numpy())
    pyrome_arr = pyr_s.reindex(cells).fillna(0).astype(int).to_numpy()

    lognorm = artifacts["lognorm_params"]
    f_event = priors["f_event"]
    lambda0 = priors["lambda0"]

    # ---- operational calendar (INV-04) -------------------------------------
    op = operational_calendar(t_days, platform_active, platform_mask)

    # ---- cloud visibility (vectorized 8d -> month -> climatology) ----------
    lats, lons = _cell_centers(cells)
    dates = T0 + pd.to_timedelta(np.arange(t_days), unit="D")
    cloud, cloud_src = cloud_fraction_matrix(
        lats, lons, dates, cloud_df, artifacts.get("cloud_climatology"))
    clear = np.clip(1.0 - cloud, *S.VIS_CLIP)                       # (B, T)
    post = (np.arange(t_days) >= DRIFT_START_T)[None, :]
    vis = np.empty((B, t_days, P), dtype=np.float64)
    abs_lat = np.abs(lats)
    for pi, p in enumerate(platforms):
        if p in S.VIIRS_PLATFORMS:
            ps = np.full((B, 1), S.PSWATH_VIIRS)
            vis[:, :, pi] = ps * clear
        else:
            base = np.clip(S.PSWATH_MODIS_BASE
                           + S.PSWATH_MODIS_SLOPE * abs_lat / 30.0,
                           S.PSWATH_MODIS_BASE, 1.0)[:, None]
            vis[:, :, pi] = np.where(post, base * S.PSWATH_MODIS_POST2022,
                                     base) * clear

    # ---- Hurdle rate coefficient k = lambda0 psi_CNT(A_avg) / F_event -----
    k_miss = np.zeros((B, P, 2), dtype=np.float64)
    lam = np.array([lambda0.get(int(b), 1.0) for b in pyrome_arr])
    fev = np.array([max(f_event.get(int(b), 1.0), 1e-6) for b in pyrome_arr])
    for pi, p in enumerate(platforms):
        fam = S.SENSOR_FAMILY[p]
        a_avg = S.PASS_GEOMETRY_AVG[fam][0]
        for s in (0, 1):
            psi = ops.psi_cnt(fam, pyrome_arr, np.full(B, a_avg),
                              DN_STATES[s], lognorm)
            k_miss[:, pi, s] = lam * np.asarray(psi, dtype=float) / fev

    # ---- detection rows -> dense tensors (vectorized scatter) --------------
    y = np.zeros((B, t_days, P, 2), dtype=np.float64)
    sat = np.zeros((B, t_days, P, 2), dtype=np.uint8)
    mu = np.zeros((B, t_days, P, 2), dtype=np.float64)
    tau2 = np.full((B, t_days, P, 2), 1.0 / 0.09, dtype=np.float64)
    if pass_table is not None and len(pass_table):
        pt = pass_table[pass_table["y_frp_mw"] > 0]
        h = pt["h3_index"].to_numpy(dtype=np.int64)
        i = np.searchsorted(cells, h)
        ok = (i < B) & (cells[np.minimum(i, B - 1)] == h)
        t = (pd.to_datetime(pt["solar_date"]).to_numpy()
             - np.datetime64(T0)).astype("timedelta64[D]").astype(np.int64)
        pi_arr = pt["platform"].map(S.PLATFORM_INDEX).to_numpy()
        ok &= pd.notna(pi_arr) & (t >= 0) & (t < t_days)
        i, t = i[ok], t[ok]
        pi_arr = pi_arr[ok].astype(np.int64)
        s_arr = np.where(pt["dn_flag"].to_numpy()[ok] == "D", 0, 1)
        y[i, t, pi_arr, s_arr] = pt["y_frp_mw"].to_numpy(dtype=float)[ok]
        sat[i, t, pi_arr, s_arr] = (pt["sat_flag"].to_numpy()[ok]
                                    .astype(float) > 0).astype(np.uint8)
        mu[i, t, pi_arr, s_arr] = pt["mu_gain_det"].to_numpy(dtype=float)[ok]
        tau2[i, t, pi_arr, s_arr] = 1.0 / np.maximum(
            pt["sigma2"].to_numpy(dtype=float)[ok], 1e-6)

    # ---- prior means (B, T, 2) from the cell-month table -------------------
    months = dates.month.to_numpy()
    mu_cm = np.full((B, 12, 2), S.Z_MIN, dtype=np.float64)
    mu_map = priors["mu_map"]
    if mu_map:
        keys = np.array([(int(k[0]), int(k[1]), 0 if k[2] == "D" else 1)
                         for k in mu_map.keys()], dtype=np.int64)
        vals = np.fromiter(mu_map.values(), dtype=np.float64,
                           count=len(mu_map))
        ci = np.searchsorted(cells, keys[:, 0])
        okk = (ci < B) & (cells[np.minimum(ci, B - 1)] == keys[:, 0])
        mu_cm[ci[okk], keys[okk, 1] - 1, keys[okk, 2]] = vals[okk]
    mu_prior = mu_cm[:, months - 1, :]

    # ---- neighbour structure (intra-chunk edges only) ----------------------
    indptr, indices = priors["indptr"], priors["indices"]
    cell_list = np.asarray(priors["cells"], dtype=np.int64)
    colors_all = np.asarray(priors["colors"])
    kpos = np.searchsorted(cell_list, cells)
    kok = (kpos < len(cell_list)) & (
        cell_list[np.minimum(kpos, len(cell_list) - 1)] == cells)
    colors = np.where(kok, colors_all[np.minimum(kpos, len(cell_list) - 1)],
                      0).astype(np.int8)
    ei, ej = [], []
    for a_i in range(B):
        if not kok[a_i]:
            continue
        k = kpos[a_i]
        nb_cells = cell_list[indices[indptr[k]:indptr[k + 1]]]
        jj = np.searchsorted(cells, nb_cells)
        good = (jj < B) & (cells[np.minimum(jj, B - 1)] == nb_cells)
        for b_j in jj[good]:
            if b_j > a_i:
                ei.append(a_i)
                ej.append(int(b_j))
    ei = np.asarray(ei, dtype=np.int64)
    ej = np.asarray(ej, dtype=np.int64)
    n_nbrs = (np.bincount(ei, minlength=B) + np.bincount(ej, minlength=B)
              ).astype(np.float64)
    omega = np.array([1.0 / max(priors["sigma_proc"].get(int(b), 0.5) ** 2,
                                1e-6) for b in pyrome_arr], dtype=np.float64)
    rho = np.array([priors["rho"].get(int(b), 0.5) for b in pyrome_arr],
                   dtype=np.float64)
    eta = np.array([priors["eta_dn"].get(int(b), 0.15) for b in pyrome_arr],
                   dtype=np.float64)
    beta = np.array([priors["beta_dn"].get(int(b), 0.6) for b in pyrome_arr],
                    dtype=np.float64)

    return SolverData(
        cells=cells, colors=torch.tensor(colors),
        y=torch.from_numpy(y), sat=torch.from_numpy(sat),
        mu=torch.from_numpy(mu), tau2=torch.from_numpy(tau2),
        k_miss=torch.from_numpy(k_miss), vis=torch.from_numpy(vis),
        op=torch.from_numpy(op), mu_prior=torch.from_numpy(
            np.ascontiguousarray(mu_prior)),
        omega=torch.from_numpy(omega), rho=torch.from_numpy(rho),
        eta_dn=torch.from_numpy(eta), beta_dn=torch.from_numpy(beta),
        n_nbrs=torch.from_numpy(n_nbrs), edge_i=torch.from_numpy(ei),
        edge_j=torch.from_numpy(ej), t_days=t_days,
        pyrome_of_cell={int(c): int(pyrome_arr[q]) for q, c in enumerate(cells)},
        in_spotlight=spotlight_mask(lats, lons), cloud=cloud,
        cloud_source=cloud_src)


# ---------------------------------------------------------------------------
# The GMRF solver
# ---------------------------------------------------------------------------
class GMRFSolver:
    """3-color Gauss-Seidel damped-Newton MAP-Laplace solver (float64)."""

    def __init__(self, data: SolverData, use_cuda: bool | None = None,
                 device: str = "cpu", seed: int = S.GMM_RANDOM_STATE):
        self.device = torch.device(device)
        # Dense (B, T, P, 2) observation tensors stay in HOST memory; only
        # the compact per-block detection lists / masks are moved to the
        # device (bounded VRAM, readme D10 < 6 GB per chunk).
        self.d = data.to(self.device, skip=SolverData.HOST_ONLY)
        self.use_cuda = bool(use_cuda) and self.device.type == "cuda"
        self.B = len(data.cells)
        self.T = data.t_days
        self.z = self.d.mu_prior.clone()
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))
        self._cache = {}
        self._init_static()

    # ---- static per-solve structures --------------------------------------
    def _init_static(self):
        d = self.d
        ei, ej = d.edge_i.long(), d.edge_j.long()
        self._ei, self._ej = ei, ej
        if len(ei):
            self._w_edge = S.SPATIAL_ALPHA_S * 0.5 * (d.omega[ei] + d.omega[ej])
        else:
            self._w_edge = torch.zeros(0, dtype=torch.float64,
                                       device=self.device)
        sp_w = torch.zeros(self.B, dtype=torch.float64, device=self.device)
        if len(ei):
            sp_w.index_add_(0, ei, self._w_edge)
            sp_w.index_add_(0, ej, self._w_edge)
        self._sp_w = sp_w
        self._opmask = d.op > 0                                    # (T, P)

    def _block(self, idx: torch.Tensor):
        """Cached detection/non-detection structure for rows ``idx``."""
        key = tuple(idx.tolist())
        blk = self._cache.get(key)
        if blk is not None:
            return blk
        d = self.d
        dev = self.device
        idx_h = idx.cpu()
        op_h = self._opmask.cpu()
        det_h = (d.y[idx_h] > 0) & op_h.unsqueeze(0).unsqueeze(-1)
        blk = {"flat": {}, "miss": {}}
        for s in (0, 1):
            flat = torch.nonzero(det_h[..., s])
            bi, ti, pi = flat[:, 0], flat[:, 1], flat[:, 2]
            rows = idx_h[bi]
            y_f = d.y[rows, ti, pi, s].to(torch.float64)
            tau2_f = d.tau2[rows, ti, pi, s].to(torch.float64)
            blk["flat"][s] = {
                "bi": bi.to(dev), "ti": ti.to(dev),
                "ln_y": torch.log(torch.clamp(y_f, min=1e-12)).to(dev),
                "mu": d.mu[rows, ti, pi, s].to(dev, torch.float64),
                "tau2": tau2_f.to(dev),
                "tau1": torch.sqrt(torch.clamp(tau2_f, min=1e-12)).to(dev),
                "sat": (d.sat[rows, ti, pi, s] > 0).to(dev)}
            blk["miss"][s] = [
                (((~det_h[:, :, p, s]) & op_h[:, p].unsqueeze(0)).to(dev), p)
                for p in range(op_h.shape[1]) if bool(op_h[:, p].any())]
        if len(self._cache) < 16:
            self._cache[key] = blk
        return blk

    # ---- observation terms (block-local) -----------------------------------
    def _obs_rows(self, z_rows: torch.Tensor, idx: torch.Tensor, s: int,
                  need_grad: bool = True):
        """(nll, grad, hess) for rows ``idx`` at state ``s``; ``z_rows`` is
        ``z[idx, :, s]`` (n, T)."""
        d = self.d
        blk = self._block(idx)
        nll = torch.zeros_like(z_rows)
        grad = torch.zeros_like(z_rows) if need_grad else None
        hess = torch.zeros_like(z_rows) if need_grad else None
        f = blk["flat"][s]
        if len(f["bi"]):
            z_f = z_rows[f["bi"], f["ti"]]
            sat = f["sat"]
            l = torch.empty_like(z_f)
            g = torch.empty_like(z_f)
            h = torch.empty_like(z_f)
            mA = ~sat
            if bool(mA.any()):
                la, ga, ha = likelihoods.case_a_nll_grad_hess(
                    z_f[mA], f["ln_y"][mA], f["mu"][mA], f["tau2"][mA])
                l[mA], g[mA], h[mA] = la, ga, ha
            if bool(sat.any()):
                lb, gb, hb = likelihoods.case_b_nll_grad_hess(
                    z_f[sat], f["ln_y"][sat], f["mu"][sat], f["tau2"][sat],
                    f["tau1"][sat])
                l[sat], g[sat], h[sat] = lb, gb, hb
            nll.index_put_((f["bi"], f["ti"]), l, accumulate=True)
            if need_grad:
                grad.index_put_((f["bi"], f["ti"]), g, accumulate=True)
                hess.index_put_((f["bi"], f["ti"]), h, accumulate=True)
        for miss, p in blk["miss"][s]:
            k = d.k_miss[idx, p, s].view(-1, 1)
            v = torch.clamp(d.vis[idx, :, p], 0.0, 1.0)
            l, g, h = likelihoods.case_c_nll_grad_hess(z_rows, k, v)
            zero = torch.zeros_like(l)
            nll += torch.where(miss, l, zero)
            if need_grad:
                grad += torch.where(miss, g, zero)
                hess += torch.where(miss, h, zero)
        return nll, grad, hess

    # ---- prior terms -------------------------------------------------------
    def _delta(self, z):
        return z - self.d.mu_prior

    def _a_temp(self, idx):
        """Temporal prior main diagonal rows for cells ``idx`` (n, T)."""
        omega = self.d.omega[idx].view(-1, 1)
        rho = self.d.rho[idx].view(-1, 1)
        a = ((1.0 + rho * rho) * omega).expand(len(idx), self.T).clone()
        a[:, 0] = omega[:, 0]
        a[:, -1] = omega[:, 0]
        return a

    def _temporal_grad(self, idx, s, delta):
        """(H_time @ delta) rows for cells idx, state s (tridiag apply)."""
        omega = self.d.omega[idx].view(-1, 1)
        rho = self.d.rho[idx].view(-1, 1)
        dl = delta[idx, :, s]                    # (n, T)
        g = (1.0 + rho * rho) * omega * dl
        g[:, 0] = omega[:, 0] * dl[:, 0]
        g[:, -1] = omega[:, 0] * dl[:, -1]
        if self.T > 1:
            off = -rho * omega
            g[:, 1:] = g[:, 1:] + off * dl[:, :-1]
            g[:, :-1] = g[:, :-1] + off * dl[:, 1:]
        return g

    def _j_time_rows(self, idx, s, z):
        omega = self.d.omega[idx].view(-1, 1)
        rho = self.d.rho[idx].view(-1, 1)
        dl = z[idx, :, s] - self.d.mu_prior[idx, :, s]
        diff = dl[:, 1:] - rho * dl[:, :-1]
        return 0.5 * omega * ((1.0 - rho * rho) * dl[:, 0:1] ** 2
                              + (diff ** 2).sum(dim=1, keepdim=True))

    def _spatial_grad_all(self, z, s):
        """Spatial gradient for ALL cells at state s: sum_j w_ij (z_i - z_j)."""
        g = torch.zeros((self.B, self.T), dtype=torch.float64,
                        device=self.device)
        if len(self._ei):
            diff = z[self._ei, :, s] - z[self._ej, :, s]
            wd = self._w_edge.view(-1, 1) * diff
            g.index_add_(0, self._ei, wd)
            g.index_add_(0, self._ej, -wd)
        return g

    def _dn_grad_rows(self, idx, s, delta):
        """Exact gradient of both day-night coupling terms w.r.t. z_s."""
        eta = self.d.eta_dn[idx].view(-1, 1)
        beta = self.d.beta_dn[idx].view(-1, 1)
        om = self.d.omega[idx].view(-1, 1)
        ds = delta[idx, :, s]
        do = delta[idx, :, 1 - s]
        return eta * om * ((1.0 + beta * beta) * ds - 2.0 * beta * do)

    def _dn_hess_rows(self, idx):
        eta = self.d.eta_dn[idx].view(-1, 1)
        beta = self.d.beta_dn[idx].view(-1, 1)
        om = self.d.omega[idx].view(-1, 1)
        return eta * om * (1.0 + beta * beta)

    # ---- full objective -------------------------------------------------------
    def objective(self, z):
        all_idx = torch.arange(self.B, device=self.device)
        j = torch.zeros((), dtype=torch.float64, device=self.device)
        for s in (0, 1):
            nll, _, _ = self._obs_rows(z[:, :, s], all_idx, s, need_grad=False)
            j = j + nll.sum() + self._j_time_rows(all_idx, s, z).sum()
            if len(self._ei):
                diff = z[self._ei, :, s] - z[self._ej, :, s]
                j = j + (0.5 * self._w_edge.view(-1, 1) * diff ** 2).sum()
        delta = self._delta(z)
        eta = self.d.eta_dn.view(-1, 1)
        beta = self.d.beta_dn.view(-1, 1)
        om = self.d.omega.view(-1, 1)
        dD, dN = delta[:, :, 0], delta[:, :, 1]
        j = j + 0.5 * (eta * om * ((dD - beta * dN) ** 2
                                   + (dN - beta * dD) ** 2)).sum()
        return j

    # ---- block-local objective delta (exact) ---------------------------------
    def _block_delta_j(self, z_old, z_new, idx, s):
        """Exact J(z_new) - J(z_old) when only ``z[idx, :, s]`` changed."""
        d = self.d
        total = 0.0
        l_new, _, _ = self._obs_rows(z_new[idx, :, s], idx, s, need_grad=False)
        l_old, _, _ = self._obs_rows(z_old[idx, :, s], idx, s, need_grad=False)
        total += float((l_new - l_old).sum())
        total += float((self._j_time_rows(idx, s, z_new)
                        - self._j_time_rows(idx, s, z_old)).sum())
        if len(self._ei):
            m = torch.isin(self._ei, idx) | torch.isin(self._ej, idx)
            if bool(m.any()):
                ei, ej = self._ei[m], self._ej[m]
                w = self._w_edge[m].view(-1, 1)
                do = z_old[ei, :, s] - z_old[ej, :, s]
                dn_ = z_new[ei, :, s] - z_new[ej, :, s]
                total += float((0.5 * w * (dn_ ** 2 - do ** 2)).sum())
        eta = d.eta_dn[idx].view(-1, 1)
        beta = d.beta_dn[idx].view(-1, 1)
        om = d.omega[idx].view(-1, 1)
        mp = d.mu_prior[idx]
        o_old = z_old[idx, :, 1 - s] - mp[:, :, 1 - s]
        s_old = z_old[idx, :, s] - mp[:, :, s]
        s_new = z_new[idx, :, s] - mp[:, :, s]

        def _dn(ds):
            return (ds - beta * o_old) ** 2 + (o_old - beta * ds) ** 2
        total += float((0.5 * eta * om * (_dn(s_new) - _dn(s_old))).sum())
        return total

    # ---- Newton block ------------------------------------------------------
    def _block_system(self, color, s, t_lo):
        """(idx, a, b, g) tridiagonal system for a (color, state) block over
        the window ``[t_lo, T-1]`` (``t_lo > 0``: ``z[:t_lo]`` held fixed)."""
        d = self.d
        idx = torch.nonzero(d.colors == color).ravel()
        if len(idx) == 0:
            return None
        return (idx,) + self._block_system_rows(idx, s, t_lo)

    def _block_system_rows(self, idx, s, t_lo):
        d = self.d
        z = self.z
        delta = self._delta(z)
        _, g_obs, h_obs = self._obs_rows(z[idx, :, s], idx, s)
        omega = d.omega[idx].view(-1, 1)
        rho = d.rho[idx].view(-1, 1)
        g = (g_obs + self._temporal_grad(idx, s, delta)
             + self._spatial_grad_all(z, s)[idx]
             + self._dn_grad_rows(idx, s, delta))
        a = (self._a_temp(idx) + self._sp_w[idx].view(-1, 1)
             + self._dn_hess_rows(idx) + h_obs)
        a = torch.clamp(a[:, t_lo:], min=S.PRIOR_DIAG_FLOOR).contiguous()
        W = self.T - t_lo
        b = (-rho * omega).expand(len(idx), max(W - 1, 1)).contiguous()
        return a, b, g[:, t_lo:].contiguous()

    def _solve_block(self, a, b, g):
        """Thomas LDL^T solve: fused CUDA kernel on GPU, NumPy recursion on
        CPU (exact parity with the pure-torch reference)."""
        if self.use_cuda:
            dz, _l, _d = cuda_kernels.thomas_solve_batched(a, b, g,
                                                           use_cuda=True)
            return dz
        if self.device.type == "cpu":
            return torch.from_numpy(tt.thomas_factor_and_solve_numpy(
                a.numpy(), b.numpy(), g.numpy())[0])
        return tt.thomas_factor_and_solve(a, b, g)[0]

    def _factor(self, a, b):
        """LDL^T factor only (posterior)."""
        if self.use_cuda:
            return cuda_kernels.thomas_factor_batched(a, b, use_cuda=True)
        if self.device.type == "cpu":
            _dz, l, dd = tt.thomas_factor_and_solve_numpy(
                a.numpy(), b.numpy(), np.zeros(a.shape))
            return torch.from_numpy(l), torch.from_numpy(dd)
        return tt.thomas_factor(a, b)

    def _newton_block(self, color, s, t_lo=0):
        built = self._block_system(color, s, t_lo)
        if built is None:
            return
        idx, a, b, g = built
        for _ in range(S.NEWTON_STEPS):
            dz = self._solve_block(a, b, g)
            gdz = float((g * dz).sum())
            if not math.isfinite(gdz) or gdz <= 0.0:
                return
            z_old = self.z
            accepted = False
            for alpha in S.ARMIJO_ALPHAS:
                z_new = z_old.clone()
                z_new[idx, t_lo:, s] = torch.clamp(
                    z_old[idx, t_lo:, s] - alpha * dz, -8.0, 14.0)
                dj = self._block_delta_j(z_old, z_new, idx, s)
                if dj <= -S.ARMIJO_C1 * alpha * gdz:
                    self.z = z_new
                    accepted = True
                    break
            if not accepted:
                return
            if float(torch.max(torch.abs(alpha * dz))) < S.NEWTON_TOL:
                return
            a, b, g = self._block_system_rows(idx, s, t_lo)

    # ---- sweeps ------------------------------------------------------------
    def _sweeps(self, sweeps, max_sweeps, tol, t_lo=0):
        """``sweeps`` mandatory 3-color x Day/Night sweeps, then continue
        (up to ``max_sweeps``) until the max per-sweep change < ``tol``."""
        n_done = 0
        max_sweeps = max(int(max_sweeps), int(sweeps))
        while n_done < max_sweeps:
            z_prev = self.z
            for color in (0, 1, 2):
                for s in (0, 1):
                    self._newton_block(color, s, t_lo=t_lo)
            n_done += 1
            change = float(torch.max(torch.abs(self.z - z_prev))) \
                if self.z.numel() else 0.0
            if n_done >= sweeps and change < tol:
                break
        self.sweeps_done = n_done
        self.last_sweep_change = change if n_done else 0.0
        return self.z

    def solve(self, sweeps: int = S.SOLVER_SWEEPS,
              max_sweeps: int | None = None, tol: float = S.SWEEP_TOL):
        """MAP solve: the spec's 6 sweeps + convergence guard.  Passing
        ``max_sweeps = sweeps`` reproduces the fixed-sweep schedule."""
        return self._sweeps(sweeps, S.MAX_SOLVER_SWEEPS if max_sweeps is None
                            else max_sweeps, tol)

    def solve_fixed_lag(self, window_days: int = 14,
                        sweeps: int = S.SOLVER_SWEEPS,
                        max_sweeps: int | None = None,
                        tol: float = S.SWEEP_TOL):
        """Re-solve the trailing window ``[t_start, T-1]`` anchored at the
        saved ``z[t_start - 1]`` (AR(1) coupling enters the gradient)."""
        t_start = max(0, self.T - window_days)
        return self._sweeps(sweeps, S.MAX_SOLVER_SWEEPS if max_sweeps is None
                            else max_sweeps, tol, t_lo=t_start)

    # ---- posterior ---------------------------------------------------------
    def posterior_factors(self):
        """Exact conditional precision factors (l, d) per state at the MAP
        and Takahashi marginal variances ``var (B, T, 2)``."""
        all_idx = torch.arange(self.B, device=self.device)
        var = torch.zeros_like(self.z)
        factors = []
        for s in (0, 1):
            _, _, h_obs = self._obs_rows(self.z[:, :, s], all_idx, s)
            a = (self._a_temp(all_idx) + self._sp_w.view(-1, 1)
                 + self._dn_hess_rows(all_idx) + h_obs)
            a = torch.clamp(a, min=S.PRIOR_DIAG_FLOOR).contiguous()
            b = (-self.d.rho.view(-1, 1) * self.d.omega.view(-1, 1)).expand(
                self.B, max(self.T - 1, 1)).contiguous()
            l, dd = self._factor(a, b)
            sig_d, _ = cuda_kernels.takahashi_batched(l, dd,
                                                      use_cuda=self.use_cuda)
            var[:, :, s] = sig_d
            factors.append((l, dd))
        return var, factors

    def posterior(self, m_samples: int = S.MC_SAMPLES,
                  sub_batch: int | None = None, daily_mode: str = "sparse",
                  keep_years=None):
        """Takahashi variances + fused-24h trajectory statistics.

        Returns ``(var, monthly_df, daily_df)``.  ``daily_mode``:
        ``'sparse'`` (export filter applied in-chunk), ``'dense'`` (tests /
        diagnostics on tiny problems) or ``'none'`` (conformal folds).
        ``keep_years``: optional (lo, hi) year range for monthly rows.
        """
        var, factors = self.posterior_factors()
        (lD, dD), (lN, dN) = factors
        if sub_batch is None:
            sub_batch = 32 if self.device.type == "cuda" else 64
        t_len = self.T
        dates = T0 + pd.to_timedelta(np.arange(t_len), unit="D")
        ym = dates.year.to_numpy() * 12 + dates.month.to_numpy() - 1
        uym, mi = np.unique(ym, return_inverse=True)
        n_m = len(uym)
        mi_t = torch.as_tensor(mi, device=self.device)
        cnt = torch.as_tensor(np.bincount(mi, minlength=n_m),
                              dtype=torch.float64, device=self.device)
        month_keep = np.ones(n_m, dtype=bool)
        if keep_years is not None:
            yrs = uym // 12
            month_keep = (yrs >= keep_years[0]) & (yrs <= keep_years[1])
        # sparse daily criteria pieces
        det_any = ((self.d.y > 0) & self._opmask.cpu().unsqueeze(0)
                   .unsqueeze(-1)).any(dim=3).any(dim=2).to(self.device)
        pad = S.DAILY_SPARSE_DETECTION_PAD_DAYS
        near = torch.nn.functional.max_pool1d(
            det_any.double().unsqueeze(1), kernel_size=2 * pad + 1, stride=1,
            padding=pad).squeeze(1) > 0
        spot = torch.as_tensor(
            self.d.in_spotlight if self.d.in_spotlight is not None
            else np.zeros(self.B, dtype=bool), device=self.device)
        qs = torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64,
                          device=self.device)
        mon_parts, day_parts = [], []
        cells = np.asarray(self.d.cells, dtype=np.int64)
        for lo in range(0, self.B, sub_batch):
            hi = min(lo + sub_batch, self.B)
            n = hi - lo
            epsD = torch.randn((n, t_len, m_samples), dtype=torch.float64,
                               device=self.device, generator=self._gen)
            uD = cuda_kernels.cholesky_sample_batched(
                lD[lo:hi], dD[lo:hi], epsD, use_cuda=self.use_cuda)
            del epsD
            phiD = torch.exp(torch.clamp(self.z[lo:hi, :, 0].unsqueeze(-1)
                                         + uD, -8.0, 14.0))
            del uD
            epsN = torch.randn((n, t_len, m_samples), dtype=torch.float64,
                               device=self.device, generator=self._gen)
            uN = cuda_kernels.cholesky_sample_batched(
                lN[lo:hi], dN[lo:hi], epsN, use_cuda=self.use_cuda)
            del epsN
            phiN = torch.exp(torch.clamp(self.z[lo:hi, :, 1].unsqueeze(-1)
                                         + uN, -8.0, 14.0))
            del uN
            # monthly means per trajectory (D, N, fused 24h) + monthly FRE
            def _mmean(x):
                acc = torch.zeros((n, n_m, m_samples), dtype=torch.float64,
                                  device=self.device)
                acc.index_add_(1, mi_t, x)
                return acc
            sD = _mmean(phiD)
            sN = _mmean(phiN)
            s24 = 0.5 * (sD + sN)                       # monthly SUM of Phi24
            c = cnt.view(1, -1, 1)
            m24 = s24 / c
            q24 = _quantiles(m24, qs)                   # (3, n, n_m)
            mon = {
                "phi_q05_mw": q24[0], "phi_q50_mw": q24[1],
                "phi_q95_mw": q24[2], "phi_mean_mw": m24.mean(dim=-1),
                "phi_day_mw": _quantiles(sD / c, qs[1:2])[0],
                "phi_night_mw": _quantiles(sN / c, qs[1:2])[0],
                "fre_tj_q50": _quantiles(S.FRE_TJ_PER_MW_DAY * s24,
                                         qs[1:2])[0],
            }
            del sD, sN, s24, m24
            mon_parts.append({k: v.cpu().numpy() for k, v in mon.items()}
                             | {"rows": np.arange(lo, hi)})
            if daily_mode != "none":
                phi24 = 0.5 * (phiD + phiN)                 # (n, T, M) fused
                del phiD, phiN
                qd = _quantiles(phi24, qs)                  # (3, n, T)
                dmean = phi24.mean(dim=-1)
                del phi24
                if daily_mode == "sparse":
                    keep = ((qd[1] > S.DAILY_SPARSE_PHI_MIN) | near[lo:hi]
                            | spot[lo:hi].view(-1, 1))
                else:
                    keep = torch.ones_like(qd[1], dtype=torch.bool)
                r, t_ix = torch.nonzero(keep, as_tuple=True)
                rr = r + lo
                day_parts.append({
                    "h3_index": cells[rr.cpu().numpy()],
                    "t": t_ix.cpu().numpy().astype(np.int64),
                    "phi_day_mw": torch.exp(self.z[rr, t_ix, 0]).cpu().numpy(),
                    "phi_night_mw": torch.exp(self.z[rr, t_ix, 1]).cpu().numpy(),
                    "phi_q05_mw": qd[0][r, t_ix].cpu().numpy(),
                    "phi_q50_mw": qd[1][r, t_ix].cpu().numpy(),
                    "phi_q95_mw": qd[2][r, t_ix].cpu().numpy(),
                    "phi_mean_mw": dmean[r, t_ix].cpu().numpy(),
                })
                del qd, dmean
            else:
                del phiD, phiN
        monthly = _monthly_frame(mon_parts, cells, uym, month_keep)
        daily = _daily_frame(day_parts)
        return var, monthly, daily


def _quantiles(x: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """Linear-interpolated quantiles over the LAST dim (numpy 'linear'),
    via sort (``torch.quantile`` caps input size at 2^24 elements)."""
    xs, _ = torch.sort(x, dim=-1)
    m = x.shape[-1]
    pos = qs * (m - 1)
    lo = torch.floor(pos).long()
    hi = torch.clamp(lo + 1, max=m - 1)
    frac = (pos - lo.double()).view(-1, *([1] * (x.dim() - 1)))
    lo_v = xs.index_select(-1, lo).movedim(-1, 0)
    hi_v = xs.index_select(-1, hi).movedim(-1, 0)
    return lo_v + frac * (hi_v - lo_v)


def _monthly_frame(parts, cells, uym, month_keep) -> pd.DataFrame:
    cols = ["phi_day_mw", "phi_night_mw", "phi_q05_mw", "phi_q50_mw",
            "phi_q95_mw", "phi_mean_mw", "fre_tj_q50"]
    if not parts:
        return pd.DataFrame(columns=["h3_index", "year", "month"] + cols)
    frames = []
    mk = np.nonzero(month_keep)[0]
    for p in parts:
        n = len(p["rows"])
        h = np.repeat(cells[p["rows"]], len(mk))
        ym = np.tile(uym[mk], n)
        f = {"h3_index": h, "year": (ym // 12).astype(np.int64),
             "month": (ym % 12 + 1).astype(np.int64)}
        for c in cols:
            f[c] = p[c][:, mk].reshape(-1)
        frames.append(pd.DataFrame(f))
    out = pd.concat(frames, ignore_index=True)
    out["phi_q05_mw"] = out["phi_q05_mw"].clip(lower=0.0)
    return out


def _daily_frame(parts) -> pd.DataFrame:
    cols = ["h3_index", "t", "phi_day_mw", "phi_night_mw", "phi_q05_mw",
            "phi_q50_mw", "phi_q95_mw", "phi_mean_mw", "fre_tj_q50"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)
    out["fre_tj_q50"] = S.FRE_TJ_PER_MW_DAY * out["phi_q50_mw"]
    return out[cols]


# ---------------------------------------------------------------------------
# Back-compat helpers (fused quantiles are produced in posterior())
# ---------------------------------------------------------------------------
def aggregate_trajectories(cells, t_days, monthly, daily=None):
    """Identity pass-through kept for API stability: ``posterior()`` already
    returns monthly/daily statistics over fused 24h trajectories."""
    if isinstance(monthly, pd.DataFrame):
        return monthly, (daily if daily is not None else pd.DataFrame())
    raise TypeError("aggregate_trajectories expects the posterior() frames")


def combine_states_to_products(monthly, daily):
    """Identity: Day/Night fusion happens per trajectory in posterior()."""
    return monthly, daily


# ---------------------------------------------------------------------------
# Chunk orchestration
# ---------------------------------------------------------------------------
def estimate_chunk_vram_bytes(batch_size: int = S.BATCH_SIZE,
                              t_days: int = S.T_DAYS,
                              n_platforms: int = S.N_PLATFORMS,
                              m_samples: int = S.MC_SAMPLES,
                              sub_batch: int = 32) -> int:
    """Analytic peak-memory estimate for one GPU chunk (float64 bytes).

    GPU-resident: vis (B, T, P) float64; Case-C miss masks for the three
    color blocks + the all-cell block (2 x (B, T, P, 2) bool); z / mu_prior /
    var / Armijo candidate / Newton temporaries (8 x (B, T, 2) float64);
    detection lists (O(#detections), bounded here by a 5% fill of the dense
    grid x 7 float64 fields).  The dense (B, T, P, 2) observation tensors
    stay in host RAM.  Posterior transient: two sampled (sub, T, M) chains,
    the fused trajectory and sort buffers (values + int64 indices).
    """
    vis = batch_size * t_days * n_platforms * 8
    masks = 2 * batch_size * t_days * n_platforms * 2 * 1
    state = 8 * batch_size * t_days * 2 * 8
    det_lists = int(0.05 * batch_size * t_days * n_platforms * 2) * 7 * 8
    mc = 5 * sub_batch * t_days * m_samples * 8
    return vis + masks + state + det_lists + mc


def solve_chunk(data: SolverData, sweeps: int = S.SOLVER_SWEEPS,
                use_cuda: bool | None = None, device: str = "cpu",
                max_sweeps: int | None = None,
                daily_mode: str = "sparse", keep_years=None,
                keep_solver: bool = True, seed: int = S.GMM_RANDOM_STATE):
    solver = GMRFSolver(data, use_cuda=use_cuda, device=device, seed=seed)
    z = solver.solve(sweeps, max_sweeps=max_sweeps)
    var, monthly, daily = solver.posterior(daily_mode=daily_mode,
                                           keep_years=keep_years)
    out = {"cells": np.asarray(data.cells, dtype=np.int64),
           "z_map": z.cpu().numpy(), "var": var.cpu().numpy(),
           "monthly": monthly, "daily": daily,
           "cloud": data.cloud, "cloud_source": data.cloud_source,
           "op": data.op.cpu().numpy(),
           "sweeps_done": solver.sweeps_done,
           "last_sweep_change": solver.last_sweep_change}
    if keep_solver:
        out["solver"] = solver
    return out


def _n_cuda_devices() -> int:
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def _chunk_job(shared, ci, chunk, device, use_cuda, opts):
    data = build_observation_tensors(
        shared["pass_table"], chunk, shared["hex_pyromes"], shared["priors"],
        shared["artifacts"], shared["cloud_df"], t_days=opts["t_days"],
        platform_active=shared["platform_active"],
        platform_mask=shared["platform_mask"])
    res = solve_chunk(data, sweeps=opts["sweeps"], use_cuda=use_cuda,
                      max_sweeps=opts.get("max_sweeps"),
                      device=device, daily_mode=opts["daily_mode"],
                      keep_years=opts["keep_years"],
                      keep_solver=opts["keep_solver"],
                      seed=S.GMM_RANDOM_STATE + ci)
    if not (opts["keep_states"] or opts.get("save_states")):
        _drop_states(res)
    return res


_STATE_KEYS = ("z_map", "var", "cloud", "cloud_source")


def _drop_states(res):
    for k in _STATE_KEYS:
        res.pop(k, None)


def _finish_chunk(res, ci, opts):
    """Checkpoint the dense states (float32) then drop them unless the
    caller asked to keep them: a (1000 x 9510) chunk carries ~240 MB of
    state arrays, which summed over hundreds of chunks in the parent
    process would need tens of GB of RAM per solve."""
    res.pop("solver", None) if not opts.get("keep_solver") else None
    if opts["checkpoint_dir"]:
        _save_chunk_checkpoint(opts["checkpoint_dir"], ci, res)
    if not opts["keep_states"]:
        _drop_states(res)
    return res


def _dump_atomic(obj, path):
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _chunk_path(out_dir, ci):
    return os.path.join(out_dir, f"chunk_{ci:06d}.pkl")


def _solve_tag(pass_table, cells, batch_size, opts, platform_mask,
               platform_active) -> str:
    """Fingerprint of everything that determines a solve's chunk results
    (inputs, cells, batching, options, platform mask) -- NOT the number of
    workers, so a retry with fewer workers reuses finished chunks."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(cells, dtype=np.int64).tobytes())
    h.update(repr(int(batch_size)).encode())
    h.update(repr(sorted((k, v) for k, v in opts.items()
                         if k not in ("checkpoint_dir", "keep_solver"))).encode())
    h.update(pickle.dumps(platform_mask))
    h.update(repr(sorted((k, len(v)) for k, v in (platform_active or {}).items())
                  ).encode())
    if pass_table is not None and len(pass_table):
        h.update(repr((len(pass_table), list(pass_table.columns))).encode())
        sample = pass_table.iloc[:: max(1, len(pass_table) // 5000)]
        h.update(pd.util.hash_pandas_object(sample, index=False)
                 .to_numpy().tobytes())
        for c in pass_table.columns:
            if pd.api.types.is_numeric_dtype(pass_table[c]):
                h.update(repr(float(np.nansum(pass_table[c].to_numpy(
                    dtype=np.float64)))).encode())
    return h.hexdigest()[:20]


def _gpu_worker(rank, n_workers, n_devices, job_file, out_dir):
    """torch.multiprocessing worker: chunks ci with ci % n_workers == rank
    on device cuda:(rank % n_devices)."""
    with open(job_file, "rb") as fh:
        job = pickle.load(fh)
    dev = f"cuda:{rank % n_devices}"
    torch.cuda.set_device(rank % n_devices)
    use_cuda = cuda_kernels.cuda_available()
    opts = dict(job["opts"], keep_solver=False)
    todo = job["todo"]                    # chunk indices not yet solved
    for k, ci in enumerate(todo):
        if k % n_workers != rank or os.path.exists(_chunk_path(out_dir, ci)):
            continue                      # other worker's chunk / resumed
        res = _chunk_job(job["shared"], ci, job["chunks"][ci], dev, use_cuda,
                         opts)
        _dump_atomic(_finish_chunk(res, ci, opts), _chunk_path(out_dir, ci))
        del res
        torch.cuda.empty_cache()


def _save_chunk_checkpoint(ckpt_dir, ci, res):
    os.makedirs(ckpt_dir, exist_ok=True)
    if "z_map" in res:
        torch.save({"cells": res["cells"],
                    "z_map": res["z_map"].astype(np.float32),
                    "var": res["var"].astype(np.float32)},
                   os.path.join(ckpt_dir, f"state_chunk_{ci:06d}.pt"))


def solve_all_chunks(pass_table, active_cells, hex_pyromes, priors,
                     artifacts, cloud_df, t_days: int = S.T_DAYS,
                     platform_active=None, platform_mask=None,
                     batch_size: int = S.BATCH_SIZE, paths=None,
                     sweeps: int = S.SOLVER_SWEEPS, daily_mode: str = "sparse",
                     keep_years=None, keep_states: bool = True,
                     workers_per_gpu: int = 1, n_devices: int | None = None,
                     max_sweeps: int | None = None):
    """Distribute active cells over all GPUs (``batch_size = 1000`` cells
    per chunk, ``< 6 GB`` VRAM each).

    * ``>= 2`` GPUs (or ``workers_per_gpu > 1``): ``torch.multiprocessing``
      spawn, one process per (GPU, slot); chunk ``ci`` runs on
      ``cuda:(ci % n_workers) % n_devices``.
    * 1 GPU: in-process on ``cuda:0``.  No GPU: in-process on CPU.

    ``keep_states=False`` drops the dense ``(N, T, 2)`` MAP/variance arrays
    from the returned dict (they are written as per-chunk float32
    checkpoints when ``paths`` is given) so host RAM stays bounded on
    hundreds of thousands of cells.
    """
    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells))),
                     dtype=np.int64)
    chunks = [cells[i:i + batch_size] for i in range(0, len(cells), batch_size)]
    n_dev = _n_cuda_devices() if n_devices is None else int(n_devices)
    checkpoint_dir = None
    if paths is not None:
        checkpoint_dir = str(paths.state_checkpoint_pt) + ".chunks"
    opts = {"t_days": t_days, "sweeps": sweeps, "max_sweeps": max_sweeps,
            "daily_mode": daily_mode,
            "keep_years": keep_years,
            "keep_states": keep_states,
            "save_states": checkpoint_dir is not None,
            "keep_solver": n_dev <= 1, "checkpoint_dir": checkpoint_dir}
    shared = {"pass_table": pass_table, "hex_pyromes": hex_pyromes,
              "priors": priors, "artifacts": artifacts, "cloud_df": cloud_df,
              "platform_active": platform_active,
              "platform_mask": platform_mask}
    n_workers = n_dev * max(1, int(workers_per_gpu))
    results = []
    # Resumable chunk results: with FIRE_HARMONIZER_RESUME_DIR set, every
    # finished chunk is kept under <dir>/<solve fingerprint>/ and a re-run
    # (e.g. after an out-of-memory kill, with fewer workers) skips it.
    resume_root = os.environ.get("FIRE_HARMONIZER_RESUME_DIR")
    tmp_ctx = tempfile.TemporaryDirectory(prefix="fh_solve_")
    with tmp_ctx as td:
        out_dir = td
        if resume_root:
            tag = _solve_tag(pass_table, cells, batch_size, opts,
                             platform_mask, platform_active)
            out_dir = os.path.join(resume_root, f"solve_{tag}")
            os.makedirs(out_dir, exist_ok=True)
        done = sum(os.path.exists(_chunk_path(out_dir, ci))
                   for ci in range(len(chunks)))
        if done:
            logger.info("resume: %d / %d chunks already solved in %s", done,
                        len(chunks), out_dir)
        if done < len(chunks):
            if n_dev >= 1 and n_workers >= 2 and len(chunks) >= 2:
                import torch.multiprocessing as tmp
                todo = [ci for ci in range(len(chunks))
                        if not os.path.exists(_chunk_path(out_dir, ci))]
                nprocs = min(n_workers, len(todo))
                job_file = os.path.join(td, "job.pkl")
                with open(job_file, "wb") as fh:
                    pickle.dump({"chunks": chunks, "shared": shared,
                                 "opts": opts, "todo": todo}, fh,
                                protocol=pickle.HIGHEST_PROTOCOL)
                tmp.spawn(_gpu_worker, args=(nprocs, n_dev, job_file,
                                             out_dir),
                          nprocs=nprocs, join=True)
            else:
                device = "cuda:0" if n_dev >= 1 else "cpu"
                use_cuda = (cuda_kernels.cuda_available() if n_dev >= 1
                            else False)
                for ci, chunk in enumerate(chunks):
                    if os.path.exists(_chunk_path(out_dir, ci)):
                        continue
                    res = _finish_chunk(_chunk_job(shared, ci, chunk, device,
                                                   use_cuda, opts), ci, opts)
                    if resume_root:
                        keep = res.pop("solver", None)
                        _dump_atomic(res, _chunk_path(out_dir, ci))
                        if keep is not None:
                            res["solver"] = keep
                    results.append((ci, res))
        in_mem = dict(results)
        results = []
        for ci in range(len(chunks)):
            if ci in in_mem:
                results.append(in_mem.pop(ci))
            else:
                with open(_chunk_path(out_dir, ci), "rb") as fh:
                    results.append(pickle.load(fh))
    out = {"cells": np.concatenate([r["cells"] for r in results])
           if results else cells,
           "monthly": pd.concat([r["monthly"] for r in results],
                                ignore_index=True) if results else
           _monthly_frame([], cells, np.zeros(0, dtype=np.int64),
                          np.zeros(0, dtype=bool)),
           "daily": pd.concat([r["daily"] for r in results],
                              ignore_index=True) if results else
           _daily_frame([]),
           "op": results[0]["op"] if results else
           operational_calendar(t_days, platform_active, platform_mask),
           "results": results}
    if keep_states and results and "z_map" in results[0]:
        out["z_map"] = np.concatenate([r["z_map"] for r in results], axis=0)
        out["var"] = np.concatenate([r["var"] for r in results], axis=0)
        out["cloud"] = np.concatenate([r["cloud"] for r in results], axis=0)
        out["cloud_source"] = np.concatenate(
            [r["cloud_source"] for r in results], axis=0)
    if paths is not None:
        torch.save({"cells": out["cells"], "checkpoint_dir": checkpoint_dir,
                    "n_chunks": len(chunks), "batch_size": batch_size},
                   paths.state_checkpoint_pt)
    return out


def run_fixed_lag_update(solver: GMRFSolver, window_days: int = 14):
    """Offline-callable trailing-window re-solve (readme D11/D12)."""
    return solver.solve_fixed_lag(window_days)
