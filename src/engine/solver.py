"""Multi-GPU float64 GMRF MAP-Laplace solver (readme Sections 4.8, 5.3.3-5.3.4).

* Partitions active cells across ``0..device_count()-1`` GPUs in chunks of
  ``batch_size = 1000`` (``< 6 GB`` VRAM per GPU; portable across T4 x2,
  1x L4, 4x L4).
* Builds dense ``(B, T, P=5)`` per-state observation + non-detection tensors
  per chunk (``y = 0`` encodes Hurdle non-detections; ``op`` masks
  non-operational passes, INV-04).
* Runs ``SOLVER_SWEEPS = 6`` outer sweeps over the H3 **3-coloring**
  (``color in {0,1,2}``) x alternating ``('D','N')``; per block up to
  ``NEWTON_STEPS = 15`` damped Newton-Raphson iterations with Armijo
  backtracking (``alpha in {1, .5, .25, .125, .0625}``,
  ``J(z_new) <= J(z) - 1e-4 alpha sum(g dz)``, early exit
  ``||alpha dz||_inf < 1e-4``), solved via the fused ``float64`` Thomas
  ``LDL^T`` (CUDA kernel with transparent CPU fallback).
* Computes exact Takahashi marginal variances and draws ``M = 200`` joint
  posterior trajectories via ``L^T u = D^{-1/2} eps`` (soft-clamped to
  ``z in [-8, 14]``), integrating ``Phi_24h = 0.5 (Phi_Day + Phi_Night)`` and
  ``FRE = 0.0864 Phi`` (TJ/day), aggregating monthly/daily
  ``q05 / q50 / q95 / mean``.
* Exposes ``run_fixed_lag_update(window_days=14)`` re-solving the trailing
  window anchored at ``t_start - 1`` via the AR(1) prior to the saved state.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from ..config import settings as S
from . import cuda_kernels, likelihoods
from . import thomas_takahashi as tt
from .priors import (build_cloud_clim_lookup, build_cloud_lookup,
                     cloud_fraction_series, visibility_matrix)

DN_STATES = ("D", "N")
T0 = pd.Timestamp(S.DATE_START)


# ---------------------------------------------------------------------------
# Observation tensor construction
# ---------------------------------------------------------------------------
class SolverData:
    """Dense per-chunk solver tensors (torch.float64)."""

    def __init__(self, cells, colors, y, sat, mu, tau2, k_miss, vis, op,
                 mu_prior, omega, rho, eta_dn, beta_dn, n_nbrs, edge_i, edge_j,
                 t_days, pyrome_of_cell):
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
        self.n_nbrs = n_nbrs          # (B,)
        self.edge_i = edge_i          # (E,) chunk-local indices
        self.edge_j = edge_j
        self.t_days = t_days
        self.pyrome_of_cell = pyrome_of_cell

    def to(self, device):
        for name in ("y", "sat", "mu", "tau2", "k_miss", "vis", "op",
                     "mu_prior", "omega", "rho", "eta_dn", "beta_dn",
                     "n_nbrs", "edge_i", "edge_j", "colors"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def build_observation_tensors(pass_table, active_cells, hex_pyromes, priors,
                              artifacts, cloud_df, t_days: int = S.T_DAYS,
                              platform_active=None,
                              platform_mask=None) -> SolverData:
    """Dense (B, T, P, 2) observation + non-detection tensors for one chunk.

    ``platform_mask(p_index, t) -> bool`` optionally masks platforms
    (VIIRS-masked ablation, leave-one-VIIRS-platform-out conformal folds).
    """
    from ..calibration import operators as ops
    from ..geometry.h3_utils import h3_cell_center

    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells))),
                     dtype=np.int64)
    B = len(cells)
    P = S.N_PLATFORMS
    platforms = list(S.PLATFORM_INDEX)
    pos = {int(c): i for i, c in enumerate(cells)}
    pyrome_of = dict(zip(hex_pyromes["h3_index"].astype(np.int64),
                         hex_pyromes["pyrome_id"].astype(int)))
    pyrome_arr = np.array([pyrome_of.get(int(c), 0) for c in cells], dtype=int)

    y = np.zeros((B, t_days, P, 2), dtype=np.float64)
    sat = np.zeros((B, t_days, P, 2), dtype=np.uint8)
    mu = np.zeros((B, t_days, P, 2), dtype=np.float64)
    tau2 = np.ones((B, t_days, P, 2), dtype=np.float64) / 0.09
    k_miss = np.zeros((B, P, 2), dtype=np.float64)
    vis = np.zeros((B, t_days, P), dtype=np.float64)
    op = np.zeros((t_days, P), dtype=np.float64)

    lognorm = artifacts["lognorm_params"]
    f_event = priors["f_event"]
    lambda0 = priors["lambda0"]

    # ---- operational calendar (static timeline AND empirical active days)
    for p, (d0, d1) in S.PLATFORM_TIMELINE.items():
        pi = S.PLATFORM_INDEX[p]
        lo = max(0, (pd.Timestamp(d0) - T0).days)
        hi = min(t_days - 1, (pd.Timestamp(d1) - T0).days)
        if hi >= lo:
            op[lo:hi + 1, pi] = 1.0
    if platform_active is not None:
        for p, days in platform_active.items():
            pi = S.PLATFORM_INDEX[p]
            dayset = {str(d)[:10] for d in days}
            for t in range(t_days):
                if op[t, pi] > 0 and (T0 + pd.Timedelta(days=t)) \
                        .strftime("%Y-%m-%d") not in dayset:
                    op[t, pi] = 0.0
    if platform_mask is not None:
        for pi in range(P):
            for t in range(t_days):
                if not platform_mask(pi, t):
                    op[t, pi] = 0.0

    # ---- cloud visibility + k_miss per cell
    cloud_lookup = build_cloud_lookup(cloud_df)
    clim_lookup = build_cloud_clim_lookup(artifacts.get("cloud_climatology"))
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t) for t in range(t_days)])
    for i, c in enumerate(cells):
        lat_c, lon_c = h3_cell_center(int(c))
        cloud, _src = cloud_fraction_series(lat_c, lon_c, dates, cloud_lookup,
                                            clim_lookup)
        for pi, p in enumerate(platforms):
            vis[i, :, pi] = visibility_matrix(
                p, lat_c, cloud, np.arange(t_days))
            fam = S.SENSOR_FAMILY[p]
            a_avg = S.PASS_GEOMETRY_AVG[fam][0]
            b = int(pyrome_arr[i])
            for s in (0, 1):
                psi = ops.psi_cnt(fam, b, a_avg, DN_STATES[s], lognorm)
                k_miss[i, pi, s] = (lambda0.get(b, 1.0) * psi
                                    / max(f_event.get(b, 1.0), 1e-6))

    # ---- detection rows -> dense tensors
    if pass_table is not None and len(pass_table):
        pt = pass_table.copy()
        pt["t"] = ((pd.to_datetime(pt["solar_date"]) - T0).dt.days).astype(int)
        for r in pt.itertuples():
            i = pos.get(int(r.h3_index))
            pi = S.PLATFORM_INDEX.get(r.platform)
            if i is None or pi is None:
                continue
            t = int(r.t)
            if not (0 <= t < t_days):
                continue
            yv = float(r.y_frp_mw)
            if yv <= 0:
                continue                     # detection rows only (Case C = 0)
            s = 0 if r.dn_flag == "D" else 1
            y[i, t, pi, s] = yv
            sat[i, t, pi, s] = 1 if int(r.sat_flag) > 0 else 0
            mu[i, t, pi, s] = float(r.mu_gain_det)
            tau2[i, t, pi, s] = 1.0 / max(float(r.sigma2), 1e-6)

    # ---- prior means (B, T, 2)
    mu_prior = np.full((B, t_days, 2), S.Z_MIN, dtype=np.float64)
    mu_map = priors["mu_map"]
    months = dates.month.to_numpy()
    for i, c in enumerate(cells):
        for m in range(1, 13):
            sel = months == m
            for s in (0, 1):
                v = mu_map.get((int(c), m, DN_STATES[s]))
                if v is not None:
                    mu_prior[i, sel, s] = v

    # ---- neighbor structure + prior parameters (chunk-local)
    indptr, indices = priors["indptr"], priors["indices"]
    cell_list = [int(c) for c in priors["cells"]]
    pos_p = {c: k for k, c in enumerate(cell_list)}
    color_of = {int(c): int(col) for c, col
                in zip(priors["cells"], np.asarray(priors["colors"]))}
    n_nbrs = np.zeros(B, dtype=np.float64)
    ei, ej = [], []
    for i, c in enumerate(cells):
        k = pos_p.get(int(c))
        if k is None:
            continue
        nbrs = indices[indptr[k]:indptr[k + 1]]
        n_nbrs[i] = len(nbrs)
        for j in nbrs:
            jj = pos.get(cell_list[int(j)])
            if jj is not None and jj != i and i < jj:
                ei.append(i)
                ej.append(jj)
    omega = np.array([1.0 / max(priors["sigma_proc"].get(int(pyrome_arr[i]), 0.5) ** 2,
                                1e-6) for i in range(B)], dtype=np.float64)
    rho = np.array([priors["rho"].get(int(pyrome_arr[i]), 0.5)
                    for i in range(B)], dtype=np.float64)
    eta = np.array([priors["eta_dn"].get(int(pyrome_arr[i]), 0.15)
                    for i in range(B)], dtype=np.float64)
    beta = np.array([priors["beta_dn"].get(int(pyrome_arr[i]), 0.6)
                     for i in range(B)], dtype=np.float64)
    colors = np.array([color_of.get(int(c), 0) for c in cells], dtype=np.int8)

    return SolverData(
        cells=cells, colors=torch.tensor(colors),
        y=torch.tensor(y), sat=torch.tensor(sat), mu=torch.tensor(mu),
        tau2=torch.tensor(tau2), k_miss=torch.tensor(k_miss),
        vis=torch.tensor(vis), op=torch.tensor(op),
        mu_prior=torch.tensor(mu_prior), omega=torch.tensor(omega),
        rho=torch.tensor(rho), eta_dn=torch.tensor(eta),
        beta_dn=torch.tensor(beta), n_nbrs=torch.tensor(n_nbrs),
        edge_i=torch.tensor(np.asarray(ei, dtype=np.int64)),
        edge_j=torch.tensor(np.asarray(ej, dtype=np.int64)),
        t_days=t_days,
        pyrome_of_cell={int(c): int(pyrome_arr[i])
                        for i, c in enumerate(cells)})


# ---------------------------------------------------------------------------
# The GMRF solver
# ---------------------------------------------------------------------------
class GMRFSolver:
    """3-color Gauss-Seidel damped-Newton MAP-Laplace solver (float64)."""

    def __init__(self, data: SolverData, use_cuda: bool | None = None,
                 device: str = "cpu"):
        self.d = data.to(device)
        self.device = device
        self.use_cuda = use_cuda
        self.B = len(data.cells)
        self.T = data.t_days
        self.z = data.mu_prior.clone()
        self._rand = torch.Generator(device="cpu")
        self._rand.manual_seed(S.GMM_RANDOM_STATE)

    # ---- observation terms ----------------------------------------------
    def _obs_terms(self, z):
        """Per-case evaluation using the spec's exact case functions.

        Case A/B (detections, a few % of entries) evaluate on flat subsets;
        Case C (non-detections) evaluates per (platform, state) on (B, T)
        tensors -- mathematically identical to
        ``likelihoods.eval_pointwise_nll_grad_hess`` (verified in tests),
        without full-size temporaries.
        """
        d = self.d
        B, T = self.B, self.T
        nll = torch.zeros((B, T, 2), dtype=torch.float64, device=z.device)
        grad = torch.zeros_like(nll)
        hess = torch.zeros_like(nll)
        # op-active detection mask (y > 0 & op)
        op_mask = d.op.unsqueeze(0).unsqueeze(-1)          # (1, T, P, 1)
        det = (d.y > 0) & (op_mask.squeeze(-1).unsqueeze(-1) > 0)  # (B,T,P,2)
        if det.any():
            flat = torch.nonzero(det)                       # (N, 4)
            b_i, t_i, p_i, s_i = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
            z_f = z[b_i, t_i, s_i]
            y_f = d.y[b_i, t_i, p_i, s_i]
            mu_f = d.mu[b_i, t_i, p_i, s_i]
            tau2_f = d.tau2[b_i, t_i, p_i, s_i]
            sat_f = d.sat[b_i, t_i, p_i, s_i]
            ln_y = torch.log(torch.clamp(y_f, min=1e-12))
            # Case A (unsaturated)
            mA = sat_f == 0
            if mA.any():
                l, g, h = likelihoods.case_a_nll_grad_hess(
                    z_f[mA], ln_y[mA], mu_f[mA], tau2_f[mA])
                nll.index_put_((b_i[mA], t_i[mA], s_i[mA]), l, accumulate=True)
                grad.index_put_((b_i[mA], t_i[mA], s_i[mA]), g, accumulate=True)
                hess.index_put_((b_i[mA], t_i[mA], s_i[mA]), h, accumulate=True)
            # Case B (Tobit right-censored saturation)
            mB = sat_f > 0
            if mB.any():
                l, g, h = likelihoods.case_b_nll_grad_hess(
                    z_f[mB], ln_y[mB], mu_f[mB], tau2_f[mB],
                    torch.sqrt(torch.clamp(tau2_f[mB], min=1e-12)))
                nll.index_put_((b_i[mB], t_i[mB], s_i[mB]), l, accumulate=True)
                grad.index_put_((b_i[mB], t_i[mB], s_i[mB]), g, accumulate=True)
                hess.index_put_((b_i[mB], t_i[mB], s_i[mB]), h, accumulate=True)
        # Case C (Hurdle non-detections) per (platform, state)
        for p in range(d.op.shape[1]):
            if not (d.op[:, p] > 0).any():
                continue
            op_p = d.op[:, p] > 0                       # (T,)
            det_p = det[:, :, p, :]                     # (B, T, 2)
            for s in range(2):
                miss = (~det_p[:, :, s]) & op_p.unsqueeze(0)   # (B, T)
                if not miss.any():
                    continue
                k = d.k_miss[:, p, s].view(-1, 1)       # (B, 1)
                v = d.vis[:, :, p]                      # (B, T)
                l, g, h = likelihoods.case_c_nll_grad_hess(
                    z[:, :, s], k, torch.clamp(v, 0.0, 1.0))
                nll[:, :, s] = nll[:, :, s] + torch.where(
                    miss, l, torch.zeros_like(l))
                grad[:, :, s] = grad[:, :, s] + torch.where(
                    miss, g, torch.zeros_like(g))
                hess[:, :, s] = hess[:, :, s] + torch.where(
                    miss, h, torch.zeros_like(h))
        return nll, grad, hess

    # ---- temporal AR(1) helpers -------------------------------------------
    def _delta(self, z):
        return z - self.d.mu_prior

    def _a_temp(self, idx):
        """Temporal prior main diagonal rows for cells ``idx`` (B_c, T)."""
        omega = self.d.omega[idx].view(-1, 1)
        rho = self.d.rho[idx].view(-1, 1)
        a = (1.0 + rho * rho) * omega * torch.ones(
            (len(idx), self.T), dtype=torch.float64, device=self.device)
        a[:, 0] = omega[:, 0]
        a[:, -1] = omega[:, 0]
        return a

    def _temporal_grad(self, idx, s, delta):
        """(H_temp @ delta) rows for cells idx, state s (tridiag apply)."""
        omega = self.d.omega[idx].view(-1, 1)
        rho = self.d.rho[idx].view(-1, 1)
        dl = delta[idx, :, s]                    # (B_c, T)
        g = (1.0 + rho * rho) * omega * dl
        g[:, 0] = omega[:, 0] * dl[:, 0]
        g[:, -1] = omega[:, 0] * dl[:, -1]
        off = -rho * omega
        if self.T > 1:
            g[:, 1:] = g[:, 1:] + off * dl[:, :-1]
            g[:, :-1] = g[:, :-1] + off * dl[:, 1:]
        return g

    def _j_time(self, delta):
        omega = self.d.omega.view(-1, 1, 1)
        rho = self.d.rho.view(-1, 1, 1)
        diff = delta[:, 1:, :] - rho * delta[:, :-1, :]
        j = 0.5 * omega * ((1.0 - rho * rho) * delta[:, 0:1, :] ** 2
                           + (diff ** 2).sum(dim=1, keepdim=True))
        return j.sum()

    # ---- full objective ----------------------------------------------------
    def objective(self, z):
        nll, _, _ = self._obs_terms(z)
        delta = self._delta(z)
        j_prior, _ = self._prior_grad(z, delta)
        return nll.sum() + j_prior

    def _prior_grad(self, z, delta):
        """(J_prior, grad_prior) for temporal + spatial + day-night terms."""
        d = self.d
        omega = d.omega.view(-1, 1, 1)
        rho = d.rho.view(-1, 1, 1)
        j = self._j_time(delta)
        grad = torch.zeros_like(z)
        for s in range(2):
            all_idx = torch.arange(self.B, device=self.device)
            grad[:, :, s] = self._temporal_grad(all_idx, s, delta)
        # spatial 1-ring coupling (held constant per color sweep)
        if len(d.edge_i):
            w = S.SPATIAL_ALPHA_S * omega.view(-1)
            for s in range(2):
                zi = z[d.edge_i, :, s]
                zj = z[d.edge_j, :, s]
                wi = w[d.edge_i].view(-1, 1)
                wj = w[d.edge_j].view(-1, 1)
                grad[d.edge_i, :, s] = grad[d.edge_i, :, s] + wi * (zi - zj)
                grad[d.edge_j, :, s] = grad[d.edge_j, :, s] + wj * (zj - zi)
                j = j + (0.5 * wi * (zi - zj) ** 2).sum()
        # day-night coupling (single beta_dn,b slope per pyrome)
        omega2 = d.omega.view(-1, 1)
        eta = d.eta_dn.view(-1, 1)
        beta = d.beta_dn.view(-1, 1)
        dd = delta[:, :, 0] - beta * delta[:, :, 1]
        dn = delta[:, :, 1] - beta * delta[:, :, 0]
        j = j + 0.5 * (eta * omega2 * dd ** 2).sum() \
            + 0.5 * (eta * omega2 * dn ** 2).sum()
        grad[:, :, 0] = grad[:, :, 0] + eta * omega2 * dd
        grad[:, :, 1] = grad[:, :, 1] + eta * omega2 * dn
        return j, grad

    def full_grad_hess(self, z):
        """(grad, hess_diag, obs_grad, obs_hess, J_total)."""
        nll, grad_map, hess_map = self._obs_terms(z)
        delta = self._delta(z)
        j_prior, grad_prior = self._prior_grad(z, delta)
        grad = grad_map + grad_prior
        d = self.d
        omega3 = d.omega.view(-1, 1, 1)
        eta3 = d.eta_dn.view(-1, 1, 1)
        hess_prior = (S.SPATIAL_ALPHA_S * omega3 * d.n_nbrs.view(-1, 1, 1)
                      + eta3 * omega3).expand_as(z).clone()
        return grad, hess_map + hess_prior, grad_map, hess_map, \
            float(nll.sum()) + float(j_prior)

    # ---- block-local objective delta (exact; everything else is unchanged)
    def _block_delta_j(self, z_old, z_new, idx, s):
        """Exact J(z_new) - J(z_old): only block cells' terms change."""
        d = self.d
        rows = idx
        total = 0.0
        # observation terms for the block's cells, state s
        op_mask = d.op > 0                                    # (T, P)
        det = (d.y > 0) & op_mask.unsqueeze(0).unsqueeze(-1)  # (B,T,P,2)
        for st in (0, 1):
            # obs nll change for (idx, :, :, st) -- state st rows only change
            # if st == s; other states' obs depend only on their own z.
            if st != s:
                continue
            det_s = det[rows, :, :, s]                         # (b, T, P)
            if det_s.any():
                flat = torch.nonzero(det_s)
                bi, ti, pi = flat[:, 0], flat[:, 1], flat[:, 2]
                y_f = d.y[rows][bi, ti, pi, s]
                mu_f = d.mu[rows][bi, ti, pi, s]
                tau2_f = d.tau2[rows][bi, ti, pi, s]
                sat_f = d.sat[rows][bi, ti, pi, s]
                ln_y = torch.log(torch.clamp(y_f, min=1e-12))
                l_new = torch.where(
                    sat_f > 0,
                    likelihoods.case_b_nll_grad_hess(
                        z_new[rows][bi, ti, s], ln_y, mu_f, tau2_f,
                        torch.sqrt(torch.clamp(tau2_f, min=1e-12)))[0],
                    likelihoods.case_a_nll_grad_hess(
                        z_new[rows][bi, ti, s], ln_y, mu_f, tau2_f)[0])
                l_old = torch.where(
                    sat_f > 0,
                    likelihoods.case_b_nll_grad_hess(
                        z_old[rows][bi, ti, s], ln_y, mu_f, tau2_f,
                        torch.sqrt(torch.clamp(tau2_f, min=1e-12)))[0],
                    likelihoods.case_a_nll_grad_hess(
                        z_old[rows][bi, ti, s], ln_y, mu_f, tau2_f)[0])
                total += float((l_new - l_old).sum())
            # Case C rows for the block's cells, state s
            for p in range(d.op.shape[1]):
                miss = (~det_s[:, :, p]) & op_mask[:, p].unsqueeze(0)
                if not miss.any():
                    continue
                k = d.k_miss[rows, p, s].view(-1, 1)
                v = torch.clamp(d.vis[rows][:, :, p], 0.0, 1.0)
                l_new, _, _ = likelihoods.case_c_nll_grad_hess(
                    z_new[rows, :, s], k, v)
                l_old, _, _ = likelihoods.case_c_nll_grad_hess(
                    z_old[rows, :, s], k, v)
                total += float(torch.where(miss, l_new - l_old,
                                           torch.zeros_like(l_new)).sum())
        # temporal AR(1) prior for (rows, s)
        omega = d.omega[rows].view(-1, 1)
        rho = d.rho[rows].view(-1, 1)
        for z_x in (z_old, z_new):
            dl = z_x[rows, :, s] - d.mu_prior[rows, :, s]
            diff = dl[:, 1:] - rho * dl[:, :-1]
            jt = 0.5 * omega * ((1.0 - rho * rho) * dl[:, 0:1] ** 2
                                + (diff ** 2).sum(dim=1, keepdim=True))
            if z_x is z_old:
                total -= float(jt.sum())
            else:
                total += float(jt.sum())
        # spatial edges touching the block's cells (state s)
        if len(d.edge_i):
            touches_i = torch.isin(d.edge_i, rows)
            touches_j = torch.isin(d.edge_j, rows)
            m = touches_i | touches_j
            if m.any():
                ei = d.edge_i[m]
                ej = d.edge_j[m]
                w = S.SPATIAL_ALPHA_S * d.omega[ei].view(-1, 1)
                do = z_old[ei, :, s] - z_old[ej, :, s]
                dn_ = z_new[ei, :, s] - z_new[ej, :, s]
                total += float((0.5 * w * (dn_ ** 2 - do ** 2)).sum())
        # day-night coupling for (rows, both directions)
        eta = d.eta_dn[rows].view(-1, 1)
        beta = d.beta_dn[rows].view(-1, 1)
        omega2 = d.omega[rows].view(-1, 1)
        for st in (0, 1):
            do_ = z_old[rows, :, st] - d.mu_prior[rows, :, st]
            dn_ = z_new[rows, :, st] - d.mu_prior[rows, :, st]
            opp_o = z_old[rows, :, 1 - st] - d.mu_prior[rows, :, 1 - st]
            opp_n = z_new[rows, :, 1 - st] - d.mu_prior[rows, :, 1 - st]
            dd_o = do_ - beta * opp_o
            dd_n = dn_ - beta * opp_n
            total += float((0.5 * eta * omega2 * (dd_n ** 2 - dd_o ** 2)).sum())
        return total

    # ---- Newton block ------------------------------------------------------
    def _block_system(self, color, s, t_lo):
        """Build the (a, b, g) tridiagonal system for a (color, state) block.

        Windowed re-solves (``t_lo > 0``) anchor row ``t_lo`` at the saved
        ``t_lo - 1`` state via the AR(1) prior.
        """
        d = self.d
        mask = d.colors == color
        idx = torch.nonzero(mask).ravel()
        if len(idx) == 0:
            return None
        grad, hess, obs_grad, obs_hess, _J = self.full_grad_hess(self.z)
        omega = d.omega[idx].view(-1, 1)
        rho = d.rho[idx].view(-1, 1)
        W = self.T - t_lo
        a = (1.0 + rho * rho) * omega * torch.ones(
            (len(idx), W), dtype=torch.float64, device=self.device)
        a[:, 0] = omega[:, 0]               # anchor row (or T edge)
        a[:, -1] = omega[:, 0] if t_lo + W >= self.T else a[:, -1]
        a = a + (S.SPATIAL_ALPHA_S * omega * d.n_nbrs[idx].view(-1, 1)
                 + d.eta_dn[idx].view(-1, 1) * omega)
        a = a + hess[idx, t_lo:, s]
        a = torch.clamp(a, min=S.PRIOR_DIAG_FLOOR)
        b = -rho * omega * torch.ones(
            (len(idx), max(W - 1, 1)), dtype=torch.float64, device=self.device)
        g = grad[idx, t_lo:, s].clone()
        if t_lo > 0:
            # replace the temporal part of row 0 with the anchored gradient
            delta = self._delta(self.z)
            delta_prev = delta[idx, t_lo - 1, s].view(-1)
            delta0 = delta[idx, t_lo, s].view(-1)
            temp_grad_full0 = self._temporal_grad(idx, s, delta)[:, 0]
            g[:, 0] = g[:, 0] - temp_grad_full0 + omega[:, 0] \
                * (delta0 - rho[:, 0] * delta_prev)
        return idx, a, b, g

    def _solve_block(self, a, b, g):
        """Thomas LDL^T solve with the fast CPU path (numpy, exact parity
        with the pure-torch reference; CUDA kernel when use_cuda)."""
        if self.use_cuda:
            dz, _l, _d = cuda_kernels.thomas_solve_batched(
                a, b, g, use_cuda=True)
            return dz
        if self.device == "cpu":
            return torch.as_tensor(
                tt.thomas_factor_and_solve_numpy(
                    a.numpy(), b.numpy(), g.numpy())[0])
        return tt.thomas_factor_and_solve(a, b, g)[0]

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
                # exact block-local objective delta (Armijo)
                dj = self._block_delta_j(z_old, z_new, idx, s)
                if dj <= -S.ARMIJO_C1 * alpha * gdz:
                    self.z = z_new
                    accepted = True
                    break
            if not accepted:
                return
            if float(torch.max(torch.abs(alpha * dz))) < S.NEWTON_TOL:
                return
            built = self._block_system(color, s, t_lo)
            if built is None:
                return
            idx, a, b, g = built

    # ---- sweeps ------------------------------------------------------------
    def solve(self, sweeps: int = S.SOLVER_SWEEPS):
        for _sw in range(sweeps):
            for color in (0, 1, 2):
                for s in (0, 1):
                    self._newton_block(color, s)
        return self.z

    def solve_fixed_lag(self, window_days: int = 14):
        """Re-solve the trailing window anchored at t_start - 1 (AR(1))."""
        t_start = max(0, self.T - window_days)
        for _sw in range(S.SOLVER_SWEEPS):
            for color in (0, 1, 2):
                for s in (0, 1):
                    self._newton_block(color, s, t_lo=t_start)
        return self.z

    # ---- posterior ---------------------------------------------------------
    def posterior(self, m_samples: int = S.MC_SAMPLES, sub_batch: int = 64):
        """Takahashi marginal variances + M joint trajectories per chain."""
        d = self.d
        var = torch.zeros_like(self.z)
        chains = []
        grad, hess, _, _, _J = self.full_grad_hess(self.z)
        for s in range(2):
            for color in (0, 1, 2):
                mask = d.colors == color
                idx = torch.nonzero(mask).ravel()
                if len(idx) == 0:
                    continue
                omega = d.omega[idx].view(-1, 1)
                rho = d.rho[idx].view(-1, 1)
                a = (1.0 + rho * rho) * omega * torch.ones(
                    (len(idx), self.T), dtype=torch.float64, device=self.device)
                a[:, 0] = omega[:, 0]
                a[:, -1] = omega[:, 0]
                a = a + (S.SPATIAL_ALPHA_S * omega * d.n_nbrs[idx].view(-1, 1)
                         + d.eta_dn[idx].view(-1, 1) * omega)
                a = a + hess[idx, :, s]
                a = torch.clamp(a, min=S.PRIOR_DIAG_FLOOR)
                b = -rho * omega * torch.ones(
                    (len(idx), max(self.T - 1, 1)), dtype=torch.float64,
                    device=self.device)
                l, dd = tt.thomas_factor(a, b)
                sig_d, _ = cuda_kernels.takahashi_batched(
                    l, dd, use_cuda=self.use_cuda)
                var[idx, :, s] = sig_d
                chains.append((idx, l, dd, s))
        monthly_rows = {}
        daily_rows = []
        for idx, l, dd, s in chains:
            for lo in range(0, len(idx), sub_batch):
                sel = idx[lo:lo + sub_batch]
                eps = torch.randn((len(sel), self.T, m_samples),
                                  dtype=torch.float64, generator=self._rand)
                eps = eps.to(self.device)
                u = cuda_kernels.cholesky_sample_batched(
                    l[lo:lo + sub_batch], dd[lo:lo + sub_batch], eps,
                    use_cuda=self.use_cuda)
                zm = torch.clamp(self.z[sel, :, s].unsqueeze(-1) + u,
                                 -8.0, 14.0)
                phim = torch.exp(zm).cpu().numpy()          # (b, T, M)
                for bi, cell_i in enumerate(sel.tolist()):
                    monthly_rows[(int(d.cells[cell_i]), s)] = phim[bi]
                    daily_rows.append((int(d.cells[cell_i]), s, phim[bi]))
        return var, monthly_rows, daily_rows


# ---------------------------------------------------------------------------
# Trajectory aggregation
# ---------------------------------------------------------------------------
def aggregate_trajectories(cells, t_days, monthly_rows):
    """Per-(cell, state) Phi trajectories -> monthly + daily quantile frames."""
    dates = pd.DatetimeIndex([T0 + pd.Timedelta(days=t) for t in range(t_days)])
    months = dates.month.to_numpy()
    years = dates.year.to_numpy()
    rows = []
    daily = []
    for (cell, s), phi in monthly_rows.items():
        for y in np.unique(years):
            for m in range(1, 13):
                sel = (years == y) & (months == m)
                if not sel.any():
                    continue
                vals = phi[sel].mean(axis=0)
                rows.append({"h3_index": int(cell), "dn": DN_STATES[s],
                             "year": int(y), "month": int(m),
                             "q05": float(np.quantile(vals, 0.05)),
                             "q50": float(np.quantile(vals, 0.50)),
                             "q95": float(np.quantile(vals, 0.95)),
                             "mean": float(vals.mean())})
        for t in range(t_days):
            vals = phi[t]
            daily.append({"h3_index": int(cell), "dn": DN_STATES[s], "t": t,
                          "q05": float(np.quantile(vals, 0.05)),
                          "q50": float(np.quantile(vals, 0.50)),
                          "q95": float(np.quantile(vals, 0.95)),
                          "mean": float(vals.mean())})
    return pd.DataFrame(rows), pd.DataFrame(daily)


def combine_states_to_products(monthly, daily):
    """Fuse Day/Night state quantiles into Phi_24h / FRE monthly + daily.

    ``Phi_24h = 0.5 (Phi_Day + Phi_Night)`` per readme 5.3.4; quantiles of
    the half-sum are combined from the per-state trajectory quantiles (the
    two half-day chains are conditionally independent given the MAP block
    structure).
    """
    out = []
    if len(monthly):
        dm = monthly[monthly["dn"] == "D"].set_index(["h3_index", "year", "month"])
        nm = monthly[monthly["dn"] == "N"].set_index(["h3_index", "year", "month"])
        for key in dm.index.intersection(nm.index):
            dr = dm.loc[key]
            nr = nm.loc[key]
            phi_d = 0.5 * (dr["q50"] + nr["q50"])
            out.append({"h3_index": key[0], "year": key[1], "month": key[2],
                        "phi_day_mw": float(dr["q50"]),
                        "phi_night_mw": float(nr["q50"]),
                        "phi_q05_mw": max(0.5 * (dr["q05"] + nr["q05"]), 0.0),
                        "phi_q50_mw": float(phi_d),
                        "phi_q95_mw": float(0.5 * (dr["q95"] + nr["q95"])),
                        "phi_mean_mw": float(0.5 * (dr["mean"] + nr["mean"])),
                        "fre_tj_q50": S.FRE_TJ_PER_MW_DAY * float(phi_d)})
    daily_out = []
    if len(daily):
        dd_ = daily[daily["dn"] == "D"].set_index(["h3_index", "t"])
        dn_ = daily[daily["dn"] == "N"].set_index(["h3_index", "t"])
        for key in dd_.index.intersection(dn_.index):
            dr = dd_.loc[key]
            nr = dn_.loc[key]
            phi_d = 0.5 * (dr["q50"] + nr["q50"])
            daily_out.append({"h3_index": key[0], "t": int(key[1]),
                              "phi_day_mw": float(dr["q50"]),
                              "phi_night_mw": float(nr["q50"]),
                              "phi_q05_mw": max(0.5 * (dr["q05"] + nr["q05"]), 0.0),
                              "phi_q50_mw": float(phi_d),
                              "phi_q95_mw": float(0.5 * (dr["q95"] + nr["q95"])),
                              "phi_mean_mw": float(0.5 * (dr["mean"] + nr["mean"])),
                              "fre_tj_q50": S.FRE_TJ_PER_MW_DAY * float(phi_d)})
    return pd.DataFrame(out), pd.DataFrame(daily_out)


# ---------------------------------------------------------------------------
# Chunk orchestration
# ---------------------------------------------------------------------------
def estimate_chunk_vram_bytes(batch_size: int = S.BATCH_SIZE,
                              t_days: int = S.T_DAYS,
                              n_platforms: int = S.N_PLATFORMS,
                              m_samples: int = S.MC_SAMPLES,
                              sub_batch: int = 64) -> int:
    """Analytic peak-memory estimate for one GPU chunk (float64 bytes)."""
    obs = 4 * batch_size * t_days * n_platforms * 2 * 8      # y, sat, mu, tau2
    vis = batch_size * t_days * n_platforms * 8
    state = 6 * batch_size * t_days * 2 * 8                  # z, mu, grad, hess, var, delta
    mc = 2 * sub_batch * t_days * m_samples * 8              # eps + u
    transient = batch_size * t_days * n_platforms * 2 * 8    # largest temp
    return obs + vis + state + mc + transient


def solve_chunk(data: SolverData, sweeps: int = S.SOLVER_SWEEPS,
                use_cuda: bool | None = None, device: str = "cpu"):
    solver = GMRFSolver(data, use_cuda=use_cuda, device=device)
    z = solver.solve(sweeps)
    var, monthly_rows, _daily_rows = solver.posterior()
    monthly, daily = aggregate_trajectories(data.cells, data.t_days,
                                            monthly_rows)
    return {"z_map": z.cpu().numpy(), "var": var.cpu().numpy(),
            "monthly": monthly, "daily": daily, "solver": solver}


def solve_all_chunks(pass_table, active_cells, hex_pyromes, priors,
                     artifacts, cloud_df, t_days: int = S.T_DAYS,
                     platform_active=None, platform_mask=None,
                     batch_size: int = S.BATCH_SIZE, paths=None):
    """Distribute active cells across GPUs (batch_size = 1000, < 6 GB VRAM)."""
    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells))),
                     dtype=np.int64)
    results = []
    try:
        n_devices = max(1, torch.cuda.device_count()) \
            if torch.cuda.is_available() else 1
    except Exception:  # noqa: BLE001
        n_devices = 1
    chunks = [cells[i:i + batch_size] for i in range(0, len(cells), batch_size)]
    for ci, chunk in enumerate(chunks):
        data = build_observation_tensors(
            pass_table, chunk, hex_pyromes, priors, artifacts, cloud_df,
            t_days=t_days, platform_active=platform_active,
            platform_mask=platform_mask)
        device, use_cuda = "cpu", False
        if torch.cuda.is_available():
            device, use_cuda = f"cuda:{ci % n_devices}", True
        results.append(solve_chunk(data, use_cuda=use_cuda, device=device))
    z_map = np.concatenate([r["z_map"] for r in results], axis=0)
    var = np.concatenate([r["var"] for r in results], axis=0)
    monthly = pd.concat([r["monthly"] for r in results], ignore_index=True)
    daily = pd.concat([r["daily"] for r in results], ignore_index=True)
    if paths is not None:
        torch.save({"z_map": z_map, "var": var, "cells": cells},
                   paths.state_checkpoint_pt)
    return {"cells": cells, "z_map": z_map, "var": var, "monthly": monthly,
            "daily": daily, "results": results}


def run_fixed_lag_update(solver: GMRFSolver, window_days: int = 14):
    """Offline-callable trailing-window re-solve (readme D11/D12)."""
    return solver.solve_fixed_lag(window_days)
