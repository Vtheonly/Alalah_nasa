"""Exact likelihood terms, analytical gradients, convexity-guarded Hessians.

Contract (readme Sections 4.8, 5.3.2) -- pure PyTorch ``torch.float64``
vectorized negative log-likelihoods for the three likelihood cases:

* **Case A** (valid unsaturated detection, ``Y > 0``, ``sat_flag == 0``):
  ``r = (z + mu) - ln Y``; ``l = 0.5 tau2 r^2``; ``g = tau2 r``; ``h = tau2``.
* **Case B** (right-censored Tobit saturation, ``Y > 0``, ``sat_flag == 1``):
  ``w = tau1 ((z + mu) - ln Y)``; for ``w >= -30`` uses
  ``torch.special.log_ndtr`` and the Mills ratio
  ``lambda(w) = exp(-0.5 w^2 - 0.5 ln(2 pi) - ln Phi(w))``; for ``w < -30``
  the exact asymptotic expansion ``-ln Phi(w) = 0.5 w^2 + ln(-w) +
  0.5 ln(2 pi)``, ``lambda(w) = -w - 1/w`` (no NaN/Inf down to w = -35+).
  ``l = -ln Phi(w)``; ``g = -tau1 lambda(w)``;
  ``h = tau2 lambda(w)(w + lambda(w)) > 0``.
* **Case C** (Hurdle zero-detection on an active operational pass):
  ``k = lambda0_b psi_CNT / F_event_b``; ``Lambda = clip(k e^z, 1e-12, 80)``;
  ``q = (1-v) + v e^{-Lambda}``; ``r = v e^{-Lambda} / q``;
  ``l = -ln q``; ``g = r Lambda``; convexity-guarded Gauss-Newton Hessian
  ``h = max(0, r Lambda (1 - Lambda(1-r)))`` (SPD guarantee).
"""
from __future__ import annotations

import math

import torch

_HALF_LN_2PI = 0.5 * math.log(2.0 * math.pi)


def eval_pointwise_nll_grad_hess(z, obs_y, obs_sat, obs_mu_gain, obs_inv_sig2,
                                 obs_inv_sig, obs_k_miss, obs_vis, is_op_pass):
    """Vectorized pointwise (nll, grad, hess_spd) for mixed observations.

    All inputs are ``torch.float64`` tensors of identical shape (..., T).
    Non-operational passes (``is_op_pass == 0``) contribute exactly zero.
    """
    z = torch.as_tensor(z, dtype=torch.float64)
    y = torch.as_tensor(obs_y, dtype=torch.float64)
    sat = torch.as_tensor(obs_sat, dtype=torch.float64)
    mu = torch.as_tensor(obs_mu_gain, dtype=torch.float64)
    tau2 = torch.as_tensor(obs_inv_sig2, dtype=torch.float64)
    tau1 = torch.as_tensor(obs_inv_sig, dtype=torch.float64)
    k_miss = torch.as_tensor(obs_k_miss, dtype=torch.float64)
    vis = torch.as_tensor(obs_vis, dtype=torch.float64)
    active = torch.as_tensor(is_op_pass, dtype=torch.float64)

    nll = torch.zeros_like(z)
    grad = torch.zeros_like(z)
    hess = torch.zeros_like(z)

    # ---- detection mask ---------------------------------------------------
    det = (y > 0) & (active > 0)
    miss = (y <= 0) & (active > 0)

    # ============================ Case A ====================================
    if det.any():
        ln_y = torch.where(det, torch.log(torch.clamp(y, min=1e-12)),
                           torch.ones_like(y))
        r_a = (z + mu) - ln_y
        l_a = 0.5 * tau2 * r_a * r_a
        g_a = tau2 * r_a
        h_a = tau2
        m = det & (sat <= 0)
        nll = nll + torch.where(m, l_a, torch.zeros_like(l_a))
        grad = grad + torch.where(m, g_a, torch.zeros_like(g_a))
        hess = hess + torch.where(m, torch.full_like(h_a, 0.0) + h_a,
                                  torch.zeros_like(h_a))

    # ============================ Case B (Tobit) ============================
    ms = det & (sat > 0)
    if ms.any():
        ln_y = torch.where(ms, torch.log(torch.clamp(y, min=1e-12)),
                           torch.ones_like(y))
        w = tau1 * ((z + mu) - ln_y)
        # exact branch w >= -30 (log_ndtr is finite down to ~ -37 in float64,
        # but the spec's asymptotic branch below -30 is exact and cheaper)
        log_phi_w = torch.where(w >= -30.0,
                                torch.special.log_ndtr(torch.clamp(w, min=-30.0)),
                                torch.zeros_like(w))
        lambda_hi = torch.exp(-0.5 * w * w - _HALF_LN_2PI - log_phi_w)
        # asymptotic branch w < -30: -ln Phi(w) = .5 w^2 + ln(-w) + .5 ln(2pi)
        neg_ln_phi_lo = 0.5 * w * w + torch.log(torch.clamp(-w, min=1e-300)) \
            + _HALF_LN_2PI
        # lambda(w) = -w - 1/w exactly as specified (w < -30 here, so 1/w
        # is finite and negative -> the correction ADDS to -w)
        lam_lo = -w - 1.0 / w
        l_lo = neg_ln_phi_lo
        use_lo = w < -30.0
        l_b = torch.where(use_lo, l_lo, -log_phi_w)
        lam = torch.where(use_lo, lam_lo, lambda_hi)
        # guard lambda_hi against exp overflow in the far right tail
        lam = torch.where(torch.isfinite(lam), lam, torch.zeros_like(lam))
        g_b = -tau1 * lam
        h_b = tau2 * lam * (w + lam)
        h_b = torch.where(torch.isfinite(h_b) & (h_b > 0.0), h_b,
                          torch.zeros_like(h_b))
        nll = nll + torch.where(ms, l_b, torch.zeros_like(l_b))
        grad = grad + torch.where(ms, g_b, torch.zeros_like(g_b))
        hess = hess + torch.where(ms, h_b, torch.zeros_like(h_b))

    # ============================ Case C (Hurdle) ===========================
    if miss.any():
        lam = torch.clamp(k_miss * torch.exp(z), 1e-12, 80.0)
        exp_neg = torch.exp(-lam)
        q = (1.0 - vis) + vis * exp_neg
        q = torch.clamp(q, 1e-300, None)
        r_post = vis * exp_neg / q
        l_c = -torch.log(q)
        g_c = r_post * lam
        h_exact = r_post * lam * (1.0 - lam * (1.0 - r_post))
        # Gauss-Newton convexity guard (SPD guarantee): max(0, h_exact)
        h_c = torch.clamp(h_exact, min=0.0)
        nll = nll + torch.where(miss, l_c, torch.zeros_like(l_c))
        grad = grad + torch.where(miss, g_c, torch.zeros_like(g_c))
        hess = hess + torch.where(miss, h_c, torch.zeros_like(h_c))

    # zero contribution for non-operational passes
    nll = nll * active
    grad = grad * active
    hess = hess * active
    return nll, grad, hess


# ---------------------------------------------------------------------------
# Pure per-case functions (unit-testable without masks)
# ---------------------------------------------------------------------------
def case_a_nll_grad_hess(z, ln_y, mu, inv_sig2):
    r = (z + mu) - ln_y
    return 0.5 * inv_sig2 * r * r, inv_sig2 * r, inv_sig2


def case_b_nll_grad_hess(z, ln_y, mu, inv_sig2, inv_sig):
    """Tobit right-censored: exact asymptotic tail for w < -30."""
    w = inv_sig * ((z + mu) - ln_y)
    use_lo = w < -30.0
    log_phi_hi = torch.special.log_ndtr(torch.clamp(w, min=-30.0))
    lam_hi = torch.exp(-0.5 * w * w - _HALF_LN_2PI - log_phi_hi)
    neg_ln_phi_lo = 0.5 * w * w + torch.log(torch.clamp(-w, min=1e-300)) \
        + _HALF_LN_2PI
    lam_lo = -w - 1.0 / w      # w < -30 here: 1/w finite and negative
    l = torch.where(use_lo, neg_ln_phi_lo, -log_phi_hi)
    lam = torch.where(use_lo, lam_lo, lam_hi)
    lam = torch.where(torch.isfinite(lam), lam, torch.zeros_like(lam))
    g = -inv_sig * lam
    h = inv_sig2 * lam * (w + lam)
    h = torch.where(torch.isfinite(h) & (h > 0.0), h, torch.zeros_like(h))
    return l, g, h


def case_c_nll_grad_hess(z, k_miss, vis):
    """Hurdle zero-detection Poisson-cloud mixture with convexity guard."""
    lam = torch.clamp(k_miss * torch.exp(z), 1e-12, 80.0)
    exp_neg = torch.exp(-lam)
    q = torch.clamp((1.0 - vis) + vis * exp_neg, 1e-300, None)
    r_post = vis * exp_neg / q
    l = -torch.log(q)
    g = r_post * lam
    h_exact = r_post * lam * (1.0 - lam * (1.0 - r_post))
    h = torch.clamp(h_exact, min=0.0)
    return l, g, h


def hurdle_lambda(k_miss, z):
    """Lambda(z) = clip(k e^z, 1e-12, 80)."""
    return torch.clamp(k_miss * torch.exp(torch.as_tensor(z, dtype=torch.float64)),
                       1e-12, 80.0)
