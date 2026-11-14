"""Pure PyTorch float64 likelihoods with exact derivatives (readme.md 5.3.2).

Three pointwise terms per operational platform pass ``(p, g, t, DN)``:

* **Case A — Valid unsaturated detection** (``Y > 0``, ``sat_flag == 0``):
  Gaussian log-FRP likelihood with exact gradient ``tau2 * r`` and constant
  Hessian ``tau2``.
* **Case B — Right-censored Tobit saturation** (``Y > 0``, ``sat_flag == 1``):
  ``log_ndtr`` plus an exact asymptotic expansion for ``w < -30`` avoiding
  ``log 0``; gradient ``-tau1 * lambda(w)``, Hessian
  ``tau2 * lambda(w) * (w + lambda(w)) > 0``.
* **Case C — Hurdle zero-detection** (``Y == 0`` on an active operational
  pass): cloud-gated Poisson-cloud mixture with convexity-guarded Hessian
  ``h = max(0, r * Lambda * (1 - Lambda * (1 - r)))`` guaranteeing global
  SPD Newton systems.

Cloud visibility ``v`` NEVER enters Case A/B (INV-05).
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.config import settings as S

_LOG_SQRT_2PI = 0.5 * torch.log(torch.tensor(2.0 * torch.pi)).item()


def _as_f64(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float64) if x.dtype != torch.float64 else x


# --------------------------------------------------------------------------- #
# Case A — Gaussian log-FRP observation likelihood
# --------------------------------------------------------------------------- #
def obs_nll_grad_hess(
    z: torch.Tensor,
    ln_y: torch.Tensor,
    mu_gain: torch.Tensor,
    inv_sig2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Case A: r = (z + mu) - ln Y; l = 0.5*tau2*r^2; g = tau2*r; h = tau2."""
    r = (z + mu_gain) - ln_y
    nll = 0.5 * inv_sig2 * r * r
    grad = inv_sig2 * r
    hess = inv_sig2.expand_as(r).clone()
    return nll, grad, hess


# --------------------------------------------------------------------------- #
# Case B — Tobit right-censored saturation likelihood
# --------------------------------------------------------------------------- #
def tobit_log_phi_and_lambda(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (log Phi(w), Mills ratio lambda(w) = phi(w)/Phi(w)).

    For ``w >= -30`` uses ``torch.special.log_ndtr``; for ``w < -30`` applies
    the exact asymptotic expansion
    ``-log Phi(w) = 0.5 w^2 + log(-w) + 0.5 log(2 pi)`` and
    ``lambda(w) = -w - 1/w`` (no NaN/Inf down to arbitrarily negative w).
    """
    w = _as_f64(w)
    tail = w < S.TOBIT_TAIL_W
    safe_w = torch.where(tail, torch.full_like(w, S.TOBIT_TAIL_W), w)

    log_phi_main = torch.special.log_ndtr(safe_w)
    lam_main = torch.exp(
        -0.5 * safe_w * safe_w - _LOG_SQRT_2PI - log_phi_main
    )

    neg_log_phi_tail = (
        0.5 * w * w + torch.log(-w) + _LOG_SQRT_2PI
    )
    log_phi_tail = -neg_log_phi_tail
    lam_tail = -w - 1.0 / w

    log_phi = torch.where(tail, log_phi_tail, log_phi_main)
    lam = torch.where(tail, lam_tail, lam_main)
    return log_phi, lam


def tobit_nll_grad_hess(
    z: torch.Tensor,
    ln_y: torch.Tensor,
    mu_gain: torch.Tensor,
    inv_sig: torch.Tensor,
    inv_sig2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Case B: l = -log Phi(w), g = -tau1*lambda(w), h = tau2*lam*(w+lam) > 0."""
    w = inv_sig * ((z + mu_gain) - ln_y)
    log_phi, lam = tobit_log_phi_and_lambda(w)
    nll = -log_phi
    grad = -inv_sig * lam
    hess = inv_sig2 * lam * (w + lam)
    return nll, grad, hess


# --------------------------------------------------------------------------- #
# Case C — Hurdle cloud-gated zero-detection likelihood
# --------------------------------------------------------------------------- #
def hurdle_lambda(z: torch.Tensor, k_miss: torch.Tensor) -> torch.Tensor:
    """Lambda = clip(k * e^z, 1e-12, 80)."""
    return torch.clamp(k_miss * torch.exp(z), *S.LAMBDA_CLIP)


def hurdle_nll_grad_hess(
    z: torch.Tensor,
    k_miss: torch.Tensor,
    vis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Case C: q = (1-v) + v e^-Lambda; l = -ln q; g = r*Lambda; h = max(0, ...).

    The exact gradient ``g = r * Lambda`` is preserved while the Hessian
    applies the positive Gauss-Newton projection
    ``h = max(0, r * Lambda * [1 - Lambda * (1 - r)])`` so the Newton system
    stays symmetric positive definite even in the partial-cloud transition
    region (v < 1, Lambda in (1, 4)) where the raw second derivative turns
    slightly negative (readme.md 5.3.2 item 3).
    """
    lam = hurdle_lambda(z, k_miss)
    exp_neg = torch.exp(-lam)
    q = (1.0 - vis) + vis * exp_neg
    q = torch.clamp(q, min=1e-300)
    nll = -torch.log(q)
    r = vis * exp_neg / q
    grad = r * lam
    raw_hess = r * lam * (1.0 - lam * (1.0 - r))
    hess = torch.clamp(raw_hess, min=0.0)
    return nll, grad, hess


# --------------------------------------------------------------------------- #
# Combined pointwise evaluator (likelihoods.py module contract)
# --------------------------------------------------------------------------- #
def eval_pointwise_nll_grad_hess(
    z: torch.Tensor,
    obs_y: torch.Tensor,
    obs_sat: torch.Tensor,
    obs_mu_gain: torch.Tensor,
    obs_inv_sig2: torch.Tensor,
    obs_inv_sig: torch.Tensor,
    obs_k_miss: torch.Tensor,
    obs_vis: torch.Tensor,
    is_op_pass: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch every elementwise pass entry to Case A / B / C.

    Shapes are broadcast-compatible elementwise tensors (typically
    ``(B, T, P)`` chunks). Entries with ``is_op_pass == 0`` (platform not
    operational or spacecraft in safe-mode outage) contribute exactly zero
    to nll, gradient, and Hessian — an outage is never treated as
    zero-fire evidence (INV-04).
    """
    z = _as_f64(z)
    obs_y = _as_f64(obs_y)
    obs_mu_gain = _as_f64(obs_mu_gain)
    obs_inv_sig2 = _as_f64(obs_inv_sig2)
    obs_inv_sig = _as_f64(obs_inv_sig)
    obs_k_miss = _as_f64(obs_k_miss)
    obs_vis = _as_f64(obs_vis)
    op = _as_f64(is_op_pass)

    detected = (obs_y > 0) & (op > 0)
    saturated = detected & (obs_sat > 0)
    unsaturated = detected & ~(obs_sat > 0)
    nondetection = (~detected) & (op > 0)

    ln_y = torch.where(
        detected, torch.log(torch.clamp(obs_y, min=1e-300)),
        torch.zeros_like(obs_y),
    )

    nll_a, g_a, h_a = obs_nll_grad_hess(
        z, ln_y, obs_mu_gain, obs_inv_sig2
    )
    nll_b, g_b, h_b = tobit_nll_grad_hess(
        z, ln_y, obs_mu_gain, obs_inv_sig, obs_inv_sig2
    )
    nll_c, g_c, h_c = hurdle_nll_grad_hess(z, obs_k_miss, obs_vis)

    nll = (
        unsaturated * nll_a + saturated * nll_b + nondetection * nll_c
    )
    grad = (
        unsaturated * g_a + saturated * g_b + nondetection * g_c
    )
    hess = (
        unsaturated * h_a + saturated * h_b + nondetection * h_c
    )
    return nll, grad, hess
