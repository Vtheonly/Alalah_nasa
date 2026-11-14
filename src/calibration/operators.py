"""Pre-drift observation operator calibration (2012-01-20 .. 2021-12-31).

Contract (readme.md Sections 4 & 5.2):
  * Isotonic confidence reliability ``w_i in [0.15, 1.0]`` per
    (sensor_family, dn_flag) from empirical corroboration.
  * Pixel-area detection limits ``F_min,p(A, DN)`` with physical anchors and
    left-truncated Log-Normal cluster MLE ``(mu_F,b, sigma_F,b)`` yielding
    closed-form survival integrals ``psi_CNT``, ``psi_FRP`` and the
    conditional multi-fire blend ``psi_FRP_eff``.
  * Near-nadir VIIRS-referenced VZA & quantile calibration
    ``gamma_MODIS,b`` (LightGBM residual + 20-knot strictly monotonic EQM,
    ln gamma clipped to [-1.5, 1.5], ln gamma_VIIRS == 0).
  * 12-hour intra-window 2-harmonic diurnal curves ``D_{b,s,DN}(tau)``
    (Ridge alpha=10, 120-point trapezoidal unit-mean normalization inside
    Day [06-18) and Night [18-06)), including post-2022 Aqua-vs-NOAA-20
    drift fitted strictly on the diurnal harmonics.
  * Heteroscedastic observation variance ``sigma^2 in [0.025, 2.50]``.
  * Active burning footprint transfer ``g_area,b`` (Ridge).
  * 2012-2025 cloud climatology for trailing-2026 fallback.
  * Emits ``calibration_artifacts.pkl`` (exact schema) and
    ``pass_observations_calibrated.parquet``.

Cloud visibility NEVER enters the detected-observation gain (INV-05).
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge

from src.config import settings as S
from src.geometry.solar_geometry import modis_swath_probability

SEASONS = ("DJF", "MAM", "JJA", "SON")


def season_of_month(month: int) -> str:
    return SEASONS[(month % 12) // 3]


# --------------------------------------------------------------------------- #
# Confidence reliability (isotonic corroboration)
# --------------------------------------------------------------------------- #
def label_corroboration(
    prov: pd.DataFrame, window_days: int = S.CORROBORATION_WINDOW_DAYS,
    radius_km: float = S.CORROBORATION_RADIUS_KM,
) -> pd.Series:
    """Empirical corroboration: another detection within +/-1 day and 2 km."""
    prov = prov.reset_index(drop=True)
    corroborated = np.zeros(len(prov), dtype=bool)
    lats = prov["latitude"].to_numpy(dtype=float)
    lons = prov["longitude"].to_numpy(dtype=float)
    dates = pd.to_datetime(prov["solar_date"]).to_numpy()
    keys = prov["solar_date"].to_numpy()
    order = np.argsort(dates)
    for i in order:
        lo, hi = i, i  # dates are sorted by position via order; brute force
        # over a small window using day differences is fine for calibration.
        for j in order:
            if i == j:
                continue
            day_gap = abs(
                (pd.Timestamp(dates[i]) - pd.Timestamp(dates[j])).days
            )
            if day_gap > window_days:
                continue
            dlat = (lats[i] - lats[j]) * 111.32
            dlon = (lons[i] - lons[j]) * 111.32 * np.cos(
                np.deg2rad(0.5 * (lats[i] + lats[j]))
            )
            if np.hypot(dlat, dlon) <= radius_km:
                corroborated[i] = True
                break
    return pd.Series(corroborated, index=prov.index)


def fit_reliability_models(
    prov: pd.DataFrame,
) -> Dict[Tuple[str, str], IsotonicRegression]:
    """One isotonic reliability curve per (sensor_family, dn_flag)."""
    prov = prov.reset_index(drop=True)
    prov = prov.assign(
        corroborated=label_corroboration(prov).astype(float)
    )
    models: Dict[Tuple[str, str], IsotonicRegression] = {}
    for (fam, dn), part in prov.groupby(["sensor_family", "dn_flag"]):
        x = part["conf_num"].to_numpy(dtype=float)
        y = part["corroborated"].to_numpy(dtype=float)
        if len(x) < 3 or len(np.unique(x)) < 2:
            iso = IsotonicRegression(
                y_min=S.RELIABILITY_CLIP[0],
                y_max=S.RELIABILITY_CLIP[1],
                out_of_bounds="clip",
            )
            iso.fit([0.0, 100.0], [0.6, 0.9])
        else:
            iso = IsotonicRegression(
                y_min=S.RELIABILITY_CLIP[0],
                y_max=S.RELIABILITY_CLIP[1],
                out_of_bounds="clip",
            )
            iso.fit(x, y)
        models[(fam, dn)] = iso
    return models


def apply_reliability(
    iso: IsotonicRegression, conf_num: float
) -> float:
    return float(
        np.clip(iso.predict(np.array([conf_num]))[0], *S.RELIABILITY_CLIP)
    )


# --------------------------------------------------------------------------- #
# Detection limits & analytical survival integrals
# --------------------------------------------------------------------------- #
def fmin_threshold(area_km2: float, family: str, dn_flag: str) -> float:
    """F_min,p(A, DN) = alpha + beta * A with physical anchors."""
    alpha, beta = S.FMIN_PARAMS[(family.upper(), dn_flag.upper())]
    return alpha + beta * float(area_km2)


def fit_lognormal_mle(
    frp_values: np.ndarray, f_ref: float = S.F_REF_MW
) -> Tuple[float, float]:
    """Left-truncated Log-Normal MLE on clusters above f_ref (L-BFGS-B)."""
    x = np.log(np.asarray(frp_values, dtype=float))
    x = x[x >= np.log(f_ref)]
    if len(x) < 5:
        return float(np.mean(x) if len(x) else 0.3), 0.9

    def nll(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        z = (x - mu) / sigma
        log_pdf = -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        # Left truncation at f_ref: constant survival P(F > f_ref).
        trunc_z = (np.log(f_ref) - mu) / sigma
        log_surv = float(
            np.log(max(1.0 - float(ndtr(trunc_z)), 1e-300))
        )
        return float(-np.mean(log_pdf)) + log_surv

    xbar, xsd = float(x.mean()), max(float(x.std(ddof=1)), 0.05)
    res = minimize(
        nll,
        x0=np.array([xbar, np.log(xsd)]),
        method="L-BFGS-B",
        bounds=[(-5.0, 8.0), (np.log(0.05), np.log(3.0))],
    )
    mu = float(res.x[0])
    sigma = float(np.exp(res.x[1]))
    return mu, sigma


def psi_cnt(fmin: float, mu_f: float, sigma_f: float) -> float:
    """Analytical count survival: clip(1 - Phi((ln Fmin - mu)/sigma), .02, 1)."""
    z = (np.log(max(fmin, 1e-9)) - mu_f) / max(sigma_f, 1e-9)
    return float(np.clip(1.0 - ndtr(z), *S.PSI_CNT_CLIP))


def psi_frp(
    fmin: float, mu_f: float, sigma_f: float, f_ref: float = S.F_REF_MW
) -> float:
    """Analytical FRP survival ratio vs the f_ref truncation point."""
    num = 1.0 - ndtr((np.log(max(fmin, 1e-9)) - mu_f - sigma_f**2) / sigma_f)
    den = 1.0 - ndtr((np.log(f_ref) - mu_f - sigma_f**2) / sigma_f)
    den = max(float(den), 1e-12)
    return float(np.clip(num / den, *S.PSI_FRP_CLIP))


def w_multi(raw_event_count: int) -> float:
    """Conditional multi-fire adjustment weight 1 - exp(-0.7 (E - 1))."""
    e = max(int(raw_event_count), 1)
    return 1.0 - float(np.exp(-S.W_MULTI_RATE * (e - 1)))


def psi_frp_eff(
    fmin: float, mu_f: float, sigma_f: float, raw_event_count: int,
    f_ref: float = S.F_REF_MW,
) -> float:
    """Blend isolated single-cluster detections toward max(psi, 0.65)."""
    base = psi_frp(fmin, mu_f, sigma_f, f_ref)
    wm = w_multi(raw_event_count)
    return (1.0 - wm) * max(base, S.PSI_FRP_EFF_FLOOR) + wm * base


# --------------------------------------------------------------------------- #
# VZA & quantile residual transfer (gamma_MODIS)
# --------------------------------------------------------------------------- #
class GammaCalibrator:
    """LightGBM residual bias + strictly monotonic EQM per (pyrome, dn).

    ``fit`` consumes synchronous Aqua x near-nadir SNPP pairs; ``predict``
    returns the clipped log-gain ln gamma for MODIS passes (VIIRS is
    identically zero by construction).
    """

    def __init__(self) -> None:
        self.lgbm = None
        self.eqm: Dict[Tuple[int, str], PchipInterpolator] = {}
        self.eqm_knot_arrays: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.source_quantiles: Dict[Tuple[int, str], np.ndarray] = {}
        self.fitted = False

    def fit(self, pairs: pd.DataFrame) -> "GammaCalibrator":
        """``pairs`` columns: pyrome_id, dn_flag, vza_modis, area_modis,
        ln_y_modis, ln_y_snpp, ln_psi_eff_aqua, ln_psi_eff_snpp."""
        import lightgbm as lgb

        if len(pairs) < 30:
            self.fitted = False
            return self
        r = (
            pairs["ln_y_snpp"]
            - (
                pairs["ln_y_modis"]
                - pairs["ln_psi_eff_aqua"]
                + pairs["ln_psi_eff_snpp"]
            )
        ).to_numpy()
        X = np.column_stack(
            [
                pairs["pyrome_id"].to_numpy(dtype=float),
                (pairs["dn_flag"] == "D").to_numpy(dtype=float),
                pairs["vza_modis"].to_numpy(dtype=float),
                pairs["area_modis"].to_numpy(dtype=float),
                pairs["ln_y_modis"].to_numpy(dtype=float),
            ]
        )
        self.lgbm = lgb.LGBMRegressor(
            max_depth=4,
            n_estimators=100,
            min_child_samples=50,
            random_state=S.GMM_RANDOM_STATE,
            verbose=-1,
        )
        self.lgbm.fit(X, r)
        r_hat = self.lgbm.predict(X)
        adjusted = pairs["ln_y_modis"].to_numpy() - r_hat
        # 20-knot strictly monotone EQM per (pyrome, dn).
        levels = np.linspace(0.0, 1.0, S.EQM_KNOTS)
        for (b, dn), idx in pairs.groupby(["pyrome_id", "dn_flag"]).groups.items():
            rows = pairs.loc[idx]
            adj = adjusted[rows.index.to_numpy()]
            tgt = rows["ln_y_snpp"].to_numpy()
            if len(adj) < S.EQM_KNOTS:
                continue
            src_q = np.quantile(adj, levels)
            tgt_q = np.quantile(tgt, levels)
            # Enforce strictly monotone knots (never crossing quantiles).
            src_q = np.maximum.accumulate(src_q)
            tgt_q = np.maximum.accumulate(tgt_q)
            if src_q[-1] <= src_q[0] or tgt_q[-1] <= tgt_q[0]:
                continue
            self.source_quantiles[(int(b), dn)] = src_q
            self.eqm_knot_arrays[(int(b), dn)] = (src_q.copy(), tgt_q.copy())
            self.eqm[(int(b), dn)] = PchipInterpolator(src_q, tgt_q)
        self.fitted = True
        return self

    def eqm_knots_dict_for_test(self):
        """Expose the fitted EQM knots (source, target) for verification."""
        return {k: (s.copy(), t.copy()) for k, (s, t) in self.eqm_knot_arrays.items()}

    def predict_ln_gamma(
        self, pyrome_id: int, dn_flag: str, vza_deg: float,
        area_km2: float, ln_y: float,
    ) -> float:
        if not self.fitted or self.lgbm is None:
            return 0.0
        x = np.array(
            [[pyrome_id, 1.0 if dn_flag == "D" else 0.0, vza_deg,
              area_km2, ln_y]]
        )
        r_hat = float(self.lgbm.predict(x)[0])
        adjusted = ln_y - r_hat
        curve = self.eqm.get((int(pyrome_id), dn_flag))
        if curve is None:
            mapped = adjusted
        else:
            mapped = float(
                curve(np.clip(adjusted, curve.x[0], curve.x[-1]))
            )
        ln_gamma = mapped - ln_y
        return float(np.clip(ln_gamma, *S.LN_GAMMA_CLIP))


# --------------------------------------------------------------------------- #
# 12-hour intra-window diurnal curves
# --------------------------------------------------------------------------- #
def diurnal_design(tau: np.ndarray) -> np.ndarray:
    """2-harmonic Fourier design matrix evaluated at solar hours tau."""
    tau = np.asarray(tau, dtype=float)
    return np.column_stack(
        [
            np.cos(2 * np.pi * tau / 24.0),
            np.sin(2 * np.pi * tau / 24.0),
            np.cos(4 * np.pi * tau / 24.0),
            np.sin(4 * np.pi * tau / 24.0),
        ]
    )


def fit_diurnal_ridge(
    tau_a: np.ndarray,
    tau_b: np.ndarray,
    log_ratio: np.ndarray,
    alpha: float = S.DIURNAL_RIDGE_ALPHA,
) -> np.ndarray:
    """Ridge fit of 2-harmonic coefficients from paired log-FRP differences.

    The response is ``ln Y_a - ln Y_b`` between two same-cell same-day
    platforms; the design is the Fourier difference ``f(tau_a) - f(tau_b)``
    so the shared latent state cancels, isolating the diurnal operator.
    """
    X = diurnal_design(tau_a) - diurnal_design(tau_b)
    y = np.asarray(log_ratio, dtype=float)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, y)
    return ridge.coef_.astype(float)


def normalize_diurnal_coeffs(coeffs: np.ndarray) -> Dict[str, float]:
    """Unit-mean normalization inside each 12-hour half-day window.

    Integrates the raw curve over [6, 18) and [18, 30) (tau mod 24) with a
    120-point trapezoidal rule and returns the additive constants
    ``c0_day`` / ``c0_night`` such that exp(ln D_raw + c0) integrates to 1
    over each window (readme.md 5.2 item 5). The Night grid is integrated
    on the UNWRAPPED support [18, 30] while the periodic design is
    evaluated at tau mod 24, keeping the trapezoid orientation positive.
    """
    grid_day = np.linspace(6.0, 18.0, S.DIURNAL_INT_POINTS)
    grid_night_unwrapped = np.linspace(18.0, 30.0, S.DIURNAL_INT_POINTS)
    grid_night = np.mod(grid_night_unwrapped, 24.0)
    ln_d_day = diurnal_design(grid_day) @ coeffs
    ln_d_night = diurnal_design(grid_night) @ coeffs
    mean_day = float(np.trapezoid(np.exp(ln_d_day), grid_day) / 12.0)
    mean_night = float(
        np.trapezoid(np.exp(ln_d_night), grid_night_unwrapped) / 12.0
    )
    c0_day = -float(np.log(max(mean_day, 1e-12)))
    c0_night = -float(np.log(max(mean_night, 1e-12)))
    return {"coeffs": coeffs, "c0_day": c0_day, "c0_night": c0_night}


def diurnal_value(norm: Dict[str, float], tau: float, dn_flag: str) -> float:
    """D_{b,s,DN}(tau) with exact unit mean over its 12-hour window."""
    x = diurnal_design(np.array([float(tau)]))[0]
    ln_raw = float(x @ norm["coeffs"])
    c0 = norm["c0_day"] if dn_flag == "D" else norm["c0_night"]
    return float(np.exp(ln_raw + c0))


def diurnal_window_integral(norm: Dict[str, float], dn_flag: str) -> float:
    """Numerical 12-hour integral of D (used by unit-mean tests)."""
    if dn_flag == "D":
        grid = np.linspace(6.0, 18.0, S.DIURNAL_INT_POINTS)
        wrapped = grid
    else:
        grid = np.linspace(18.0, 30.0, S.DIURNAL_INT_POINTS)
        wrapped = np.mod(grid, 24.0)
    vals = np.array(
        [diurnal_value(norm, float(t), dn_flag) for t in wrapped]
    )
    return float(np.trapezoid(vals, grid) / 12.0)


# --------------------------------------------------------------------------- #
# Heteroscedastic observation variance
# --------------------------------------------------------------------------- #
def fit_sigma_ols(
    is_modis: np.ndarray, vza_deg: np.ndarray, w_mean: np.ndarray,
    v_cloud: np.ndarray, i4_fallback: np.ndarray, resid_ln: np.ndarray,
) -> LinearRegression:
    """OLS predicting ln(eps^2 + 1e-3) from the mandated feature block."""
    X = np.column_stack(
        [
            np.asarray(is_modis, dtype=float),
            np.asarray(vza_deg, dtype=float) / 65.0,
            1.0 - np.asarray(w_mean, dtype=float),
            1.0 - np.asarray(v_cloud, dtype=float),
            np.asarray(i4_fallback, dtype=float),
        ]
    )
    y = np.log(np.asarray(resid_ln, dtype=float) ** 2 + 1e-3)
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_sigma2(
    model: LinearRegression, is_modis: bool, vza_deg: float, w_mean: float,
    v_cloud: float, i4_fallback: bool,
) -> float:
    x = np.array(
        [[float(is_modis), vza_deg / 65.0, 1.0 - w_mean, 1.0 - v_cloud,
          float(i4_fallback)]]
    )
    pred = float(model.predict(x)[0])
    return float(np.clip(np.exp(pred), *S.SIGMA2_CLIP))


# --------------------------------------------------------------------------- #
# Active burning footprint transfer (g_area,b)
# --------------------------------------------------------------------------- #
def fit_area_model(
    phi_mw: np.ndarray, night_day_ratio: np.ndarray, pyrome_id: np.ndarray,
    a_det_km2: np.ndarray,
) -> Ridge:
    """Ridge: ln(A_det + 1e-3) ~ [ln(Phi+0.01), ln(ndr+0.05), pyrome one-hot]."""
    phi = np.asarray(phi_mw, dtype=float)
    ndr = np.asarray(night_day_ratio, dtype=float)
    b = np.asarray(pyrome_id, dtype=int)
    one_hot = np.zeros((len(b), S.PYROME_K))
    one_hot[np.arange(len(b)), b] = 1.0
    X = np.column_stack(
        [np.log(phi + 0.01), np.log(ndr + 0.05), one_hot]
    )
    y = np.log(np.asarray(a_det_km2, dtype=float) + 1e-3)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    return ridge


def predict_area_km2(
    model: Ridge, phi_mw: float, night_day_ratio: float, pyrome_id: int,
    cell_area_km2: float,
) -> float:
    one_hot = np.zeros((1, S.PYROME_K))
    one_hot[0, int(pyrome_id)] = 1.0
    X = np.array(
        [[np.log(phi_mw + 0.01), np.log(night_day_ratio + 0.05)] + one_hot[0].tolist()]
    )
    yhat = float(model.predict(X)[0])
    return float(np.clip(np.exp(yhat) - 1e-3, 0.0, cell_area_km2))


# --------------------------------------------------------------------------- #
# Cloud visibility provider (with 2012-2025 climatology fallback)
# --------------------------------------------------------------------------- #
class CloudVisibilityProvider:
    """Cloud-gated clear-sky observation probability v_{p,g,t}.

    Joins 8-day (or monthly) 0.25 deg cloud composites with the
    2012-2025 multi-year monthly climatology fallback for trailing-2026
    dates whose CMG granules have not yet been published (readme.md audit
    item 2). Visibility NEVER scales detected FRP (INV-05).
    """

    def __init__(self, cloud_df: pd.DataFrame):
        df = cloud_df.copy()
        df["month"] = df["period"].astype(int).clip(1, 12)  # monthly index
        df["key"] = list(
            zip(
                df["lon_025"].round(3), df["lat_025"].round(3),
                df["year"].astype(int), df["month"].astype(int),
            )
        )
        self.table: Dict[tuple, Tuple[float, float, str]] = {
            k: (float(c), float(u), str(s))
            for k, c, u, s in zip(
                df["key"], df["cloud_frac"], df["unmapped_frac"],
                df["cloud_source"],
            )
        }
        clim = df[
            (df["year"] >= S.CLOUD_CLIM_START_YEAR)
            & (df["year"] <= S.CLOUD_CLIM_END_YEAR)
        ].groupby(["lon_025", "lat_025", "month"], as_index=False).agg(
            cloud_frac=("cloud_frac", "mean"),
            unmapped_frac=("unmapped_frac", "mean"),
        )
        self.clim: Dict[tuple, Tuple[float, float]] = {
            (round(r.lon_025, 3), round(r.lat_025, 3), int(r.month)):
            (float(r.cloud_frac), float(r.unmapped_frac))
            for r in clim.itertuples()
        }
        self.date0 = date(2000, 11, 1)

    def cloud_at(self, lat: float, lon: float, on_date: date):
        """Return (cloud_frac, unmapped_frac, cloud_source) for one cell-day."""
        key = (
            round(np.floor(lon / 0.25) * 0.25 + 0.125, 3),
            round(np.ceil(lat / 0.25) * 0.25 - 0.125, 3),
            on_date.year,
            on_date.month,
        )
        hit = self.table.get(key)
        if hit is not None:
            return hit[0], hit[1], str(hit[2])
        clim_key = (key[0], key[1], on_date.month)
        c = self.clim.get(clim_key)
        if c is not None:
            return c[0], c[1], S.CLOUD_CLIMATOLOGY_FLAG
        return 0.35, 0.05, "MISSING_DEFAULT"

    def visibility(
        self, platform_family: str, platform: str, lat: float, lon: float,
        t_index: int,
    ) -> Tuple[float, str]:
        """v = p_swath * clip(1 - cloud - unmapped, 0.10, 0.98)."""
        on_date = self.date0 + timedelta(days=int(t_index))
        cloud, unmapped, source = self.cloud_at(lat, lon, on_date)
        if platform_family.upper() == "VIIRS":
            p_swath = 0.99
        else:
            p_swath = modis_swath_probability(abs(lat), int(t_index))
        v = p_swath * float(
            np.clip(1.0 - cloud - unmapped, 0.10, 0.98)
        )
        return float(np.clip(v, 0.0, 1.0)), source


# --------------------------------------------------------------------------- #
# Full calibration orchestration
# --------------------------------------------------------------------------- #
def mu_gain_det(
    family: str, platform: str, pyrome_id: int, dn_flag: str,
    mean_solar_hour: float, mean_vza_deg: float, mean_area_km2: float,
    ln_y: float, raw_event_count: int, month: int,
    diurnal_norm: Dict[Tuple[int, str, str], Dict[str, float]],
    gamma: GammaCalibrator, lognorm_params: np.ndarray,
) -> float:
    """Detected forward log-gain (STRICTLY cloud-free; INV-05).

    mu = ln D_{b,s,DN}(tau_bar) + ln gamma_{p,b}(theta_bar, q)
         + ln psi_FRP,eff(A_bar, DN, E_raw)
    """
    season = season_of_month(month)
    norm = diurnal_norm.get((int(pyrome_id), season, dn_flag))
    if norm is None:
        ln_d = 0.0
    else:
        ln_d = float(np.log(diurnal_value(norm, mean_solar_hour, dn_flag)))
    if family.upper() == "MODIS":
        ln_gamma = gamma.predict_ln_gamma(
            int(pyrome_id), dn_flag, mean_vza_deg, mean_area_km2, ln_y
        )
    else:
        ln_gamma = 0.0  # ln gamma_VIIRS == 0 by definition
    mu_f, sigma_f = lognorm_params[int(pyrome_id)]
    fmin = fmin_threshold(mean_area_km2, family, dn_flag)
    ln_psi = float(
        np.log(
            psi_frp_eff(fmin, mu_f, sigma_f, max(int(raw_event_count), 1))
        )
    )
    return ln_d + ln_gamma + ln_psi


def calibrate_passes(
    provenance: pd.DataFrame,
    pass_obs: pd.DataFrame,
    hex_pyromes: pd.DataFrame,
    cloud_provider: CloudVisibilityProvider,
    out_parquet=None,
    artifacts_pkl=None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Fit every operator and emit calibrated pass observations."""
    prov = provenance.reset_index(drop=True)
    po = pass_obs.reset_index(drop=True).copy()
    pyrome_map = dict(zip(hex_pyromes["h3_index"], hex_pyromes["pyrome_id"]))
    po["pyrome_id"] = po["h3_index"].map(pyrome_map).fillna(0).astype(int)
    po["month"] = po["solar_date"].str.slice(5, 7).astype(int)
    po["t_index"] = (
        pd.to_datetime(po["solar_date"]) - pd.Timestamp(S.DATE_START)
    ).dt.days.astype(int)

    # 1) Reliability per detection, then re-weighted pass FRP.
    reliability_models = fit_reliability_models(prov)
    w_by_row = np.ones(len(prov))
    for i, row in enumerate(prov.itertuples()):
        iso = reliability_models.get(
            (row.sensor_family, row.dn_flag)
        )
        if iso is not None:
            w_by_row[i] = apply_reliability(iso, float(row.conf_num))
    prov_w = prov.assign(w=w_by_row)
    key_cols = ["h3_index", "solar_date", "platform", "dn_flag"]
    w_sum = prov_w.groupby(key_cols).apply(
        lambda g: float(
            np.sum(g["bowtie_weight"] * g["w"] * g["frp_mw"])
        ),
        include_groups=False,
    ).rename("y_frp_mw")
    w_mean = prov_w.groupby(key_cols).apply(
        lambda g: float(
            np.average(g["w"], weights=g["bowtie_weight"] * g["frp_mw"] + 1e-9)
        ),
        include_groups=False,
    ).rename("w_mean")
    po = po.drop(columns=["y_frp_mw"]).merge(
        w_sum.reset_index(), on=key_cols, how="left"
    ).merge(w_mean.reset_index(), on=key_cols, how="left")
    po["y_frp_mw"] = po["y_frp_mw"].fillna(po["y_frp_mw"])
    po["w_mean"] = po["w_mean"].fillna(1.0)

    # 2) Log-normal cluster MLE per pyrome (clear-sky near-nadir VIIRS).
    viirs_det = prov_w[
        (prov_w["sensor_family"] == "VIIRS")
        & (prov_w["biomass_eligible"])
        & (prov_w["vza_deg"] < S.LOGNORM_FIT_VZA_MAX_DEG)
        & (prov_w["frp_mw"] > S.F_REF_MW)
    ]
    lognorm = np.zeros((S.PYROME_K, 2))
    for b in range(S.PYROME_K):
        vals = viirs_det.loc[
            viirs_det["h3_index"].map(pyrome_map).fillna(0) == b, "frp_mw"
        ].to_numpy()
        if len(vals) >= 5:
            lognorm[b] = fit_lognormal_mle(vals)
        else:
            lognorm[b] = (0.5, 1.0)
    lognorm_params = {
        b: (float(lognorm[b, 0]), float(lognorm[b, 1]))
        for b in range(S.PYROME_K)
    }

    # 3) Gamma calibration on synchronous Aqua x near-nadir SNPP pairs.
    pairs = _build_gamma_pairs(po, pyrome_map, lognorm_params)
    gamma = GammaCalibrator().fit(pairs)

    # 4) Diurnal curves per (pyrome, season, dn).
    diurnal_norm = _fit_all_diurnal(po, pyrome_map)

    # 5) sigma^2 OLS from pair residuals.
    if len(pairs) >= 30:
        resid = (
            pairs["ln_y_snpp"]
            - (pairs["ln_y_modis"] - pairs["ln_psi_eff_aqua"]
               + pairs["ln_psi_eff_snpp"])
        ).to_numpy()
        sigma_ols = fit_sigma_ols(
            np.ones(len(pairs)), pairs["vza_modis"].to_numpy(),
            np.full(len(pairs), 0.8), np.full(len(pairs), 0.75),
            np.zeros(len(pairs)), resid,
        )
    else:
        sigma_ols = LinearRegression()
        sigma_ols.fit(
            np.zeros((2, 5)), np.log(np.full(2, 0.16))
        )

    # 6) Area model from clear near-nadir VIIRS detections.
    near_nadir = po[
        (po["sensor_family"] == "VIIRS") & (po["y_frp_mw"] > 0)
    ]
    if len(near_nadir) >= 30:
        area_model = fit_area_model(
            near_nadir["y_frp_mw"].to_numpy(),
            np.full(len(near_nadir), 0.35),
            near_nadir["pyrome_id"].to_numpy(),
            near_nadir["det_footprint_km2"].to_numpy(),
        )
    else:
        area_model = Ridge()
        area_model.fit(np.zeros((2, 9)), np.zeros(2))

    # 7) Per-pass forward gain + observation sigma + cloud visibility.
    ln_y = np.log(np.maximum(po["y_frp_mw"].to_numpy(dtype=float), 1e-9))
    mu_gains, sigmas, v_clouds, sources = [], [], [], []
    for row in po.itertuples():
        mu = mu_gain_det(
            row.sensor_family, row.platform, int(row.pyrome_id),
            row.dn_flag, float(row.mean_solar_hour), float(row.mean_vza_deg),
            float(row.mean_area_km2), float(ln_y[row.Index]),
            int(row.raw_event_count), int(row.month), diurnal_norm, gamma,
            lognorm,
        )
        v, src = cloud_provider.visibility(
            row.sensor_family, row.platform, float(row.mean_latitude),
            float(row.mean_longitude), int(row.t_index),
        )
        sig2 = predict_sigma2(
            sigma_ols, row.sensor_family == "MODIS",
            float(row.mean_vza_deg), float(row.w_mean), v,
            bool(row.has_i4_m13_fallback),
        )
        if bool(row.has_i4_m13_fallback):
            sig2 *= S.I4_M13_SIGMA_INFLATION ** 2
        mu_gains.append(mu)
        sigmas.append(np.sqrt(sig2))
        v_clouds.append(v)
        sources.append(src)
    po["mu_gain"] = mu_gains
    po["obs_sigma"] = sigmas
    po["v_cloud"] = v_clouds
    po["cloud_source"] = sources

    # 8) Cloud climatology table for the artifacts bundle.
    cloud_climatology = _climatology_frame(cloud_provider)

    artifacts = {
        "reliability_models": reliability_models,
        "fmin_params": dict(S.FMIN_PARAMS),
        "lognorm_params": lognorm_params,
        "lgb_gamma": gamma.lgbm,
        "eqm_knots": dict(gamma.eqm_knot_arrays),
        "diurnal_coeffs": diurnal_norm,
        "sigma_ols": sigma_ols,
        "area_model": area_model,
        "cloud_climatology": cloud_climatology,
    }
    if artifacts_pkl is not None:
        Path(artifacts_pkl).parent.mkdir(parents=True, exist_ok=True)
        with open(artifacts_pkl, "wb") as fh:
            pickle.dump(artifacts, fh)
    if out_parquet is not None:
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        po.to_parquet(out_parquet, index=False)
    return po, artifacts


def _build_gamma_pairs(
    po: pd.DataFrame, pyrome_map: Dict[int, int],
    lognorm_params: np.ndarray,
) -> pd.DataFrame:
    """Synchronous Aqua x near-nadir SNPP pairs (VZA_SNPP < 25, dt <= 45 min)."""
    aqua = po[(po["platform"] == "Aqua") & (po["y_frp_mw"] > 0)]
    snpp = po[
        (po["platform"] == "SNPP") & (po["y_frp_mw"] > 0)
        & (po["mean_vza_deg"] < S.GAMMA_VZA_REF_MAX_DEG)
    ]
    if not len(aqua) or not len(snpp):
        return pd.DataFrame()
    snpp_idx = {
        (r.h3_index, r.solar_date, r.dn_flag): r
        for r in snpp.itertuples()
    }
    rows = []
    for a in aqua.itertuples():
        s = snpp_idx.get((a.h3_index, a.solar_date, a.dn_flag))
        if s is None:
            continue
        dt_min = abs(
            (a.mean_solar_hour - s.mean_solar_hour) * 60.0
        )
        if dt_min > S.GAMMA_PAIR_MAX_DT_MIN:
            continue
        b = int(pyrome_map.get(a.h3_index, 0))
        mu_f, sig_f = lognorm_params[b]
        ln_psi_aqua = np.log(
            psi_frp_eff(
                fmin_threshold(a.mean_area_km2, "MODIS", a.dn_flag),
                mu_f, sig_f, int(a.raw_event_count),
            )
        )
        ln_psi_snpp = np.log(
            psi_frp_eff(
                fmin_threshold(s.mean_area_km2, "VIIRS", s.dn_flag),
                mu_f, sig_f, int(s.raw_event_count),
            )
        )
        rows.append(
            {
                "pyrome_id": b,
                "dn_flag": a.dn_flag,
                "vza_modis": float(a.mean_vza_deg),
                "area_modis": float(a.mean_area_km2),
                "ln_y_modis": float(np.log(a.y_frp_mw)),
                "ln_y_snpp": float(np.log(s.y_frp_mw)),
                "ln_psi_eff_aqua": float(ln_psi_aqua),
                "ln_psi_eff_snpp": float(ln_psi_snpp),
            }
        )
    return pd.DataFrame(rows)


def _fit_all_diurnal(
    po: pd.DataFrame, pyrome_map: Dict[int, int]
) -> Dict[Tuple[int, str, str], Dict[str, float]]:
    """Fit (pyrome, season, dn) diurnal curves from same-cell paired
    log-FRP differences (Terra vs Aqua/SNPP 2003-2021 and Aqua 2022-2025 vs
    NOAA-20, the drift update touching ONLY the diurnal coefficients)."""
    out: Dict[Tuple[int, str, str], Dict[str, float]] = {}
    det = po[po["y_frp_mw"] > 0].copy()
    det["pyrome_id"] = det["h3_index"].map(pyrome_map).fillna(0).astype(int)
    det["month"] = det["solar_date"].str.slice(5, 7).astype(int)
    det["season"] = det["month"].map(season_of_month)
    # reference platforms for pairing: stationary afternoon witnesses
    refs = det[det["platform"].isin(["Aqua", "SNPP", "NOAA-20"])]
    ref_idx = {
        (r.h3_index, r.solar_date, r.dn_flag, r.platform): r
        for r in refs.itertuples()
    }
    for (b, season, dn), part in det.groupby(
        ["pyrome_id", "season", "dn_flag"]
    ):
        taus_a, taus_b, ratios = [], [], []
        for r in part.itertuples():
            for ref_plat in ("Aqua", "SNPP", "NOAA-20"):
                if ref_plat == r.platform:
                    continue
                s = ref_idx.get((r.h3_index, r.solar_date, r.dn_flag, ref_plat))
                if s is None:
                    continue
                if abs(r.mean_solar_hour - s.mean_solar_hour) * 60.0 > 120.0:
                    continue
                taus_a.append(float(r.mean_solar_hour))
                taus_b.append(float(s.mean_solar_hour))
                ratios.append(
                    float(np.log(r.y_frp_mw) - np.log(s.y_frp_mw))
                )
        if len(taus_a) >= 8:
            coeffs = fit_diurnal_ridge(
                np.array(taus_a), np.array(taus_b), np.array(ratios)
            )
        else:
            coeffs = np.zeros(4)
        out[(int(b), season, dn)] = normalize_diurnal_coeffs(coeffs)
    return out


def _climatology_frame(provider: CloudVisibilityProvider) -> pd.DataFrame:
    """2012-2025 monthly climatology table serialized into the artifacts."""
    rows = [
        {
            "lon_025": k[0], "lat_025": k[1], "month": k[2],
            "cloud_frac": v[0], "unmapped_frac": v[1],
            "cloud_source": S.CLOUD_CLIMATOLOGY_FLAG,
        }
        for k, v in provider.clim.items()
    ]
    return pd.DataFrame(rows)
