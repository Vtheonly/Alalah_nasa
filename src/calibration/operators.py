"""Pre-drift observation operator calibration (2012-01-20 .. 2021-12-31).

Contract (readme Sections 4.7, 5.2, INV-03/INV-05):
* Isotonic confidence reliability w_i in [0.15, 1.0] per (sensor_family,
  dn_flag); recompute clean weighted pass FRP ``y_frp_mw = sum(omega_bt *
  w_i * frp_mw)`` (w_i defaults to 1.0 where calibrated w_i >= 0.90).
* Physical detection limits F_min(A, DN) = alpha + beta A with spec anchors
  and per-pyrome left-truncated Log-Normal MLE (L-BFGS-B, F_ref = 0.5 MW,
  clear-sky near-nadir VIIRS clusters), yielding closed-form survival
  integrals psi_CNT, psi_FRP and the conditional psi_FRP_eff blend
  (w_multi(E) = 1 - exp(-0.7 (E-1))).
* Near-nadir VIIRS (VZA_SNPP < 25 deg, dt <= 45 min) EQM + LightGBM residual
  calibration ln gamma_MODIS in [-1.5, +1.5]; ln gamma_VIIRS == 0.
* 12-hour intra-window 2-harmonic Fourier diurnal curves D_{b,s,DN}(tau)
  (Ridge alpha=10 on same-day paired log-FRP differences incl. post-2022
  Aqua-vs-NOAA-20 drift strictly on diurnal coefficients), normalized by
  120-point trapezoidal integration independently per half-day window.
* Heteroscedastic observation variance sigma^2 in [0.025, 2.50] (OLS on
  squared calibration residuals).
* Active burning footprint Ridge model g_area,b.
* Emits ``calibration_artifacts.pkl`` with the exact dictionary schema
  {"reliability_models", "fmin_params", "lognorm_params", "lgb_gamma",
  "eqm_knots", "diurnal_coeffs", "sigma_ols", "area_model",
  "cloud_climatology"} and ``pass_observations_calibrated.parquet``.

CLOUD VISIBILITY NEVER ENTERS THE DETECTED GAIN (INV-05 / Math Bug #1).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge

from ..config import settings as S
from ..geometry.solar_geometry import latlon_to_ecef
from .pyromes import behavioral_features, extract_cluster_table

SEASONS = ("DJF", "MAM", "JJA", "SON")
EQM_KNOTS = 20


def season_of(month: int) -> str:
    return SEASONS[(int(month) % 12) // 3]


# ---------------------------------------------------------------------------
# 1) Isotonic confidence reliability
# ---------------------------------------------------------------------------
def fit_isotonic_reliability(prov_bio: pd.DataFrame,
                             calib_start: str = S.CALIB_START,
                             calib_end: str = S.CALIB_END) -> dict:
    """Corroboration-based reliability per (sensor_family, dn_flag).

    A detection is corroborated (1) when another detection from a different
    pass or platform occurs within +-1 day and <= 2 km.
    """
    df = prov_bio.copy()
    df["solar_date"] = pd.to_datetime(df["solar_date"])
    df = df[(df["solar_date"] >= calib_start) & (df["solar_date"] <= calib_end)]
    models: dict[tuple, IsotonicRegression] = {}
    if df.empty:
        return models
    day_key = (df["solar_date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    ecef = latlon_to_ecef(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    pass_ids = df["pass_id"].fillna(-1).to_numpy()
    platforms = df["platform"].to_numpy()
    days = np.unique(day_key)
    day_pos = {d: np.where(day_key == d)[0] for d in days}
    corroborated = np.zeros(len(df), dtype=bool)
    for d in days:
        idx = np.concatenate([day_pos.get(d - 1, []), day_pos.get(d, []),
                              day_pos.get(d + 1, [])]).astype(int)
        if len(idx) < 2:
            continue
        tree = cKDTree(ecef[idx])
        pairs = tree.query_pairs(r=2.0, output_type="ndarray")
        for i, j in pairs:
            gi, gj = int(idx[i]), int(idx[j])
            if pass_ids[gi] != pass_ids[gj] or platforms[gi] != platforms[gj]:
                corroborated[gi] = True
                corroborated[gj] = True
    df["corroborated"] = corroborated.astype(float)
    for (fam, dn), g in df.groupby(["sensor_family", "dn_flag"]):
        if len(g) < 8:
            continue
        iso = IsotonicRegression(y_min=S.RELIABILITY_CLIP[0],
                                 y_max=S.RELIABILITY_CLIP[1],
                                 out_of_bounds="clip")
        iso.fit(g["conf_num"].to_numpy(dtype=float),
                g["corroborated"].to_numpy(dtype=float))
        models[(fam, dn)] = iso
    return models


def reliability_weights(prov_bio: pd.DataFrame, models: dict) -> np.ndarray:
    """Per-detection calibrated reliability w_i (1.0 where w >= 0.90)."""
    w = np.ones(len(prov_bio), dtype=np.float64)
    fam = prov_bio["sensor_family"].to_numpy()
    dn = prov_bio["dn_flag"].to_numpy()
    conf = prov_bio["conf_num"].fillna(65.0).to_numpy(dtype=float)
    for i in range(len(prov_bio)):
        m = models.get((fam[i], dn[i]))
        if m is None:
            continue
        wi = float(m.predict([conf[i]])[0])
        w[i] = 1.0 if wi >= 0.90 else wi
    return w


# ---------------------------------------------------------------------------
# 2) Detection limits + truncated Log-Normal MLE + survival integrals
# ---------------------------------------------------------------------------
def fmin_mw(family: str, area_km2, dn):
    """F_min,p(A, DN) = alpha + beta * A with the physical spec anchors.

    Supports scalar and vectorized (area, dn) inputs.
    """
    area = np.asarray(area_km2, dtype=np.float64)
    dn_arr = np.asarray(dn, dtype=object)
    if dn_arr.ndim == 0:
        is_day = str(dn_arr.item()).strip().upper().startswith("D")
        a, b = S.FMIN_PARAMS[(family, "D" if is_day else "N")]
        return a + b * area
    is_day = np.array([str(d).strip().upper().startswith("D")
                       for d in dn_arr.ravel()])
    shape = dn_arr.shape
    aa = np.where(is_day, S.FMIN_PARAMS[(family, "D")][0],
                  S.FMIN_PARAMS[(family, "N")][0]).reshape(shape)
    bb = np.where(is_day, S.FMIN_PARAMS[(family, "D")][1],
                  S.FMIN_PARAMS[(family, "N")][1]).reshape(shape)
    if area.shape == shape:
        return aa + bb * area
    return aa + bb * np.broadcast_to(area, shape)


def fit_lognormal_truncated(frp: np.ndarray, f_ref: float = S.F_REF_MW
                            ) -> tuple[float, float]:
    """Left-truncated Log-Normal MLE (L-BFGS-B) for clusters above F_ref."""
    x = np.log(np.asarray(frp, dtype=np.float64)[np.asarray(frp) > f_ref])
    if len(x) < 5:
        return (0.0, 1.0)

    def nll(theta):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        z = (x - mu) / sigma
        ll = (-np.log(sigma) - 0.5 * z * z).sum()
        # truncation: P(X > ln f_ref) = 1 - Phi((ln f_ref - mu)/sigma)
        zc = (np.log(f_ref) - mu) / sigma
        surv = np.clip(1.0 - ndtr(zc), 1e-12, 1.0)
        return -(ll - len(x) * np.log(surv))

    res = minimize(nll, x0=[float(np.median(x)), 0.0], method="L-BFGS-B",
                   bounds=[(-5.0, 8.0), (-4.0, 2.0)])
    mu = float(res.x[0])
    sigma = float(np.exp(res.x[1]))
    return (mu, sigma)


def fit_lognorm_params(clusters: pd.DataFrame, f_ref: float = S.F_REF_MW) -> dict:
    """Per-pyrome truncated Log-Normal (mu_F, sigma_F) on clear-sky near-nadir
    (VZA < 20 deg, cloud_frac < 0.30) VIIRS clusters above F_ref."""
    out: dict[int, tuple[float, float]] = {}
    if "pyrome_id" not in clusters.columns:
        clusters = clusters.assign(pyrome_id=0)
    for b, g in clusters.groupby("pyrome_id"):
        sel = g[(g["vza_deg"] < 20.0)
                & (g.get("cloud_frac", pd.Series(0.0, index=g.index)) < 0.30)
                & (g["cluster_frp_mw"] > f_ref)]
        vals = sel["cluster_frp_mw"].to_numpy() if len(sel) else \
            g[g["cluster_frp_mw"] > f_ref]["cluster_frp_mw"].to_numpy()
        if len(vals) < 5:
            vals = g["cluster_frp_mw"].to_numpy()
        out[int(b)] = fit_lognormal_truncated(vals, f_ref)
    if not out:
        out[0] = (0.0, 1.0)
    return out


def _mu_sigma(lognorm_params: dict, pyrome):
    """Elementwise (mu_F, sigma_F) lookup supporting scalars and arrays."""
    arr = np.asarray(pyrome)
    flat = arr.ravel().tolist()
    mus = np.empty(len(flat))
    sgs = np.empty(len(flat))
    for i, b in enumerate(flat):
        mu, sg = lognorm_params.get(int(b), (0.0, 1.0))
        mus[i] = mu
        sgs[i] = sg
    return mus.reshape(arr.shape), sgs.reshape(arr.shape)


def psi_cnt(family: str, pyrome, area_km2, dn: str,
            lognorm_params: dict):
    """psi_CNT_{p,b}(A, DN) = clip(1 - Phi((ln F_min - mu_F)/sigma_F), .02, 1)."""
    mu, sigma = _mu_sigma(lognorm_params, pyrome)
    ln_fmin = np.log(np.maximum(fmin_mw(family, area_km2, dn), 1e-6))
    z = (ln_fmin - mu) / np.maximum(sigma, 1e-6)
    return np.clip(1.0 - ndtr(z), *S.PSI_CNT_CLIP)


def psi_frp(family: str, pyrome, area_km2, dn: str,
            lognorm_params: dict, f_ref: float = S.F_REF_MW):
    """psi_FRP: mean detected-FRP fraction above F_min, truncated at F_ref."""
    mu, sigma = _mu_sigma(lognorm_params, pyrome)
    sigma = np.maximum(sigma, 1e-6)
    ln_fmin = np.log(np.maximum(fmin_mw(family, area_km2, dn), 1e-6))
    z_min = (ln_fmin - mu - sigma * sigma) / sigma
    z_ref = (np.log(f_ref) - mu - sigma * sigma) / sigma
    num = 1.0 - ndtr(z_min)
    den = 1.0 - ndtr(z_ref)
    return np.clip(num / np.maximum(den, 1e-12), *S.PSI_FRP_CLIP)


def w_multi(e_raw) -> np.ndarray | float:
    """w_multi(E) = 1 - exp(-0.7 (E - 1))."""
    e = np.maximum(np.asarray(e_raw, dtype=np.float64) - 1.0, 0.0)
    v = 1.0 - np.exp(-S.W_MULTI_RATE * e)
    return float(v) if v.ndim == 0 else v


def psi_frp_eff(family: str, pyrome: int, area_km2, dn: str, e_raw,
                lognorm_params: dict, f_ref: float = S.F_REF_MW):
    """Conditional multi-fire adjustment (readme 5.2.3):

    psi_FRP_eff = (1 - w_multi) * max(psi_FRP, 0.65) + w_multi * psi_FRP.
    """
    base = psi_frp(family, pyrome, area_km2, dn, lognorm_params, f_ref)
    wm = w_multi(e_raw)
    return (1.0 - wm) * np.maximum(base, S.PSI_EFF_ISOLATED_FLOOR) + wm * base


# ---------------------------------------------------------------------------
# 3) VZA & quantile residual transfer (gamma_{p,b}) -- Aqua vs near-nadir SNPP
# ---------------------------------------------------------------------------
def build_gamma_pairs(pass_obs: pd.DataFrame) -> pd.DataFrame:
    """Synchronous Aqua / near-nadir SNPP pairs (VZA_SNPP < 25 deg, dt <= 45
    min, same h3_index, same dn_flag, both > 0)."""
    p = pass_obs[(pass_obs["y_frp_mw"] > 0)].copy()
    aqua = p[p["platform"] == "Aqua"]
    snpp = p[(p["platform"] == "SNPP") & (p["mean_vza_deg"] < 25.0)]
    if not len(aqua) or not len(snpp):
        return pd.DataFrame()
    keys = ["h3_index", "solar_date", "dn_flag"]
    m = aqua.merge(snpp, on=keys, suffixes=("_m", "_v"))
    if not len(m):
        return pd.DataFrame()
    # dt <= 45 min between overpass solar hours
    dt = (m["mean_solar_hour_m"] - m["mean_solar_hour_v"]).abs()
    dt = np.minimum(dt, 24.0 - dt) * 60.0
    m = m[dt <= 45.0]
    return m


def fit_gamma_calibration(pass_obs: pd.DataFrame, lognorm_params: dict):
    """LightGBM residual + 20-knot monotonic EQM per (pyrome, dn).

    The LightGBM carries the conditional-mean transfer ``r_k``; the EQM
    refines the monotone quantile ordering of the transfer.  The EQM's
    LEVEL is anchored to zero mean over the calibration-window MODIS pass
    population (no spurious net level transfer: the twin's gamma_T = 0 is
    recovered exactly), so quantile curves never cross while the net
    transfer stays unbiased.

    Returns (lgbm model or None, eqm_knots dict, eqm_anchors dict,
    fallback dict).
    """
    pairs = build_gamma_pairs(pass_obs)
    if pairs.empty or len(pairs) < 30:
        return None, {}, {}, {}
    ln_psi_m = np.log(psi_frp_eff(
        "MODIS", pairs["pyrome_id_m"], pairs["mean_area_km2_m"],
        pairs["dn_flag"], pairs["raw_event_count_m"], lognorm_params))
    ln_psi_v = np.log(psi_frp_eff(
        "VIIRS", pairs["pyrome_id_m"], pairs["mean_area_km2_v"],
        pairs["dn_flag"], pairs["raw_event_count_v"], lognorm_params))
    ln_y_m = np.log(pairs["y_frp_mw_m"].to_numpy(dtype=float))
    ln_y_v = np.log(pairs["y_frp_mw_v"].to_numpy(dtype=float))
    x_adj = ln_y_m - ln_psi_m + ln_psi_v
    r_k = ln_y_v - x_adj

    feats = np.column_stack([
        pairs["pyrome_id_m"].to_numpy(dtype=float),
        (pairs["dn_flag"].to_numpy() == "D").astype(float),
        pairs["mean_vza_deg_m"].to_numpy(dtype=float),
        pairs["mean_area_km2_m"].to_numpy(dtype=float),
        ln_y_m])
    lgbm = None
    r_hat = np.zeros(len(r_k))
    if len(r_k) >= 60:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(max_depth=4, n_estimators=100,
                                 min_child_samples=50, verbose=-1)
        # cross-validated gate: apply the feature transfer only if it
        # generalizes (CV R^2 > 0.02); otherwise r_hat = 0 (no spurious
        # pair-selection-bias transfer to the pass population).
        from sklearn.model_selection import cross_val_score
        try:
            r2 = float(np.mean(cross_val_score(
                lgb.LGBMRegressor(max_depth=4, n_estimators=100,
                                  min_child_samples=50, verbose=-1),
                feats, r_k, cv=3, scoring="r2")))
        except Exception:  # noqa: BLE001
            r2 = -1.0
        if r2 > 0.02:
            lgbm.fit(feats, r_k)
            r_hat = lgbm.predict(feats)
        else:
            lgbm = None
            r_hat = np.zeros(len(r_k))
    else:
        r_hat = np.full(len(r_k), float(np.median(r_k)))

    # 20-knot strictly monotonic EQM per (pyrome, dn): quantile knots of
    # (x_adj + r_hat) mapped onto the same quantile ranks of ln Y_SNPP.
    eqm_knots: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    df = pd.DataFrame({
        "pyrome": pairs["pyrome_id_m"].to_numpy(),
        "dn": pairs["dn_flag"].to_numpy(),
        "x": x_adj + r_hat, "y": ln_y_v})
    q = np.linspace(0.5 / EQM_KNOTS, 1.0 - 0.5 / EQM_KNOTS, EQM_KNOTS)
    for (b, dn), g in df.groupby(["pyrome", "dn"]):
        if len(g) < 8:
            continue
        xk = np.quantile(g["x"], q)
        yk = np.quantile(g["y"], q)
        xk = np.maximum.accumulate(xk)
        yk = np.maximum.accumulate(yk)
        eqm_knots[(int(b), str(dn))] = (xk, yk)

    # Level anchors: mean EQM shift over the calibration MODIS pass
    # population per (pyrome, dn) -> the EQM contributes quantile SHAPE
    # only (zero net level transfer).
    eqm_anchors: dict[tuple, float] = {}
    mod = pass_obs[(pass_obs["sensor_family"] == "MODIS")
                   & (pass_obs["y_frp_mw"] > 0)]
    if len(mod):
        ln_y_all = np.log(mod["y_frp_mw"].to_numpy(dtype=float))
        r_all = _predict_r(lgbm, mod, ln_y_all, float(np.median(r_k)))
        ln_psi_m_all = np.log(psi_frp_eff(
            "MODIS", mod["pyrome_id"], mod["mean_area_km2"], mod["dn_flag"],
            mod["raw_event_count"], lognorm_params))
        ln_psi_v_all = np.log(psi_frp_eff(
            "VIIRS", mod["pyrome_id"], S.PASS_GEOMETRY_AVG["VIIRS"][0],
            mod["dn_flag"], mod["raw_event_count"], lognorm_params))
        x_all = ln_y_all - ln_psi_m_all + ln_psi_v_all + r_all
        pyr = mod["pyrome_id"].to_numpy()
        dns = mod["dn_flag"].to_numpy()
        for (b, dn), (xk, yk) in eqm_knots.items():
            m = (pyr == b) & (dns == dn)
            if m.any():
                shift = _pchip_eval(xk, yk, x_all[m]) - x_all[m]
                eqm_anchors[(int(b), str(dn))] = float(np.mean(shift))
    fallback = {"log_gamma": float(-np.mean(r_hat))}
    return lgbm, eqm_knots, eqm_anchors, fallback


def _predict_r(lgbm, frame, ln_y, fallback_value: float):
    """Predict r_hat for a pass frame with the fitted LightGBM."""
    if lgbm is None:
        return np.full(len(ln_y), fallback_value)
    feats = np.column_stack([
        frame["pyrome_id"].to_numpy(dtype=float),
        (frame["dn_flag"].to_numpy() == "D").astype(float),
        frame["mean_vza_deg"].to_numpy(dtype=float),
        frame["mean_area_km2"].to_numpy(dtype=float),
        ln_y])
    return np.asarray(lgbm.predict(feats), dtype=float)


def _pchip_eval(xk: np.ndarray, yk: np.ndarray, x) -> np.ndarray:
    """Monotone PCHIP-style interpolation (scipy) with edge clamping."""
    from scipy.interpolate import PchipInterpolator
    if len(xk) < 2:
        return np.full_like(np.asarray(x, dtype=float), float(yk[0]) if len(yk) else 0.0,
                            dtype=float)
    f = PchipInterpolator(xk, yk, extrapolate=False)
    xs = np.clip(np.asarray(x, dtype=float), xk[0], xk[-1])
    return np.asarray(f(xs), dtype=float)


def log_gamma_modis(ln_y, pyrome, dn_is_day, vza, area, e_raw,
                    lgbm, eqm_knots, lognorm_params,
                    fallback: dict | None = None,
                    eqm_anchors: dict | None = None):
    """ln gamma for a MODIS pass: -(r_hat + anchored EQM shift), clipped."""
    ln_y = np.asarray(ln_y, dtype=np.float64)
    dn = np.where(np.asarray(dn_is_day, dtype=bool), "D", "N")
    ln_psi_m = np.log(psi_frp_eff("MODIS", pyrome, area, dn, e_raw,
                                  lognorm_params))
    ln_psi_v = np.log(psi_frp_eff("VIIRS", pyrome, S.PASS_GEOMETRY_AVG["VIIRS"][0],
                                  dn, e_raw, lognorm_params))
    x_adj = ln_y - ln_psi_m + ln_psi_v
    r_hat = np.zeros_like(ln_y)
    if lgbm is not None:
        feats = np.column_stack([np.asarray(pyrome, dtype=float),
                                 np.asarray(dn_is_day, dtype=float),
                                 np.asarray(vza, dtype=float),
                                 np.asarray(area, dtype=float), ln_y])
        r_hat = np.asarray(lgbm.predict(feats), dtype=float)
    elif fallback and "log_gamma" in fallback:
        r_hat = np.full_like(ln_y, -float(fallback["log_gamma"]))
    eqm_anchors = eqm_anchors or {}
    eqm_shift = np.zeros_like(ln_y)
    pyrome = np.asarray(pyrome)
    for (b, d), (xk, yk) in eqm_knots.items():
        m = (pyrome == b) & (dn == d)
        if m.any():
            x_m = (x_adj + r_hat)[m]
            shift = _pchip_eval(xk, yk, x_m) - x_m
            anchor = float(eqm_anchors.get((int(b), str(d)), 0.0))
            eqm_shift[m] = shift - anchor
    # ln gamma = x_adj - x_mapped (== -r_hat under identity EQM)
    ln_gamma = -(r_hat + eqm_shift)
    return np.clip(ln_gamma, *S.LN_GAMMA_CLIP)


# ---------------------------------------------------------------------------
# 4) 12-hour intra-window diurnal & drift curves
# ---------------------------------------------------------------------------
def _fourier_design(tau: np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    return np.column_stack([np.cos(2 * np.pi * t / 24.0), np.sin(2 * np.pi * t / 24.0),
                            np.cos(4 * np.pi * t / 24.0), np.sin(4 * np.pi * t / 24.0)])


def _fourier_eval(coef: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return _fourier_design(tau) @ np.asarray(coef, dtype=np.float64)


def build_diurnal_pairs(pass_obs: pd.DataFrame, lgbm, eqm_knots,
                        eqm_anchors, lognorm_params: dict) -> pd.DataFrame:
    """Same-day paired log-FRP differences for the diurnal fit.

    * Terra vs Aqua/SNPP (psi/gamma-adjusted to a common reference).
    * Aqua 2022-2025 drifting solar hours vs stationary NOAA-20 (strictly
      diurnal coefficients; psi/gamma frozen).
    """
    p = pass_obs[pass_obs["y_frp_mw"] > 0].copy()
    if p.empty:
        return pd.DataFrame()
    p["ln_y"] = np.log(p["y_frp_mw"].to_numpy(dtype=float))
    fam = p["sensor_family"].to_numpy()
    ln_psi = np.zeros(len(p))
    for family in ("MODIS", "VIIRS"):
        m = fam == family
        if m.any():
            ln_psi[m] = np.log(psi_frp_eff(
                family, p.loc[m, "pyrome_id"], p.loc[m, "mean_area_km2"],
                p.loc[m, "dn_flag"], p.loc[m, "raw_event_count"],
                lognorm_params))
    ln_gamma = np.zeros(len(p))
    m_modis = fam == "MODIS"
    if m_modis.any():
        ln_gamma[m_modis] = log_gamma_modis(
            p.loc[m_modis, "ln_y"], p.loc[m_modis, "pyrome_id"],
            (p.loc[m_modis, "dn_flag"] == "D"), p.loc[m_modis, "mean_vza_deg"],
            p.loc[m_modis, "mean_area_km2"], p.loc[m_modis, "raw_event_count"],
            lgbm, eqm_knots, lognorm_params, None, eqm_anchors)
    p["ln_y_adj"] = p["ln_y"] - ln_gamma - ln_psi
    p["v_cloud"] = p["v_cloud"] if "v_cloud" in p.columns else 0.0

    keys = ["h3_index", "solar_date", "dn_flag"]
    pairs = []
    d = p[p["platform"].isin(["Terra", "Aqua", "SNPP", "NOAA-20"])]
    for (_h, dt, dn), g in d.groupby(keys):
        b = int(g["pyrome_id"].iloc[0])
        terras = g[g["platform"] == "Terra"]
        ref = g[g["platform"].isin(["Aqua", "SNPP"])]
        for _, t in terras.iterrows():
            for _, r in ref.iterrows():
                pairs.append((b, dt, dn, t["mean_solar_hour"], r["mean_solar_hour"],
                              t["ln_y_adj"], r["ln_y_adj"],
                              float((t["sensor_family"] == "MODIS")
                                    & (r["sensor_family"] == "MODIS")),
                              (t["mean_vza_deg"] + r["mean_vza_deg"]) / 2.0,
                              (t["w_mean"] + r["w_mean"]) / 2.0,
                              float((t["v_cloud"] + r["v_cloud"]) / 2.0),
                              max(t["has_i4_m13_fallback"],
                                  r["has_i4_m13_fallback"])))
        if pd.Timestamp(dt) >= pd.Timestamp(S.DRIFT_START):
            aquas = g[g["platform"] == "Aqua"]
            n20 = g[g["platform"] == "NOAA-20"]
            for _, a in aquas.iterrows():
                for _, n in n20.iterrows():
                    pairs.append((b, dt, dn, a["mean_solar_hour"],
                                  n["mean_solar_hour"], a["ln_y_adj"],
                                  n["ln_y_adj"],
                                  float((a["sensor_family"] == "MODIS")
                                        & (n["sensor_family"] == "MODIS")),
                                  (a["mean_vza_deg"] + n["mean_vza_deg"]) / 2.0,
                                  (a["w_mean"] + n["w_mean"]) / 2.0,
                                  float((a["v_cloud"] + n["v_cloud"]) / 2.0),
                                  max(a["has_i4_m13_fallback"],
                                      n["has_i4_m13_fallback"])))
    cols = ["pyrome_id", "solar_date", "dn_flag", "tau1", "tau2", "ln1",
            "ln2", "is_modis", "vza_mean", "w_mean", "v_cloud", "m13"]
    if not pairs:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(pairs, columns=cols)


def fit_diurnal_curves(pairs: pd.DataFrame) -> dict:
    """Ridge (alpha=10) 2-harmonic diurnal coefficients per (pyrome, season,
    dn), normalized independently within each 12-hour half-day window."""
    coeffs: dict[tuple, dict] = {}
    if pairs.empty:
        return coeffs
    df = pairs.copy()
    df["solar_date"] = pd.to_datetime(df["solar_date"])
    df["season"] = df["solar_date"].dt.month.map(season_of)
    df["diff"] = df["ln1"] - df["ln2"]
    for (b, s, dn), g in df.groupby(["pyrome_id", "season", "dn_flag"]):
        X1 = _fourier_design(g["tau1"].to_numpy())
        X2 = _fourier_design(g["tau2"].to_numpy())
        X = X1 - X2
        y = g["diff"].to_numpy()
        if len(g) < 8 or np.allclose(X, 0.0):
            coef = np.zeros(4)
        else:
            ridge = Ridge(alpha=10.0, fit_intercept=False)
            ridge.fit(X, y)
            coef = ridge.coef_.astype(np.float64)
        # 12-hour normalization via 120-point trapezoidal integration of
        # D_raw = exp(fourier) within each half-day window
        c0 = {}
        for window, (lo, hi) in (("D", (6.0, 18.0)), ("N", (18.0, 30.0))):
            tau = np.linspace(lo, hi, S.DIURNAL_GRID_POINTS)
            vals = np.exp(_fourier_eval(coef, np.mod(tau, 24.0)))
            mean = float(np.trapezoid(vals, tau) / (hi - lo))
            c0[window] = -np.log(mean) if mean > 1e-12 else 0.0
        coeffs[(int(b), s, dn)] = {"coef": coef, "c0": c0}
    return coeffs


def diurnal_eval(coeffs: dict, pyrome, season, dn, tau) -> np.ndarray:
    """D_{b,s,DN}(tau): unit mean within its 12-hour window."""
    tau = np.mod(np.asarray(tau, dtype=np.float64), 24.0)
    pyrome = np.asarray(pyrome)
    dn_arr = np.asarray(dn)
    if isinstance(season, str):
        season_arr = np.full(tau.shape, season, dtype=object)
    else:
        season_arr = np.asarray(season, dtype=object)
    out = np.zeros_like(tau)
    for (b, s, d), spec in coeffs.items():
        m = (pyrome == b) & (dn_arr == d) & (season_arr == s)
        if m.any():
            out[m] = _fourier_eval(spec["coef"], tau[m]) + spec["c0"][d]
    return np.exp(out)


def diurnal_gain(coeffs: dict, pyrome, month, dn, tau) -> np.ndarray:
    """Convenience: season resolved from month."""
    months = np.asarray(month)
    if months.ndim == 0:
        seasons = season_of(int(months))
    else:
        seasons = [season_of(int(m)) for m in months]
    return diurnal_eval(coeffs, pyrome, seasons, dn, tau)


# ---------------------------------------------------------------------------
# 5) Heteroscedastic sigma^2 OLS
# ---------------------------------------------------------------------------
def fit_sigma_ols(pairs: pd.DataFrame, coeffs: dict) -> LinearRegression:
    """OLS predicting ln(eps_hat^2 + 1e-3) from [is_modis, vza/65,
    1-w_mean, 1-v_cloud, i4_m13_fallback], clipped to [0.025, 2.50]."""
    xs, ys = [], []
    if not pairs.empty:
        df = pairs.copy()
        df["solar_date"] = pd.to_datetime(df["solar_date"])
        df["season"] = df["solar_date"].dt.month.map(season_of)
        for (b, s, dn), g in df.groupby(["pyrome_id", "season", "dn_flag"]):
            spec = coeffs.get((int(b), s, dn))
            if spec is None:
                continue
            d1 = _fourier_eval(spec["coef"], g["tau1"]) + spec["c0"][dn]
            d2 = _fourier_eval(spec["coef"], g["tau2"]) + spec["c0"][dn]
            eps = (g["ln1"] - g["ln2"]) - (d1 - d2)
            xs.append(np.column_stack([
                g["is_modis"].to_numpy(dtype=float),
                g["vza_mean"].to_numpy(dtype=float) / 65.0,
                1.0 - g["w_mean"].to_numpy(dtype=float),
                1.0 - g["v_cloud"].to_numpy(dtype=float),
                g["m13"].to_numpy(dtype=float)]))
            ys.append(np.log(eps.to_numpy(dtype=float) ** 2 + 1e-3))
    if not xs:
        ols = LinearRegression()
        ols.coef_ = np.zeros(5)
        ols.intercept_ = float(np.log(0.25))
        ols.n_features_in_ = 5
        return ols
    X = np.vstack(xs)
    y = np.concatenate(ys)
    # Stable OLS via rank-truncated least squares: the pair features are
    # strongly collinear (e.g. is_modis vs mean VZA), so a raw normal-equation
    # fit produces huge cancelling coefficients that explode on the pass
    # population; lstsq(rcond) keeps the exact OLS fit on the identifiable
    # subspace with bounded extrapolation.
    Xc = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=1e-4)
    var = X.var(axis=0)
    coef[1:] = np.where(var > 1e-12, coef[1:], 0.0)
    ols = LinearRegression()
    ols.coef_ = coef[1:]
    ols.intercept_ = float(coef[0])
    ols.n_features_in_ = X.shape[1]
    return ols


def sigma2_of(ols: LinearRegression, is_modis, vza, w_mean, v_cloud,
              m13) -> np.ndarray:
    """sigma^2 = clip(exp(OLS prediction), 0.025, 2.50) (log-space model)."""
    feats = np.column_stack([np.asarray(is_modis, dtype=float),
                             np.asarray(vza, dtype=float) / 65.0,
                             1.0 - np.asarray(w_mean, dtype=float),
                             1.0 - np.asarray(v_cloud, dtype=float),
                             np.asarray(m13, dtype=float)])
    return np.clip(np.exp(ols.predict(feats)), *S.SIGMA2_CLIP)


# ---------------------------------------------------------------------------
# 6) Active burning footprint model g_area,b
# ---------------------------------------------------------------------------
def fit_area_model(pass_obs: pd.DataFrame, coeffs: dict,
                   lognorm_params: dict | None = None) -> Ridge:
    """Ridge on clear-sky near-nadir VIIRS detections predicting
    ln(A_det + 1e-3) from [ln(Phi + 0.01), ln(ndr + 0.05), pyrome one-hot]."""
    p = pass_obs[(pass_obs["sensor_family"] == "VIIRS")
                 & (pass_obs["y_frp_mw"] > 0)
                 & (pass_obs["mean_vza_deg"] < 25.0)].copy()
    p["month"] = pd.to_datetime(p["solar_date"]).dt.month
    if lognorm_params is None:
        c = extract_cluster_table_from_pass(p)
        lognorm_params = fit_lognorm_params(c)
    n_features = 2 + S.PYROME_K
    if len(p) < 10:
        ridge = Ridge(alpha=1.0)
        ridge.fit(np.zeros((1, n_features)), np.zeros(1))
        ridge.coef_ = np.zeros(n_features)
        ridge.intercept_ = float(np.log(1.0 + 1e-3))
        return ridge
    ln_psi = np.log(psi_frp_eff(
        "VIIRS", p["pyrome_id"], p["mean_area_km2"], p["dn_flag"],
        p["raw_event_count"], lognorm_params))
    ln_d = np.log(np.maximum(
        diurnal_gain(coeffs, p["pyrome_id"], p["month"], p["dn_flag"],
                     p["mean_solar_hour"]), 1e-6))
    p["ln_phi"] = p["y_frp_mw"].to_numpy(dtype=float) \
        / np.exp(ln_d + ln_psi)
    p["ln_phi"] = np.log(p["ln_phi"].clip(lower=1e-3) + 0.01)
    # night/day ratio per (h3_index, month) from implied day/night log-flux
    gm = (p.groupby(["h3_index", "month", "dn_flag"])["ln_phi"].mean()
            .unstack("dn_flag"))
    if {"D", "N"} <= set(gm.columns):
        ndr = np.exp(gm["N"] - gm["D"]).rename("ndr")
    else:
        ndr = pd.Series(1.0, index=gm.index, name="ndr")
    ndr = ndr.reset_index()
    p = p.merge(ndr, on=["h3_index", "month"], how="left")
    p["ndr"] = p["ndr"].fillna(1.0)
    onehot = np.eye(S.PYROME_K)[p["pyrome_id"].to_numpy(dtype=int)]
    X = np.column_stack([p["ln_phi"].to_numpy(dtype=float),
                         np.log(p["ndr"].to_numpy(dtype=float) + 0.05), onehot])
    y = np.log(p["det_footprint_km2"].to_numpy(dtype=float) + 1e-3)
    return Ridge(alpha=1.0).fit(X, y)


def extract_cluster_table_from_pass(p: pd.DataFrame) -> pd.DataFrame:
    c = p.rename(columns={"y_frp_mw": "cluster_frp_mw",
                          "mean_vza_deg": "vza_deg",
                          "raw_event_count": "n_clusters"})
    c["cluster_frp_mw"] = c["cluster_frp_mw"] / np.maximum(
        c["n_clusters"].clip(lower=1), 1)
    if "cloud_frac" not in c.columns:
        c["cloud_frac"] = 0.0
    return c


def predict_burn_area(area_model: Ridge, ln_phi, ndr, pyrome,
                      hex_area_km2: float) -> np.ndarray:
    onehot = np.eye(S.PYROME_K)[np.asarray(pyrome, dtype=int)]
    X = np.column_stack([np.asarray(ln_phi, dtype=float),
                         np.log(np.asarray(ndr, dtype=float) + 0.05), onehot])
    yhat = area_model.predict(X)
    return np.clip(np.exp(yhat) - 1e-3, 0.0, hex_area_km2)


# ---------------------------------------------------------------------------
# 7) Full calibration orchestrator
# ---------------------------------------------------------------------------
def calibrate(prov_bio: pd.DataFrame, pass_obs: pd.DataFrame,
              hex_pyromes: pd.DataFrame, cloud_df: pd.DataFrame | None = None):
    """Run the full 2012-01-20..2021-12-31 calibration.

    Returns (calibration_artifacts dict, pass_observations_calibrated frame).
    """
    artifacts = {}
    # ---- reliability -----------------------------------------------------
    reliability_models = fit_isotonic_reliability(prov_bio)
    w = reliability_weights(prov_bio, reliability_models)
    prov_bio = prov_bio.copy()
    prov_bio["w_rel_cal"] = w
    artifacts["reliability_models"] = reliability_models
    # recompute clean weighted pass FRP (w enters AFTER calibration here)
    prov_bio["w_frp"] = prov_bio["bowtie_weight"] * prov_bio["w_rel_cal"] \
        * prov_bio["frp_mw"].fillna(0.0)
    wsum = prov_bio.groupby(["h3_index", "solar_date", "platform", "dn_flag"],
                            as_index=False).agg(
        y_frp_mw=("w_frp", "sum"), w_mean=("w_rel_cal", "mean"))
    pass_obs = pass_obs.merge(
        wsum, on=["h3_index", "solar_date", "platform", "dn_flag"],
        how="left", suffixes=("", "_cal"))
    pass_obs["y_frp_mw"] = pass_obs["y_frp_mw_cal"].fillna(
        pass_obs["y_frp_mw"])
    pass_obs["w_mean"] = pass_obs["w_mean_cal"].fillna(pass_obs.get(
        "w_mean", 1.0))
    pass_obs = pass_obs.drop(columns=[c for c in pass_obs.columns
                                      if c.endswith("_cal")])

    # ---- join pyromes -----------------------------------------------------
    pass_obs = pass_obs.merge(
        hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index", how="left")
    pass_obs["pyrome_id"] = pass_obs["pyrome_id"].fillna(0).astype(int)

    # ---- log-normal cluster intensity + survival integrals ----------------
    clusters = extract_cluster_table(prov_bio)
    clusters = clusters.merge(
        hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index", how="left")
    clusters["pyrome_id"] = clusters["pyrome_id"].fillna(0).astype(int)
    clusters["vza_deg"] = 20.0        # cluster-level VZA unknown here
    clusters["cloud_frac"] = 0.0
    lognorm_params = fit_lognorm_params(clusters)
    artifacts["fmin_params"] = {k: v for k, v in S.FMIN_PARAMS.items()}
    artifacts["lognorm_params"] = lognorm_params

    # ---- gamma transfer ----------------------------------------------------
    lgbm, eqm_knots, eqm_anchors, fallback = fit_gamma_calibration(
        pass_obs, lognorm_params)
    artifacts["lgb_gamma"] = lgbm
    artifacts["eqm_knots"] = eqm_knots
    artifacts["eqm_anchors"] = eqm_anchors
    artifacts["_gamma_fallback"] = fallback

    # ---- diurnal curves -----------------------------------------------------
    pairs = build_diurnal_pairs(pass_obs, lgbm, eqm_knots, eqm_anchors,
                                lognorm_params)
    diurnal_coeffs = fit_diurnal_curves(pairs)
    if not diurnal_coeffs:
        diurnal_coeffs = {(0, "DJF", "D"): {"coef": np.zeros(4),
                                            "c0": {"D": 0.0, "N": 0.0}}}
    artifacts["diurnal_coeffs"] = diurnal_coeffs

    # ---- sigma^2 OLS ---------------------------------------------------------
    sigma_ols = fit_sigma_ols(pairs, diurnal_coeffs)
    artifacts["sigma_ols"] = sigma_ols

    # ---- area model -----------------------------------------------------------
    area_model = fit_area_model(pass_obs, diurnal_coeffs, lognorm_params)
    artifacts["area_model"] = area_model

    # ---- cloud climatology (trailing-2026 fallback) -----------------------------
    if cloud_df is not None and len(cloud_df):
        artifacts["cloud_climatology"] = build_cloud_climatology(cloud_df)
    else:
        artifacts["cloud_climatology"] = pd.DataFrame(
            columns=["lat_025", "lon_025", "month", "cloud_frac_clim"])

    # ---- assemble calibrated pass observations --------------------------------
    out = pass_obs.copy()
    fam = out["sensor_family"].to_numpy()
    ln_gamma = np.zeros(len(out))
    m = fam == "MODIS"
    if m.any():
        ln_gamma[m] = log_gamma_modis(
            np.log(np.maximum(out.loc[m, "y_frp_mw"].to_numpy(dtype=float), 1e-6)),
            out.loc[m, "pyrome_id"], out.loc[m, "dn_flag"].to_numpy() == "D",
            out.loc[m, "mean_vza_deg"], out.loc[m, "mean_area_km2"],
            out.loc[m, "raw_event_count"], lgbm, eqm_knots, lognorm_params,
            fallback, eqm_anchors)
    out["ln_gamma"] = ln_gamma
    months = pd.to_datetime(out["solar_date"]).dt.month
    out["ln_diurnal"] = np.log(np.maximum(diurnal_gain(
        diurnal_coeffs, out["pyrome_id"], months, out["dn_flag"],
        out["mean_solar_hour"]), 1e-6))
    ln_psi_eff = np.zeros(len(out))
    for family in ("MODIS", "VIIRS"):
        mm = fam == family
        if mm.any():
            ln_psi_eff[mm] = np.log(psi_frp_eff(
                family, out.loc[mm, "pyrome_id"], out.loc[mm, "mean_area_km2"],
                out.loc[mm, "dn_flag"], out.loc[mm, "raw_event_count"],
                lognorm_params))
    out["ln_psi_eff"] = ln_psi_eff
    out["mu_gain_det"] = out["ln_diurnal"] + out["ln_gamma"] + out["ln_psi_eff"]
    out["sigma2"] = sigma2_of(
        sigma_ols, fam == "MODIS", out["mean_vza_deg"], out["w_mean"],
        np.zeros(len(out)), out["has_i4_m13_fallback"])
    out.loc[out["has_i4_m13_fallback"] > 0, "sigma2"] = np.clip(
        out.loc[out["has_i4_m13_fallback"] > 0, "sigma2"]
        * S.VIIRS_M13_SIGMA_INFLATION ** 2, *S.SIGMA2_CLIP)
    return artifacts, out


def build_cloud_climatology(cloud_df: pd.DataFrame) -> pd.DataFrame:
    """2012-2025 multi-year mean cloud fraction per (0.25 deg cell, month)."""
    df = cloud_df.copy()
    df["month"] = np.where(df["period_kind"] == "month", df["period"],
                           ((df["period"] - 1) * 8 // 30) + 1)
    clim = (df[df["year"].between(2012, 2025)]
            .groupby(["lat_025", "lon_025", "month"], as_index=False)["cloud_frac"]
            .mean().rename(columns={"cloud_frac": "cloud_frac_clim"}))
    return clim


def save_artifacts(artifacts: dict, path) -> None:
    with open(str(path), "wb") as fh:
        pickle.dump(artifacts, fh)
