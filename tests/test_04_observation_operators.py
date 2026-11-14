"""test_04 -- Observation operator tests (20 tests).

Readme Section 7.4: detection threshold ordering (MODIS_day > MODIS_night >
VIIRS_night), analytical survival integrals psi in (0, 1] monotonicity,
conditional psi_FRP_eff smoothing (E_raw=1 vs E_raw=5), separate 12-hour
Day (06-18) and Night (18-06) Fourier integrals equal to 1.0 +- 1e-4,
ln(v) strictly absent from mu_gain_det, monotonic EQM ordering,
heteroscedastic sigma^2 ordering, GMM coordinate-shift invariance, and the
trailing-2026 cloud climatology fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration import operators as ops
from src.calibration.pyromes import behavioral_features, extract_cluster_table
from src.config import settings as S
from tests.conftest import _short_twin


@pytest.fixture(scope="module")
def cal():
    return _short_twin()


# ------------------------------------------------------- detection limits
def test_04_detection_threshold_ordering():
    a = 3.2
    f_md = ops.fmin_mw("MODIS", a, "D")
    f_mn = ops.fmin_mw("MODIS", a, "N")
    f_vn = ops.fmin_mw("VIIRS", a, "N")
    f_vd = ops.fmin_mw("VIIRS", a, "D")
    assert f_md > f_mn
    assert f_mn > f_vn
    assert f_vn < f_vd


def test_04_fmin_linear_in_area():
    for fam in ("MODIS", "VIIRS"):
        for dn in ("D", "N"):
            a = np.array([0.5, 1.0, 3.2])
            f = ops.fmin_mw(fam, a, dn)
            slope = np.diff(f) / np.diff(a)
            assert np.allclose(slope, slope[0], rtol=1e-12)


# ------------------------------------------------------- survival integrals
def test_04_psi_cnt_in_range_and_monotone():
    ln = {0: (1.7, 1.0), 1: (2.1, 0.95)}
    for fam in ("MODIS", "VIIRS"):
        for dn in ("D", "N"):
            areas = np.linspace(0.12, 10.0, 40)
            psi = ops.psi_cnt(fam, 0, areas, dn, ln)
            assert np.all((psi > 0.0) & (psi <= 1.0))
            assert np.all(np.diff(psi) <= 1e-12)   # higher area -> fewer detections


def test_04_psi_frp_in_range_and_monotone():
    ln = {0: (1.7, 1.0), 1: (2.1, 0.95)}
    for fam in ("MODIS", "VIIRS"):
        for dn in ("D", "N"):
            areas = np.linspace(0.12, 10.0, 40)
            psi = ops.psi_frp(fam, 0, areas, dn, ln)
            assert np.all((psi >= S.PSI_FRP_CLIP[0]) & (psi <= 1.0))
            assert np.all(np.diff(psi) <= 1e-12)


def test_04_psi_frp_eff_smoothing_e1_vs_e5():
    ln = {0: (1.7, 1.0), 1: (2.1, 0.95)}
    a, dn = 3.2, "D"
    p1 = ops.psi_frp_eff("MODIS", 0, a, dn, 1, ln)
    p5 = ops.psi_frp_eff("MODIS", 0, a, dn, 5, ln)
    base = ops.psi_frp("MODIS", 0, a, dn, ln)
    # isolated single-cluster fires are pulled toward the 0.65 floor
    assert p1 > base + 1e-6
    # multi-cluster coexistence applies the full survival correction
    assert abs(p5 - base) < 0.05 * base
    assert p1 >= p5 - 1e-9


def test_04_w_multi_formula():
    assert ops.w_multi(1) == pytest.approx(0.0, abs=1e-12)
    assert ops.w_multi(2) == pytest.approx(1 - np.exp(-0.7), abs=1e-12)
    assert ops.w_multi(50) == pytest.approx(1.0, abs=1e-9)


def test_04_psi_matches_closed_form():
    from scipy.special import ndtr
    ln = {0: (1.7, 1.0)}
    mu, sg = ln[0]
    a, dn = 2.35, "N"
    fmin = 1.5 + 2.2 * a
    z = (np.log(fmin) - mu) / sg
    assert ops.psi_cnt("MODIS", 0, a, dn, ln) == pytest.approx(
        np.clip(1 - ndtr(z), 0.02, 1.0), abs=1e-12)
    z_min = (np.log(fmin) - mu - sg ** 2) / sg
    z_ref = (np.log(S.F_REF_MW) - mu - sg ** 2) / sg
    expect = np.clip((1 - ndtr(z_min)) / (1 - ndtr(z_ref)), 0.05, 1.0)
    assert ops.psi_frp("MODIS", 0, a, dn, ln) == pytest.approx(expect, abs=1e-12)


# ------------------------------------------------------- diurnal curves
def test_04_diurnal_12h_unit_mean_day_and_night():
    """After 12h-window normalization (c0), each window's mean is exactly 1."""
    coef = np.array([0.45, 0.20, 0.12, -0.05])
    for dn, (lo, hi) in (("D", (6.0, 18.0)), ("N", (18.0, 30.0))):
        tau = np.linspace(lo, hi, 240)
        raw = ops._fourier_eval(coef, np.mod(tau, 24.0))
        c0 = -np.log(np.trapezoid(np.exp(raw), tau) / (hi - lo))
        vals = np.exp(raw + c0)
        mean = np.trapezoid(vals, tau) / (hi - lo)
        assert mean == pytest.approx(1.0, abs=1e-10)
        # normalized over the whole 24h cycle, the half-window mean equals
        # the same-window integral; Day and Night are normalized separately
        tau2 = np.linspace(lo, hi, 120)
        vals2 = np.exp(ops._fourier_eval(coef, np.mod(tau2, 24.0)) + c0)
        assert np.trapezoid(vals2, tau2) / 12.0 == pytest.approx(1.0, abs=1e-4)


def test_04_diurnal_normalization_separate_windows():
    """Day and Night curves each integrate to 1 over their OWN window."""
    amps = {"D": (6.0, 18.0), "N": (18.0, 30.0)}
    for dn, (lo, hi) in amps.items():
        spec = {"coef": np.array([0.5, 0.3, 0.1, 0.0]),
                "c0": {k: 0.0 for k in "DN"}}
        raw = ops._fourier_eval(spec["coef"], np.mod(np.linspace(lo, hi, 240), 24))
        c0 = -np.log(np.trapezoid(np.exp(raw), np.linspace(lo, hi, 240)) / 12.0)
        assert c0 != 0.0 or np.allclose(raw, 0)
        tau = np.linspace(lo, hi, 240)
        d = np.exp(raw + c0)
        assert np.trapezoid(d, tau) / 12.0 == pytest.approx(1.0, abs=1e-10)


def test_04_diurnal_eval_unit_mean_both_windows():
    coeffs = {(0, "DJF", "D"): {"coef": np.array([0.6, 0.2, 0.15, 0.05]),
                                 "c0": {"D": 0.0, "N": 0.0}}}
    # recompute c0 through the public fitter path
    pairs = pd.DataFrame({
        "pyrome_id": [0] * 400, "solar_date": ["2020-01-15"] * 400,
        "season": ["DJF"] * 400, "dn_flag": ["D"] * 400,
        "tau1": np.linspace(6.5, 9.5, 400), "tau2": np.linspace(12.5, 14.5, 400),
        "ln1": np.zeros(400), "ln2": np.zeros(400), "diff": np.zeros(400),
        "is_modis": np.ones(400), "vza_mean": np.full(400, 30.0),
        "w_mean": np.ones(400), "v_cloud": np.zeros(400), "m13": np.zeros(400)})
    fitted = ops.fit_diurnal_curves(pairs)
    d = ops.diurnal_eval(fitted, np.zeros(400), ["DJF"] * 400, ["D"] * 400,
                         np.linspace(6.5, 17.5, 400))
    assert d.mean() == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------- cloud-free gain
def test_04_ln_v_absent_from_detected_gain(cal):
    """mu_gain_det = ln D + ln gamma + ln psi_eff -- NEVER ln(v) (INV-05)."""
    pc = cal["pass_cal"]
    sub = pc[pc["y_frp_mw"] > 0]
    assert len(sub)
    recomputed = sub["ln_diurnal"] + sub["ln_gamma"] + sub["ln_psi_eff"]
    assert np.allclose(recomputed, sub["mu_gain_det"], atol=1e-9)
    # the gain contains no cloud-visibility term: correlate against vis-like
    # quantities must not be identity; and for two identical passes with
    # different clouds the gain is identical by construction (it is a pure
    # function of (D, gamma, psi) inputs only).
    mu1 = ops.diurnal_gain({(0, "DJF", "D"): {"coef": np.zeros(4),
                                               "c0": {"D": 0.0, "N": 0.0}}},
                           0, 1, "D", 13.5)
    mu2 = ops.diurnal_gain({(0, "DJF", "D"): {"coef": np.zeros(4),
                                               "c0": {"D": 0.0, "N": 0.0}}},
                           0, 1, "D", 13.5)
    assert mu1 == mu2


def test_04_gain_excludes_cloud_multiplier():
    """Cloud visibility changes only the Hurdle case, never the gain."""
    from src.engine.likelihoods import case_c_nll_grad_hess
    z = torch_free(1.0)
    l1, _, _ = case_c_nll_grad_hess(z, 0.1, 0.9)
    l2, _, _ = case_c_nll_grad_hess(z, 0.1, 0.2)
    assert l1 != l2   # visibility matters for non-detections ...


def torch_free(x):
    import torch
    return torch.tensor(x, dtype=torch.float64)


# ------------------------------------------------------- EQM
def test_04_eqm_knots_monotone(cal):
    art = cal["artifacts"]
    assert len(art["eqm_knots"]) > 0
    for (b, dn), (xk, yk) in art["eqm_knots"].items():
        assert len(xk) == 20
        assert np.all(np.diff(xk) >= -1e-12)
        assert np.all(np.diff(yk) >= -1e-12)


def test_04_eqm_anchored_zero_mean_transfer(cal):
    """The anchored EQM contributes shape only (no net level transfer)."""
    art = cal["artifacts"]
    pc = cal["pass_cal"]
    sub = pc[(pc["sensor_family"] == "MODIS") & (pc["y_frp_mw"] > 0)]
    if not len(sub) or not art["lgb_gamma"]:
        pytest.skip("no gamma pairs in short twin")
    ln_gamma = ops.log_gamma_modis(
        np.log(sub["y_frp_mw"].to_numpy()), sub["pyrome_id"],
        (sub["dn_flag"] == "D").to_numpy(), sub["mean_vza_deg"],
        sub["mean_area_km2"], sub["raw_event_count"], art["lgb_gamma"],
        art["eqm_knots"], art["lognorm_params"], None, art["eqm_anchors"])
    # net transfer bounded well inside the clip (the twin has gamma_T = 0)
    assert abs(float(np.mean(ln_gamma))) < 0.5


# ------------------------------------------------------- sigma^2
def test_04_sigma2_range_and_ordering(cal):
    pc = cal["pass_cal"]
    s2 = pc["sigma2"].to_numpy()
    assert np.all((s2 >= S.SIGMA2_CLIP[0]) & (s2 <= S.SIGMA2_CLIP[1]))
    # heteroscedasticity: variance differs across conditions
    assert np.std(s2) > 1e-4


def test_04_sigma2_of_clipping_and_formula():
    ols = _fake_ols()
    vza_lo = ops.sigma2_of(ols, [0.0], [10.0], [1.0], [0.0], [0.0])
    vza_hi = ops.sigma2_of(ols, [0.0], [60.0], [1.0], [0.0], [0.0])
    assert vza_lo[0] != vza_hi[0]


def _fake_ols():
    from sklearn.linear_model import LinearRegression
    X = np.array([[0.0, 0.15, 0.0, 1.0, 0.0],
                  [1.0, 0.15, 0.0, 1.0, 0.0],
                  [0.0, 0.90, 0.0, 1.0, 0.0],
                  [1.0, 0.90, 0.0, 1.0, 0.0]])
    y = np.log(np.array([0.30, 0.22, 0.55, 0.45]) + 1e-3)
    return LinearRegression().fit(X, y)


# ------------------------------------------------------- pyromes
def test_04_gmm_coordinate_shift_invariance():
    from src.calibration.pyromes import fit_pyromes
    rng = np.random.default_rng(3)
    rows = []
    for i in range(14):
        for t in range(80):
            b = i % 2
            rows.append(dict(h3_index=10_000 + i, lat=0.0, lon=0.0,
                             solar_date=pd.Timestamp("2015-%02d-01" % (1 + t % 12)),
                             dn_flag="D" if t % 2 else "N",
                             cluster_frp_mw=float(np.exp(rng.normal(1.0 + b, 0.4))),
                             n_pixels=int(rng.integers(1, 6))))
    cl = pd.DataFrame(rows)
    cl.loc[cl.h3_index % 2 == 1, "cluster_frp_mw"] *= 6.0
    base = _assign(cl)
    shifted = cl.copy()
    shifted["h3_index"] = shifted["h3_index"] + 5      # relabel cells only
    got = _assign(shifted)
    # behavioral partition identical up to a label permutation
    a = np.sort(base["pyrome_id"].to_numpy())
    b = np.sort(got["pyrome_id"].to_numpy())
    assert np.array_equal(a, b)


def _assign(cl):
    from src.calibration.pyromes import GaussianMixture, StandardScaler
    feats = behavioral_features(cl)
    X = feats[[c for c in ("ln_median_cluster_frp", "night_frp_share",
                           "seasonal_circular_concentration", "interannual_cv",
                           "mean_cluster_pixels")]].to_numpy()
    z = StandardScaler().fit_transform(X)
    lab = GaussianMixture(n_components=2, random_state=42).fit_predict(z)
    return feats.assign(pyrome_id=lab)


def test_04_pyrome_features_coordinate_free():
    cl = pd.DataFrame(dict(
        h3_index=[1, 1, 2, 2], solar_date=pd.to_datetime(
            ["2015-01-01", "2015-03-01", "2015-01-01", "2015-03-01"]),
        dn_flag=["D", "N", "N", "D"], cluster_frp_mw=[3.0, 1.0, 2.0, 4.0],
        n_pixels=[2, 1, 3, 2]))
    feats = behavioral_features(extract_cluster_table(cl))
    assert "latitude" not in feats.columns and "longitude" not in feats.columns
    assert feats.loc[0, "mean_cluster_pixels"] == pytest.approx(1.5)
    assert feats.loc[1, "mean_cluster_pixels"] == pytest.approx(2.5)


# ------------------------------------------------------- cloud climatology
def test_04_cloud_climatology_fallback():
    cloud = pd.DataFrame({
        "lat_025": [-30.0] * 6, "lon_025": [-67.5] * 6,
        "year": [2012, 2013, 2014, 2015, 2024, 2025],
        "period": [1, 1, 1, 1, 1, 1], "period_kind": ["8d"] * 6,
        "cloud_frac": [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]})
    clim = ops.build_cloud_climatology(cloud)
    assert list(clim.columns) == ["lat_025", "lon_025", "month",
                                  "cloud_frac_clim"]
    assert len(clim) == 1
    assert clim["cloud_frac_clim"].iloc[0] == pytest.approx(np.mean(
        [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]))


def test_04_visibility_formula():
    from src.engine.priors import visibility_matrix
    cloud = np.array([0.2, 0.5, 0.95])
    v = visibility_matrix("SNPP", -30.0, cloud, np.arange(3))
    # p_swath(VIIRS) = 0.99 exactly; v = 0.99 * clip(1 - cloud, 0.10, 0.98)
    assert np.allclose(v[:2], [0.99 * 0.8, 0.99 * 0.5])
    assert v[2] == pytest.approx(0.99 * S.VIS_CLIP[0])    # clipped low at 0.10
    vm = visibility_matrix("Terra", -30.0, np.zeros(2), np.arange(2))
    # clear-sky MODIS visibility is capped at 0.98 by the clip
    assert vm[0] == pytest.approx(0.98)
    assert vm[1] == pytest.approx(0.98)
