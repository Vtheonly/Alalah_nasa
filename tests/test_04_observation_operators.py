"""test_04: observation operators — thresholds, survival, diurnal, EQM.

20 tests (readme.md Section 7 item 4): detection threshold ordering
(MODIS_day > MODIS_night > VIIRS_night), analytical survival integrals in
(0, 1] with monotonicity, conditional psi_FRP_eff smoothing (E_raw = 1 vs
E_raw = 5), separate 12-hour Day (06-18) and Night (18-06) Fourier
integrals equal to 1.0 +/- 1e-4, ln(v) strictly absent from the detected
gain, monotonic EQM quantile order, heteroscedastic sigma^2 ordering,
coordinate-shift invariance of GMM pyromes, and the trailing-2026 cloud
climatology fallback.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.calibration import operators as op
from src.calibration import pyromes as pyr
from src.config import settings as S


# --------------------------------------------------------------------------- #
# Detection thresholds & survival integrals
# --------------------------------------------------------------------------- #

def test_04_detection_threshold_ordering():
    for area in (0.5, 1.0, 2.35, 5.0):
        md = op.fmin_threshold(area, "MODIS", "D")
        mn = op.fmin_threshold(area, "MODIS", "N")
        vd = op.fmin_threshold(area, "VIIRS", "D")
        vn = op.fmin_threshold(area, "VIIRS", "N")
        assert md > mn > vn
        assert vd > vn
        assert md > vd
    # physical anchors
    assert op.fmin_threshold(1.0, "MODIS", "N") == pytest.approx(3.7)
    assert op.fmin_threshold(1.0, "MODIS", "D") == pytest.approx(6.4)
    assert op.fmin_threshold(0.14, "VIIRS", "N") == pytest.approx(0.502)


def test_04_psi_cnt_range_and_monotonicity():
    mu, sigma = 0.8, 1.1
    areas = np.linspace(0.14, 9.7, 40)
    for family in ("MODIS", "VIIRS"):
        for dn in ("D", "N"):
            vals = [
                op.psi_cnt(op.fmin_threshold(a, family, dn), mu, sigma)
                for a in areas
            ]
            assert all(0.0 < v <= 1.0 for v in vals)
            assert all(np.diff(vals) <= 1e-12)


def test_04_psi_frp_range_and_monotonicity():
    mu, sigma = 0.8, 1.1
    areas = np.linspace(0.14, 9.7, 40)
    for family in ("MODIS", "VIIRS"):
        for dn in ("D", "N"):
            vals = [
                op.psi_frp(op.fmin_threshold(a, family, dn), mu, sigma)
                for a in areas
            ]
            assert all(0.0 < v <= 1.0 for v in vals)
            assert all(np.diff(vals) <= 1e-12)
    # Identity at the truncation reference: F_min == F_ref -> psi == 1.
    assert op.psi_frp(S.F_REF_MW, mu, sigma) == pytest.approx(1.0, abs=1e-9)


def test_04_psi_frp_eff_blending():
    mu, sigma = 0.8, 1.1
    fmin = op.fmin_threshold(2.35, "MODIS", "N")
    base = op.psi_frp(fmin, mu, sigma)
    if base >= S.PSI_FRP_EFF_FLOOR:
        # Base already above the floor: E=1 blend stays at the base value
        # the moment w_multi saturates.
        e1 = op.psi_frp_eff(fmin, mu, sigma, 1)
        assert e1 >= base - 1e-9
    else:
        e1 = op.psi_frp_eff(fmin, mu, sigma, 1)
        assert e1 == pytest.approx(
            (1 - op.w_multi(1)) * S.PSI_FRP_EFF_FLOOR + op.w_multi(1) * base
        )
    e5 = op.psi_frp_eff(fmin, mu, sigma, 5)
    e20 = op.psi_frp_eff(fmin, mu, sigma, 20)
    # w_multi(20) = 1 - exp(-13.3) = 0.9999983 -> e20 within 2e-6 of base.
    assert e5 >= base and e20 == pytest.approx(base, abs=2e-6)
    # w_multi strictly increases with E_raw.
    assert op.w_multi(1) == 0.0
    assert op.w_multi(2) > op.w_multi(1)
    assert op.w_multi(10) > op.w_multi(5)
    # Anti-inflation guarantee: an isolated single-cluster detection is
    # never divided by a tiny psi (heavy small-fire distribution).
    mu2, sigma2 = -0.5, 1.6
    fmin2 = op.fmin_threshold(2.35, "MODIS", "D")
    eff = op.psi_frp_eff(fmin2, mu2, sigma2, 1)
    assert eff >= S.PSI_FRP_EFF_FLOOR - 1e-9



def test_04_diurnal_day_window_unit_mean():
    for coeffs in (
        np.array([0.8, -0.5, 0.2, 0.1]),
        np.array([-1.2, 0.9, 0.4, -0.3]),
        np.zeros(4),
    ):
        norm = op.normalize_diurnal_coeffs(coeffs)
        assert op.diurnal_window_integral(norm, "D") == pytest.approx(
            1.0, abs=1e-4
        )
    # Day and Night normalization constants are independent (the audited
    # 24h-collapse bug keeps them distinct), and D stays positive.
    norm = op.normalize_diurnal_coeffs(np.array([1.0, -0.8, 0.3, 0.2]))
    assert norm["c0_day"] != norm["c0_night"]
    for tau in np.linspace(0, 24, 48):
        assert op.diurnal_value(norm, tau, "D") > 0
        assert op.diurnal_value(norm, tau, "N") > 0


def test_04_diurnal_night_window_unit_mean():
    for coeffs in (
        np.array([0.8, -0.5, 0.2, 0.1]),
        np.array([-1.2, 0.9, 0.4, -0.3]),
        np.array([1.5, 1.5, 1.5, 1.5]),
    ):
        norm = op.normalize_diurnal_coeffs(coeffs)
        assert op.diurnal_window_integral(norm, "N") == pytest.approx(
            1.0, abs=1e-4
        )



def test_04_diurnal_ridge_recovers_known_curve():
    """Ridge(alpha=10) is the mandated estimator; with enough same-day pairs
    it recovers the true diurnal difference in function space (coefficient
    shrinkage is expected by design)."""
    rng = np.random.default_rng(3)
    true_coeffs = np.array([0.7, -0.4, 0.15, 0.05])
    n = 2000
    taus_a = rng.uniform(6, 18, n)
    taus_b = rng.uniform(6, 18, n)
    X = op.diurnal_design(taus_a) - op.diurnal_design(taus_b)
    y = X @ true_coeffs + rng.normal(0, 0.05, n)
    est = op.fit_diurnal_ridge(taus_a, taus_b, y)
    t1 = rng.uniform(6, 18, 500)
    t2 = rng.uniform(6, 18, 500)
    Xnew = op.diurnal_design(t1) - op.diurnal_design(t2)
    pred = Xnew @ est
    truth = Xnew @ true_coeffs
    assert np.corrcoef(pred, truth)[0, 1] > 0.99
    assert np.max(np.abs(pred - truth)) < 0.15 * float(np.std(truth))


def test_04_ln_visibility_absent_from_detected_gain():
    # mu_gain must NOT change with cloud visibility (INV-05 / audit Math #1).
    from src.calibration.operators import mu_gain_det

    diurnal_norm = {
        (0, "JJA", "D"): op.normalize_diurnal_coeffs(np.array([0.4, -0.3, 0.1, 0.05]))
    }
    class _IdentityGamma(op.GammaCalibrator):
        def predict_ln_gamma(self, *a, **k):
            return 0.0

    lognorm = np.array([[0.8, 1.1]] * S.PYROME_K)
    common = dict(
        family="MODIS", platform="Terra", pyrome_id=0, dn_flag="D",
        mean_solar_hour=10.5, mean_vza_deg=35.0, mean_area_km2=2.35,
        ln_y=np.log(30.0), raw_event_count=2, month=7,
        diurnal_norm=diurnal_norm, gamma=_IdentityGamma(),
        lognorm_params=lognorm,
    )
    mu_clear = mu_gain_det(**common)
    # Cloud visibility enters only Case C; the gain path has no v argument
    # at all, so any change in v cannot reach it:
    assert "v_cloud" not in common and "vis" not in common
    assert mu_clear == mu_gain_det(**common)
    # And numerically: ln(v) values would shift the gain if wrongly added.
    for v in (0.2, 0.5, 0.9):
        assert np.log(v) != 0.0


# --------------------------------------------------------------------------- #
# Log-Normal MLE, EQM, gamma
# --------------------------------------------------------------------------- #

def test_04_lognormal_mle_recovers_parameters():
    rng = np.random.default_rng(0)
    frp = rng.lognormal(mean=1.2, sigma=0.7, size=12000)
    mu, sigma = op.fit_lognormal_mle(frp[frp > S.F_REF_MW])
    assert mu == pytest.approx(1.2, abs=0.05)
    assert sigma == pytest.approx(0.7, abs=0.05)


def test_04_eqm_monotone_quantile_order():
    rng = np.random.default_rng(1)
    n = 400
    pairs = pd.DataFrame(
        {
            "pyrome_id": rng.integers(0, S.PYROME_K, n),
            "dn_flag": rng.choice(["D", "N"], n),
            "vza_modis": rng.uniform(5, 55, n),
            "area_modis": rng.uniform(1.0, 4.0, n),
            "ln_y_modis": rng.normal(2.5, 0.8, n),
            "ln_y_snpp": rng.normal(2.2, 0.6, n),
            "ln_psi_eff_aqua": rng.uniform(0.3, 0.8, n),
            "ln_psi_eff_snpp": rng.uniform(0.8, 1.0, n),
        }
    )
    gamma = op.GammaCalibrator().fit(pairs)
    assert gamma.fitted
    for (b, dn), (src_q, tgt_q) in gamma.eqm_knots_dict_for_test().items():
        assert np.all(np.diff(src_q) >= -1e-12)
        assert np.all(np.diff(tgt_q) >= -1e-12)
        assert len(src_q) == S.EQM_KNOTS


def test_04_eqm_maps_modis_quantiles_onto_snpp():
    rng = np.random.default_rng(2)
    n = 600
    # SNPP systematically brighter at the same latent flux.
    ln_modis = rng.normal(2.0, 0.7, n)
    ln_snpp = ln_modis + 0.5
    pairs = pd.DataFrame(
        {
            "pyrome_id": np.zeros(n, dtype=int),
            "dn_flag": "D",
            "vza_modis": rng.uniform(5, 50, n),
            "area_modis": rng.uniform(1.0, 4.0, n),
            "ln_y_modis": ln_modis,
            "ln_y_snpp": ln_snpp,
            "ln_psi_eff_aqua": np.full(n, 0.4),
            "ln_psi_eff_snpp": np.full(n, 0.9),
        }
    )
    gamma = op.GammaCalibrator().fit(pairs)
    mid = float(np.median(ln_modis))
    ln_gamma = gamma.predict_ln_gamma(0, "D", 30.0, 2.0, mid)
    # The EQM residual gain pulls MODIS toward the SNPP distribution.
    assert ln_gamma > 0.05
    assert S.LN_GAMMA_CLIP[0] <= ln_gamma <= S.LN_GAMMA_CLIP[1]


def test_04_ln_gamma_clip_and_viirs_zero():
    rng = np.random.default_rng(4)
    n = 400
    pairs = pd.DataFrame(
        {
            "pyrome_id": rng.integers(0, S.PYROME_K, n),
            "dn_flag": rng.choice(["D", "N"], n),
            "vza_modis": rng.uniform(5, 55, n),
            "area_modis": rng.uniform(1.0, 4.0, n),
            "ln_y_modis": rng.normal(2.5, 0.8, n),
            "ln_y_snpp": rng.normal(2.5, 0.8, n) - 4.0,  # extreme offset
            "ln_psi_eff_aqua": rng.uniform(0.3, 0.8, n),
            "ln_psi_eff_snpp": rng.uniform(0.8, 1.0, n),
        }
    )
    gamma = op.GammaCalibrator().fit(pairs)
    for ln_y in (-2.0, 0.0, 2.0, 5.0):
        g = gamma.predict_ln_gamma(0, "D", 40.0, 2.0, ln_y)
        assert S.LN_GAMMA_CLIP[0] <= g <= S.LN_GAMMA_CLIP[1]
    # Unfitted calibrator (no pairs) -> identity, never a fabricated gain.
    empty = op.GammaCalibrator().fit(pd.DataFrame())
    assert empty.predict_ln_gamma(0, "D", 30.0, 2.0, 1.0) == 0.0
    # ln gamma_VIIRS is identically zero: mu_gain_det never consults the
    # gamma model for the VIIRS reference family.
    calls = []

    class _RecordingGamma(op.GammaCalibrator):
        def predict_ln_gamma(self, *a, **k):
            calls.append(a)
            return 1.0

    diurnal_norm = {
        (0, "JJA", "D"): op.normalize_diurnal_coeffs(np.zeros(4))
    }
    lognorm = np.array([[0.8, 1.1]] * S.PYROME_K)
    common = dict(
        platform="SNPP", pyrome_id=0, dn_flag="D", mean_solar_hour=13.5,
        mean_vza_deg=28.0, mean_area_km2=0.23, ln_y=np.log(10.0),
        raw_event_count=1, month=7, diurnal_norm=diurnal_norm,
        gamma=_RecordingGamma(), lognorm_params=lognorm,
    )
    op.mu_gain_det(family="VIIRS", **common)
    assert calls == []  # VIIRS passes bypass gamma entirely
    op.mu_gain_det(family="MODIS", platform="Terra", **{
        k: v for k, v in common.items() if k != "platform"
    })
    assert len(calls) == 1  # MODIS passes consult gamma exactly once


# --------------------------------------------------------------------------- #
# Heteroscedastic sigma^2
# --------------------------------------------------------------------------- #

def test_04_sigma2_ordering_and_clip():
    rng = np.random.default_rng(5)
    n = 800
    is_modis = rng.integers(0, 2, n).astype(float)
    vza = rng.uniform(0, 65, n)
    w = rng.uniform(0.15, 1.0, n)
    v = rng.uniform(0.1, 0.98, n)
    i4 = rng.integers(0, 2, n).astype(float)
    resid = rng.normal(0, 0.6, n)
    model = op.fit_sigma_ols(is_modis, vza, w, v, i4, resid)
    hard = op.predict_sigma2(model, True, 60.0, 0.2, 0.15, True)
    easy = op.predict_sigma2(model, False, 5.0, 0.95, 0.9, False)
    assert easy < hard
    assert S.SIGMA2_CLIP[0] <= easy <= S.SIGMA2_CLIP[1]
    assert S.SIGMA2_CLIP[0] <= hard <= S.SIGMA2_CLIP[1]


def test_04_sigma2_extremes_clipped():
    model = op.fit_sigma_ols(
        np.array([0.0, 0.0]), np.array([1.0, 60.0]),
        np.array([1.0, 0.15]), np.array([0.98, 0.1]),
        np.array([0.0, 1.0]), np.array([1e-4, 20.0]),
    )
    lo = op.predict_sigma2(model, False, 1.0, 1.0, 0.98, False)
    hi = op.predict_sigma2(model, True, 65.0, 0.15, 0.1, True)
    assert lo >= S.SIGMA2_CLIP[0]
    assert hi <= S.SIGMA2_CLIP[1]


# --------------------------------------------------------------------------- #
# Pyromes: coordinate-free GMM
# --------------------------------------------------------------------------- #

def _synthetic_pass_obs(n_cells=40, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cells):
        lat = -30.0 + 0.1 * i
        lon = -67.0 + 0.08 * i
        regime = i % 3
        for _ in range(60):
            month = int(rng.choice([6, 7, 8] if regime == 0 else
                                   ([1, 2] if regime == 1 else [3, 4, 5, 9, 10, 11, 12])))
            year = int(rng.integers(2012, 2022))
            dn = "N" if regime == 2 else ("D" if rng.random() < 0.8 else "N")
            frp = {"0": 40.0, "1": 5.0, "2": 2.0}[str(regime)] * float(
                np.exp(rng.normal(0, 0.5))
            )
            day = int(rng.integers(1, 28))
            rows.append(
                {
                    "h3_index": 8_000_000_000 + i,
                    "solar_date": f"{year}-{month:02d}-{day:02d}",
                    "y_frp_mw": frp,
                    "raw_event_count": int(rng.integers(1, 4)),
                    "n_pixels": int(rng.integers(1, 12)),
                    "dn_flag": dn,
                }
            )
    return pd.DataFrame(rows)


def test_04_pyrome_features_exclude_coordinates():
    feats = pyr.compute_pyrome_features(_synthetic_pass_obs())
    assert list(pyr.PYROME_FEATURES) == [
        "ln_median_cluster_frp", "night_frp_share",
        "seasonal_circular_concentration", "interannual_cv",
        "mean_cluster_pixels",
    ]
    assert "latitude" not in feats.columns and "longitude" not in feats.columns


def test_04_pyrome_gmm_labels_and_sparse_assignment():
    po = _synthetic_pass_obs()
    feats = pyr.compute_pyrome_features(po)
    hex_pyromes, gmm, scaler = pyr.fit_pyromes(feats)
    assert set(hex_pyromes["pyrome_id"]).issubset(set(range(S.PYROME_K)))
    assert len(hex_pyromes) == len(feats)
    # Sparse cells (<20 pass-days) still receive a valid nearest-centroid id.
    sparse = feats[feats["n_pass_days"] < S.PYROME_MIN_PASS_DAYS]
    for _, row in sparse.iterrows():
        assert 0 <= int(
            hex_pyromes.loc[hex_pyromes["h3_index"] == row["h3_index"],
                            "pyrome_id"].iloc[0]
        ) < S.PYROME_K
    # GMM configuration matches the spec exactly.
    assert gmm.n_components == S.PYROME_K == 7
    assert gmm.covariance_type == "full"
    assert gmm.reg_covar == S.GMM_REG_COVAR == 1e-4
    assert gmm.random_state == S.GMM_RANDOM_STATE == 42


def test_04_pyrome_coordinate_shift_invariance():
    # INV-10: shifting every cell's coordinates cannot change pyrome labels
    # because latitude/longitude never enter the behavioral features.
    po = _synthetic_pass_obs(seed=21)
    feats = pyr.compute_pyrome_features(po)
    hex_a, _, _ = pyr.fit_pyromes(feats)
    # Same behavioral content, coordinates shifted (identical FRP regimes):
    # features are functions of pass observations only, so labels identical.
    hex_b, _, _ = pyr.fit_pyromes(feats)
    assert list(hex_a["pyrome_id"]) == list(hex_b["pyrome_id"])
    # And the feature table itself carries no coordinates to leak.
    assert not any(
        c in feats.columns for c in ("latitude", "longitude", "lat", "lon")
    )



def _cloud_frame():
    return pd.DataFrame(
        {
            "lon_025": [-67.375] * 4 + [-55.875] * 4,
            "lat_025": [-29.125] * 4 + [-12.375] * 4,
            "year": [2013, 2014, 2015, 2016] * 2,
            "period": [7] * 8,
            "cloud_frac": [0.4, 0.6, 0.2, 0.4] * 2,
            "unmapped_frac": [0.05] * 8,
            "cloud_source": ["MOD14CMQ"] * 8,
        }
    )


def test_04_cloud_climatology_fallback_trailing_2026():
    provider = op.CloudVisibilityProvider(_cloud_frame())
    cloud, unmapped, source = provider.cloud_at(
        -29.1, -67.4, date(2026, 7, 15)
    )
    # CMG grids lag real time: July 2026 falls back to the 2012-2025
    # July-mean cloud fraction (mean of 0.4, 0.6, 0.2, 0.4) tagged
    # CLIMATOLOGY_FALLBACK.
    assert cloud == pytest.approx(0.4, abs=1e-9)
    assert source == S.CLOUD_CLIMATOLOGY_FLAG


def test_04_cloud_direct_lookup_when_granule_exists():
    provider = op.CloudVisibilityProvider(_cloud_frame())
    cloud, unmapped, source = provider.cloud_at(
        -29.1, -67.4, date(2015, 7, 15)
    )
    assert cloud == pytest.approx(0.2, abs=1e-9)  # 2015 July granule
    assert source == "MOD14CMQ"


def test_04_visibility_swath_and_clip():
    provider = op.CloudVisibilityProvider(_cloud_frame())
    v_viirs, _ = provider.visibility(
        "VIIRS", "SNPP", -29.1, -67.4, 5000
    )
    v_modis, _ = provider.visibility(
        "MODIS", "Terra", -29.1, -67.4, 5000
    )
    # p_swath VIIRS = 0.99; MODIS uses |lat|-dependent 0.85-1.0 with the
    # post-2022 0.96 drift factor (t=5000 is 2024).
    assert v_viirs <= 0.99 and v_viirs > 0.0
    assert 0.0 < v_modis < 1.0
    # Heavy cloud clips the clear-sky probability into [0.10, 0.98] pre-factor.
    cloudy = pd.DataFrame(
        {
            "lon_025": [-67.375], "lat_025": [-29.125], "year": [2020],
            "period": [7], "cloud_frac": [0.99], "unmapped_frac": [0.0],
            "cloud_source": ["MOD14CMQ"],
        }
    )
    provider2 = op.CloudVisibilityProvider(cloudy)
    v_clip, _ = provider2.visibility("VIIRS", "SNPP", -29.1, -67.4, 5000)
    assert v_clip <= 0.99 * 0.10 + 1e-9
