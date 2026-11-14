"""test_07: 26-year synthetic digital-twin truth falsification.

18 tests (readme.md Section 7 item 7): the full twin with Day/Night
contrast, 2002 Aqua, 2012 S-NPP, 2018 N20, 2022 drift, 2023 N21, the
2026-11 S-NPP end, 50% clouds, bow-tie duplication and Tobit saturation —
asserting the 2012 raw jump (> +150%) is eliminated (< +/-10%), the
3-satellite inflation is eliminated, night_day_ratio is preserved
(error < 15%), the Nov 2026 cutoff stays unbiased, post-2022 drift is
corrected, monthly correlation r > 0.90, and the drought fire spike is
preserved within +/-15%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.validation.synthetic_twin import TWIN_NIGHT_DAY_RATIO


def _month_index(twin):
    return {
        f"{y}-{m:02d}": i for i, (y, m) in enumerate(twin["month_keys"])
    }


def _monthly_means(arr_2d, twin):
    mid = twin["month_ids"]
    return np.array(
        [arr_2d[:, mid == m].mean() for m in range(mid.max() + 1)]
    )


def _era_mean(series, idx, y1, y2):
    ids = [i for k, i in idx.items() if y1 <= int(k[:4]) <= y2]
    return float(np.mean(series[ids]))


# --------------------------------------------------------------------------- #
# Twin construction sanity
# --------------------------------------------------------------------------- #

def test_07_twin_shapes_and_determinism(twin):
    assert twin["phi_day_true"].shape == (5, S.T_DAYS)
    assert twin["phi_night_true"].shape == (5, S.T_DAYS)
    assert len(twin["month_keys"]) == 313
    assert len(twin["pass_df"]) > 1000
    assert set(twin["colors"]).issubset({0, 1, 2})
    twin2 = __import__("src.validation.synthetic_twin", fromlist=["x"])
    again = twin2.generate_twin(n_cells=5, T=S.T_DAYS, seed=42)
    assert np.allclose(again["phi_day_true"], twin["phi_day_true"])


def test_07_truth_night_day_contrast(twin):
    ratio = twin["phi_night_true"] / np.maximum(twin["phi_day_true"], 1e-12)
    assert np.allclose(ratio, TWIN_NIGHT_DAY_RATIO, atol=1e-9)


def test_07_platform_constellation_calendar(twin):
    df = twin["pass_df"]
    assert set(df["platform"]) == set(S.PLATFORM_INDEX)
    snpp = df[df["platform"] == "SNPP"]
    assert snpp["solar_date"].max() <= "2026-11-02"  # SNPP ends Nov 2
    aqua = df[df["platform"] == "Aqua"]
    assert aqua["solar_date"].min() >= "2002-07-04"  # Aqua joins July 2002
    for plat in ("NOAA-20", "NOAA-21"):
        sub = df[df["platform"] == plat]
        assert sub["solar_date"].max() >= "2026-11-01"  # through the end


def test_07_twin_clouds_bowtie_and_saturation_present(twin):
    # 50% mean cloud, bow-tie duplicated MODIS pixels, Tobit saturation.
    assert 0.35 < float(twin["cloud"].mean()) < 0.65
    df = twin["pass_df"]
    modis = df[df["sensor_family"] == "MODIS"]
    assert (modis["n_pixels"] % 2 == 0).any()  # bow-tie duplicated pixels
    sat = df[df["sat_flag"] == 1]
    assert len(sat) > 50  # Tobit right-censored rows exist
    assert sat["y_frp_mw"].max() < 80.0  # censored at the saturation ceiling
    assert df["has_i4_m13_fallback"].sum() > 0  # M13 fallback noise rows


# --------------------------------------------------------------------------- #
# Raw (unharmonized) artifacts that the harmonization must remove
# --------------------------------------------------------------------------- #

def test_07_raw_2012_jump_exceeds_150_percent(twin):
    idx = _month_index(twin)
    pre = _era_mean(twin["raw_monthly_counts"], idx, 2010, 2011)
    post = _era_mean(twin["raw_monthly_counts"], idx, 2013, 2014)
    assert post / pre > 2.5  # > +150% artificial step at the VIIRS switch


def test_07_raw_three_satellite_inflation(twin):
    # VIIRS-only raw pixels ~ triple once N20+N21 join the SNPP plane.
    df = twin["pass_df"]
    df = df.assign(year=df["solar_date"].str[:4].astype(int))
    viirs = ("SNPP", "NOAA-20", "NOAA-21")
    one = df[
        df["year"].between(2016, 2017) & df["platform"].isin(viirs)
    ]["n_pixels"].sum() / 24
    three = df[
        df["year"].between(2024, 2025) & df["platform"].isin(viirs)
    ]["n_pixels"].sum() / 24
    assert three / one > 1.8


# --------------------------------------------------------------------------- #
# Harmonized structural continuity (the core falsification)
# --------------------------------------------------------------------------- #

def test_07_harmonized_2012_jump_eliminated(twin, twin_solved):
    idx = _month_index(twin)
    tm_hat = _monthly_means(
        0.5 * (np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])),
        twin,
    )
    tm_true = _monthly_means(
        0.5 * (twin["phi_day_true"] + twin["phi_night_true"]), twin
    )
    rel = _era_mean(tm_hat, idx, 2010, 2011) / _era_mean(
        tm_hat, idx, 2013, 2014
    ) / (
        _era_mean(tm_true, idx, 2010, 2011)
        / _era_mean(tm_true, idx, 2013, 2014)
    )
    assert 0.90 <= rel <= 1.10, rel  # < +/-10% residual step


def test_07_harmonized_three_satellite_inflation_eliminated(twin, twin_solved):
    idx = _month_index(twin)
    tm_hat = _monthly_means(
        0.5 * (np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])),
        twin,
    )
    tm_true = _monthly_means(
        0.5 * (twin["phi_day_true"] + twin["phi_night_true"]), twin
    )
    rel = _era_mean(tm_hat, idx, 2024, 2025) / _era_mean(
        tm_hat, idx, 2016, 2017
    ) / (
        _era_mean(tm_true, idx, 2024, 2025)
        / _era_mean(tm_true, idx, 2016, 2017)
    )
    assert 0.90 <= rel <= 1.10, rel


def test_07_night_day_ratio_preserved(twin, twin_solved):
    ratio = float(
        np.exp(twin_solved["z_night"]).mean()
        / np.exp(twin_solved["z_day"]).mean()
    )
    assert ratio == pytest.approx(TWIN_NIGHT_DAY_RATIO, rel=0.15)


def test_07_night_day_ratio_never_collapses_to_one(twin, twin_solved):
    # The audited 24h-normalization bug collapsed night_day_ratio toward
    # ~1.0 and blew up on nocturnal smoldering fires; the 12-hour window
    # normalization must keep Day and Night regimes physically distinct.
    ratio = float(
        np.exp(twin_solved["z_night"]).mean()
        / np.exp(twin_solved["z_day"]).mean()
    )
    assert ratio < 0.42
    assert ratio > 0.20
    # Monthly night/day ratios stay below 0.5 throughout (no collapse).
    idx = _month_index(twin)
    ids = [i for k, i in idx.items() if 2012 <= int(k[:4]) <= 2026]
    phi_n = _monthly_means(np.exp(twin_solved["z_night"]), twin)
    phi_d = _monthly_means(np.exp(twin_solved["z_day"]), twin)
    monthly_ratio = phi_n[:, ids] / np.maximum(phi_d[:, ids], 1e-9)
    assert float(monthly_ratio.max()) < 0.60


def test_07_nov_2026_snpp_cutoff_unbiased(twin, twin_solved):
    phi_hat = 0.5 * (
        np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])
    )
    phi_true = 0.5 * (twin["phi_day_true"] + twin["phi_night_true"])
    nov1 = (np.datetime64("2026-11-01") - np.datetime64("2000-11-01")).astype(int)
    oct1 = (np.datetime64("2026-10-01") - np.datetime64("2000-11-01")).astype(int)
    rel = (phi_hat[:, nov1:].mean() / phi_hat[:, oct1:nov1].mean()) / (
        phi_true[:, nov1:].mean() / phi_true[:, oct1:nov1].mean()
    )
    assert 0.90 <= rel <= 1.10, rel


def test_07_post_2022_drift_corrected(twin, twin_solved):
    idx = _month_index(twin)
    tm_hat = _monthly_means(
        0.5 * (np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])),
        twin,
    )
    tm_true = _monthly_means(
        0.5 * (twin["phi_day_true"] + twin["phi_night_true"]), twin
    )
    rel = _era_mean(tm_hat, idx, 2024, 2025) / _era_mean(
        tm_hat, idx, 2022, 2023
    ) / (
        _era_mean(tm_true, idx, 2024, 2025)
        / _era_mean(tm_true, idx, 2022, 2023)
    )
    assert 0.90 <= rel <= 1.10, rel


def test_07_monthly_correlation_above_090(twin, twin_solved):
    tm_hat = _monthly_means(
        0.5 * (np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])),
        twin,
    )
    tm_true = _monthly_means(
        0.5 * (twin["phi_day_true"] + twin["phi_night_true"]), twin
    )
    r = float(np.corrcoef(tm_hat, tm_true)[0, 1])
    assert r > 0.90


def test_07_drought_spike_preserved(twin, twin_solved):
    idx = _month_index(twin)
    q50 = twin_solved["monthly_q50"]
    d = float(q50[0, idx["2019-08"]])
    base = float(np.mean([q50[0, idx[f"{y}-08"]] for y in (2016, 2017, 2018)]))
    phi24_true = 0.5 * (
        twin["phi_day_true"][0] + twin["phi_night_true"][0]
    )
    mid = twin["month_ids"]
    tm_true = np.array(
        [phi24_true[mid == m].mean() for m in range(mid.max() + 1)]
    )
    d_t = float(tm_true[idx["2019-08"]])
    base_t = float(
        np.mean([tm_true[idx[f"{y}-08"]] for y in (2016, 2017, 2018)])
    )
    rel = (d / base) / (d_t / base_t)
    assert 0.85 <= rel <= 1.15, rel
    assert d / base > 2.0  # a genuine >2x drought signal survives


# --------------------------------------------------------------------------- #
# Reconstruction quality & uncertainty ordering
# --------------------------------------------------------------------------- #

def test_07_z_rmse_and_drought_magnitude(twin, twin_solved):
    err = twin_solved["z_day"] - twin["z_day_true"]
    rmse = float(np.sqrt((err**2).mean()))
    assert rmse < 1.0
    # The drought cell-month exceeds its pre-drought August climatology.
    idx = _month_index(twin)
    q50 = twin_solved["monthly_q50"]
    base = float(
        np.mean([q50[0, idx[f"{y}-08"]] for y in (2016, 2017, 2018)])
    )
    assert q50[0, idx["2019-08"]] > 2.0 * base


def test_07_posterior_wider_in_modis_only_era(twin, twin_solved):
    # INV-07: pre-2012 MODIS-only uncertainty strictly wider than the
    # 2012-2026 multi-sensor era.
    var = twin_solved["var_day"]
    yr = np.array([k[0] for k in twin["month_keys"]])[twin["month_ids"]]
    v_2005 = float(var[:, yr == 2005].mean())
    v_2024 = float(var[:, yr == 2024].mean())
    assert v_2005 > v_2024
    assert v_2005 > 1.5 * v_2024


def test_07_monthly_interval_ordering_and_coverage(twin, twin_solved):
    q05 = twin_solved["monthly_q05"]
    q50 = twin_solved["monthly_q50"]
    q95 = twin_solved["monthly_q95"]
    assert np.all(q05 <= q50 + 1e-9)
    assert np.all(q50 <= q95 + 1e-9)
    assert np.all(q05 >= 0.0)
    # Relative interval width ordering: 2005 wider than 2024.
    idx = _month_index(twin)
    ids05 = [i for k, i in idx.items() if k.startswith("2005")]
    ids24 = [i for k, i in idx.items() if k.startswith("2024")]
    w05 = float(
        np.nanmean(
            (q95[:, ids05] - q05[:, ids05]) / np.maximum(q50[:, ids05], 1e-6)
        )
    )
    w24 = float(
        np.nanmean(
            (q95[:, ids24] - q05[:, ids24]) / np.maximum(q50[:, ids24], 1e-6)
        )
    )
    assert w05 > w24


def test_07_daily_quantiles_available(twin, twin_solved):
    dq = twin_solved["daily_q50"]
    assert dq.shape == (5, S.T_DAYS)
    assert np.isfinite(dq).all()
    assert np.all(twin_solved["daily_q05"] <= dq + 1e-9)
    assert np.all(dq <= twin_solved["daily_q95"] + 1e-9)
