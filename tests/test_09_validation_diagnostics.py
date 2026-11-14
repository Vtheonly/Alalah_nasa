"""test_09: structural breaks, ablation gains, burned-area cross-checks.

12 tests (readme.md Section 7 item 9): Chow & Pettitt break tests catching
the raw 2012 jump (p < 0.001) and passing the harmonized control (p > 0.05);
the 2019-2021 VIIRS-masked ablation outperforming raw MODIS on CRPS and
Spearman rho; the MCD64CMQ burned-area correlation improvement; and the
Product 1/2/3 schema & sparse daily export invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.products.export_duckdb import (
    PRODUCT1_COLUMNS,
    PRODUCT2_MONTHLY_COLUMNS,
    PRODUCT3_COLUMNS,
)
from src.validation import diagnostics as dg


def _month_index(twin):
    return {
        f"{y}-{m:02d}": i for i, (y, m) in enumerate(twin["month_keys"])
    }


def _monthly_means(arr, twin):
    mid = twin["month_ids"]
    return np.array([arr[:, mid == m].mean() for m in range(mid.max() + 1)])


# --------------------------------------------------------------------------- #
# Structural break tests
# --------------------------------------------------------------------------- #

def test_09_chow_catches_raw_2012_break(twin):
    raw = twin["raw_monthly_counts"]
    keys = [f"{y}-{m:02d}" for (y, m) in twin["month_keys"]]
    t_break = next(i for i, k in enumerate(keys) if k >= "2012-01")
    res = dg.chow_test(raw, t_break)
    assert res["p_value"] < 0.001


def test_09_chow_passes_harmonized_control(twin, twin_solved):
    tm_hat = _monthly_means(
        0.5 * (np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])),
        twin,
    )
    keys = [f"{y}-{m:02d}" for (y, m) in twin["month_keys"]]
    t_break = next(i for i, k in enumerate(keys) if k >= "2012-01")
    res = dg.chow_test(tm_hat, t_break)
    assert res["p_value"] > 0.05


def test_09_pettitt_catches_raw_break_near_2012(twin):
    raw = twin["raw_monthly_counts"]
    res = dg.pettitt_test(raw)
    assert res["p_value"] < 0.001
    keys = [f"{y}-{m:02d}" for (y, m) in twin["month_keys"]]
    true_break = next(i for i, k in enumerate(keys) if k >= "2012-01")
    assert abs(res["change_index"] - true_break) <= 24  # within 2 years


def test_09_pettitt_passes_harmonized(twin, twin_solved):
    tm_hat = _monthly_means(
        0.5 * (np.exp(twin_solved["z_day"]) + np.exp(twin_solved["z_night"])),
        twin,
    )
    res = dg.pettitt_test(tm_hat)
    assert res["p_value"] > 0.05


# --------------------------------------------------------------------------- #
# 2019-2021 VIIRS-masked ablation
# --------------------------------------------------------------------------- #

def _ablation_and_raw_2019_2021(twin, twin_ablation_solved):
    """Monthly truth vs (a) MODIS-only harmonized ablation and (b) the raw
    MODIS estimator (per-cell mean of observed MODIS pass FRP, no
    harmonization) for 2019-2021."""
    idx = _month_index(twin)
    ids = np.array(
        [i for k, i in idx.items() if 2019 <= int(k[:4]) <= 2021]
    )
    mid = twin["month_ids"]
    n_cells = len(twin["cells"])
    truth = _monthly_means(
        0.5 * (twin["phi_day_true"] + twin["phi_night_true"]), twin
    )  # (n_months,) cell-averaged truth
    truth_cells = np.stack(
        [
            np.array(
                [
                    0.5
                    * (
                        twin["phi_day_true"][b][mid == m]
                        + twin["phi_night_true"][b][mid == m]
                    ).mean()
                    for m in range(mid.max() + 1)
                ]
            )
            for b in range(n_cells)
        ]
    )
    ablation = _monthly_means(
        0.5
        * (np.exp(twin_ablation_solved["z_day"])
           + np.exp(twin_ablation_solved["z_night"])),
        twin,
    )[ids]
    # raw MODIS estimator: per-cell monthly mean of daily-mean MODIS FRP.
    df = twin["pass_df"]
    modis = df[df["sensor_family"] == "MODIS"]
    raw_cells = np.zeros((n_cells, len(ids)))
    for j, m in enumerate(ids):
        days = np.flatnonzero(mid == m)
        sub = modis[modis["t_index"].isin(days)]
        for b in range(n_cells):
            sb = sub[sub["cell_pos"] == b]
            if len(sb):
                by_day = sb.groupby("t_index")["y_frp_mw"].mean()
                raw_cells[b, j] = float(by_day.mean())
    truth = truth_cells[:, ids].mean(axis=0)
    raw = raw_cells.mean(axis=0)
    return truth, ablation, raw


def test_09_ablation_beats_raw_on_crps(twin, twin_ablation_solved):
    truth, ablation, raw = _ablation_and_raw_2019_2021(
        twin, twin_ablation_solved
    )
    crps_abl = dg.crps_point(truth, ablation)
    crps_raw = dg.crps_point(truth, raw)
    assert crps_abl < crps_raw, (crps_abl, crps_raw)


def test_09_ablation_beats_raw_on_spearman(twin, twin_ablation_solved):
    truth, ablation, raw = _ablation_and_raw_2019_2021(
        twin, twin_ablation_solved
    )
    rho_abl = dg.spearman_rho(truth, ablation)
    rho_raw = dg.spearman_rho(truth, raw)
    assert rho_abl > rho_raw, (rho_abl, rho_raw)
    assert rho_abl > 0.85


def test_09_ablation_log_rmse_and_peak_months(twin, twin_ablation_solved):
    truth, ablation, _ = _ablation_and_raw_2019_2021(
        twin, twin_ablation_solved
    )
    assert dg.log_rmse(truth, ablation) < dg.log_rmse(truth, truth * 0.5)
    assert dg.top3_peak_month_match(ablation, truth) >= 2


# --------------------------------------------------------------------------- #
# MCD64CMQ burned-area cross-check
# --------------------------------------------------------------------------- #

def test_09_burned_area_correlation_improvement(twin, twin_solved):
    """Independent burned area (truth FRE + multiplicative noise) correlates
    more strongly with harmonized FRE than with raw hotspot counts."""
    rng = np.random.default_rng(21)
    idx = _month_index(twin)
    n = len(idx)
    truth = _monthly_means(
        0.5 * (twin["phi_day_true"] + twin["phi_night_true"]), twin
    )
    burned = truth * rng.lognormal(0.0, 0.25, size=n)  # independent-ish
    harmonized_fre = np.nansum(twin_solved["monthly_q50"], axis=0) * (
        S.FRE_TJ_PER_MW_DAY
    )
    raw_counts = twin["raw_monthly_counts"]
    result = dg.burned_area_correlation(harmonized_fre, raw_counts, burned)
    assert result["harmonized_fre_corr"] > result["raw_counts_corr"]
    assert result["harmonized_fre_corr"] > 0.8


# --------------------------------------------------------------------------- #
# Product schemas & sparse daily export
# --------------------------------------------------------------------------- #

def test_09_product1_schema(twin_products):
    p1 = twin_products["p1"]
    assert list(p1.columns) == PRODUCT1_COLUMNS
    assert len(p1) > 0
    # Raw provenance fields preserved untouched alongside derived columns.
    for col in ("platform", "acq_date", "frp_mw", "conf_raw", "type_raw"):
        assert col in p1.columns
    assert p1["h3_index"].notna().all()


def test_09_product2_monthly_schema_and_invariants(twin_products):
    p2m = twin_products["p2m"]
    assert list(p2m.columns) == PRODUCT2_MONTHLY_COLUMNS
    assert (p2m["phi_q05_mw"] <= p2m["phi_q50_mw"] + 1e-9).all()
    assert (p2m["phi_q50_mw"] <= p2m["phi_q95_mw"] + 1e-9).all()
    assert (p2m["phi_q05_mw"] >= 0).all()
    assert (p2m["fre_tj_q50"] >= 0).all()
    assert set(p2m["year"].unique()) >= set(range(2001, 2027))
    # Calendar indices present and bounded.
    assert p2m["anomaly_ratio"].notna().all()
    assert (p2m["climatology_pctile"].between(0.0, 1.0)).all()


def test_09_product2_daily_sparse_invariants(twin, twin_products, twin_solved):
    p2d = twin_products["p2d"]
    assert {"h3_index", "date", "phi_q50_mw"} <= set(p2d.columns)
    dense = len(twin["cells"]) * S.T_DAYS
    assert 0 < len(p2d) <= dense
    # Every emitted row satisfies at least one sparse-filter clause, and
    # all three clauses contribute rows for the twin (its background flux
    # sits above the 0.25 MW threshold, so the phi clause fires widely).
    det_days = {
        (int(r.cell_pos), int(r.t_index))
        for r in twin["pass_df"].itertuples()
    }
    spotlight = S.SPOTLIGHT_PRESETS["Chilecito ARG 2026"]["bbox"]
    lat_by_pos = {
        int(i): float(lat) for i, lat in enumerate(twin["latlon"][:, 0])
    }
    lon_by_pos = {
        int(i): float(lon) for i, lon in enumerate(twin["latlon"][:, 1])
    }
    h3_to_pos = {int(h): i for i, h in enumerate(twin["cells"])}
    t0 = np.datetime64("2000-11-01")
    n_phi = n_det = n_spot = 0
    for row in p2d.itertuples():
        t = (np.datetime64(row.date) - t0).astype(int)
        pos = h3_to_pos[int(row.h3_index)]
        near_det = any(
            (pos, t + d) in det_days
            for d in range(
                -S.DAILY_EXPORT_DETECTION_WINDOW_DAYS,
                S.DAILY_EXPORT_DETECTION_WINDOW_DAYS + 1,
            )
        )
        in_spot = (
            spotlight[0] <= lon_by_pos[pos] <= spotlight[2]
            and spotlight[1] <= lat_by_pos[pos] <= spotlight[3]
        )
        assert (
            row.phi_q50_mw > S.DAILY_EXPORT_PHI_MIN_MW
            or near_det or in_spot
        )
        n_phi += row.phi_q50_mw > S.DAILY_EXPORT_PHI_MIN_MW
        n_det += (
            near_det and row.phi_q50_mw <= S.DAILY_EXPORT_PHI_MIN_MW
        )
        n_spot += (
            in_spot and not near_det
            and row.phi_q50_mw <= S.DAILY_EXPORT_PHI_MIN_MW
        )
    assert n_phi > 0 and n_det > 0
    # (the spotlight clause alone rarely fires on this twin because its
    # background flux sits above the export threshold; it is exercised
    # directly in the builder-level checks below)

    # Builder-level sparse filter unit check: a dormant series with one
    # 4-day episode and one detection day exports ONLY the phi-episode and
    # the +/-3-day detection window rows (cells outside every spotlight).
    from src.products.export_duckdb import build_product2_daily

    T = S.T_DAYS
    n = len(twin["cells"])
    solved = dict(twin_solved)
    solved["month_ids"] = twin["month_ids"]
    solved["daily_q50"] = np.zeros((n, T))
    solved["daily_q05"] = np.zeros((n, T))
    solved["daily_q95"] = np.zeros((n, T))
    solved["daily_q50"][:, 100:104] = 30.0  # 4-day episode
    meta = twin_products["cell_meta"].copy()
    meta["latitude"] = -35.0   # outside all SPOTLIGHT_PRESETS boxes
    meta["longitude"] = -60.0
    empty_pass = twin_products["pass_df"].iloc[0:0]
    p2d_phi = build_product2_daily(solved, meta, twin["month_keys"],
                                   empty_pass)
    assert len(p2d_phi) == n * 4  # phi clause only

    one_det = pd.DataFrame(
        {"cell_pos": [0], "t_index": [400], "n_pixels": [1],
         "y_frp_mw": [10.0], "raw_event_count": [1]}
    )
    solved["daily_q50"] = np.zeros((n, T))
    p2d_det = build_product2_daily(solved, meta, twin["month_keys"],
                                   one_det)
    assert len(p2d_det) == 7  # t in [397, 403] for the detecting cell only

    # Spotlight clause: dormant cells inside the Chilecito ROI export every
    # day even with no detections and sub-threshold flux.
    meta_spot = meta.copy()
    meta_spot["latitude"] = -29.16
    meta_spot["longitude"] = -67.50
    solved["daily_q50"] = np.zeros((n, T))
    p2d_spot = build_product2_daily(solved, meta_spot, twin["month_keys"],
                                    empty_pass)
    assert len(p2d_spot) == n * T  # fully retained inside the ROI


def test_09_product3_schema_and_one_to_one(twin_products):
    p2m = twin_products["p2m"]
    p3 = twin_products["p3"]
    assert list(p3.columns) == PRODUCT3_COLUMNS
    # 1-to-1 with the monthly product on (h3_index, year, month).
    key2 = set(zip(p2m["h3_index"], p2m["year"], p2m["month"]))
    key3 = set(zip(p3["h3_index"], p3["year"], p3["month"]))
    assert key2 == key3
    assert len(key3) == len(p3)
    assert (p3["measured_fraction"].between(0.0, 1.0)).all()
    assert set(p3["attribution_flag"]) <= {
        "REAL_FIRE_CHANGE", "SENSOR_ARTIFACT_CORRECTED",
        "AMBIGUOUS_OVERLAPPING_CI", "WITHIN_NORMAL_VARIABILITY",
    }
    assert set(p3["quality_flag"]) <= {
        "NOMINAL", "ABSTAIN_LOW_EVIDENCE", "CLIMATOLOGY_CLOUD_FALLBACK",
    }
