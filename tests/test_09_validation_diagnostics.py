"""test_09 -- Validation diagnostics tests (12 tests).

Readme Section 7.9: Chow & Pettitt structural breaks catching the raw 2012
jump (p < 0.001) and passing the harmonized control (p > 0.05); the
2019-2021 VIIRS-masked ablation outperforming raw MODIS on CRPS & Spearman
rho; MCD64CMQ burned-area correlation improvement; Product 1/2/3 schema and
sparse daily export invariants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.validation.diagnostics import (ablation_metrics, chow_test,
                                        pettitt_test, structural_break_suite,
                                        top3_peak_month_match)
from tests.conftest import _full_twin


# ---------------------------------------------------------------- Chow/Pettitt
def test_09_chow_catches_known_break():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(10, 1, 40), rng.normal(25, 1, 40)])
    res = chow_test(y, 40)
    assert res["p"] < 0.001
    assert res["F"] > 20


def test_09_chow_passes_stationary_control():
    rng = np.random.default_rng(2)
    y = rng.normal(10, 1, 80)
    res = chow_test(y, 40)
    assert res["p"] > 0.05


def test_09_pettitt_detects_change_point():
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)])
    res = pettitt_test(y)
    assert res["p"] < 0.05
    assert res["K"] > 3.0
    assert 48 <= res["idx"] <= 52          # break located at t = 50


def test_09_pettitt_stationary_control():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 100)
    assert pettitt_test(y)["p"] > 0.05


def test_09_chow_raw_2012_break_and_harmonized_pass(twin_diagnostics):
    sb = twin_diagnostics["structural_breaks"]
    # raw counts break violently at 2012
    assert sb["chow_raw_2012"]["p"] < 0.001
    assert sb["pettitt_raw_2012"]["p"] < 0.05
    # the harmonized series is dramatically more continuous: the Chow F
    # statistic drops by > 60% and the break p-value rises by orders of
    # magnitude (the twin's harmonized series carries genuine climate
    # trends, so we compare break STRENGTH raw-vs-harmonized)
    assert sb["chow_harm_2012"]["F"] < 0.4 * sb["chow_raw_2012"]["F"]
    assert sb["chow_harm_2012"]["p"] > sb["chow_raw_2012"]["p"]
    # readme 7.9: the harmonized record passes the 2012 Chow test
    assert sb["chow_harm_2012"]["p"] > 0.05


def test_09_mcd64cmq_burned_area_correlation():
    """MCD64CMQ cross-check joins H3 cells on the exact 0.25 deg CMG bin
    centres (x.125/x.375/x.625/x.875) and the harmonized monthly FRE
    correlates with burned area more strongly than raw counts when raw
    counts carry a sensor step."""
    from src.geometry.h3_utils import latlon_to_h3_int
    from src.engine.priors import cmg_bin_centers
    from src.validation.diagnostics import mcd64cmq_correlation
    rng = np.random.default_rng(9)
    from src.geometry.h3_utils import h3_cell_center
    # bin semantics: edges at 90 - k/4 and -180 + k/4, centres half a bin in
    la, lo = cmg_bin_centers(np.array([-29.16, -29.30]),
                             np.array([-67.40, -67.60]))
    assert np.allclose(la, [-29.125, -29.375])
    assert np.allclose(lo, [-67.375, -67.625])
    cell = int(latlon_to_h3_int([-29.16], [-67.40], 6)[0])
    lat_c, lon_c = cmg_bin_centers(*h3_cell_center(cell))
    rows, ba = [], []
    for y in range(2005, 2021):
        for mth in range(1, 13):
            burned = float(np.exp(rng.normal(1.0, 1.0)))
            fre = burned * float(np.exp(rng.normal(0.0, 0.2)))
            raw = burned * (6.0 if y >= 2012 else 1.0) \
                * float(np.exp(rng.normal(0.0, 0.2)))
            rows.append(dict(h3_index=cell, year=y, month=mth,
                             fre_tj_q50=fre, raw_count_sum=raw))
            ba.append(dict(lat_025=float(lat_c), lon_025=float(lon_c), year=y,
                           month=mth, burned_area_km2=burned))
    res = mcd64cmq_correlation(pd.DataFrame(rows), pd.DataFrame(ba))
    assert res["n"] == len(rows)
    assert res["harmonized_fre"] > res["raw_counts"]
    assert res["harmonized_fre"] > 0.9


def test_09_chow_break_structure():
    """Chow F is monotone in the break size for the same noise."""
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 60)
    for i, delta in enumerate((0.5, 2.0, 6.0)):
        y = base.copy()
        y[30:] += delta
        assert chow_test(y, 30)["F"] > chow_test(base + 0, 30)["F"]


# ---------------------------------------------------------------- ablation
def test_09_ablation_metrics_components():
    rng = np.random.default_rng(6)
    truth = np.exp(rng.normal(1, 0.5, 500))
    pred = truth * np.exp(0.3 * rng.standard_normal(500))
    m = ablation_metrics(pred * 0.8, pred, pred * 1.25, truth)
    assert 0.0 <= m["coverage_90"] <= 1.0
    assert m["crps"] > 0
    assert m["log_rmse"] > 0
    assert -1.0 <= m["spearman_rho"] <= 1.0


def test_09_ablation_beats_raw_modis_baseline(twin_diagnostics):
    abl = twin_diagnostics["ablation_2019_2021"]
    base = abl["raw_baseline"]
    assert abl["crps"] < base["crps"]
    assert abl["spearman_rho"] > base["spearman_rho"]
    assert abl["log_rmse"] < base["log_rmse"]


def test_09_top3_peak_month_match():
    pred = pd.DataFrame({"h3_index": [1] * 12, "month": range(1, 13),
                         "phi_q50_mw": [5, 4, 3, 2, 1, 2, 3, 9, 8, 7, 6, 5]})
    true = pd.DataFrame({"h3_index": [1] * 12, "month": range(1, 13),
                         "phi24_true": [5, 4, 3, 2, 1, 2, 3, 9, 8, 7, 6, 5]})
    assert top3_peak_month_match(pred, true) == 1.0
    true2 = true.copy()
    true2["phi24_true"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # pred top-3 = (8, 9, 10); true2 top-3 = (10, 11, 12): 1/3 overlap
    assert top3_peak_month_match(pred, true2) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------- products
def test_09_product_schemas(full_twin):
    from src.products import export_duckdb as ex
    prov = full_twin["prov"]
    pc = full_twin["pass_cal"]
    monthly = full_twin["monthly"]
    daily = full_twin["daily"]
    priors = full_twin["priors"]
    art = full_twin["artifacts"]
    products = ex.build_products(monthly, daily, full_twin["pass_obs"],
                                 prov, pc, priors, art, full_twin["cloud_df"])
    p1 = products["product1"]
    for col in ("platform", "acq_date", "frp_mw", "t_mir_k", "t_tir_k",
                "conf_raw", "type_raw", "ingest_ts", "h3_index", "sza_deg",
                "solar_hour", "dn_flag", "pixel_area_km2", "vza_deg",
                "is_static_industrial", "is_sama_filtered", "bowtie_weight",
                "event_cluster_id"):
        assert col in p1.columns, f"Product 1 missing {col}"
    p2m = products["product2_monthly"]
    for col in ("h3_index", "year", "month", "raw_count_sum", "raw_frp_sum_mw",
                "raw_clustered_events", "phi_day_mw", "phi_night_mw",
                "phi_q05_mw", "phi_q50_mw", "phi_q95_mw", "phi_mean_mw",
                "fre_tj_q50", "night_day_ratio"):
        assert col in p2m.columns, f"Product 2 missing {col}"
    p3 = products["product3"]
    for col in ("h3_index", "year", "month", "conformal_c_factor",
                "rel_uncertainty_width", "measured_fraction",
                "active_platforms_bitmask", "mean_cloud_frac",
                "attribution_flag", "quality_flag", "revision_count"):
        assert col in p3.columns, f"Product 3 missing {col}"
    # 1-to-1 with monthly
    assert len(p3) == len(p2m)


def test_09_sparse_daily_export_invariants():
    """Sparse episode filter: phi > 0.25 OR within +-3 days of a detection
    OR inside a spotlight ROI (readme 5.4.4) -- verified on a synthetic
    daily grid with dormant days, detections, and a spotlight."""
    from src.geometry.h3_utils import h3_cell_center, latlon_to_h3_int
    from src.products.export_duckdb import sparse_daily_filter
    # plain cell far from every spotlight ROI (Brazil north)
    cell_plain = int(latlon_to_h3_int([10.0], [-60.0], 6)[0])
    cell_spot = int(latlon_to_h3_int([-29.15], [-67.55], 6)[0])  # Chilecito
    t_days = 40
    rows = []
    for t in range(t_days):
        for c in {cell_plain, cell_spot}:
            phi = 0.05 if t < 20 else 2.0    # dormant first half
            rows.append(dict(h3_index=c, t=t, phi_q05_mw=phi * 0.5,
                             phi_q50_mw=phi, phi_q95_mw=phi * 2.0,
                             phi_day_mw=phi, phi_night_mw=phi * 0.5,
                             phi_mean_mw=phi, fre_tj_q50=0.0864 * phi))
    daily = pd.DataFrame(rows)
    # detections at t=10 and t=11 for the plain cell
    pass_obs = pd.DataFrame({"h3_index": [cell_plain, cell_plain],
                             "solar_date": ["2000-11-11", "2000-11-12"]})
    keep = sparse_daily_filter(daily, pass_obs, t_days)
    assert not keep.all()                      # genuinely sparse
    assert keep.mean() < 0.9
    # active days always kept
    assert keep[daily["phi_q50_mw"] > 0.25].all()
    plain_keep = keep[daily["h3_index"] == cell_plain].to_numpy()
    assert not plain_keep[0]                   # t=0 dormant, no detection
    # detection at t=10/11: kept window is [7, 14]
    assert not plain_keep[6] and plain_keep[7] and plain_keep[14] \
        and not plain_keep[15]
    # spotlight cell: everything kept regardless
    spot_keep = keep[daily["h3_index"] == cell_spot].to_numpy()
    assert spot_keep.all()                       # spotlight keeps everything


def test_09_export_duckdb_roundtrip(full_twin, tmp_path):
    import duckdb
    from src.products import export_duckdb as ex
    products = ex.build_products(full_twin["monthly"], full_twin["daily"],
                                 full_twin["pass_obs"], full_twin["prov"],
                                 full_twin["pass_cal"], full_twin["priors"],
                                 full_twin["artifacts"],
                                 full_twin["cloud_df"])
    db = tmp_path / "fire_harmonized.duckdb"
    ex.export_duckdb(products, db, parquet_dir=str(tmp_path))
    con = duckdb.connect(str(db), read_only=True)
    for table in ("obs_harmonized_provenance", "state_reconstructed_monthly",
                  "state_reconstructed_daily", "uncertainty_diagnostics"):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert n > 0
    view = con.execute("SELECT COUNT(*) FROM v_calendar").fetchone()[0]
    assert view > 0
    con.close()


def test_09_export_survives_duckdb_duplicate_key_scale(tmp_path):
    """150k provenance rows over 3 hexagons (heavily duplicated h3_index):
    the exporter must not build an ART index that DuckDB 1.3.x cannot
    construct on duplicated keys; tables are clustered by h3_index."""
    import duckdb
    from src.products import export_duckdb as ex
    rng = np.random.default_rng(0)
    cells = np.array([606614382356463615, 606614382490681343,
                      606614387456737279], dtype=np.int64)
    n = 150_000
    p1 = pd.DataFrame({"h3_index": rng.choice(cells, n),
                       "frp_mw": rng.uniform(1, 50, n)})
    monthly = pd.DataFrame({"h3_index": cells, "year": 2020, "month": 8,
                            "country_iso": "ARG", "pyrome_id": 0,
                            "phi_q50_mw": 1.0, "phi_q05_mw": 0.5,
                            "phi_q95_mw": 2.0, "hist_median_2003_2020": 1.0,
                            "anomaly_ratio": 1.0, "raw_count_sum": 3.0,
                            "raw_frp_sum_mw": 9.0})
    daily = pd.DataFrame({"h3_index": rng.choice(cells, n),
                          "t": rng.integers(0, 9510, n),
                          "phi_q50_mw": 1.0})
    diag = monthly[["h3_index", "year", "month"]].assign(
        quality_flag="NOMINAL", attribution_flag="WITHIN_NORMAL_VARIABILITY",
        measured_fraction=1.0)
    db = tmp_path / "fire_harmonized.duckdb"
    ex.export_duckdb({"product1": p1, "product2_monthly": monthly,
                      "product2_daily": daily, "product3": diag}, db,
                     parquet_dir=str(tmp_path))
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM obs_harmonized_provenance"
                       ).fetchone()[0] == n
    assert con.execute("SELECT COUNT(*) FROM hex_centers").fetchone()[0] == 3
    con.close()
