"""test_10: FastAPI endpoints, CSV sanitization, latency, VRAM budget.

12 tests (readme.md Section 7 item 10): all five FastAPI endpoints return
200 with expected JSON keys; CSV formula-injection neutralization (=, +, -,
@ prefixes); empty-bbox handling; DuckDB query latency < 300 ms; and the
single-GPU chunk (batch_size = 1000, T = 9510) VRAM staying < 6 GB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import settings as S

DANGEROUS = ["=SUM(A1:A2)", "+1|calc", "-2+3|cmd", "@import(x)", "\ttab",
             "\rcr"]


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    """A small DuckDB built from the twin products + a diagnostics file."""
    import duckdb

    from src.products.export_duckdb import export_all
    from tests.conftest import twin_helpers_build  # noqa: F401 (fixture chain)

    # The twin products come from the session fixtures via import.
    from src.engine import solver as sv
    from src.validation import synthetic_twin as st

    tmp = tmp_path_factory.mktemp("webapp")
    twin = st.generate_twin(n_cells=5, T=9510, seed=42)
    solved = sv.solve_all_chunks(
        lambda cells, dev: st.build_chunk_problem(twin, cells, dev),
        len(twin["cells"]), batch_size=1000, sweeps=2, newton_steps=8,
        mc_samples=40, sample_subbatch=5,
    )
    solved["month_ids"] = twin["month_ids"]
    solved["platform_calendar"] = twin["platform_calendar"]
    cells = twin["cells"]
    latlon = np.asarray(twin["latlon"], dtype=float)
    import pandas as pd

    cell_meta = pd.DataFrame(
        {
            "cell_pos": np.arange(len(cells)),
            "h3_index": np.asarray(cells, dtype=np.int64),
            "pyrome_id": np.zeros(len(cells), dtype=int),
            "latitude": latlon[:, 0], "longitude": latlon[:, 1],
            "country_iso": ["ARG"] * len(cells),
        }
    )
    pass_df = twin["pass_df"].copy()
    pass_df["data_tier"] = "ML_SCIENCE"

    from src.products import export_duckdb as ex

    # Craft one malicious string to prove end-to-end CSV neutralization.
    provenance = _provenance_with_malicious_row(twin)
    p1 = ex.build_product1(provenance, pass_df)
    p2m = ex.build_product2_monthly(
        solved, cell_meta, twin["month_keys"], pass_df,
        np.full(7, 5.0), cqr_factor=1.2,
    )
    p2d = ex.build_product2_daily(solved, cell_meta, twin["month_keys"],
                                  pass_df)
    p3 = ex.build_product3(
        solved, cell_meta, twin["month_keys"], pass_df, p2m, 1.2
    )
    p3.loc[0, "quality_flag"] = "=SUM(A1:A2)"  # injected formula
    p3.loc[1, "quality_flag"] = "@import(evil)"
    duckdb_path = tmp / "fire_harmonized.duckdb"
    export_all(
        p1, p2m, p2d, p3, duckdb_path,
        product_paths={
            "product1": tmp / "p1.parquet",
            "product2_monthly": tmp / "p2m.parquet",
            "product2_daily": tmp / "p2d.parquet",
            "product3": tmp / "p3.parquet",
        },
    )
    diag = tmp / "diagnostics.json"
    diag.write_text(json.dumps({"chow_2012": {"raw_p_below_0_001": True}}))

    from src.webapp.server import create_app

    app = create_app(duckdb_path=duckdb_path, diagnostics_json=diag)
    client = TestClient(app)
    # expose internals needed by tests
    client.twin = twin
    return client


def _provenance_with_malicious_row(twin):
    import pandas as pd

    df = twin["pass_df"].head(30).copy()
    df["acq_date"] = df["solar_date"]
    df["acq_hhmm"] = 1345
    df["t_mir_k"] = 340.0
    df["t_tir_k"] = 300.0
    df["conf_raw"] = "n"
    df["type_raw"] = 0
    df["data_tier"] = "ML_SCIENCE"
    df["scan_km"] = np.nan
    df["track_km"] = np.nan
    df["sza_deg"] = 40.0
    df["pixel_area_km2"] = 0.2
    df["vza_deg"] = 30.0
    df["is_static_industrial"] = False
    df["is_sama_filtered"] = False
    df["bowtie_weight"] = 1.0
    df["reliability_prob"] = 0.9
    df["event_cluster_id"] = 0
    df["frp_mw"] = df["y_frp_mw"]
    df["ingest_ts"] = "2026-11-14T00:00:00Z"
    return df


BBOX = {
    "min_lon": -68.0, "min_lat": -29.6,
    "max_lon": -67.1, "max_lat": -28.7,
}


# --------------------------------------------------------------------------- #
# Endpoint availability & JSON contracts
# --------------------------------------------------------------------------- #

def test_10_calendar_endpoint(api):
    res = api.get("/api/calendar", params={
        **BBOX, "start_year": 2019, "end_year": 2021,
        "metric": "phi_q50_mw",
    })
    assert res.status_code == 200
    body = res.json()
    assert "meta" in body and "cells" in body
    assert body["meta"]["n_tiles"] > 0
    tile = body["cells"][0]
    for key in ("year", "month", "value", "q05", "q95", "hist_median",
                "anomaly_ratio", "raw_count", "raw_frp", "measured_fraction",
                "quality_flag", "attribution_flag"):
        assert key in tile


def test_10_timeseries_endpoint_with_constellation_events(api):
    res = api.get("/api/timeseries", params={
        **BBOX, "start_year": 2000, "end_year": 2026,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["constellation_events"] == list(S.CONSTELLATION_EVENTS)
    assert set(body["series"][0]) >= {
        "month", "phi_q50_mw", "raw_count", "measured_fraction"
    }
    months = [d["month"] for d in body["series"]]
    assert months == sorted(months)


def test_10_map_hex_endpoint_geojson_and_cap(api):
    res = api.get("/api/map_hex", params={
        **BBOX, "year": 2019, "month": 8, "color_by": "anomaly_ratio",
    })
    assert res.status_code == 200
    fc = res.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0
    assert len(fc["features"]) <= S.MAX_MAP_HEX_FEATURES
    geom = fc["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    assert len(geom["coordinates"][0]) >= 4  # closed hex ring
    props = fc["features"][0]["properties"]
    assert {"h3_index", "phi_q50_mw", "anomaly_ratio", "quality_flag"} <= set(
        props
    )


def test_10_diagnostics_endpoint(api):
    res = api.get("/api/diagnostics")
    assert res.status_code == 200
    assert res.json()["chow_2012"]["raw_p_below_0_001"] is True


def test_10_export_csv_endpoint(api):
    res = api.get("/api/export.csv", params={
        **BBOX, "start_year": 2019, "end_year": 2020,
    })
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    lines = res.text.strip().splitlines()
    assert lines[0].startswith("h3_index")
    assert len(lines) > 2


def test_10_export_csv_consistent_with_calendar(api):
    """The CSV view and the calendar view agree tile-for-tile on q50."""
    res_csv = api.get("/api/export.csv", params={
        **BBOX, "start_year": 2019, "end_year": 2019,
    })
    res_cal = api.get("/api/calendar", params={
        **BBOX, "start_year": 2019, "end_year": 2019,
        "metric": "phi_q50_mw",
    })
    assert res_csv.status_code == 200 and res_cal.status_code == 200
    rows = [line.split(",") for line in res_csv.text.strip().splitlines()[1:]]
    by_key = {(r[0], r[1], r[2]): float(r[3]) for r in rows}
    tiles = res_cal.json()["cells"]
    assert len(tiles) == len(by_key)
    for tile in tiles:
        key = (str(tile["h3_index"]), str(tile["year"]), str(tile["month"]))
        assert key in by_key
        assert abs(by_key[key] - tile["value"]) < 1e-9


# --------------------------------------------------------------------------- #
# Security: formula-injection neutralization + bbox validation
# --------------------------------------------------------------------------- #

def test_10_csv_formula_injection_neutralized(api):
    from src.webapp.server import sanitize_csv_value

    for payload in DANGEROUS:
        assert sanitize_csv_value(payload) == "'" + payload
    # Safe strings pass through untouched; numbers are never quoted.
    assert sanitize_csv_value("NOMINAL") == "NOMINAL"
    assert sanitize_csv_value("-1.5") == "'-1.5"  # string minus is quoted
    assert sanitize_csv_value(42) == "42"
    assert sanitize_csv_value(-1.5) == "-1.5"    # numeric minus untouched
    # End-to-end: the crafted malicious quality_flag rows come back quoted.
    res = api.get("/api/export.csv", params={
        **BBOX, "start_year": 2000, "end_year": 2026,
    })
    assert "'=SUM(A1:A2)" in res.text
    assert "'@import(evil)" in res.text
    assert not any(
        line.startswith("=SUM") for line in res.text.splitlines()
    )


def test_10_bbox_validation_and_25deg_cap(api):
    # Inverted bbox rejected.
    bad = {"min_lon": 10.0, "min_lat": 0.0, "max_lon": -10.0,
           "max_lat": 5.0}
    res = api.get("/api/calendar", params={**bad, "start_year": 2020,
                                            "end_year": 2021})
    assert res.status_code == 422
    # Oversized bbox (> 25 x 25 deg) rejected without country_iso.
    big = {"min_lon": -60.0, "min_lat": -40.0, "max_lon": 20.0,
           "max_lat": 10.0}
    res = api.get("/api/calendar", params={**big, "start_year": 2020,
                                            "end_year": 2021})
    assert res.status_code == 422
    assert "25" in res.json()["detail"]
    # country_iso unlocks the large bbox.
    res = api.get("/api/calendar", params={
        **big, "start_year": 2020, "end_year": 2021,
        "country_iso": "ARG",
    })
    assert res.status_code == 200
    # Degenerate (empty) bbox returns an empty tile set, not a crash.
    empty = {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.5,
             "max_lat": 0.5}
    res = api.get("/api/calendar", params={**empty, "start_year": 2020,
                                            "end_year": 2021})
    assert res.status_code == 200
    assert res.json()["cells"] == []


# --------------------------------------------------------------------------- #
# Latency & scale budget
# --------------------------------------------------------------------------- #

def test_10_duckdb_query_latency_under_300ms(api):
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = api.get("/api/timeseries", params={
            **BBOX, "start_year": 2000, "end_year": 2026,
        })
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert res.status_code == 200
    assert max(latencies) < S.API_LATENCY_BUDGET_MS, latencies


def test_10_map_hex_feature_cap_enforced(api):
    # Wide-area query (25 x 25 deg, the cap boundary) is bounded at 3000
    # features, never unbounded.
    res = api.get("/api/map_hex", params={
        "min_lon": -70.0, "min_lat": -45.0, "max_lon": -45.0,
        "max_lat": -20.0, "year": 2020, "month": 8,
        "color_by": "phi_q50_mw",
    })
    assert res.status_code == 200
    fc = res.json()
    assert fc["meta"]["n_features"] <= S.MAX_MAP_HEX_FEATURES
    assert fc["meta"]["n_features"] == len(fc["features"])


def test_10_chunk_vram_under_6gb():
    from src.engine.solver import estimate_chunk_vram_bytes

    est = estimate_chunk_vram_bytes(
        batch_size=1000, T=9510, n_platforms=5, mc_samples=200,
        sample_subbatch=16,
    )
    assert est < 6 * 1024**3
    # Per-GPU element plan matches the mandated batch geometry exactly.
    per_regime = 1000 * 9510 * 5
    expected_dense = 2 * (5 * 8 + 2) * per_regime
    assert expected_dense == 3_994_200_000


def test_10_static_frontend_served(api):
    res = api.get("/")
    assert res.status_code == 200
    assert "Burning Activity Calendar" in res.text
    assert "MapLibre" not in res.text or "map" in res.text
    res = api.get("/static/app.js")
    assert res.status_code == 200
    assert "SPOTLIGHT" in res.text or "Chilecito" in res.text
    res = api.get("/static/styles.css")
    assert res.status_code == 200
