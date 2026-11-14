"""test_10 -- Web app API & scale tests (12 tests).

Readme Section 7.10: all 5 FastAPI endpoints return 200 and expected JSON
keys; CSV formula-injection neutralization (=, +, -, @); empty-bbox
handling; DuckDB query latency < 300 ms; single-GPU chunk (batch_size =
1000, T = 9510) VRAM < 6 GB.
"""
from __future__ import annotations

import os
import time

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from tests.conftest import _full_twin


@pytest.fixture(scope="module")
def client(full_twin, tmp_path_factory):
    """FastAPI TestClient against a DuckDB built from the twin products."""
    from fastapi.testclient import TestClient
    from src.products import export_duckdb as ex
    from src.webapp import server as srv

    tmp = tmp_path_factory.mktemp("web")
    products = ex.build_products(full_twin["monthly"], full_twin["daily"],
                                 full_twin["pass_obs"], full_twin["prov"],
                                 full_twin["pass_cal"], full_twin["priors"],
                                 full_twin["artifacts"],
                                 full_twin["cloud_df"])
    db = tmp / "fire_harmonized.duckdb"
    ex.export_duckdb(products, db, parquet_dir=str(tmp))
    srv._con = None
    os.environ["FIRE_HARMONIZED_DB"] = str(db)
    try:
        with TestClient(srv.app) as c:
            yield c
    finally:
        os.environ.pop("FIRE_HARMONIZED_DB", None)
        srv._con = None


# ---------------------------------------------------------------- endpoints
def test_10_calendar_endpoint(client):
    r = client.get("/api/calendar", params=dict(
        min_lon=-69.0, min_lat=-31.5, max_lon=-66.0, max_lat=-28.5,
        start_year=2000, end_year=2026))
    assert r.status_code == 200
    body = r.json()
    assert "meta" in body and "cells" in body
    assert len(body["cells"]) > 0
    keys = body["cells"][0].keys()
    for k in ("year", "month", "value", "q05", "q95", "hist_median",
              "anomaly_ratio", "raw_count", "raw_frp", "measured_fraction",
              "quality_flag", "attribution_flag"):
        assert k in keys


def test_10_timeseries_endpoint(client):
    r = client.get("/api/timeseries", params=dict(
        min_lon=-69.0, min_lat=-31.5, max_lon=-66.0, max_lat=-28.5))
    assert r.status_code == 200
    body = r.json()
    assert "months" in body and "values" in body and "raw_counts" in body
    assert len(body["months"]) == len(body["values"])
    events = body["constellation_events"]
    assert {e["month"] for e in events} >= {"2002-07", "2012-01", "2018-04",
                                            "2022-01", "2023-03", "2026-11"}


def test_10_map_hex_endpoint(client):
    r = client.get("/api/map_hex", params=dict(
        min_lon=-69.0, min_lat=-31.5, max_lon=-66.0, max_lat=-28.5,
        year=2026, month=8))
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) <= S.MAX_HEX_FEATURES
    if fc["features"]:
        f0 = fc["features"][0]
        assert f0["geometry"]["type"] == "Polygon"
        assert "anomaly_ratio" in f0["properties"]


def test_10_diagnostics_endpoint(client):
    r = client.get("/api/diagnostics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_10_export_csv_endpoint(client):
    r = client.get("/api/export.csv", params=dict(
        min_lon=-69.0, min_lat=-31.5, max_lon=-66.0, max_lat=-28.5))
    assert r.status_code == 200
    text = r.text
    assert text.splitlines()[0].startswith("year,month")
    assert len(text.splitlines()) > 10


# ---------------------------------------------------------------- security
def test_10_csv_formula_injection_sanitized():
    from src.webapp.server import sanitize_csv_field
    assert sanitize_csv_field("=1+1") == "'=1+1"
    assert sanitize_csv_field("+CMD") == "'+CMD"
    assert sanitize_csv_field("-2") == "'-2"
    assert sanitize_csv_field("@SUM(A1)") == "'@SUM(A1)"
    assert sanitize_csv_field("\tTAB") == "'\tTAB"
    assert sanitize_csv_field("\rCR") == "'\rCR"
    assert sanitize_csv_field("2026") == "2026"
    assert sanitize_csv_field("NOMINAL") == "NOMINAL"
    assert sanitize_csv_field(None) == ""


def test_10_export_csv_never_emits_formula_cells(client):
    r = client.get("/api/export.csv", params=dict(
        min_lon=-69.0, min_lat=-31.5, max_lon=-66.0, max_lat=-28.5))
    for line in r.text.splitlines()[1:]:
        for cell in line.split(","):
            assert not (cell[:1] in ("=", "+", "-", "@", "\t", "\r")
                        and len(cell) > 1), f"unsanitized cell: {cell!r}"


# ---------------------------------------------------------------- robustness
def test_10_empty_bbox_rejected(client):
    r = client.get("/api/calendar", params=dict(
        min_lon=-67.0, min_lat=-30.0, max_lon=-67.0, max_lat=-30.0))
    assert r.status_code == 422


def test_10_inverted_bbox_rejected(client):
    r = client.get("/api/calendar", params=dict(
        min_lon=-66.0, min_lat=-30.0, max_lon=-69.0, max_lat=-28.0))
    assert r.status_code == 422


def test_10_oversized_bbox_rejected_without_country(client):
    r = client.get("/api/calendar", params=dict(
        min_lon=-170.0, min_lat=-80.0, max_lon=170.0, max_lat=80.0))
    assert r.status_code == 422


# ---------------------------------------------------------------- scale
def test_10_query_latency_under_300ms(client):
    t0 = time.time()
    for _ in range(5):
        r = client.get("/api/calendar", params=dict(
            min_lon=-69.0, min_lat=-31.5, max_lon=-66.0, max_lat=-28.5,
            start_year=2010, end_year=2026))
        assert r.status_code == 200
    per_call_ms = 1000 * (time.time() - t0) / 5
    assert per_call_ms < S.API_LATENCY_BUDGET_MS, \
        f"avg latency {per_call_ms:.1f} ms"


def test_10_single_gpu_chunk_vram_under_6gb():
    from src.engine.solver import estimate_chunk_vram_bytes
    est = estimate_chunk_vram_bytes(batch_size=S.BATCH_SIZE, t_days=S.T_DAYS,
                                    n_platforms=S.N_PLATFORMS,
                                    m_samples=S.MC_SAMPLES, sub_batch=32)
    assert est < 6 * 1024 ** 3
