"""Server-side data provider layer: real engine vs clearly labelled synthetic
demonstration data, validation gate, fallback rules, identical API shape."""
from __future__ import annotations

import os

import duckdb
import pytest

from src.products import export_duckdb as ex

ENV_KEYS = ("FH_DATA_PROVIDER", "FH_SYNTHETIC_FALLBACK", "FIRE_HARMONIZED_DB",
            "FH_SYNTHETIC_DB")


@pytest.fixture(scope="module")
def synthetic_db(tmp_path_factory):
    from src.webapp.providers.synthetic import build_synthetic_store
    path = tmp_path_factory.mktemp("synth") / "synthetic_demo.duckdb"
    meta = build_synthetic_store(path)
    assert meta["source"] == "SYNTHETIC_DEMO"
    return path


@pytest.fixture()
def env(monkeypatch):
    from src.webapp import providers
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    providers.reset()

    def client(**kv):
        from fastapi.testclient import TestClient
        from src.webapp import server as srv
        for k, v in kv.items():
            monkeypatch.setenv(k, str(v))
        providers.reset()
        srv._challenge_cached.cache_clear()
        srv._insights_cached.cache_clear()
        return TestClient(srv.app)
    yield client
    providers.reset()


def _real_unvalidated_db(path, synthetic_db):
    """A 'real engine' file with the export schema but no diagnostics."""
    con = duckdb.connect(str(path))
    con.execute(f"ATTACH '{synthetic_db}' AS s (READ_ONLY)")
    for t in ("state_reconstructed_monthly", "uncertainty_diagnostics",
              "hex_centers"):
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM s.{t}")
    con.close()


def test_11_synthetic_store_has_the_engine_export_schema(synthetic_db):
    con = duckdb.connect(str(synthetic_db), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    for t in ex.PRODUCT_TABLES.values():
        assert t in tables
    cols = lambda t: {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}  # noqa: E731
    assert set(ex.PRODUCT2_MONTHLY_COLUMNS) <= cols("state_reconstructed_monthly")
    assert set(ex.PRODUCT3_COLUMNS) <= cols("uncertainty_diagnostics")
    assert set(ex.PRODUCT1_COLUMNS) <= cols("obs_harmonized_provenance")
    prov = dict(con.execute("SELECT key, value FROM dataset_provenance").fetchall())
    assert '"SYNTHETIC_DEMO"' == prov["source"]
    # raw monthly counts are exact aggregates of the Product-1 detections
    bad = con.execute("""
        SELECT COUNT(*) FROM state_reconstructed_monthly m LEFT JOIN (
          SELECT h3_index, YEAR(solar_date) y, MONTH(solar_date) mo, COUNT(*) n
          FROM obs_harmonized_provenance WHERE is_biomass GROUP BY 1, 2, 3) d
        ON m.h3_index = d.h3_index AND m.year = d.y AND m.month = d.mo
        WHERE m.raw_count_sum <> COALESCE(d.n, 0)""").fetchone()[0]
    assert bad == 0


def test_11_synthetic_mode_is_labelled_and_never_validated(env, synthetic_db):
    c = env(FH_DATA_PROVIDER="synthetic", FH_SYNTHETIC_DB=synthetic_db)
    s = c.get("/api/status").json()
    p = s["provenance"]
    assert p["source"] == "SYNTHETIC_DEMO" and p["is_synthetic"]
    assert p["validated"] is False
    assert "SYNTHETIC" in p["label"]
    for path in ("/api/overview", "/api/validation", "/api/insights",
                 "/api/challenges", "/api/challenges/sensor-transition"):
        assert c.get(path).json()["provenance"]["is_synthetic"] is True


def test_11_auto_falls_back_to_synthetic_when_real_missing(env, synthetic_db,
                                                          tmp_path):
    c = env(FH_DATA_PROVIDER="auto", FH_SYNTHETIC_DB=synthetic_db,
            FIRE_HARMONIZED_DB=tmp_path / "missing.duckdb")
    s = c.get("/api/status").json()
    assert s["provenance"]["source"] == "SYNTHETIC_DEMO"
    assert s["real_engine"]["available"] is False
    assert "missing.duckdb" in s["provenance"]["reason"]
    assert str(tmp_path) not in str(s)          # no server paths leak


def test_11_unvalidated_real_output_falls_back_or_is_flagged(env, synthetic_db,
                                                            tmp_path):
    real = tmp_path / "fire_harmonized.duckdb"
    _real_unvalidated_db(real, synthetic_db)
    c = env(FH_DATA_PROVIDER="real", FH_SYNTHETIC_DB=synthetic_db,
            FIRE_HARMONIZED_DB=real)
    s = c.get("/api/status").json()
    assert s["provenance"]["source"] == "SYNTHETIC_DEMO"   # fallback on
    assert s["real_engine"]["available"] and not s["real_engine"]["validated"]
    c = env(FH_DATA_PROVIDER="real", FH_SYNTHETIC_FALLBACK="false",
            FH_SYNTHETIC_DB=synthetic_db, FIRE_HARMONIZED_DB=real)
    s = c.get("/api/status").json()
    assert s["provenance"]["source"] == "REAL_ENGINE"
    assert s["provenance"]["validated"] is False           # never claimed


def test_11_validated_real_output_is_selected(env, synthetic_db, tmp_path):
    """A real-labelled store whose diagnostics pass the critical checks is
    served as REAL_ENGINE validated (the synthetic store's diagnostics are
    used here only to exercise the gate)."""
    import shutil
    real = tmp_path / "fire_harmonized.duckdb"
    shutil.copy(synthetic_db, real)
    con = duckdb.connect(str(real))
    for t in ("dataset_provenance", "synthetic_truth", "synthetic_regions"):
        con.execute(f"DROP TABLE {t}")
    con.close()
    c = env(FH_DATA_PROVIDER="auto", FIRE_HARMONIZED_DB=real,
            FH_SYNTHETIC_DB=synthetic_db)
    s = c.get("/api/status").json()
    assert s["provenance"]["source"] == "REAL_ENGINE"
    assert s["provenance"]["validated"] is True
    assert s["validation_gate"]["passed"]


def test_11_no_provider_returns_503(env, tmp_path):
    c = env(FH_DATA_PROVIDER="real", FH_SYNTHETIC_FALLBACK="false",
            FIRE_HARMONIZED_DB=tmp_path / "missing.duckdb")
    assert c.get("/api/overview").status_code == 503


def test_11_every_challenge_has_problem_solution_and_evidence(env, synthetic_db):
    c = env(FH_DATA_PROVIDER="synthetic", FH_SYNTHETIC_DB=synthetic_db)
    ids = [x["id"] for x in c.get("/api/challenges").json()["challenges"]]
    assert len(ids) == 8
    for cid in ids:
        for iso in (None, "BRA"):
            r = c.get(f"/api/challenges/{cid}", params={"country_iso": iso} if iso else {})
            assert r.status_code == 200, cid
            j = r.json()
            assert j["problem"] and j["solution"] and j["pipeline"]
            assert j["charts"], cid
            for ch in j["charts"]:
                assert ch["series"] and all(len(s["data"]) for s in ch["series"])
    assert c.get("/api/challenges/nope").status_code == 404


def test_11_validation_insights_and_traceability(env, synthetic_db):
    c = env(FH_DATA_PROVIDER="synthetic", FH_SYNTHETIC_DB=synthetic_db)
    v = c.get("/api/validation").json()
    by = {x["id"]: x for x in v["checks"]}
    assert by["break_2012"]["status"] == "pass"
    assert by["truth_recovery"]["status"] == "pass"
    assert v["gate"]["passed"]
    ins = c.get("/api/insights", params={"country_iso": "IDN"}).json()
    assert ins["seasonality"]["peak_month"] in range(1, 13)
    assert len(ins["anomalies"]) > 0
    fc = c.get("/api/map_hex", params=dict(min_lon=-74.5, min_lat=-34, max_lon=-34,
                                           max_lat=5.5, year=2020, month=9,
                                           country_iso="BRA")).json()
    h3 = fc["features"][0]["properties"]["h3_index"]
    cell = c.get(f"/api/cell/{h3}", params={"year": 2020, "month": 9}).json()
    assert cell["reconciliation"]["match"] is True
    assert len(cell["detections"]) == cell["reconciliation"]["n_detections_total"] \
        or len(cell["detections"]) == 400
    assert c.get("/api/cell/123").status_code == 404
    assert c.get("/api/cell/abc").status_code == 422
