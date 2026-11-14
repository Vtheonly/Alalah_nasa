"""FastAPI + read-only DuckDB Burning Activity Calendar web app.

Endpoints (readme Section 5.5):
* ``GET /api/calendar`` -- Year x Month calendar aggregates for an AOI bbox
  (bbox span <= 25 deg unless ``country_iso`` supplied).
* ``GET /api/timeseries`` -- monthly array + constellation events.
* ``GET /api/map_hex`` -- GeoJSON FeatureCollection of H3 polygons
  (<= 3000 features).
* ``GET /api/diagnostics`` -- diagnostics.json contents.
* ``GET /api/export.csv`` -- CSV download sanitized against formula
  injection (``=``/``+``/``-``/``@``/tab/CR prefixed with ``'``).
* ``GET /`` -- static MapLibre GL JS + Apache ECharts frontend.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..config import settings as S
from ..geometry.h3_utils import h3_boundary_geojson

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DB = Path(os.environ.get("FIRE_HARMONIZED_DB", "output/fire_harmonized.duckdb"))

app = FastAPI(title="Burning Activity Calendar", version="1.0.0")
_con = None


def get_con() -> duckdb.DuckDBPyConnection:
    """Read-only DuckDB connection to fire_harmonized.duckdb."""
    global _con
    if _con is None:
        path = Path(os.environ.get("FIRE_HARMONIZED_DB", str(DEFAULT_DB)))
        if not path.exists():
            raise HTTPException(503, "fire_harmonized.duckdb not found; run "
                                     "the Env 2 pipeline first")
        _con = duckdb.connect(str(path), read_only=True)
    return _con


def _validate_bbox(min_lon, min_lat, max_lon, max_lat, country_iso=None):
    if not (min_lon < max_lon and min_lat < max_lat):
        raise HTTPException(422, "empty or inverted bbox: require "
                                 "min_lon < max_lon and min_lat < max_lat")
    span = max(max_lon - min_lon, max_lat - min_lat)
    if span > S.MAX_BBOX_SPAN_DEG and not country_iso:
        raise HTTPException(422, f"bbox span {span:.1f} deg exceeds "
                                 f"{S.MAX_BBOX_SPAN_DEG} deg (supply "
                                 "country_iso for country-wide queries)")


# Strict column whitelists: user input never reaches SQL as an identifier.
CALENDAR_METRICS = ("phi_q50_mw", "phi_mean_mw", "phi_day_mw", "phi_night_mw",
                    "fre_tj_q50", "energy_equiv_events_q50",
                    "hurdle_adj_events", "est_active_area_km2",
                    "phi_flux_mw_km2", "anomaly_ratio", "raw_count_sum",
                    "raw_frp_sum_mw")
MAP_COLOR_FIELDS = ("anomaly_ratio", "phi_q50_mw", "anomaly_zscore",
                    "climatology_pctile", "fre_tj_q50")


def _metric_column(metric: str) -> str:
    if metric not in CALENDAR_METRICS:
        raise HTTPException(422, f"unsupported metric '{metric}'; allowed: "
                                 f"{', '.join(CALENDAR_METRICS)}")
    return metric


def _country(country_iso: str | None) -> str | None:
    if country_iso is None or country_iso == "":
        return None
    iso = country_iso.upper()
    if iso not in S.TARGET_COUNTRY_ISOS:
        raise HTTPException(422, f"unknown country_iso '{country_iso}'; "
                                 f"allowed: {', '.join(S.TARGET_COUNTRY_ISOS)}")
    return iso


def _cell_filter_sql(country_iso: str | None):
    """Spatial cell filter on the materialized ``hex_centers`` index: bbox
    AND (optionally) the country."""
    sql = ("SELECT h3_index FROM hex_centers WHERE longitude BETWEEN ? AND ? "
           "AND latitude BETWEEN ? AND ?")
    if country_iso:
        sql += " AND country_iso = ?"
    return sql


def _calendar_rows(con, min_lon, min_lat, max_lon, max_lat, y0, y1, metric,
                   country_iso):
    col = _metric_column(metric)
    q = f"""
        SELECT m.year, m.month, COUNT(*) AS n_cells,
               AVG(m.{col}) AS value, AVG(m.phi_q05_mw) AS q05,
               AVG(m.phi_q95_mw) AS q95,
               AVG(m.hist_median_2003_2020) AS hist_median,
               AVG(m.anomaly_ratio) AS anomaly_ratio,
               SUM(m.raw_count_sum) AS raw_count,
               SUM(m.raw_frp_sum_mw) AS raw_frp,
               AVG(u.measured_fraction) AS measured_fraction,
               CASE WHEN BOOL_OR(u.quality_flag = 'ABSTAIN_LOW_EVIDENCE')
                    THEN 'ABSTAIN_LOW_EVIDENCE'
                    WHEN BOOL_OR(u.quality_flag = 'CLIMATOLOGY_CLOUD_FALLBACK')
                    THEN 'CLIMATOLOGY_CLOUD_FALLBACK' ELSE 'NOMINAL' END
                    AS quality_flag,
               MODE(u.attribution_flag) AS attribution_flag
        FROM state_reconstructed_monthly m
        JOIN uncertainty_diagnostics u
          ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
        WHERE m.h3_index IN ({_cell_filter_sql(country_iso)})
          AND m.year BETWEEN ? AND ?
        GROUP BY m.year, m.month ORDER BY m.year, m.month
    """
    params = [min_lon, max_lon, min_lat, max_lat]
    if country_iso:
        params.append(country_iso)
    params += [y0, y1]
    return con.execute(q, params).fetchall()


@app.get("/api/calendar")
def api_calendar(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 start_year: int = 2000, end_year: int = 2026,
                 metric: str = "phi_q50_mw", country_iso: str | None = None):
    _metric_column(metric)
    iso = _country(country_iso)
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, iso)
    t0 = time.time()
    con = get_con()
    rows = _calendar_rows(con, min_lon, min_lat, max_lon, max_lat, start_year,
                          end_year, metric, iso)
    cells = [dict(zip(("year", "month", "n_cells", "value", "q05", "q95",
                       "hist_median", "anomaly_ratio", "raw_count", "raw_frp",
                       "measured_fraction", "quality_flag",
                       "attribution_flag"), r)) for r in rows]
    return {"meta": {"bbox": [min_lon, min_lat, max_lon, max_lat],
                     "country_iso": iso,
                     "start_year": start_year, "end_year": end_year,
                     "metric": metric, "n_rows": len(cells),
                     "elapsed_ms": round(1000 * (time.time() - t0), 2)},
            "cells": cells}


@app.get("/api/timeseries")
def api_timeseries(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                   start_year: int = 2000, end_year: int = 2026,
                   metric: str = "phi_q50_mw", country_iso: str | None = None):
    res = api_calendar(min_lon, min_lat, max_lon, max_lat, start_year, end_year,
                       metric, country_iso)
    return {"meta": res["meta"],
            "months": [f"{c['year']}-{c['month']:02d}" for c in res["cells"]],
            "values": [c["value"] for c in res["cells"]],
            "raw_counts": [c["raw_count"] for c in res["cells"]],
            "q05": [c["q05"] for c in res["cells"]],
            "q95": [c["q95"] for c in res["cells"]],
            "hist_median": [c["hist_median"] for c in res["cells"]],
            "constellation_events": [
                {"month": m, "label": lbl} for m, lbl in S.CONSTELLATION_EVENTS]}


@app.get("/api/map_hex")
def api_map_hex(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                year: int = 2026, month: int = 8,
                color_by: str = "anomaly_ratio",
                country_iso: str | None = None):
    if color_by not in MAP_COLOR_FIELDS:
        raise HTTPException(422, f"unsupported color_by '{color_by}'; allowed: "
                                 f"{', '.join(MAP_COLOR_FIELDS)}")
    iso = _country(country_iso)
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, iso)
    con = get_con()
    q = f"""
        SELECT m.h3_index, m.phi_q50_mw, m.{color_by} AS color_value,
               m.phi_q95_mw - m.phi_q05_mw AS width, u.quality_flag,
               u.measured_fraction
        FROM state_reconstructed_monthly m
        JOIN uncertainty_diagnostics u
          ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
        WHERE m.h3_index IN ({_cell_filter_sql(iso)})
          AND m.year = ? AND m.month = ?
        LIMIT {int(S.MAX_HEX_FEATURES)}
    """
    params = [min_lon, max_lon, min_lat, max_lat]
    if iso:
        params.append(iso)
    params += [year, month]
    rows = con.execute(q, params).fetchall()
    features = []
    for h3, phi, cval, width, qf, mf in rows:
        features.append({
            "type": "Feature",
            "properties": {"h3_index": str(int(h3)), "phi_q50_mw": float(phi),
                           color_by: float(cval) if cval is not None else 0.0,
                           "width": float(width) if width is not None else 0.0,
                           "quality_flag": qf,
                           "measured_fraction": float(mf) if mf is not None
                           else 0.0},
            "geometry": {"type": "Polygon",
                         "coordinates": [h3_boundary_geojson(int(h3))]},
        })
    return {"type": "FeatureCollection", "features": features,
            "meta": {"year": year, "month": month, "color_by": color_by,
                     "n": len(features), "max_features": S.MAX_HEX_FEATURES}}


@app.get("/api/diagnostics")
def api_diagnostics():
    con = get_con()
    try:
        row = con.execute(
            "SELECT diagnostics_json FROM diagnostics LIMIT 1").fetchone()
        return json.loads(row[0]) if row else {}
    except duckdb.Error:
        diag_path = Path(os.environ.get("FIRE_HARMONIZED_DIAGNOSTICS",
                                        "output/diagnostics.json"))
        if diag_path.exists():
            return json.loads(diag_path.read_text())
        return {"note": "diagnostics.json not yet generated"}


def sanitize_csv_field(v) -> str:
    """CSV formula-injection sanitization (=/+/-/@/tab/CR prefixed with ')."""
    s = "" if v is None else str(v)
    if s[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


@app.get("/api/export.csv", response_class=PlainTextResponse)
def api_export_csv(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                   start_year: int = 2000, end_year: int = 2026,
                   metric: str = "phi_q50_mw", country_iso: str | None = None):
    res = api_calendar(min_lon, min_lat, max_lon, max_lat, start_year, end_year,
                       metric, country_iso)
    cols = ["year", "month", "value", "q05", "q95", "hist_median",
            "anomaly_ratio", "raw_count", "raw_frp", "measured_fraction",
            "quality_flag", "attribution_flag"]
    lines = [",".join(cols)]
    for c in res["cells"]:
        lines.append(",".join(sanitize_csv_field(c.get(col)) for col in cols))
    return "\n".join(lines) + "\n"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
