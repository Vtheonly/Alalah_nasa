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


def _hex_center_con(con, h3_index):
    row = con.execute(
        "SELECT AVG(latitude), AVG(longitude) FROM obs_harmonized_provenance "
        "WHERE h3_index = ?", [int(h3_index)]).fetchone()
    return row


@app.get("/api/calendar")
def api_calendar(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 start_year: int = 2000, end_year: int = 2026,
                 metric: str = "phi_q50_mw", country_iso: str | None = None):
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, country_iso)
    t0 = time.time()
    con = get_con()
    q = f"""
        SELECT m.year, m.month, COUNT(*) AS n_cells,
               AVG(m.{metric}) AS value, AVG(m.phi_q05_mw) AS q05,
               AVG(m.phi_q95_mw) AS q95,
               AVG(m.hist_median_2003_2020) AS hist_median,
               AVG(m.anomaly_ratio) AS anomaly_ratio,
               SUM(m.raw_count_sum) AS raw_count,
               SUM(m.raw_frp_sum_mw) AS raw_frp,
               AVG(u.measured_fraction) AS measured_fraction,
               MIN(u.quality_flag) AS quality_flag,
               MIN(u.attribution_flag) AS attribution_flag
        FROM state_reconstructed_monthly m
        JOIN uncertainty_diagnostics u
          ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
        JOIN (SELECT DISTINCT h3_index FROM hex_centers
              WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?) c
          ON m.h3_index = c.h3_index
        WHERE m.year BETWEEN ? AND ?
        GROUP BY m.year, m.month ORDER BY m.year, m.month
    """
    try:
        rows = con.execute(q, [min_lon, max_lon, min_lat, max_lat,
                               start_year, end_year]).fetchall()
    except duckdb.Error:
        rows = _calendar_fallback(con, min_lon, min_lat, max_lon, max_lat,
                                  start_year, end_year, metric)
    cells = [dict(zip(("year", "month", "n_cells", "value", "q05", "q95",
                       "hist_median", "anomaly_ratio", "raw_count", "raw_frp",
                       "measured_fraction", "quality_flag",
                       "attribution_flag"), r)) for r in rows]
    return {"meta": {"bbox": [min_lon, min_lat, max_lon, max_lat],
                     "start_year": start_year, "end_year": end_year,
                     "metric": metric, "n_rows": len(cells),
                     "elapsed_ms": round(1000 * (time.time() - t0), 2)},
            "cells": cells}


def _calendar_fallback(con, min_lon, min_lat, max_lon, max_lat, y0, y1, metric):
    """Calendar join without the precomputed hex_centers view."""
    q = f"""
        WITH centers AS (
            SELECT h3_index, AVG(latitude) AS latitude, AVG(longitude) AS longitude
            FROM obs_harmonized_provenance
            WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?
            GROUP BY h3_index
        )
        SELECT m.year, m.month, COUNT(*) AS n_cells,
               AVG(m.{metric}) AS value, AVG(m.phi_q05_mw) AS q05,
               AVG(m.phi_q95_mw) AS q95,
               AVG(m.hist_median_2003_2020) AS hist_median,
               AVG(m.anomaly_ratio) AS anomaly_ratio,
               SUM(m.raw_count_sum) AS raw_count,
               SUM(m.raw_frp_sum_mw) AS raw_frp,
               AVG(u.measured_fraction) AS measured_fraction,
               MIN(u.quality_flag) AS quality_flag,
               MIN(u.attribution_flag) AS attribution_flag
        FROM state_reconstructed_monthly m
        JOIN centers c ON m.h3_index = c.h3_index
        JOIN uncertainty_diagnostics u
          ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
        WHERE m.year BETWEEN ? AND ?
        GROUP BY m.year, m.month ORDER BY m.year, m.month
    """
    return con.execute(q, [min_lon, max_lon, min_lat, max_lat, y0, y1]).fetchall()


@app.get("/api/timeseries")
def api_timeseries(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                   start_year: int = 2000, end_year: int = 2026,
                   metric: str = "phi_q50_mw"):
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, None)
    res = api_calendar(min_lon, min_lat, max_lon, max_lat, start_year, end_year,
                       metric)
    return {"meta": res["meta"],
            "months": [f"{c['year']}-{c['month']:02d}" for c in res["cells"]],
            "values": [c["value"] for c in res["cells"]],
            "raw_counts": [c["raw_count"] for c in res["cells"]],
            "q05": [c["q05"] for c in res["cells"]],
            "q95": [c["q95"] for c in res["cells"]],
            "constellation_events": [
                {"month": m, "label": lbl} for m, lbl in S.CONSTELLATION_EVENTS]}


@app.get("/api/map_hex")
def api_map_hex(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                year: int = 2026, month: int = 8,
                color_by: str = "anomaly_ratio"):
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, None)
    con = get_con()
    q = f"""
        WITH centers AS (
            SELECT h3_index, AVG(latitude) AS latitude, AVG(longitude) AS longitude
            FROM obs_harmonized_provenance
            WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?
            GROUP BY h3_index
        )
        SELECT m.h3_index, m.phi_q50_mw, m.anomaly_ratio, m.phi_q95_mw - m.phi_q05_mw AS width
        FROM state_reconstructed_monthly m JOIN centers c USING (h3_index)
        WHERE m.year = ? AND m.month = ?
        LIMIT {S.MAX_HEX_FEATURES}
    """
    try:
        rows = con.execute(q, [min_lon, max_lon, min_lat, max_lat,
                               year, month]).fetchall()
    except duckdb.Error:
        rows = []
    features = []
    for h3, phi, anom, width in rows:
        features.append({
            "type": "Feature",
            "properties": {"h3_index": int(h3), "phi_q50_mw": float(phi),
                           "anomaly_ratio": float(anom) if anom is not None else 0.0,
                           "width": float(width) if width is not None else 0.0},
            "geometry": {"type": "Polygon",
                         "coordinates": [h3_boundary_geojson(int(h3))]},
        })
    return {"type": "FeatureCollection", "features": features,
            "meta": {"year": year, "month": month, "color_by": color_by,
                     "n": len(features)}}


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
                   start_year: int = 2000, end_year: int = 2026):
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, None)
    res = api_calendar(min_lon, min_lat, max_lon, max_lat, start_year, end_year)
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
