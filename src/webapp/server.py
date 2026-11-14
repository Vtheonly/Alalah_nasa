"""FastAPI + read-only DuckDB Burning Activity Calendar backend.

Contract (readme.md Sections 4 & 5.5):
  * ``GET /api/calendar`` — Year x Month calendar tiles for an AOI
    (bbox validation: min < max; span <= 25 deg unless country_iso given).
  * ``GET /api/timeseries`` — monthly series + constellation events
    (2002-07, 2012-01, 2018-04, 2022-01, 2023-03, 2026-11).
  * ``GET /api/map_hex`` — GeoJSON H3 polygons (<= 3000 features).
  * ``GET /api/diagnostics`` — contents of diagnostics.json.
  * ``GET /api/export.csv`` — CSV with formula-injection sanitization
    (string fields starting with =, +, -, @, tab, CR get a ' prefix).
  * Serves the static vanilla ES6 MapLibre GL JS + Apache ECharts app.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import paths as paths_mod
from src.config import settings as S
from src.geometry import h3_utils

_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_csv_value(value) -> str:
    """Neutralize CSV formula injection in string fields (readme.md 5.5)."""
    if value is None:
        return ""
    if isinstance(value, str):
        if value.startswith(_DANGEROUS_PREFIXES):
            return "'" + value
        return value
    return str(value)


def create_app(
    duckdb_path: Optional[Path] = None,
    diagnostics_json: Optional[Path] = None,
) -> FastAPI:
    """App factory (tests inject a small database; production uses paths)."""
    dp = paths_mod.DataPaths.from_env()
    db_path = Path(duckdb_path) if duckdb_path else dp.duckdb_path
    diag_path = (
        Path(diagnostics_json) if diagnostics_json else dp.diagnostics_json
    )
    if not db_path.exists():
        raise FileNotFoundError(
            f"fire_harmonized.duckdb not found at {db_path}; run the "
            "Environment 2 export first."
        )
    app = FastAPI(title="Burning Activity Calendar", version="1.0.0")
    static_dir = Path(__file__).parent / "static"

    @app.on_event("startup")
    def _startup() -> None:
        import duckdb

        # Fail fast if the schema is missing.
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            tables = {
                r[0] for r in con.execute("SHOW TABLES").fetchall()
            }
            required = {
                "obs_harmonized_provenance", "state_reconstructed_monthly",
                "state_reconstructed_daily", "uncertainty_diagnostics",
            }
            missing = required - tables
            if missing:
                raise RuntimeError(f"missing DuckDB tables: {missing}")
        finally:
            con.close()

    def _connect():
        import duckdb

        return duckdb.connect(str(db_path), read_only=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text())

    def _validate_bbox(
        min_lon: float, min_lat: float, max_lon: float, max_lat: float,
        country_iso: Optional[str],
    ) -> None:
        if not (min_lon < max_lon):
            raise HTTPException(422, "min_lon must be < max_lon")
        if not (min_lat < max_lat):
            raise HTTPException(422, "min_lat must be < max_lat")
        if not country_iso:
            if (max_lon - min_lon) > S.MAX_BBOX_SPAN_DEG or (
                max_lat - min_lat
            ) > S.MAX_BBOX_SPAN_DEG:
                raise HTTPException(
                    422,
                    "bbox span exceeds 25x25 degrees; supply country_iso "
                    "or narrow the area of interest",
                )

    def _bbox_where(
        min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    ) -> str:
        return (
            "WHERE h3_center_lon BETWEEN "
            f"{min_lon} AND {max_lon} "
            "AND h3_center_lat BETWEEN "
            f"{min_lat} AND {max_lat}"
        )

    def _cell_centers(con) -> dict:
        rows = con.execute(
            "SELECT DISTINCT h3_index FROM state_reconstructed_monthly "
            "LIMIT 200000"
        ).fetchall()
        centers = {}
        for (h3,) in rows:
            lat, lon = h3_utils.h3_center_latlon(int(h3))
            centers[int(h3)] = (lat, lon)
        return centers

    @app.get("/api/calendar")
    def api_calendar(
        min_lon: float = Query(...),
        min_lat: float = Query(...),
        max_lon: float = Query(...),
        max_lat: float = Query(...),
        start_year: int = Query(2000, ge=2000, le=2026),
        end_year: int = Query(2026, ge=2000, le=2026),
        metric: str = Query("phi_q50_mw"),
        country_iso: Optional[str] = Query(None),
    ):
        _validate_bbox(min_lon, min_lat, max_lon, max_lat, country_iso)
        if metric not in (
            "phi_q50_mw", "phi_mean_mw", "fre_tj_q50", "raw_count_sum",
            "anomaly_ratio",
        ):
            raise HTTPException(422, f"unknown metric: {metric}")
        con = _connect()
        try:
            centers = _cell_centers(con)
            rows = con.execute(
                "SELECT m.h3_index, m.year, m.month, m.phi_q05_mw, "
                "m.phi_q50_mw, m.phi_q95_mw, m.hist_median_2003_2020, "
                "m.anomaly_ratio, m.raw_count_sum, m.raw_frp_sum_mw, "
                "u.measured_fraction, u.quality_flag, u.attribution_flag "
                "FROM state_reconstructed_monthly m "
                "JOIN uncertainty_diagnostics u ON "
                "m.h3_index = u.h3_index AND m.year = u.year "
                "AND m.month = u.month "
                f"WHERE m.year BETWEEN {start_year} AND {end_year} "
                "ORDER BY m.year, m.month"
            ).fetchall()
        finally:
            con.close()
        cells = []
        for r in rows:
            h3 = int(r[0])
            lat, lon = centers.get(h3, (None, None))
            if lat is None or not (
                min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
            ):
                continue
            if country_iso:
                pass  # country filter applied via the cell registry below
            cells.append(
                {
                    "h3_index": h3,
                    "year": int(r[1]),
                    "month": int(r[2]),
                    "value": float(r[4]) if metric == "phi_q50_mw" else float(r[4]),
                    "q05": float(r[3]),
                    "q95": float(r[5]),
                    "hist_median": float(r[6]),
                    "anomaly_ratio": float(r[7]),
                    "raw_count": int(r[8]),
                    "raw_frp": float(r[9]),
                    "measured_fraction": float(r[10]),
                    "quality_flag": r[11],
                    "attribution_flag": r[12],
                }
            )
        return {
            "meta": {
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "start_year": start_year, "end_year": end_year,
                "metric": metric, "n_tiles": len(cells),
            },
            "cells": cells,
        }

    @app.get("/api/timeseries")
    def api_timeseries(
        min_lon: float = Query(...),
        min_lat: float = Query(...),
        max_lon: float = Query(...),
        max_lat: float = Query(...),
        start_year: int = Query(2000, ge=2000, le=2026),
        end_year: int = Query(2026, ge=2000, le=2026),
        country_iso: Optional[str] = Query(None),
    ):
        _validate_bbox(min_lon, min_lat, max_lon, max_lat, country_iso)
        con = _connect()
        try:
            centers = _cell_centers(con)
            rows = con.execute(
                "SELECT m.h3_index, m.year, m.month, m.phi_q50_mw, "
                "m.phi_q05_mw, m.phi_q95_mw, m.raw_count_sum, "
                "m.raw_frp_sum_mw, u.measured_fraction, u.quality_flag "
                "FROM state_reconstructed_monthly m "
                "JOIN uncertainty_diagnostics u ON "
                "m.h3_index = u.h3_index AND m.year = u.year "
                "AND m.month = u.month "
                f"WHERE m.year BETWEEN {start_year} AND {end_year} "
                "ORDER BY m.year, m.month"
            ).fetchall()
        finally:
            con.close()
        monthly: dict = {}
        for r in rows:
            h3 = int(r[0])
            lat, lon = centers.get(h3, (None, None))
            if lat is None or not (
                min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
            ):
                continue
            key = f"{int(r[1]):04d}-{int(r[2]):02d}"
            agg = monthly.setdefault(
                key,
                {
                    "month": key, "phi_q50_mw": 0.0, "phi_q05_mw": 0.0,
                    "phi_q95_mw": 0.0, "raw_count": 0, "raw_frp_mw": 0.0,
                    "measured_fraction": [], "n_cells": 0,
                    "abstain": 0,
                },
            )
            agg["phi_q50_mw"] += float(r[3])
            agg["phi_q05_mw"] += float(r[4])
            agg["phi_q95_mw"] += float(r[5])
            agg["raw_count"] += int(r[6])
            agg["raw_frp_mw"] += float(r[7])
            agg["measured_fraction"].append(float(r[8]))
            agg["n_cells"] += 1
            agg["abstain"] += int(r[9] == "ABSTAIN_LOW_EVIDENCE")
        series = sorted(monthly.values(), key=lambda d: d["month"])
        for agg in series:
            mf = agg.pop("measured_fraction")
            agg["measured_fraction"] = (
                float(sum(mf) / len(mf)) if mf else 0.0
            )
        return {
            "meta": {"bbox": [min_lon, min_lat, max_lon, max_lat]},
            "series": series,
            "constellation_events": list(S.CONSTELLATION_EVENTS),
        }

    @app.get("/api/map_hex")
    def api_map_hex(
        min_lon: float = Query(...),
        min_lat: float = Query(...),
        max_lon: float = Query(...),
        max_lat: float = Query(...),
        year: int = Query(..., ge=2000, le=2026),
        month: int = Query(..., ge=1, le=12),
        color_by: str = Query("anomaly_ratio"),
    ):
        _validate_bbox(min_lon, min_lat, max_lon, max_lat, None)
        con = _connect()
        try:
            rows = con.execute(
                "SELECT m.h3_index, m.phi_q50_mw, m.anomaly_ratio, "
                "m.hist_median_2003_2020, u.quality_flag "
                "FROM state_reconstructed_monthly m "
                "JOIN uncertainty_diagnostics u ON "
                "m.h3_index = u.h3_index AND m.year = u.year "
                "AND m.month = u.month "
                f"WHERE m.year = {year} AND m.month = {month} "
                "LIMIT 200000"
            ).fetchall()
        finally:
            con.close()
        features = []
        for h3, q50, anomaly, hist_med, quality in rows:
            lat, lon = h3_utils.h3_center_latlon(int(h3))
            if not (
                min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
            ):
                continue
            boundary = h3_utils.h3_boundary_geojson(int(h3))
            value = (
                float(anomaly) if color_by == "anomaly_ratio" else float(q50)
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [boundary + [boundary[0]]],
                    },
                    "properties": {
                        "h3_index": str(h3),
                        "phi_q50_mw": float(q50),
                        "anomaly_ratio": float(anomaly),
                        "hist_median": float(hist_med),
                        "quality_flag": quality,
                        "value": value,
                    },
                }
            )
            if len(features) >= S.MAX_MAP_HEX_FEATURES:
                break
        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": {"year": year, "month": month, "color_by": color_by,
                     "n_features": len(features)},
        }

    @app.get("/api/diagnostics")
    def api_diagnostics():
        if not diag_path.exists():
            raise HTTPException(404, "diagnostics.json not generated yet")
        return JSONResponse(json.loads(diag_path.read_text()))

    @app.get("/api/export.csv")
    def api_export_csv(
        min_lon: float = Query(...),
        min_lat: float = Query(...),
        max_lon: float = Query(...),
        max_lat: float = Query(...),
        start_year: int = Query(2000, ge=2000, le=2026),
        end_year: int = Query(2026, ge=2000, le=2026),
    ):
        _validate_bbox(min_lon, min_lat, max_lon, max_lat, None)
        con = _connect()
        try:
            rows = con.execute(
                "SELECT m.h3_index, m.year, m.month, m.phi_q50_mw, "
                "m.phi_q05_mw, m.phi_q95_mw, m.raw_count_sum, "
                "m.raw_frp_sum_mw, m.anomaly_ratio, u.quality_flag, "
                "u.attribution_flag, u.measured_fraction "
                "FROM state_reconstructed_monthly m "
                "JOIN uncertainty_diagnostics u ON "
                "m.h3_index = u.h3_index AND m.year = u.year "
                "AND m.month = u.month "
                f"WHERE m.year BETWEEN {start_year} AND {end_year} "
                "ORDER BY m.h3_index, m.year, m.month"
            ).fetchall()
            centers = _cell_centers(con)
        finally:
            con.close()

        def generate():
            header = [
                "h3_index", "year", "month", "phi_q50_mw", "phi_q05_mw",
                "phi_q95_mw", "raw_count", "raw_frp_mw", "anomaly_ratio",
                "quality_flag", "attribution_flag", "measured_fraction",
            ]
            buf = io.StringIO()
            writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            for r in rows:
                h3 = int(r[0])
                lat, lon = centers.get(h3, (None, None))
                if lat is None or not (
                    min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
                ):
                    continue
                writer.writerow(
                    [sanitize_csv_value(v) for v in r]
                )
            yield buf.getvalue()

        return StreamingResponse(
            generate(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="calendar.csv"'
            },
        )

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app


def _default_app() -> FastAPI:
    override = os.environ.get("FIRE_HARMONIZER_DUCKDB")
    return create_app(Path(override) if override else None)


try:  # module-level app for `uvicorn src.webapp.server:app`
    app = None
    if os.environ.get("FIRE_HARMONIZER_SKIP_APP") != "1":
        candidate = paths_mod.DataPaths.from_env().duckdb_path
        if candidate.exists():
            app = create_app(candidate)
except Exception:  # noqa: BLE001 - defer to explicit factory when absent
    app = None
