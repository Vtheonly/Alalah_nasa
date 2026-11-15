"""FastAPI web service for the harmonized MODIS/VIIRS fire record.

All data comes from the server-side provider layer (``src.webapp.providers``):
the real engine output or, when it is unavailable / not validated and the
fallback is enabled, a clearly labelled SYNTHETIC demonstration store with
the identical schema.  Every response carries ``provenance``; the frontend
is identical for both providers.

Endpoints:
* ``GET /api/status`` -- active provider, validation gate, configuration.
* ``GET /api/overview`` / ``/api/regions``.
* ``GET /api/calendar`` / ``/api/timeseries`` / ``/api/map_hex`` (readme 5.5).
* ``GET /api/challenges`` + ``/api/challenges/{id}`` -- each scientific /
  technical challenge with the evidence of how it is solved.
* ``GET /api/validation`` -- validation checks with evidence.
* ``GET /api/insights`` -- trends, seasonality, anomalies.
* ``GET /api/cell/{h3_index}`` -- traceability down to every satellite
  detection behind a harmonized cell-month.
* ``GET /api/diagnostics`` / ``/api/export.csv``.
"""
from __future__ import annotations

import functools
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..config import settings as S
from ..geometry.h3_utils import h3_boundary_geojson
from . import providers

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = Path(os.environ.get("FH_FRONTEND_DIST",
                                    str(REPO_ROOT / "Frontend" / "dist")))

app = FastAPI(title="Fire Harmonizer - Burning Activity Calendar",
              version="2.0.0")
_con = None          # kept for backward compatibility with older tests


def active() -> dict:
    try:
        return providers.get_active()
    except RuntimeError as exc:          # no provider available
        raise HTTPException(503, str(exc)) from exc


def store():
    return active()["store"]


def prov() -> dict:
    active()
    return providers.provenance()


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


# ------------------------------------------------------------------ status
@app.get("/api/status")
def api_status():
    a = active()
    st = a["store"]
    val = st.validation()
    cfg = dict(a["config"])
    for k in ("real_db", "synthetic_db"):
        cfg[k] = Path(cfg[k]).name
    return {"provenance": prov(),
            "dataset": st.dataset_provenance(),
            "config": cfg,
            "real_engine": a["real_engine"],
            "validation_gate": val["gate"],
            "constellation_events": [{"month": m, "label": lbl}
                                     for m, lbl in S.CONSTELLATION_EVENTS]}


@app.get("/api/overview")
def api_overview():
    return {"provenance": prov(), **store().overview()}


@app.get("/api/regions")
def api_regions():
    return {"provenance": prov(), "regions": store().regions()}


# ------------------------------------------------- calendar / series / map
@app.get("/api/calendar")
def api_calendar(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 start_year: int = 2000, end_year: int = 2026,
                 metric: str = "phi_q50_mw", country_iso: str | None = None):
    _metric_column(metric)
    iso = _country(country_iso)
    _validate_bbox(min_lon, min_lat, max_lon, max_lat, iso)
    t0 = time.time()
    rows = store().calendar(min_lon, min_lat, max_lon, max_lat, start_year,
                            end_year, metric, iso)
    cells = [dict(zip(("year", "month", "n_cells", "value", "q05", "q95",
                       "hist_median", "anomaly_ratio", "raw_count", "raw_frp",
                       "measured_fraction", "quality_flag",
                       "attribution_flag", "fre_tj"), r)) for r in rows]
    return {"meta": {"bbox": [min_lon, min_lat, max_lon, max_lat],
                     "country_iso": iso,
                     "start_year": start_year, "end_year": end_year,
                     "metric": metric, "n_rows": len(cells),
                     "elapsed_ms": round(1000 * (time.time() - t0), 2),
                     "provenance": prov()},
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
    rows = store().map_hex(min_lon, min_lat, max_lon, max_lat, year, month,
                           color_by, iso, S.MAX_HEX_FEATURES)
    features = []
    for h3, phi, cval, width, qf, mf, rc, af, ar, fre in rows:
        features.append({
            "type": "Feature",
            "properties": {"h3_index": str(int(h3)), "phi_q50_mw": float(phi),
                           color_by: float(cval) if cval is not None else 0.0,
                           "anomaly_ratio": float(ar) if ar is not None else 0.0,
                           "width": float(width) if width is not None else 0.0,
                           "quality_flag": qf, "attribution_flag": af,
                           "raw_count": float(rc or 0.0),
                           "fre_tj": float(fre or 0.0),
                           "measured_fraction": float(mf) if mf is not None
                           else 0.0},
            "geometry": {"type": "Polygon",
                         "coordinates": [h3_boundary_geojson(int(h3))]},
        })
    return {"type": "FeatureCollection", "features": features,
            "meta": {"year": year, "month": month, "color_by": color_by,
                     "n": len(features), "max_features": S.MAX_HEX_FEATURES,
                     "provenance": prov()}}


# --------------------------------------------------------------- challenges
CHALLENGES = [
    {"id": "sensor-transition", "icon": "git-merge",
     "title": "The 2012 sensor jump",
     "problem": "VIIRS (375 m) joined MODIS (1 km) in 2012 and sees far more "
                "fire pixels: raw counts jump although the fires did not change.",
     "solution": "Sensor-specific observation operators, calibrated on the "
                 "2012-2021 overlap and then frozen, map every detection onto "
                 "one latent fire-energy state, so the record stays continuous.",
     "pipeline": ["src/calibration/operators.py", "src/engine/solver.py"],
     "method": "_sensor_transition"},
    {"id": "resolution", "icon": "scan",
     "title": "Different pixel sizes and detection limits",
     "problem": "A 1 km MODIS pixel misses small fires a 375 m VIIRS pixel "
                "catches, and FRP per detection differs several-fold.",
     "solution": "Sensor-specific forward gains plus a hurdle likelihood for "
                 "non-detections translate counts and FRP of every sensor into "
                 "the same physical quantity (fire radiative power).",
     "pipeline": ["src/engine/likelihoods.py", "src/calibration/operators.py"],
     "method": "_resolution"},
    {"id": "overpass-timing", "icon": "clock",
     "title": "Different overpass times and orbit drift",
     "problem": "Terra passes at ~10:30, Aqua and VIIRS at ~13:30: each samples "
                "a different moment of the daily fire cycle, and Terra/Aqua "
                "drift after 2022.",
     "solution": "Each detection is placed on its local solar time; a diurnal "
                 "operator normalizes every pass to a 24-hour mean with separate "
                 "day and night latent states.",
     "pipeline": ["src/geometry/solar_geometry.py", "src/engine/solver.py"],
     "method": "_overpass"},
    {"id": "missing-observations", "icon": "cloud-off",
     "title": "Clouds and missing observations",
     "problem": "Clouds hide fires: a month without detections may mean "
                "'no fire' or simply 'not seen'.",
     "solution": "Cloud-aware visibility enters the likelihood, uncertainty "
                 "widens when evidence is thin, and months with too little "
                 "evidence are flagged ABSTAIN instead of guessed.",
     "pipeline": ["src/engine/priors.py", "src/engine/conformal.py"],
     "method": "_missing"},
    {"id": "constellation", "icon": "satellite",
     "title": "A constellation that keeps changing",
     "problem": "Five satellites launched, drifted, had outages or ended "
                "between 2000 and 2026 (S-NPP delivery ended Nov 2026).",
     "solution": "Operational calendars mask each platform exactly when it "
                 "flew, so the harmonized state stays continuous across every "
                 "transition.",
     "pipeline": ["src/ingest/normalize_dedup.py", "src/engine/solver.py"],
     "method": "_constellation"},
    {"id": "non-fire-sources", "icon": "factory",
     "title": "Gas flares and other non-fire heat",
     "problem": "Gas flares, industry and South Atlantic Anomaly noise look "
                "like fires to a thermal sensor.",
     "solution": "A persistent-source mask on a fine H3 grid (>=3 years, >=15 "
                 "days/yr, day~night) and a SAMA filter remove them before "
                 "fusion; they stay visible in the provenance table.",
     "pipeline": ["src/geometry/preprocess_geometry.py"],
     "method": "_flares"},
    {"id": "duplicates", "icon": "copy",
     "title": "Duplicate and overlapping detections",
     "problem": "Overlapping MODIS scan lines (bow-tie) and multi-pixel fires "
                "make one fire look like many.",
     "solution": "Bow-tie weights down-weight duplicated pixels and per-pass "
                 "clustering groups pixels into independent fire events.",
     "pipeline": ["src/geometry/preprocess_geometry.py"],
     "method": "_duplicates"},
    {"id": "uncertainty", "icon": "gauge",
     "title": "Knowing how sure we are",
     "problem": "A fused estimate is meaningless without an honest statement "
                "of its uncertainty.",
     "solution": "Posterior intervals are recalibrated with conformal hold-out "
                 "folds (VIIRS withheld, one satellite left out) so that 90% "
                 "intervals really cover ~90%.",
     "pipeline": ["src/engine/conformal.py", "src/validation/diagnostics.py"],
     "method": "_uncertainty"},
]
_CH_METHODS = {"_sensor_transition": "challenge_sensor_transition",
               "_resolution": "challenge_resolution",
               "_overpass": "challenge_overpass",
               "_missing": "challenge_missing",
               "_constellation": "challenge_constellation",
               "_flares": "challenge_flares",
               "_duplicates": "challenge_duplicates",
               "_uncertainty": "challenge_uncertainty"}


@functools.lru_cache(maxsize=128)
def _challenge_cached(store_id: int, cid: str, iso: str | None):
    ch = next(c for c in CHALLENGES if c["id"] == cid)
    fn = getattr(store(), _CH_METHODS[ch["method"]])
    return fn(iso)


@app.get("/api/challenges")
def api_challenges():
    return {"provenance": prov(),
            "challenges": [{k: v for k, v in c.items() if k != "method"}
                           for c in CHALLENGES]}


@app.get("/api/challenges/{cid}")
def api_challenge(cid: str, country_iso: str | None = None):
    if cid not in {c["id"] for c in CHALLENGES}:
        raise HTTPException(404, f"unknown challenge '{cid}'")
    iso = _country(country_iso)
    ch = next(c for c in CHALLENGES if c["id"] == cid)
    ev = _challenge_cached(id(store()), cid, iso)
    return {"provenance": prov(), "country_iso": iso,
            **{k: v for k, v in ch.items() if k != "method"}, **ev}


# --------------------------------------------- validation / insights / cell
@app.get("/api/validation")
def api_validation():
    return {"provenance": prov(), **store().validation()}


@functools.lru_cache(maxsize=16)
def _insights_cached(store_id: int, iso: str | None):
    return store().insights(iso)


@app.get("/api/insights")
def api_insights(country_iso: str | None = None):
    iso = _country(country_iso)
    return {"provenance": prov(), "country_iso": iso,
            **_insights_cached(id(store()), iso)}


@app.get("/api/cell/{h3_index}")
def api_cell(h3_index: str, year: int | None = None, month: int | None = None):
    try:
        h3 = int(h3_index)
    except ValueError as exc:
        raise HTTPException(422, "h3_index must be the integer cell id") from exc
    if month is not None and not 1 <= month <= 12:
        raise HTTPException(422, "month must be 1..12")
    out = store().cell(h3, year, month)
    if not out:
        raise HTTPException(404, f"cell {h3_index} not in the harmonized product")
    return {"provenance": prov(), **out}


@app.get("/api/replay")
def api_replay(year: int, month: int, country_iso: str | None = None,
               step: str = "day"):
    """Burning-calendar replay of one month: harmonized daily hexagon states
    + every satellite detection, by day or by overpass hour."""
    if step not in ("day", "hour"):
        raise HTTPException(422, "step must be 'day' or 'hour'")
    if not 1 <= month <= 12:
        raise HTTPException(422, "month must be 1..12")
    iso = _country(country_iso)
    return {"provenance": prov(), "country_iso": iso,
            **store().replay(iso, year, month, step)}


@app.get("/api/diagnostics")
def api_diagnostics():
    d = store().diagnostics()
    return d if d else {"note": "diagnostics.json not yet generated"}


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
    p = res["meta"]["provenance"]
    cols = ["year", "month", "value", "q05", "q95", "hist_median",
            "anomaly_ratio", "raw_count", "raw_frp", "measured_fraction",
            "quality_flag", "attribution_flag"]
    lines = [",".join(cols)]
    for c in res["cells"]:
        lines.append(",".join(sanitize_csv_field(c.get(col)) for col in cols))
    lines.append(sanitize_csv_field(f"# provenance: {p['source']} "
                                    f"validated={p['validated']}"))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- frontend
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")),
              name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    built = FRONTEND_DIST / "index.html"
    if built.exists():
        return FileResponse(built)
    return (STATIC_DIR / "index.html").read_text()
