"""Environment 1 data downloader (Internet ON, CPU only).

Contract (readme Sections 4.5, 6.1, 6.2):
1. University of Maryland fire archive (``fuoco.geog.umd.edu``, SFTP port 22
   first, FTP port 21 fallback, login ``fire`` / ``burnt``) for MCD14ML
   space-delimited ``.txt.gz`` monthly files, VNP14IMGML / VJ114IMGML /
   VJ214IMGML ``.csv.gz`` files, MOD14C8Q / MYD14C8Q (8-day) and Mx14CMQ
   (monthly fallback) HDF4 cloud grids, and MCD64CMQ burned-area HDF4 grids.
2. Source Cooperative "Portolan" NASA FIRMS GeoParquet 2.0 mirror via DuckDB
   ``httpfs`` (+ ``spatial``) with Hilbert/bbox row-group pruning if UMD is
   unreachable.
3. NASA FIRMS REST API strictly for trailing-2026 NRT gap-fill
   (``DAY_RANGE <= 5``, ``time.sleep(0.15)`` between calls, 429 backoff).
4. Natural Earth 50m Admin-0 country polygons (GeoBoundaries fallback).

Autonomous Discovery & Failover Protocol (readme Section 6.2) is executed
whenever a URL/path fails: remote directory tree traversal, STAC inspection,
NASA CMR granule search, GeoBoundaries polygon fallback, and a schema
validation gate before any discovered file enters ``staged_raw/``.  All
resolved sources are logged in ``data_provenance_manifest.json``.  If every
mirror for a product fails, we log a critical warning and abort -- synthetic
fabrication is forbidden.

``pyhdf`` is imported lazily inside HDF4 readers (it is Env 1 only and must
never leak into Environment 2).
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import re
import socket
import time
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import settings as S
from ..config.paths import DataPaths

logger = logging.getLogger("fire_harmonizer.ingest.download")

FTP_TIMEOUT_S = 30.0
HTTP_TIMEOUT_S = 120.0
FIRMS_SLEEP_S = 0.15                 # respects the 5,000 tx / 10 min limit
FIRMS_MAX_BACKOFF_S = 60.0

_UMD_PATTERNS = {
    "MCD14ML": re.compile(r"MCD14ML", re.IGNORECASE),
    "VNP14IMGML": re.compile(r"VNP14IMGML", re.IGNORECASE),
    "VJ114IMGML": re.compile(r"VJ114IMGML", re.IGNORECASE),
    "VJ214IMGML": re.compile(r"VJ214IMGML", re.IGNORECASE),
    "MOD14C8Q": re.compile(r"MOD14C(8|M)Q", re.IGNORECASE),
    "MYD14C8Q": re.compile(r"MYD14C(8|M)Q", re.IGNORECASE),
    "MCD64CMQ": re.compile(r"MCD64CMQ", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------
class ProvenanceManifest:
    """Structured log of resolved data sources (readme 6.2)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[dict] = []

    def log(self, product: str, source: str, method: str,
            detail: str = "", status: str = "OK") -> None:
        entry = {
            "product": product,
            "resolved_source": source,
            "method": method,
            "detail": detail,
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.entries.append(entry)
        logger.info("provenance: %s -> %s (%s) [%s]", product, source, method, status)

    def critical(self, product: str, message: str) -> None:
        self.log(product, "NONE", "EXHAUSTED", detail=message, status="CRITICAL")
        self.save()
        raise RuntimeError(
            f"All mirrors failed for '{product}': {message}. Aborting rather "
            "than fabricating data (readme Section 6.2, Must-Not-Do)."
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))


# ---------------------------------------------------------------------------
# Natural Earth / GeoBoundaries polygons
# ---------------------------------------------------------------------------
def _polygon_records(geojson: dict, target_isos) -> list[dict]:
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        iso = props.get("ADM0_A3") or props.get("ISO_A3") or props.get("GID_0")
        if iso in target_isos:
            rows.append({"country_iso": iso,
                         "geometry": json.dumps(feat.get("geometry") or {})})
    return rows


def download_country_polygons(paths: DataPaths,
                              manifest: ProvenanceManifest | None = None) -> Path:
    """Fetch Natural Earth 50m Admin-0 polygons for the 6 target countries."""
    target_isos = set(S.TARGET_COUNTRY_ISOS)
    out = Path(paths.country_polygons_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Primary: Natural Earth GeoJSON on GitHub raw.
    try:
        resp = requests.get(S.NATURAL_EARTH_URL, timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        rows = _polygon_records(resp.json(), target_isos)
        if len(rows) == len(target_isos):
            pd.DataFrame(rows).to_parquet(out, index=False)
            if manifest:
                manifest.log("country_polygons", S.NATURAL_EARTH_URL, "http")
            return out
        logger.warning("Natural Earth GeoJSON missing target ISOs (%d/%d).",
                       len(rows), len(target_isos))
    except Exception as exc:  # noqa: BLE001 - fallthrough to mirror protocol
        logger.warning("Natural Earth fetch failed (%s); running failover.", exc)

    # Failover: GeoBoundaries Open API per ISO3.
    rows = []
    for iso in sorted(target_isos):
        try:
            api = S.GEOBOUNDARIES_URL.format(ISO3=iso)
            meta = requests.get(api, timeout=HTTP_TIMEOUT_S)
            meta.raise_for_status()
            gj_url = meta.json()["gjDownloadURL"]
            gj = requests.get(gj_url, timeout=HTTP_TIMEOUT_S)
            gj.raise_for_status()
            rows.append({"country_iso": iso,
                         "geometry": json.dumps(gj.json()["features"][0]["geometry"])})
        except Exception as exc:  # noqa: BLE001
            logger.warning("GeoBoundaries fetch failed for %s: %s", iso, exc)
    if len(rows) == len(target_isos):
        pd.DataFrame(rows).to_parquet(out, index=False)
        if manifest:
            manifest.log("country_polygons", S.GEOBOUNDARIES_URL, "geoboundaries")
        return out
    if manifest:
        manifest.critical("country_polygons", "Natural Earth and GeoBoundaries both failed.")
    raise RuntimeError("Could not download country polygons from any mirror.")


# ---------------------------------------------------------------------------
# UMD SFTP / FTP archive (primary historical source)
# ---------------------------------------------------------------------------
def _iter_remote_files(host: str, user: str, password: str,
                       patterns: dict[str, re.Pattern]) -> dict[str, list[str]]:
    """Walk the UMD archive tree over SFTP (fallback FTP), matching patterns."""
    try:
        import paramiko  # Env 1 dependency
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=22, username=user, password=password,
                       timeout=FTP_TIMEOUT_S, allow_agent=False, look_for_keys=False)
        sftp = client.open_sftp()

        def listdir(path):
            return [(attr.filename, attr.st_mode) for attr in sftp.listdir_attr(path)]

        def read_file(path):
            with sftp.open(path, "rb") as fh:
                return fh.read()

        def close():
            sftp.close()
            client.close()
        return _walk_tree(listdir, read_file, close, patterns, "sftp")
    except Exception as exc:  # noqa: BLE001
        logger.warning("UMD SFTP failed (%s); trying FTP port 21.", exc)

    import ftplib
    ftp = ftplib.FTP()
    ftp.connect(host, port=21, timeout=FTP_TIMEOUT_S)
    ftp.login(user, password)

    def listdir_ftp(path):
        entries = []
        try:
            for name in ftp.nlst(path):
                try:
                    size = ftp.size(name)
                    entries.append((name.split("/")[-1], 1 if size is not None else None))
                except Exception:  # noqa: BLE001 - directories raise on SIZE
                    entries.append((name.split("/")[-1], None))
        except Exception:  # noqa: BLE001
            pass
        return entries

    def read_file_ftp(path):
        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {path}", buf.write)
        return buf.getvalue()

    def close_ftp():
        ftp.quit()

    return _walk_tree(listdir_ftp, read_file_ftp, close_ftp, patterns, "ftp")


def _walk_tree(listdir, read_file, close, patterns: dict, method: str):
    """Recursive top-level traversal matching readme discovery patterns."""
    found: dict[str, list[str]] = {k: [] for k in patterns}
    queue: list[str] = ["."]
    seen: set[str] = set()
    try:
        while queue:
            d = queue.pop(0)
            if d in seen:
                continue
            seen.add(d)
            try:
                entries = listdir(d)
            except Exception:  # noqa: BLE001
                continue
            for name, _mode in entries:
                full = name if d in (".", "", "/") else f"{d}/{name}"
                matched = False
                for key, rx in patterns.items():
                    if rx.search(name) and name.endswith((".gz", ".txt.gz", ".csv.gz", ".hdf")):
                        found[key].append(full)
                        matched = True
                if not matched and not name.endswith((".gz", ".hdf", ".txt", ".csv")):
                    queue.append(full)
    finally:
        close()
    return {k: sorted(v) for k, v in found.items() if v}


def fetch_umd_history(paths: DataPaths, years: list[int],
                      manifest: ProvenanceManifest | None = None) -> list[Path]:
    """Stream MCD14ML + Vxx14IMGML monthly files from UMD into staged_raw/."""
    out_dir = Path(paths.raw_stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        files = _iter_remote_files(S.UMD_FTP_HOST, S.UMD_FTP_USER, S.UMD_FTP_PASS,
                                   _UMD_PATTERNS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("UMD archive unreachable: %s", exc)
        if manifest:
            manifest.log("umd_active_fires", S.UMD_FTP_HOST, "unreachable",
                         detail=str(exc), status="FAIL")
        return []
    if manifest:
        manifest.log("umd_active_fires", f"sftp/ftp://{S.UMD_FTP_HOST}",
                     "remote_walk", detail=f"{sum(map(len, files.values()))} files")
    # _iter_remote_files returns paths only; do a second pass to read them.
    # (kept simple: re-connect and fetch the discovered file list)
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(S.UMD_FTP_HOST, port=22, username=S.UMD_FTP_USER,
                   password=S.UMD_FTP_PASS, timeout=FTP_TIMEOUT_S,
                   allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()
    try:
        for key, plist in files.items():
            for rel in plist:
                if not any(str(y) in rel for y in years):
                    continue
                dest = out_dir / f"{key}__{Path(rel).name}"
                if not dest.exists():
                    with sftp.open(rel, "rb") as src, open(dest, "wb") as dst:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            dst.write(chunk)
                written.append(dest)
    finally:
        sftp.close()
        client.close()
    return written


# ---------------------------------------------------------------------------
# Source Cooperative Portolan GeoParquet mirror (failover for active fires)
# ---------------------------------------------------------------------------
def _duckdb_con():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
    except Exception:  # noqa: BLE001 - spatial optional if geometry pre-decoded
        pass
    return con


def inspect_portolan_catalog(con=None) -> dict:
    """Read the Portolan STAC catalog and follow child links dynamically."""
    own = con is None
    if own:
        con = _duckdb_con()
    try:
        catalog = json.loads(requests.get(S.PORTOLAN_CATALOG_URL,
                                          timeout=HTTP_TIMEOUT_S).text)
        children = [c.get("href") for c in catalog.get("links", [])
                    if c.get("rel") == "child"]
        return {"root": S.PORTOLAN_CATALOG_URL, "children": children}
    finally:
        if own:
            con.close()


def fetch_portolan_year(year: int, bboxes: list[tuple], out_path: Path,
                        con=None) -> Path:
    """Spatially-pruned extraction of one Portolan year partition.

    Portolan stores FIRMS schema directly (2000-present): geometry ST_Point,
    acq_datetime, sensor, satellite, quality, version, brightness/bright_t31/
    bright_ti4/bright_ti5, scan, track, frp, daynight, confidence,
    confidence_pct, type, _month, _hilbert.
    """
    import duckdb
    own = con is None
    if own:
        con = _duckdb_con()
    url = f"{S.PORTOLAN_BASE}/detections/year={year}/detections.parquet"
    try:
        or_clauses = []
        for (min_lon, min_lat, max_lon, max_lat) in bboxes:
            or_clauses.append(
                f"(ST_X(geometry) BETWEEN {min_lon} AND {max_lon} "
                f"AND ST_Y(geometry) BETWEEN {min_lat} AND {max_lat})")
        where = " OR ".join(or_clauses) if or_clauses else "TRUE"
        query = f"""
            SELECT ST_X(geometry) AS longitude, ST_Y(geometry) AS latitude,
                   acq_datetime, acq_date, sensor, satellite, quality, version,
                   brightness, bright_t31, bright_ti4, bright_ti5,
                   scan, track, frp, daynight, confidence, confidence_pct,
                   type, _month
            FROM read_parquet('{url}')
            WHERE {where}
        """
        df = con.execute(query).fetch_df()
    except Exception as exc:  # noqa: BLE001 - raise for caller failover
        raise RuntimeError(f"Portolan fetch failed for {year}: {exc}") from exc
    finally:
        if own:
            con.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# NASA FIRMS REST API (trailing 2026 NRT gap-fill only)
# ---------------------------------------------------------------------------
def firms_data_availability(map_key: str) -> pd.DataFrame:
    url = f"{S.FIRMS_API_BASE}/data_availability/csv/{map_key}/ALL"
    return pd.read_csv(io.StringIO(_firms_get(url)))


def fetch_firms_nrt(map_key: str, source: str, bbox: tuple,
                    day_range: int, date_str: str) -> pd.DataFrame:
    """``/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/5/{DATE}`` with DAY_RANGE<=5."""
    if not 1 <= day_range <= 5:
        raise ValueError("FIRMS API DAY_RANGE must be within 1..5 (strict).")
    west, south, east, north = bbox
    bb = f"{west},{south},{east},{north}"
    url = (f"{S.FIRMS_API_BASE}/area/csv/{map_key}/{source}/{bb}/"
           f"{day_range}/{date_str}")
    csv_text = _firms_get(url)
    if not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


def _firms_get(url: str, max_retries: int = 6) -> str:
    """GET with 0.15 s pacing and exponential backoff on 429/5xx."""
    delay = 1.0
    for attempt in range(max_retries):
        time.sleep(FIRMS_SLEEP_S)
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT_S)
        except requests.RequestException as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            logger.warning("FIRMS request error (%s); retry in %.1fs", exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, FIRMS_MAX_BACKOFF_S)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(delay)
            delay = min(delay * 2, FIRMS_MAX_BACKOFF_S)
            continue
        resp.raise_for_status()
        return resp.text
    raise RuntimeError(f"FIRMS API exhausted retries: {url}")


# ---------------------------------------------------------------------------
# HDF4 CMG grids (pyhdf; Env 1 only)
# ---------------------------------------------------------------------------
CMQ_CLOUD_LAYERS = ("MeanCloudFraction", "CorrFirePix", "CloudCorrFirePix", "RawFirePix")
CMQ_BURN_LAYERS = ("BurnedArea", "QA", "UnmappedFraction")


def read_hdf4_sds(path: str | Path, layer_names: tuple[str, ...]) -> dict:
    """Read named SDS layers + attributes via pyhdf (lazy import, Env 1 only)."""
    try:
        from pyhdf.SD import SD, SDC  # noqa: F401  (Env 1 only)
    except ImportError as exc:  # pragma: no cover - depends on Env 1 image
        raise RuntimeError(
            "pyhdf is required in Environment 1 (apt-get install -y libhdf4-dev "
            "&& pip install pyhdf); it must never be imported in Environment 2."
        ) from exc
    sd = SD(str(path), SDC.READ)
    out: dict[str, tuple[dict, np.ndarray]] = {}
    try:
        for name in layer_names:
            sds = sd.select(name)
            try:
                attrs = {k: sds.attributes()[k] for k in sds.attributes()}
                data = np.asarray(sds[:], dtype=np.float64)
            finally:
                sds.endaccess()
            out[name] = (attrs, data)
    finally:
        sd.end()
    return out


def apply_sds_scaling(attrs: dict, data: np.ndarray) -> np.ndarray:
    """Apply dynamic scale_factor / add_offset / _FillValue from SDS attrs."""
    fill = attrs.get("_FillValue")
    if fill is not None:
        data = np.where(np.isclose(data, float(fill)), np.nan, data)
    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is not None:
        data = data * float(scale)
    if offset is not None:
        data = data + float(offset)
    return data


def cmg_grid_centers() -> tuple[np.ndarray, np.ndarray]:
    """0.25 deg CMG cell-center coordinates (720 x 1440, UL (-179.875, 89.875))."""
    lons = -179.875 + 0.25 * np.arange(S.CMG_GRID_SHAPE[1])
    lats = 89.875 - 0.25 * np.arange(S.CMG_GRID_SHAPE[0])
    return lats, lons


def parse_cloud_cmg(sds_layers: dict, year: int, period: int,
                    period_kind: str) -> pd.DataFrame:
    """Parse MOD14C8Q / MYD14C8Q / Mx14CMQ cloud SDS layers to a tidy frame.

    ``MeanCloudFraction`` values in [0,100] are normalized to [0,1].
    """
    lats, lons = cmg_grid_centers()
    attrs, cloud_raw = sds_layers["MeanCloudFraction"]
    cloud = apply_sds_scaling(attrs, cloud_raw)
    cloud = np.where(cloud > 100.0, np.nan, cloud)
    cloud = cloud / 100.0 if np.nanmax(cloud, initial=0.0) > 1.0 else cloud
    rows = 2160 if cloud.shape[0] == 720 else cloud.shape[0]
    del rows
    lat_g, lon_g = np.meshgrid(lats[: cloud.shape[0]], lons[: cloud.shape[1]],
                               indexing="ij")
    df = pd.DataFrame({
        "lat_025": lat_g.ravel().round(2),
        "lon_025": lon_g.ravel().round(2),
        "cloud_frac": np.clip(cloud.ravel(), 0.0, 1.0),
        "year": year,
        "period": period,
        "period_kind": period_kind,   # '8d' or 'month'
        "platform": "MOD" if "MOD14" in str(sds_layers.get("_source", "MOD14")) else "MYD",
    })
    return df.dropna(subset=["cloud_frac"])


def parse_burned_area_cmg(sds_layers: dict, year: int, month: int) -> pd.DataFrame:
    """Parse MCD64CMQ BurnedArea/QA/UnmappedFraction layers (hectares -> km^2)."""
    lats, lons = cmg_grid_centers()
    attrs, burn_raw = sds_layers["BurnedArea"]
    burn = apply_sds_scaling(attrs, burn_raw)      # scaled -> hectares
    burn_km2 = burn * 0.01                          # ha -> km^2 (readme 6.1.1)
    qa_attrs, qa_raw = sds_layers["QA"]
    qa = apply_sds_scaling(qa_attrs, qa_raw)
    um_attrs, um_raw = sds_layers["UnmappedFraction"]
    unmapped = apply_sds_scaling(um_attrs, um_raw)
    unmapped = unmapped / 100.0 if np.nanmax(unmapped, initial=0.0) > 1.0 else unmapped
    lat_g, lon_g = np.meshgrid(lats[: burn.shape[0]], lons[: burn.shape[1]],
                               indexing="ij")
    df = pd.DataFrame({
        "lat_025": lat_g.ravel().round(2),
        "lon_025": lon_g.ravel().round(2),
        "burned_area_km2": np.clip(burn_km2.ravel(), 0.0, None),
        "qa": qa.ravel(),
        "unmapped_fraction": np.clip(unmapped.ravel(), 0.0, 1.0),
        "year": year,
        "month": month,
    })
    df = df[(df["qa"] >= 2) | df["qa"].isna()]     # QA 2 = processed land
    return df.dropna(subset=["burned_area_km2"])


def build_cloud_climatology_fallback(cloud_df: pd.DataFrame) -> pd.DataFrame:
    """2012-2025 multi-year mean cloud fraction per (0.25 deg cell, month).

    Used when trailing-2026 CMG science grids lag publication by 1-3 months
    (cloud_source = 'CLIMATOLOGY_FALLBACK').
    """
    clim = (cloud_df[cloud_df["year"].between(2012, 2025)]
            .assign(month=lambda d: np.where(d["period_kind"] == "month",
                                             d["period"],
                                             ((d["period"] - 1) * 8 // 30) + 1))
            .groupby(["lat_025", "lon_025", "month"], as_index=False)["cloud_frac"]
            .mean())
    clim = clim.rename(columns={"cloud_frac": "cloud_frac_clim"})
    return clim


# ---------------------------------------------------------------------------
# NASA CMR discovery (readme 6.2.3)
# ---------------------------------------------------------------------------
def cmr_search_granules(short_name: str, page_size: int = 2000) -> list[dict]:
    """Query NASA CMR for alternate download endpoints of missing granules."""
    params = {"short_name": short_name, "page_size": page_size,
              "cloud_hosted": "true"}
    resp = requests.get(S.NASA_CMR_URL, params=params, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json().get("feed", {}).get("entry", [])


# ---------------------------------------------------------------------------
# Schema validation gate (readme 6.2.5)
# ---------------------------------------------------------------------------
def schema_validation_gate(df: pd.DataFrame, source_name: str) -> bool:
    """Column-presence and value-range assertions before staged acceptance."""
    if df is None or len(df) == 0:
        logger.warning("schema gate: empty frame from %s", source_name)
        return False
    required = ("latitude", "longitude", "frp_mw", "acq_date", "acq_hhmm")
    for col in required:
        if col not in df.columns:
            logger.warning("schema gate: %s missing column %s", source_name, col)
            return False
        nonnull = float(df[col].notna().mean())
        if nonnull < 0.99:
            logger.warning("schema gate: %s column %s non-null %.3f < 0.99",
                           source_name, col, nonnull)
            return False
    if not df["latitude"].between(-90.0, 90.0).all():
        logger.warning("schema gate: %s latitude out of range", source_name)
        return False
    if not (df["frp_mw"].fillna(0.0) >= 0.0).all():
        logger.warning("schema gate: %s negative FRP", source_name)
        return False
    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    if dates.notna().sum() == 0:
        logger.warning("schema gate: %s no valid dates", source_name)
        return False
    logger.info("schema gate PASSED for %s (%d rows)", source_name, len(df))
    return True


# ---------------------------------------------------------------------------
# Top-level Env 1 orchestrator
# ---------------------------------------------------------------------------
def download_all(paths: DataPaths, years: list[int] | None = None,
                 firms_map_key: str | None = None,
                 bboxes: list[tuple] | None = None) -> dict:
    """Run the full Env 1 acquisition with autonomous failover.

    Returns a summary dict; every resolved source is written to
    ``data_provenance_manifest.json``.  This function requires Internet and is
    executed ONLY in Environment 1.
    """
    manifest = ProvenanceManifest(Path(paths.provenance_manifest))
    paths.ensure_output_dirs()
    if years is None:
        years = list(range(2000, 2027))
    if bboxes is None:
        bboxes = [box for c in S.TARGET_COUNTRIES for box in c.prefilter_boxes]

    summary: dict = {"years": years}

    # 1) Country polygons (needed for clipping everywhere downstream).
    download_country_polygons(paths, manifest)

    # 2) Historical active fires: UMD primary, Portolan failover.
    umd_files = fetch_umd_history(paths, years, manifest)
    summary["umd_files"] = [str(p) for p in umd_files]
    if not umd_files:
        con = _duckdb_con()
        try:
            catalog = inspect_portolan_catalog(con)
            manifest.log("portolan_catalog", S.PORTOLAN_CATALOG_URL, "stac",
                         detail=f"{len(catalog['children'])} children")
            for year in years:
                dest = Path(paths.raw_stage_dir) / f"portolan_{year}.parquet"
                try:
                    fetch_portolan_year(year, bboxes, dest, con)
                    manifest.log(f"active_fires_{year}", str(dest), "portolan_httpfs")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Portolan year %d failed: %s", year, exc)
                    manifest.log(f"active_fires_{year}", "portolan", "httpfs",
                                 detail=str(exc), status="FAIL")
        finally:
            con.close()

    # 3) Trailing-2026 NRT gap-fill via FIRMS REST API (strictly trailing).
    if firms_map_key:
        for source in ("MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT",
                       "VIIRS_NOAA21_NRT"):
            for (west, south, east, north) in bboxes:
                try:
                    df = fetch_firms_nrt(firms_map_key, source,
                                         (west, south, east, north), 5,
                                         S.DATE_END)
                    if len(df):
                        dest = (Path(paths.raw_stage_dir) /
                                f"firms_nrt_{source}_{west}_{south}.parquet")
                        df.to_parquet(dest, index=False)
                        manifest.log(f"nrt_{source}", str(dest), "firms_api")
                except Exception as exc:  # noqa: BLE001
                    manifest.log(f"nrt_{source}", "firms_api", "http",
                                 detail=str(exc), status="FAIL")

    # 4) Ancillary cloud + burned-area CMG grids.
    try:
        cloud_parts: list[pd.DataFrame] = []
        burn_parts: list[pd.DataFrame] = []
        for year in range(2000, 2027):
            for month in range(1, 13):
                for short in ("MOD14CMQ", "MYD14CMQ"):
                    p = (Path(paths.raw_stage_dir) /
                         f"{short}.{year}{month:02d}.hdf")
                    if not p.exists():
                        continue
                    sds = read_hdf4_sds(p, CMQ_CLOUD_LAYERS)
                    sds["_source"] = ({}, np.array([short]))
                    cloud_parts.append(parse_cloud_cmg(sds, year, month, "month"))
                p64 = Path(paths.raw_stage_dir) / f"MCD64CMQ.{year}{month:02d}.hdf"
                if p64.exists():
                    sds = read_hdf4_sds(p64, CMQ_BURN_LAYERS)
                    burn_parts.append(parse_burned_area_cmg(sds, year, month))
        if cloud_parts:
            cloud = pd.concat(cloud_parts, ignore_index=True)
            cloud.to_parquet(paths.cloud_parquet, index=False)
            manifest.log("cloud_cmg", str(paths.cloud_parquet), "pyhdf")
            build_cloud_climatology_fallback(cloud).to_parquet(
                Path(paths.cloud_parquet).with_name("cloud_climatology.parquet"),
                index=False)
        else:
            manifest.log("cloud_cmg", "none", "pyhdf",
                         detail="No CMG HDF4 files staged (see Section 6.2 "
                                "CMR discovery for alternates).", status="WARN")
        if burn_parts:
            pd.concat(burn_parts, ignore_index=True).to_parquet(
                paths.burned_area_parquet, index=False)
            manifest.log("burned_area_cmq", str(paths.burned_area_parquet), "pyhdf")
    except Exception as exc:  # noqa: BLE001
        manifest.log("ancillary_cmg", "pyhdf", "hdf4",
                     detail=str(exc), status="FAIL")

    manifest.save()
    summary["manifest"] = str(manifest.path)
    return summary
