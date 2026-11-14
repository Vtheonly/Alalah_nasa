"""Environment 1 data & ancillary grid downloader (internet ON, CPU only).

Contract (readme.md Sections 4, 6.1, 6.2):
  * Primary Science archives from ``fuoco.geog.umd.edu`` (SFTP ``fire`` /
    ``burnt``, FTP fallback): ``MCD14ML`` (space-delimited .txt.gz),
    ``VNP14IMGML`` / ``VJ114IMGML`` / ``VJ214IMGML`` (.csv.gz),
    ``MOD14C8Q``/``MYD14C8Q``/``Mx14CMQ`` HDF4 cloud grids and ``MCD64CMQ``
    burned-area HDF4 grids (pyhdf, strictly Environment 1 only).
  * Source Cooperative Portolan GeoParquet 2.0 mirror via DuckDB ``httpfs``
    when UMD is unreachable.
  * NASA FIRMS REST API strictly for trailing-2026 NRT backfill
    (``DAY_RANGE <= 5``, 5,000 tx / 10 min budget discipline).
  * Natural Earth 50m Admin-0 GeoJSON (GeoBoundaries fallback).
  * Autonomous Data Source Discovery & Failover Protocol with a
    ``data_provenance_manifest.json`` log; never fabricate synthetic data.

This module performs network I/O only when its public functions are invoked
(explicitly from ``scripts/run_env1_stage.py``); importing it is side-effect
free so unit tests can exercise parsers against fixtures.
"""

from __future__ import annotations

import ftplib
import gzip
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.config import paths as paths_mod
from src.config import settings as S

logger = logging.getLogger("fire_harmonizer.ingest.download_data")

# Regex patterns used by the autonomous remote-directory discovery protocol.
DISCOVERY_PATTERNS = (
    re.compile(r"MCD14ML", re.IGNORECASE),
    re.compile(r"V(NP|J1|J2)14IMGML", re.IGNORECASE),
    re.compile(r"M[OY]D14C(8|M)Q", re.IGNORECASE),
    re.compile(r"MCD64CMQ", re.IGNORECASE),
)


# --------------------------------------------------------------------------- #
# Provenance manifest
# --------------------------------------------------------------------------- #
class ProvenanceManifest:
    """Structured log of every resolved data source (readme.md 6.2)."""

    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self.entries: List[Dict[str, object]] = []

    def log(self, dataset: str, source: str, status: str, detail: str = "",
            n_rows: int = 0) -> None:
        entry = {
            "dataset": dataset,
            "source": source,
            "status": status,
            "detail": detail,
            "n_rows": n_rows,
            "logged_at": datetime.utcnow().isoformat() + "Z",
        }
        self.entries.append(entry)
        if status == "OK":
            logger.info("provenance OK %s <- %s (%s rows)", dataset, source, n_rows)
        elif status == "CRITICAL":
            logger.critical("provenance CRITICAL %s: %s", dataset, detail)
        else:
            logger.warning("provenance %s %s: %s", status, dataset, detail)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries}, fh, indent=2)


# --------------------------------------------------------------------------- #
# Schema validation gate (readme.md 6.2 item 5)
# --------------------------------------------------------------------------- #
def validate_detection_schema(df: pd.DataFrame, source: str) -> bool:
    """Column-presence and value-range assertions before staging acceptance."""
    required = ("latitude", "longitude", "acq_date", "frp_mw")
    for col in required:
        if col not in df.columns:
            logger.error("schema gate FAIL (%s): missing column %s", source, col)
            return False
    ok_rows = int(
        df[["latitude", "longitude", "acq_date"]].notna().all(axis=1).sum()
    )
    nonnull_rate = ok_rows / max(len(df), 1)
    lat_ok = bool(df["latitude"].between(-90.0, 90.0).all())
    frp_ok = bool((df["frp_mw"].fillna(0.0) >= 0.0).all())
    if nonnull_rate < 0.99 or not lat_ok or not frp_ok:
        logger.error(
            "schema gate FAIL (%s): nonnull=%.4f lat_ok=%s frp_ok=%s",
            source, nonnull_rate, lat_ok, frp_ok,
        )
        return False
    logger.info("schema gate OK (%s): %d rows", source, len(df))
    return True


# --------------------------------------------------------------------------- #
# UMD fuoco SFTP / FTP access with autonomous directory discovery
# --------------------------------------------------------------------------- #
_UMD_DIRS_PROBED: List[str] = []


def _umd_password() -> str:
    """Password for the UMD fire archive (default 'burnt', env override)."""
    return os.environ.get(S.UMD_FUOCO_PASSWORD_ENV, "burnt")


def _sftp_connect(host: str, user: str, password: str, timeout: float = 30.0):
    import paramiko  # Environment 1 only; lazy import keeps Env 2 clean

    transport = paramiko.Transport((host, 22))
    transport.connect(username=user, password=password)
    transport.banner_timeout = timeout
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport


def _sftp_walk_matches(sftp, base: str = "/", depth: int = 3) -> List[str]:
    """Recursively walk the SFTP tree looking for the discovery patterns."""
    matches: List[str] = []
    _UMD_DIRS_PROBED.append(base)

    def _walk(dirpath: str, level: int) -> None:
        if level > depth:
            return
        try:
            entries = sftp.listdir_attr(dirpath)
        except IOError:
            return
        for entry in entries:
            name = entry.filename
            full = f"{dirpath.rstrip('/')}/{name}"
            if any(p.search(name) for p in DISCOVERY_PATTERNS):
                matches.append(full)
            try:
                import stat as stat_mod

                if stat_mod.S_ISDIR(entry.st_mode):
                    _walk(full, level + 1)
            except (OSError, AttributeError):
                continue

    _walk(base, 0)
    return matches


def _ftp_walk_matches(ftp: ftplib.FTP, base: str = "/", depth: int = 3) -> List[str]:
    """Recursive FTP (port 21) fallback directory traversal."""
    matches: List[str] = []

    def _walk(dirpath: str, level: int) -> None:
        if level > depth:
            return
        try:
            names = ftp.nlst(dirpath)
        except ftplib.all_errors:
            return
        for name in names:
            if any(p.search(Path(name).name) for p in DISCOVERY_PATTERNS):
                matches.append(name)
            if "." not in Path(name).name and level < depth:
                _walk(name, level + 1)

    _walk(base, 0)
    return matches


def open_umd_archive(manifest: ProvenanceManifest) -> Tuple[str, object]:
    """Connect to UMD fuoco trying SFTP first, FTP fallback.

    Returns ``(mode, handle)`` where mode is ``"sftp"`` or ``"ftp"``; raises
    ``ConnectionError`` when both transports fail so the caller can execute
    the Portolan mirror failover.
    """
    host, user, password = S.UMD_FUOCO_HOST, S.UMD_FUOCO_USER, _umd_password()
    try:
        sftp, transport = _sftp_connect(host, user, password)
        manifest.log("umd_connection", f"sftp://{host}:22", "OK")
        return "sftp", (sftp, transport)
    except Exception as exc:  # noqa: BLE001 - fall back to FTP explicitly
        manifest.log(
            "umd_connection", f"sftp://{host}:22", "FALLBACK", str(exc)[:400]
        )
    try:
        ftp = ftplib.FTP(host, timeout=30)
        ftp.login(user, password)
        manifest.log("umd_connection", f"ftp://{host}:21", "OK")
        return "ftp", ftp
    except Exception as exc:  # noqa: BLE001
        manifest.log("umd_connection", host, "UNREACHABLE", str(exc)[:400])
        raise ConnectionError(f"UMD fuoco unreachable via SFTP and FTP: {exc}")


def discover_umd_files(handle, mode: str, dataset_regex: re.Pattern) -> List[str]:
    """Autonomous discovery of remote paths matching ``dataset_regex``."""
    if mode == "sftp":
        sftp, _transport = handle
        found = _sftp_walk_matches(sftp, base="/")
    else:
        found = _ftp_walk_matches(handle, base="/")
    return sorted(p for p in found if dataset_regex.search(Path(p).name))


def stream_remote_file(handle, mode: str, remote_path: str) -> bytes:
    """Read one remote file fully into memory (monthly chunks are small)."""
    if mode == "sftp":
        sftp, _transport = handle
        with sftp.open(remote_path, "rb") as fh:
            return fh.read()
    buf = io.BytesIO()
    handle.retrbinary(f"RETR {remote_path}", buf.write)
    return buf.getvalue()


def close_umd_archive(handle, mode: str) -> None:
    if mode == "sftp":
        sftp, transport = handle
        sftp.close()
        transport.close()
    else:
        handle.quit()


# --------------------------------------------------------------------------- #
# HDF4 CMG grid extraction (pyhdf; Environment 1 ONLY)
# --------------------------------------------------------------------------- #
def read_hdf4_cmg_layer(hdf_path: Path, sds_name: str) -> np.ndarray:
    """Read one SDS layer from a 720x1440 CMG HDF4 file with dynamic scaling.

    Applies ``scale_factor`` / ``add_offset`` and masks ``_FillValue`` /
    negative fill entries dynamically from the SDS attributes (readme.md
    Section 6.1 item 1), returning a float array with NaN where unmapped.
    """
    from pyhdf.SD import SD, SDC  # Environment 1 only

    ds = SD(str(hdf_path), SDC.READ)
    try:
        sds = ds.select(sds_name)
        attrs = sds.attributes()
        data = np.asarray(sds.get(), dtype=np.float64)
        fill = attrs.get("_FillValue")
        if fill is not None:
            data = np.where(data == float(fill), np.nan, data)
        scale = attrs.get("scale_factor")
        offset = attrs.get("add_offset")
        if scale is not None:
            data = data * float(scale)
        if offset is not None:
            data = data + float(offset)
        # Negative values other than NaN are fill/unused -> mask.
        data = np.where(data < 0, np.nan, data)
        if data.shape != (S.CMG_GRID_ROWS, S.CMG_GRID_COLS):
            raise ValueError(
                f"unexpected CMG grid shape {data.shape} for {sds_name}"
            )
        return data
    finally:
        ds.end()


def cmg_cell_centers() -> Tuple[np.ndarray, np.ndarray]:
    """Lon/lat cell-center coordinate arrays for the 0.25 deg CMG grid."""
    lons = S.CMG_LON0 + np.arange(S.CMG_GRID_COLS) * S.CMG_CELL_DEG
    lats = S.CMG_LAT0 - np.arange(S.CMG_GRID_ROWS) * S.CMG_CELL_DEG
    return lons, lats


def extract_cloud_grid(
    hdf_path: Path, source_name: str, period_key: Tuple[int, int]
) -> pd.DataFrame:
    """Extract MeanCloudFraction + unmapped fraction for one CMG granule.

    Returns long-format rows ``(lon_025, lat_025, year, period, cloud_frac,
    unmapped_frac, cloud_source)``. ``period_key`` is ``(year, period_index)``
    where period_index enumerates 8-day or monthly compositing windows.
    """
    cloud = read_hdf4_cmg_layer(hdf_path, "MeanCloudFraction")
    # Unmapped fraction is derived from the QA layer when present; MOD14C8Q /
    # Mx14CMQ expose the needed denominator via CorrFirePix-style counts, so
    # the unmapped fraction uses the MCD64CMQ UnmappedFraction convention
    # where available and falls back to 0 for cloud grids.
    try:
        unmapped = read_hdf4_cmg_layer(hdf_path, "UnmappedFraction")
    except Exception:  # noqa: BLE001 - layer absent on fire CMG products
        unmapped = np.zeros_like(cloud)
    lons, lats = cmg_cell_centers()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    valid = ~np.isnan(cloud)
    # Values arrive in [0, 100] percentages; normalize to [0, 1].
    cloud_frac = np.clip(cloud[valid] / 100.0, 0.0, 1.0)
    unmap_frac = np.clip(np.nan_to_num(unmapped[valid]) / 100.0, 0.0, 1.0)
    year, period = period_key
    return pd.DataFrame(
        {
            "lon_025": lon_grid[valid],
            "lat_025": lat_grid[valid],
            "year": year,
            "period": period,
            "cloud_frac": cloud_frac,
            "unmapped_frac": unmap_frac,
            "cloud_source": source_name,
        }
    )


def extract_burned_area_grid(
    hdf_path: Path, period_key: Tuple[int, int]
) -> pd.DataFrame:
    """Extract MCD64CMQ BurnedArea (hectares -> km2) per 0.25 deg cell."""
    burned_ha = read_hdf4_cmg_layer(hdf_path, "BurnedArea")
    try:
        qa = read_hdf4_cmg_layer(hdf_path, "QA")
    except Exception:  # noqa: BLE001
        qa = np.full_like(burned_ha, 2.0)
    try:
        unmapped = read_hdf4_cmg_layer(hdf_path, "UnmappedFraction")
    except Exception:  # noqa: BLE001
        unmapped = np.zeros_like(burned_ha)
    lons, lats = cmg_cell_centers()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    keep = ~np.isnan(burned_ha)
    # QA == 2 marks processed land pixels (readme.md 6.1 item 1).
    keep &= np.nan_to_num(qa) == 2.0
    year, month = period_key
    return pd.DataFrame(
        {
            "lon_025": lon_grid[keep],
            "lat_025": lat_grid[keep],
            "year": year,
            "month": month,
            # scale_factor 0.01 converts raw counts to hectares; 1 ha = 0.01 km2.
            "burned_area_km2": np.clip(burned_ha[keep] * 0.01, 0.0, None) * 0.01,
            "unmapped_frac": np.clip(np.nan_to_num(unmapped[keep]) / 100.0, 0.0, 1.0),
        }
    )


# --------------------------------------------------------------------------- #
# NASA FIRMS REST API (trailing-2026 NRT backfill only)
# --------------------------------------------------------------------------- #
FIRMS_SOURCE_ALIASES = {
    ("MODIS", "Terra"): "MODIS_T",
    ("MODIS", "Aqua"): "MODIS_A",
    ("MODIS", None): "MODIS",
    ("VIIRS", "SNPP"): "VIIRS_SNPP",
    ("VIIRS", "NOAA-20"): "VIIRS_NOAA20",
    ("VIIRS", "NOAA-21"): "VIIRS_NOAA21",
}


def firms_area_csv(
    map_key: str,
    source: str,
    bbox: Tuple[float, float, float, float],
    day_range: int,
    on_date: str,
    session: Optional[requests.Session] = None,
) -> str:
    """One FIRMS /api/area/csv call with DAY_RANGE <= 5 and backoff discipline."""
    if day_range > S.FIRMS_MAX_DAY_RANGE:
        raise ValueError("FIRMS DAY_RANGE must be <= 5")
    west, south, east, north = bbox
    url = (
        f"{S.FIRMS_API_BASE}/api/area/csv/{map_key}/{source}/"
        f"{west},{south},{east},{north}/{day_range}/{on_date}"
    )
    sess = session or requests.Session()
    for attempt in range(6):
        time.sleep(S.FIRMS_RATE_LIMIT_SLEEP_S)
        resp = sess.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 429:
            wait = 2.0 ** attempt
            logger.warning("FIRMS 429; backing off %.1fs", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"FIRMS rate limit exhausted for {url}")


def firms_data_availability(map_key: str) -> pd.DataFrame:
    """Query /api/data_availability/csv/{MAP_KEY}/ALL for NRT coverage dates."""
    url = f"{S.FIRMS_API_BASE}/api/data_availability/csv/{map_key}/ALL"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


# --------------------------------------------------------------------------- #
# Portolan GeoParquet mirror via DuckDB httpfs
# --------------------------------------------------------------------------- #
def portolan_year_detections(
    year: int,
    bbox: Tuple[float, float, float, float],
    country_iso: str,
    out_parquet: Path,
    manifest: ProvenanceManifest,
) -> int:
    """Spatially-pruned Portolan extraction for one year into staged_raw/.

    Uses DuckDB ``httpfs`` (+ ``spatial``) with ST_X/ST_Y bounding-box
    pruning per the verified Portolan schema (readme.md 6.1 item 2).
    """
    import duckdb

    west, south, east, north = bbox
    url = S.PORTOLAN_DETECTION_URL.format(year=year)
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        try:
            con.execute("INSTALL spatial; LOAD spatial;")
            geom_expr = "ST_X(geometry) AS longitude, ST_Y(geometry) AS latitude"
        except Exception:  # noqa: BLE001 - spatial ext optional
            geom_expr = (
                "struct_extract(geometry, 'x') AS longitude, "
                "struct_extract(geometry, 'y') AS latitude"
            )
        query = f"""
            SELECT {geom_expr},
                   acq_datetime, acq_date, sensor, satellite, quality, version,
                   brightness, bright_t31, bright_ti4, bright_ti5,
                   scan, track, frp, daynight, confidence, confidence_pct,
                   type, _month
            FROM read_parquet('{url}')
            WHERE longitude BETWEEN {west} AND {east}
              AND latitude BETWEEN {south} AND {north}
        """
        df = con.execute(query).fetch_df()
    except Exception as exc:  # noqa: BLE001
        manifest.log("portolan", url, "UNREACHABLE", str(exc)[:400])
        return 0
    finally:
        con.close()
    df["country_prefilter"] = country_iso
    if not validate_detection_schema(
        df.rename(columns={"frp": "frp_mw"}), f"portolan/{year}"
    ):
        manifest.log("portolan", url, "SCHEMA_FAIL")
        return 0
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    manifest.log("portolan", url, "OK", n_rows=len(df))
    return len(df)


# --------------------------------------------------------------------------- #
# Country polygons (Natural Earth with GeoBoundaries fallback)
# --------------------------------------------------------------------------- #
TARGET_NATURAL_EARTH_A3 = {"USA": "United States of America"}


def download_country_polygons(out_parquet: Path) -> int:
    """Fetch Natural Earth 50m Admin-0 GeoJSON; GeoBoundaries failover.

    Stores one row per target country with its polygon geometry serialized as
    WKB plus the ISO_A3 label. Returns the number of country rows written.
    """
    from shapely.geometry import shape
    from shapely import wkb as shapely_wkb
    from shapely.geometry import Polygon, MultiPolygon

    resp = requests.get(S.NATURAL_EARTH_GEOJSON_URL, timeout=120)
    rows: List[Dict[str, object]] = []
    source = "natural-earth-50m"
    if resp.status_code == 200:
        gj = resp.json()
        features = gj.get("features", [])
    else:
        # GeoBoundaries failover (readme.md 6.2 item 4).
        source = "geoboundaries-gbOpen-ADM0"
        features = []
        for iso3 in S.TARGET_COUNTRY_ISOS:
            meta = requests.get(
                S.GEOBOUNDARIES_API_URL.format(iso3=iso3), timeout=60
            ).json()
            gj_url = meta["gjDownloadURL"]
            geo = requests.get(gj_url, timeout=120).json()
            for feat in geo.get("features", []):
                feat["properties"] = {"ISO_A3": iso3, "ADMIN": iso3}
                features.append(feat)
    for feat in features:
        props = feat.get("properties", {})
        iso3 = props.get("ISO_A3") or props.get("ADM0_A3")
        if iso3 not in S.TARGET_COUNTRY_ISOS:
            continue
        geom = shape(feat["geometry"])
        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        rows.append(
            {
                "country_iso": iso3,
                "admin_name": props.get("ADMIN", iso3),
                "geometry_wkb": shapely_wkb.dumps(geom),
                "source": source,
            }
        )
    if len(rows) < len(S.TARGET_COUNTRY_ISOS):
        raise RuntimeError(
            f"country polygon fetch incomplete: {len(rows)}/{len(S.TARGET_COUNTRY_ISOS)}"
        )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    return len(rows)


# --------------------------------------------------------------------------- #
# Cloud climatology fallback for trailing-2026 CMG lag (readme.md audit #2)
# --------------------------------------------------------------------------- #
def build_cloud_climatology_fallback(
    cloud_df: pd.DataFrame,
) -> pd.DataFrame:
    """2012-2025 multi-year mean cloud fraction per (0.25 deg cell, month)."""
    cell = cloud_df.groupby(["lon_025", "lat_025", "month"], as_index=False)[
        "cloud_frac"
    ].mean()
    cell = cell.rename(columns={"cloud_frac": "cloud_frac_clim"})
    cell["cloud_source"] = S.CLOUD_CLIMATOLOGY_FLAG
    return cell


# --------------------------------------------------------------------------- #
# Orchestration entry point (Environment 1)
# --------------------------------------------------------------------------- #
def fetch_umd_monthly_detections(
    manifest: ProvenanceManifest,
    staged_raw_dir: Path,
    year: int,
    month: int,
) -> int:
    """Stream one month of UMD Science detections into staged_raw/.

    Follows the discovery protocol: MCD14ML space-delimited .txt.gz and
    Vxx14IMGML .csv.gz files are located by regex walk and written as
    normalized raw parquet chunks for downstream normalize_dedup.py.
    """
    total = 0
    try:
        mode, handle = open_umd_archive(manifest)
    except ConnectionError:
        return 0
    try:
        for regex, tag in (
            (re.compile(r"MCD14ML", re.I), "MCD14ML"),
            (re.compile(r"V(NP|J1|J2)14IMGML", re.I), "Vxx14IMGML"),
        ):
            candidates = discover_umd_files(handle, mode, regex)
            ym = f"{year}{month:02d}"
            for path in candidates:
                if ym not in Path(path).name:
                    continue
                raw = stream_remote_file(handle, mode, path)
                out = staged_raw_dir / f"{Path(path).stem}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                _decompress_to_parquet(raw, Path(path).name, out)
                total += pq.ParquetFile(out).metadata.num_rows
                manifest.log(tag, f"umd:{path}", "OK")
    finally:
        close_umd_archive(handle, mode)
    return total


def _decompress_to_parquet(raw: bytes, filename: str, out: Path) -> None:
    """Gunzip a raw UMD chunk and store it verbatim as parquet lines."""
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    df = pd.DataFrame({"line": text.splitlines(), "source_file": filename})
    df.to_parquet(out, index=False)


def fetch_nrt_backfill_2026(
    manifest: ProvenanceManifest,
    staged_raw_dir: Path,
    latest_science_date: date,
    end_date: date,
    country_boxes: Dict[str, Tuple[Tuple[float, float, float, float], ...]],
    map_key: Optional[str] = None,
) -> int:
    """Trailing-2026 FIRMS NRT gap-fill strictly with DAY_RANGE <= 5."""
    map_key = map_key or os.environ.get(S.FIRMS_MAP_KEY_ENV, "")
    if not map_key:
        manifest.log(
            "firms_nrt", "api", "SKIPPED",
            "no FIRMS_MAP_KEY in environment; trailing NRT window not filled",
        )
        return 0
    total = 0
    sess = requests.Session()
    cursor = latest_science_date
    while cursor <= end_date:
        for iso, boxes in country_boxes.items():
            for bbox in boxes:
                for source in (
                    "MODIS_T", "MODIS_A",
                    "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21",
                ):
                    text = firms_area_csv(
                        map_key, source, bbox,
                        min(S.FIRMS_MAX_DAY_RANGE, (end_date - cursor).days + 1),
                        cursor.isoformat(), sess,
                    )
                    if text.strip():
                        df = pd.read_csv(io.StringIO(text))
                        df["source_file"] = f"firms_{source}_{cursor.isoformat()}"
                        df["country_prefilter"] = iso
                        out = (
                            staged_raw_dir
                            / f"firms_{source}_{iso}_{cursor.isoformat()}.parquet"
                        )
                        df.to_parquet(out, index=False)
                        total += len(df)
                cursor = cursor + timedelta(days=S.FIRMS_MAX_DAY_RANGE)
        break
    if total:
        manifest.log("firms_nrt", "api/area/csv", "OK", n_rows=total)
    return total


def run_env1_download(
    data_paths: Optional[paths_mod.DataPaths] = None,
    start_year: int = 2000,
    end_year: int = 2026,
    include_umd: bool = True,
    include_nrt: bool = True,
) -> Dict[str, object]:
    """Full Environment-1 acquisition run (explicit orchestration entry).

    Returns a summary dict; raises only when every source for a mandatory
    dataset fails (never silently fabricating data, readme.md 6.2 item 3).
    """
    dp = data_paths or paths_mod.DataPaths.from_env()
    dp.ensure_dirs()
    manifest = ProvenanceManifest(dp.provenance_manifest_json)
    summary: Dict[str, object] = {"staged_rows": 0, "countries": 0}

    # 1) Country polygons (mandatory; both primary and fallback paths).
    summary["countries"] = download_country_polygons(dp.country_polygons_parquet)
    manifest.log(
        "country_polygons", "natural-earth/geoboundaries", "OK",
        n_rows=summary["countries"],
    )

    # 2) Science detections: UMD primary, Portolan mirror failover.
    umd_rows = 0
    if include_umd:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                umd_rows += fetch_umd_monthly_detections(
                    manifest, dp.raw_stage_dir, year, month
                )
    summary["staged_rows"] = umd_rows
    if umd_rows == 0:
        logger.warning("UMD unreachable -> Portolan GeoParquet mirror failover")
        for year in range(start_year, end_year + 1):
            for iso, spec in S.TARGET_COUNTRIES.items():
                for bbox in spec.prefilter_boxes:
                    summary["staged_rows"] += portolan_year_detections(
                        year, bbox, iso,
                        dp.raw_stage_dir / f"portolan_{iso}_{year}.parquet",
                        manifest,
                    )
        if summary["staged_rows"] == 0:
            raise RuntimeError(
                "All primary detection sources failed; aborting rather than "
                "fabricating data (readme.md 6.2)."
            )

    # 3) Trailing-2026 NRT backfill via FIRMS REST API.
    if include_nrt:
        latest = date(end_year, 1, 1)
        boxes = {iso: spec.prefilter_boxes for iso, spec in S.TARGET_COUNTRIES.items()}
        fetch_nrt_backfill_2026(
            manifest, dp.raw_stage_dir, latest, date(2026, 11, 14), boxes
        )

    # NOTE: cloud / burned-area HDF4 CMG granules are fetched by the SFTP/FTP
    # discovery walk inside scripts/run_env1_stage.py, which routes each file
    # through extract_cloud_grid / extract_burned_area_grid and enforces the
    # climatology fallback for trailing 2026 (readme.md audit item 2).
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry (``python -m src.ingest.download_data``)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Environment 1 NASA/UMD fire-archive downloader"
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--skip-umd", action="store_true", help="force Portolan mirror failover"
    )
    parser.add_argument(
        "--skip-nrt", action="store_true", help="skip FIRMS trailing backfill"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    summary = run_env1_download(
        start_year=args.start_year,
        end_year=args.end_year,
        include_umd=not args.skip_umd,
        include_nrt=not args.skip_nrt,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
