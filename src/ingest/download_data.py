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
class UMDSession:
    """One authenticated UMD archive session: SFTP (port 22) first, FTP
    (port 21) fallback -- the SAME transport is reused for the directory
    walk and every download (readme 6.2.1)."""

    def __init__(self, host: str = S.UMD_FTP_HOST, user: str = S.UMD_FTP_USER,
                 password: str = S.UMD_FTP_PASS):
        self.method = None
        self._client = self._sftp = self._ftp = None
        try:
            import paramiko  # Env 1 dependency
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, port=22, username=user, password=password,
                           timeout=FTP_TIMEOUT_S, allow_agent=False,
                           look_for_keys=False)
            self._client, self._sftp = client, client.open_sftp()
            self.method = "sftp"
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMD SFTP failed (%s); trying FTP port 21.", exc)
            import ftplib
            ftp = ftplib.FTP()
            ftp.connect(host, port=21, timeout=FTP_TIMEOUT_S)
            ftp.login(user, password)
            self._ftp = ftp
            self.method = "ftp"

    def listdir(self, path: str):
        if self.method == "sftp":
            import stat
            return [(a.filename, stat.S_ISDIR(a.st_mode))
                    for a in self._sftp.listdir_attr(path)]
        entries = []
        try:
            lines: list[str] = []
            self._ftp.retrlines(f"LIST {path}", lines.append)
            for ln in lines:
                parts = ln.split(None, 8)
                if len(parts) == 9:
                    entries.append((parts[8], ln.startswith("d")))
        except Exception:  # noqa: BLE001 - fall back to NLST
            for name in self._ftp.nlst(path):
                base = name.rstrip("/").split("/")[-1]
                entries.append((base, "." not in base))
        return entries

    def download(self, remote: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.method == "sftp":
            self._sftp.get(remote, str(dest))
        else:
            with open(dest, "wb") as fh:
                self._ftp.retrbinary(f"RETR {remote}", fh.write)
        return dest

    def close(self):
        try:
            if self._sftp is not None:
                self._sftp.close()
                self._client.close()
            if self._ftp is not None:
                self._ftp.quit()
        except Exception:  # noqa: BLE001
            pass


def walk_umd_tree(session, patterns: dict[str, re.Pattern] = _UMD_PATTERNS,
                  max_dirs: int = 5000) -> dict[str, list[str]]:
    """Breadth-first walk of the archive matching the readme regexes
    (``MCD14ML``, ``V(NP|J1|J2)14IMGML``, ``M[OY]D14C(8|M)Q``, ``MCD64CMQ``)."""
    found: dict[str, list[str]] = {k: [] for k in patterns}
    queue, seen = ["."], set()
    while queue and len(seen) < max_dirs:
        d = queue.pop(0)
        if d in seen:
            continue
        seen.add(d)
        try:
            entries = session.listdir(d)
        except Exception:  # noqa: BLE001
            continue
        for name, is_dir in entries:
            full = name if d in (".", "", "/") else f"{d}/{name}"
            if is_dir:
                queue.append(full)
                continue
            if not name.endswith((".gz", ".hdf")):
                continue
            for key, rx in patterns.items():
                if rx.search(name):
                    found[key].append(full)
                    break
    return {k: sorted(v) for k, v in found.items() if v}


_A_DATE = re.compile(r"\.A(\d{4})(\d{3})\.")
_YM = re.compile(r"[._](\d{4})[._]?(\d{2})[._]")


def parse_cmg_filename(name: str) -> dict | None:
    """Product, year and period of a CMG HDF4 file name.

    * ``MOD14C8Q`` / ``MYD14C8Q`` (8-day): ``.AYYYYDDD.`` -> period
      ``(DDD-1)//8 + 1`` (1..46), ``period_kind = '8d'``.
    * ``MOD14CMQ`` / ``MYD14CMQ`` (monthly): ``.AYYYYDDD.`` or ``.YYYYMM.`` ->
      month, ``period_kind = 'month'``.
    * ``MCD64CMQ`` (monthly burned area): month from the same patterns.
    """
    base = Path(name).name
    m = re.search(r"(MOD14C8Q|MYD14C8Q|MOD14CMQ|MYD14CMQ|MCD64CMQ)", base,
                  re.IGNORECASE)
    if not m:
        return None
    product = m.group(1).upper()
    year = period = None
    a = _A_DATE.search(base)
    if a:
        year, doy = int(a.group(1)), int(a.group(2))
        if product.endswith("C8Q"):
            period = (doy - 1) // 8 + 1
        else:
            period = int((pd.Timestamp(year, 1, 1)
                          + pd.Timedelta(days=doy - 1)).month)
    else:
        ym = _YM.search(base)
        if ym:
            year, period = int(ym.group(1)), int(ym.group(2))
    if year is None or period is None:
        return None
    kind = "8d" if product.endswith("C8Q") else "month"
    return {"product": product, "year": year, "period": period,
            "period_kind": kind}


def _year_of_name(name: str) -> int | None:
    a = _A_DATE.search(name)
    if a:
        return int(a.group(1))
    m = re.search(r"(19|20)\d{2}", Path(name).name)
    return int(m.group(0)) if m else None


def _in_boxes(lat, lon, bboxes) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    keep = np.zeros(len(lat), dtype=bool)
    for (w, so, e, n) in bboxes:
        keep |= (lon >= w) & (lon <= e) & (lat >= so) & (lat <= n)
    return keep


def fetch_umd_history(paths: DataPaths, years: list[int],
                      manifest: ProvenanceManifest | None = None,
                      bboxes: list[tuple] | None = None,
                      session=None, originals_dir: Path | None = None,
                      keep_originals: str = "all") -> dict:
    """Stream UMD files ONE AT A TIME through a temp file (deleted
    immediately) and stage compact outputs:

    * active fires -> canonical ``staged_raw/umd_<PRODUCT>_<file>.parquet``
      (bbox-prefiltered; exact polygon clipping happens in normalization);
    * cloud CMG (8-day + monthly, Terra + Aqua) and MCD64CMQ -> bbox-pruned
      tidy frames returned to the caller (written once as Parquet).

    Every original file's size + SHA-256 is logged in the provenance
    manifest.  With ``originals_dir`` the untouched original is MOVED there
    instead of being deleted (and an existing original is reused instead of
    re-downloaded -- resumable acquisition).  ``keep_originals``:
    ``"all"`` or ``"ancillary"`` (only the small CMG cloud / MCD64CMQ HDF4
    grids; the ~100 GB of global active-fire originals are then deleted
    after parsing, their SHA-256 still logged).
    """
    if keep_originals not in ("all", "ancillary"):
        raise ValueError("keep_originals must be 'all' or 'ancillary'")
    import tempfile

    from .normalize_dedup import (parse_mcd14ml_rows, parse_vxx14imgml_rows,
                                  platform_from_vxx_filename)
    bboxes = bboxes or [b for c in S.TARGET_COUNTRIES for b in c.prefilter_boxes]
    out_dir = Path(paths.raw_stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"fires": [], "cloud": [], "burned": [], "method": None}
    own = session is None
    try:
        session = session or UMDSession()
    except Exception as exc:  # noqa: BLE001
        logger.warning("UMD archive unreachable: %s", exc)
        if manifest:
            manifest.log("umd_archive", S.UMD_FTP_HOST, "unreachable",
                         detail=str(exc), status="FAIL")
        return result
    result["method"] = session.method
    try:
        files = walk_umd_tree(session)
        if manifest:
            manifest.log("umd_archive", f"{session.method}://{S.UMD_FTP_HOST}",
                         "remote_walk",
                         detail=json.dumps({k: len(v) for k, v in files.items()}))
        years_set = set(int(y) for y in years)
        with tempfile.TemporaryDirectory(prefix="umd_") as tmp:
            for key, plist in files.items():
                for rel in plist:
                    y = _year_of_name(rel)
                    if y is None or y not in years_set:
                        continue
                    name = Path(rel).name
                    if key in ("MCD14ML", "VNP14IMGML", "VJ114IMGML",
                               "VJ214IMGML"):
                        dest = out_dir / f"umd_{key}_{name}.parquet"
                        if dest.exists():
                            result["fires"].append(dest)
                            continue
                    tmp_path = Path(tmp) / name
                    keep_this = originals_dir is not None and (
                        keep_originals == "all"
                        or key in ("MOD14C8Q", "MYD14C8Q", "MCD64CMQ"))
                    kept = (Path(originals_dir) / key / name) if keep_this \
                        else None
                    try:
                        if kept is not None and kept.exists() and \
                                kept.stat().st_size > 0:
                            import shutil as _sh
                            _sh.copyfile(kept, tmp_path)
                        else:
                            session.download(rel, tmp_path)
                            _record_original(manifest, key, tmp_path,
                                             f"{session.method}://"
                                             f"{S.UMD_FTP_HOST}/{rel}",
                                             session.method)
                        if key == "MCD14ML" or key.startswith("V"):
                            with gzip.open(tmp_path, "rt", errors="replace") as fh:
                                lines = fh.read().splitlines()
                            df = (parse_mcd14ml_rows(lines) if key == "MCD14ML"
                                  else parse_vxx14imgml_rows(
                                      lines, platform_hint=(
                                          platform_from_vxx_filename(name))))
                            df = df[_in_boxes(df["latitude"], df["longitude"],
                                              bboxes)]
                            if schema_validation_gate(df, name) or not len(df):
                                df.to_parquet(dest, index=False,
                                              compression="zstd")
                                result["fires"].append(dest)
                        elif key in ("MOD14C8Q", "MYD14C8Q"):
                            meta = parse_cmg_filename(name)
                            if meta:
                                sds = read_hdf4_sds(tmp_path, CMQ_CLOUD_LAYERS)
                                f = parse_cloud_cmg(sds, meta["year"],
                                                    meta["period"],
                                                    meta["period_kind"])
                                f["platform"] = meta["product"][:3]
                                result["cloud"].append(
                                    f[_in_boxes(f["lat_025"], f["lon_025"],
                                                bboxes)])
                        elif key == "MCD64CMQ":
                            meta = parse_cmg_filename(name)
                            if meta:
                                sds = read_hdf4_sds(tmp_path, CMQ_BURN_LAYERS)
                                f = parse_burned_area_cmg(sds, meta["year"],
                                                          meta["period"])
                                result["burned"].append(
                                    f[_in_boxes(f["lat_025"], f["lon_025"],
                                                bboxes)])
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("UMD file %s failed: %s", rel, exc)
                        if manifest:
                            manifest.log(key, rel, session.method,
                                         detail=str(exc), status="FAIL")
                    finally:
                        if kept is not None and tmp_path.exists() and \
                                not kept.exists():
                            kept.parent.mkdir(parents=True, exist_ok=True)
                            import shutil as _sh
                            _sh.move(str(tmp_path), str(kept))
                        tmp_path.unlink(missing_ok=True)   # delete immediately
    finally:
        if own:
            session.close()
    return result


def file_sha256(path, chunk: int = 1 << 22) -> str:
    """SHA-256 of a file (streamed)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _record_original(manifest, product: str, path: Path, remote: str,
                     method: str) -> dict:
    """Log an original downloaded file (size + SHA-256) in the manifest."""
    info = {"file": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": file_sha256(path), "remote": remote}
    if manifest is not None:
        manifest.log(product, remote, method, detail=json.dumps(info))
    return info


def probe_sources(firms_map_key: str | None = None,
                  timeout_s: float = 20.0) -> list[dict]:
    """Readme Phase 1 "Minute 15" connectivity check of every Env 1 source.

    Returns one dict per source: ``{source, endpoint, reachable, detail}``.
    ``reachable`` means the endpoint answered as the product is served
    (e.g. FIRMS answers 401 without a valid MAP_KEY: the endpoint is live
    but a key is required).  No data is fabricated or cached.
    """
    results: list[dict] = []

    def http(source, url, ok_codes=(200,), method="GET", key_note=None):
        try:
            r = requests.request(method, url, timeout=timeout_s,
                                 stream=True, allow_redirects=True)
            code = r.status_code
            r.close()
            ok = code in ok_codes
            detail = f"HTTP {code}" + (f" ({key_note})" if key_note and
                                        code in (401, 403) else "")
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:200]
        results.append({"source": source, "endpoint": url.replace(
            firms_map_key or "\0", "<MAP_KEY>"), "reachable": ok,
            "detail": detail})

    def tcp(source, host, port):
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                ok, detail = True, f"TCP {port} open"
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"TCP {port}: {type(exc).__name__}: {exc}"[:200]
        results.append({"source": source, "endpoint": f"{host}:{port}",
                        "reachable": ok, "detail": detail})

    tcp("UMD archive SFTP (MCD14ML, Vxx14IMGML, Mx14C8Q/CMQ, MCD64CMQ)",
        S.UMD_FTP_HOST, 22)
    tcp("UMD archive FTP fallback", S.UMD_FTP_HOST, 21)
    http("Portolan STAC catalog (active-fire failover)", S.PORTOLAN_CATALOG_URL)
    http("Portolan year partition (HEAD, 2012)",
         f"{S.PORTOLAN_BASE}/detections/year=2012/detections.parquet",
         method="HEAD")
    key = firms_map_key or "INVALID_KEY_PROBE"
    http("FIRMS API data_availability (NRT gap-fill)",
         f"{S.FIRMS_API_BASE}/data_availability/csv/{key}/ALL",
         ok_codes=(200,) if firms_map_key else (200, 400, 401, 403),
         key_note="endpoint live; valid MAP_KEY required")
    http("Natural Earth 50m Admin-0 GeoJSON", S.NATURAL_EARTH_URL,
         method="HEAD")
    http("GeoBoundaries polygon fallback", S.GEOBOUNDARIES_URL.format(ISO3="ARG"))
    http("NASA CMR search (granule discovery)",
         f"{S.NASA_CMR_URL}?short_name=MCD64A1&page_size=1")
    http("DuckDB extension repository (httpfs/spatial for Portolan)",
         "http://extensions.duckdb.org/", ok_codes=(200, 301, 302, 403, 404))
    return results


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
    catalog = json.loads(requests.get(S.PORTOLAN_CATALOG_URL,
                                      timeout=HTTP_TIMEOUT_S).text)
    children = [c.get("href") for c in catalog.get("links", [])
                if c.get("rel") == "child"]
    return {"root": S.PORTOLAN_CATALOG_URL, "children": children}


def _wkb_points_xy(col) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized lon/lat from a GeoParquet WKB POINT column (pyarrow)."""
    import pyarrow as pa
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    if isinstance(arr, pa.ChunkedArray):
        arr = pa.concat_arrays(arr.chunks)
    offsets = np.frombuffer(arr.buffers()[1], dtype=np.int32)[
        arr.offset:arr.offset + len(arr) + 1]
    data = np.frombuffer(arr.buffers()[2], dtype=np.uint8)
    lens = np.diff(offsets)
    x = np.full(len(arr), np.nan)
    y = np.full(len(arr), np.nan)
    ok = lens >= 21
    starts = offsets[:-1][ok]
    le = data[starts] == 1
    raw = np.stack([data[starts + 5 + k] for k in range(16)], axis=1)
    for big in (False, True):
        sel = (~le) if big else le
        if sel.any():
            buf = raw[sel].copy()
            if big:
                buf[:, :8] = buf[:, 7::-1]
                buf[:, 8:] = buf[:, :7:-1]
            xy = buf.view(np.float64).reshape(-1, 2)
            idx = np.nonzero(ok)[0][sel]
            x[idx], y[idx] = xy[:, 0], xy[:, 1]
    return x, y


def fetch_portolan_year_arrow(year: int, bboxes: list[tuple], out_path: Path,
                              tmp_dir: Path | None = None) -> Path:
    """Failover for :func:`fetch_portolan_year` when DuckDB's httpfs /
    spatial extensions cannot be installed: stream the year partition over
    HTTPS to a temp file, scan it row group by row group with PyArrow,
    decode the WKB points vectorially and keep rows inside ``bboxes``."""
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    url = f"{S.PORTOLAN_BASE}/detections/year={year}/detections.parquet"
    cols = ["acq_datetime", "acq_date", "sensor", "satellite", "quality",
            "version", "brightness", "bright_t31", "bright_ti4", "bright_ti5",
            "scan", "track", "frp", "daynight", "confidence", "confidence_pct",
            "type"]
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        local = Path(td) / f"portolan_{year}_full.parquet"
        with requests.get(url, timeout=HTTP_TIMEOUT_S, stream=True) as r:
            r.raise_for_status()
            with open(local, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
        pf = pq.ParquetFile(local)
        have = [c for c in cols if c in pf.schema_arrow.names]
        parts = []
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=have + ["geometry"])
            lon, lat = _wkb_points_xy(t.column("geometry"))
            keep = _in_boxes(lat, lon, bboxes)
            if keep.any():
                sub = t.drop(["geometry"]).filter(pa.array(keep)).to_pandas()
                sub.insert(0, "latitude", lat[keep])
                sub.insert(0, "longitude", lon[keep])
                parts.append(sub)
    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=["longitude", "latitude"] + cols))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


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
    cmax = np.nanmax(cloud) if np.isfinite(cloud).any() else 0.0
    cloud = cloud / 100.0 if cmax > 1.0 else cloud
    lat_g, lon_g = np.meshgrid(lats[: cloud.shape[0]], lons[: cloud.shape[1]],
                               indexing="ij")
    df = pd.DataFrame({
        "lat_025": lat_g.ravel().round(2),
        "lon_025": lon_g.ravel().round(2),
        "cloud_frac": np.clip(cloud.ravel(), 0.0, 1.0),
        "year": year,
        "period": period,
        "period_kind": period_kind,   # '8d' or 'month'
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
    umax = np.nanmax(unmapped) if np.isfinite(unmapped).any() else 0.0
    unmapped = unmapped / 100.0 if umax > 1.0 else unmapped
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
            .assign(month=lambda d: np.where(
                d["period_kind"] == "month", d["period"],
                (pd.Timestamp("2001-01-01") + pd.to_timedelta(
                    np.clip((d["period"].to_numpy(int) - 1) * 8 + 4, 1, 365)
                    - 1, unit="D")).month))
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
def nrt_windows(last_science: str, end: str = S.DATE_END,
                max_range: int = 5) -> list[tuple[str, int]]:
    """Consecutive FIRMS ``/api/area`` windows ``(start_date, day_range)``
    covering ``last_science + 1 .. end`` with ``DAY_RANGE <= 5`` (the API
    returns DAY_RANGE days starting at DATE)."""
    start = pd.Timestamp(last_science) + pd.Timedelta(days=1)
    stop = pd.Timestamp(end)
    out = []
    while start <= stop:
        rng = int(min(max_range, (stop - start).days + 1))
        out.append((start.strftime("%Y-%m-%d"), rng))
        start += pd.Timedelta(days=rng)
    return out


def last_science_dates(raw_dir) -> dict:
    """Latest staged Science (ML/SP) acq_date per platform."""
    import duckdb
    files = [str(p) for p in Path(raw_dir).glob("*.parquet")
             if p.name.startswith(("umd_", "portolan_"))]
    out: dict = {}
    if not files:
        return out
    con = duckdb.connect()
    try:
        for f in files:
            cols = [r[0] for r in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [f]).fetchall()]
            if {"platform", "acq_date", "data_tier"} <= set(cols):
                q = ("SELECT platform, MAX(acq_date) FROM read_parquet(?) "
                     "WHERE data_tier <> 'NRT' GROUP BY platform")
            elif {"satellite", "acq_date"} <= set(cols):
                from .normalize_dedup import parse_portolan_frame
                df = parse_portolan_frame(pd.read_parquet(f))
                df = df[df["data_tier"] != "NRT"]
                for p, d in df.groupby("platform")["acq_date"].max().items():
                    out[p] = max(out.get(p, d), d)
                continue
            else:
                continue
            for p, d in con.execute(q, [f]).fetchall():
                d = str(d)[:10]
                out[p] = max(out.get(p, d), d)
    finally:
        con.close()
    return out


FIRMS_NRT_SOURCES = {"MODIS_NRT": ("Terra", "Aqua"),
                     "VIIRS_SNPP_NRT": ("SNPP",),
                     "VIIRS_NOAA20_NRT": ("NOAA-20",),
                     "VIIRS_NOAA21_NRT": ("NOAA-21",)}


def cmr_discover_cmg(manifest: ProvenanceManifest,
                     short_names=("MOD14C8Q", "MYD14C8Q", "MOD14CMQ",
                                  "MYD14CMQ", "MCD64CMQ")) -> dict:
    """Readme 6.2.3: query NASA CMR for alternate CMG granule endpoints and
    log every candidate download URL in the provenance manifest."""
    found = {}
    for sn in short_names:
        try:
            entries = cmr_search_granules(sn)
            urls = [lk.get("href") for e in entries for lk in e.get("links", [])
                    if str(lk.get("rel", "")).endswith("/data#")]
            found[sn] = urls
            manifest.log(f"cmr_{sn}", S.NASA_CMR_URL, "cmr_search",
                         detail=f"{len(urls)} granule URLs")
        except Exception as exc:  # noqa: BLE001
            manifest.log(f"cmr_{sn}", S.NASA_CMR_URL, "cmr_search",
                         detail=str(exc), status="FAIL")
    return found


def download_cmr_granules(urls: list[str], tmp_dir: Path, bboxes,
                          years: set[int], token: str | None) -> dict:
    """Download CMR-discovered CMG granules (Earthdata bearer token from
    ``EARTHDATA_TOKEN``), parse with pyhdf, delete each file at once."""
    out = {"cloud": [], "burned": []}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for url in urls:
        meta = parse_cmg_filename(url)
        if not meta or meta["year"] not in years:
            continue
        dest = Path(tmp_dir) / Path(urllib.parse.urlparse(url).path).name
        try:
            with requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_S,
                              stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if meta["product"] == "MCD64CMQ":
                f = parse_burned_area_cmg(read_hdf4_sds(dest, CMQ_BURN_LAYERS),
                                          meta["year"], meta["period"])
                out["burned"].append(f[_in_boxes(f["lat_025"], f["lon_025"],
                                                 bboxes)])
            else:
                f = parse_cloud_cmg(read_hdf4_sds(dest, CMQ_CLOUD_LAYERS),
                                    meta["year"], meta["period"],
                                    meta["period_kind"])
                f["platform"] = meta["product"][:3]
                out["cloud"].append(f[_in_boxes(f["lat_025"], f["lon_025"],
                                                bboxes)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("CMR granule %s failed: %s", url, exc)
        finally:
            dest.unlink(missing_ok=True)
    return out


def finalize_cloud_table(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge Terra (MOD) + Aqua (MYD) CMG cloud fractions: mean per
    (0.25 deg bin, year, period, period_kind)."""
    cloud = pd.concat(parts, ignore_index=True)
    return (cloud.groupby(["lat_025", "lon_025", "year", "period",
                           "period_kind"], as_index=False)["cloud_frac"]
            .mean())


def download_all(paths: DataPaths, years: list[int] | None = None,
                 firms_map_key: str | None = None,
                 bboxes: list[tuple] | None = None,
                 earthdata_token: str | None = None,
                 originals_dir: str | Path | None = None,
                 keep_originals: str = "all") -> dict:
    """Run the full Env 1 acquisition with autonomous failover.

    Returns a summary dict; every resolved source is written to
    ``data_provenance_manifest.json``.  This function requires Internet and is
    executed ONLY in Environment 1.
    """
    import os
    import tempfile

    manifest = ProvenanceManifest(Path(paths.provenance_manifest))
    paths.ensure_output_dirs()
    if years is None:
        years = list(range(2000, 2027))
    if bboxes is None:
        bboxes = [box for c in S.TARGET_COUNTRIES for box in c.prefilter_boxes]
    earthdata_token = earthdata_token or os.environ.get("EARTHDATA_TOKEN")
    summary: dict = {"years": years}

    # 1) Country polygons (needed for clipping everywhere downstream).
    download_country_polygons(paths, manifest)

    # 2) Historical active fires + CMG grids: UMD primary (SFTP -> FTP).
    umd = fetch_umd_history(paths, years, manifest, bboxes,
                            originals_dir=originals_dir,
                            keep_originals=keep_originals)
    summary["umd_method"] = umd["method"]
    summary["umd_fire_files"] = len(umd["fires"])
    if not umd["fires"]:
        # Portolan GeoParquet failover for active fires (readme 6.2.2)
        try:
            con = _duckdb_con()
        except Exception as exc:  # noqa: BLE001 - extensions not installable
            logger.warning("DuckDB httpfs/spatial unavailable (%s); Portolan "
                           "via PyArrow streaming failover.", exc)
            con = None
        try:
            try:
                catalog = inspect_portolan_catalog(con)
                manifest.log("portolan_catalog", S.PORTOLAN_CATALOG_URL,
                             "stac", detail=f"{len(catalog['children'])} children")
            except Exception as exc:  # noqa: BLE001
                manifest.log("portolan_catalog", S.PORTOLAN_CATALOG_URL,
                             "stac", detail=str(exc), status="FAIL")
            n_ok = 0
            for year in years:
                dest = Path(paths.raw_stage_dir) / f"portolan_{year}.parquet"
                if dest.exists() and dest.stat().st_size > 0:
                    n_ok += 1                       # resume: already staged
                    continue
                try:
                    if con is not None:
                        try:
                            fetch_portolan_year(year, bboxes, dest, con)
                            method = "portolan_duckdb_httpfs"
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Portolan httpfs %d failed (%s); "
                                           "PyArrow failover.", year, exc)
                            fetch_portolan_year_arrow(year, bboxes, dest)
                            method = "portolan_pyarrow_stream"
                    else:
                        fetch_portolan_year_arrow(year, bboxes, dest)
                        method = "portolan_pyarrow_stream"
                    _record_original(
                        manifest, f"active_fires_{year}", dest,
                        f"{S.PORTOLAN_BASE}/detections/year={year}/"
                        "detections.parquet", method)
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Portolan year %d failed: %s", year, exc)
                    manifest.log(f"active_fires_{year}", "portolan", "httpfs",
                                 detail=str(exc), status="FAIL")
            if n_ok == 0:
                manifest.critical("active_fires",
                                  "UMD and Portolan both unreachable")
        finally:
            if con is not None:
                con.close()

    # 3) Trailing NRT gap-fill: 5-day windows from the last Science date of
    #    each platform through DATE_END (DAY_RANGE <= 5 strictly).
    if firms_map_key:
        last_sci = last_science_dates(paths.raw_stage_dir)
        try:
            avail = firms_data_availability(firms_map_key)
            manifest.log("firms_data_availability", S.FIRMS_API_BASE, "http",
                         detail=f"{len(avail)} sources")
        except Exception as exc:  # noqa: BLE001
            avail = None
            manifest.log("firms_data_availability", S.FIRMS_API_BASE, "http",
                         detail=str(exc), status="FAIL")
        for source, plats in FIRMS_NRT_SOURCES.items():
            known = [last_sci[p] for p in plats if p in last_sci]
            start = min(known) if known else "2026-01-01"
            end = S.DATE_END
            if avail is not None and "data_id" in avail.columns:
                row = avail[avail["data_id"] == source]
                if len(row) and "max_date" in row.columns:
                    end = min(end, str(row["max_date"].iloc[0])[:10])
            for bi, box in enumerate(bboxes):
                for day0, rng in nrt_windows(start, end):
                    try:
                        df = fetch_firms_nrt(firms_map_key, source, box, rng,
                                             day0)
                        if len(df):
                            dest = (Path(paths.raw_stage_dir) /
                                    f"firms_nrt_{source}_{bi}_{day0}.parquet")
                            df.to_parquet(dest, index=False)
                    except Exception as exc:  # noqa: BLE001
                        manifest.log(f"nrt_{source}", "firms_api", "http",
                                     detail=f"{day0}: {exc}", status="FAIL")
            manifest.log(f"nrt_{source}", "firms_api", "area_csv",
                         detail=f"from {start} to {end}")

    # 4) Ancillary cloud + burned-area CMG grids (UMD -> CMR discovery).
    cloud_parts, burn_parts = list(umd["cloud"]), list(umd["burned"])
    if not cloud_parts or not burn_parts:
        found = cmr_discover_cmg(manifest)
        urls_c = [u for k, v in found.items() if k != "MCD64CMQ" for u in v]
        urls_b = found.get("MCD64CMQ", [])
        with tempfile.TemporaryDirectory(prefix="cmr_") as tmp:
            got = download_cmr_granules(
                (urls_c if not cloud_parts else []) +
                (urls_b if not burn_parts else []), Path(tmp), bboxes,
                set(int(y) for y in years), earthdata_token)
        cloud_parts += got["cloud"]
        burn_parts += got["burned"]
    if cloud_parts:
        cloud = finalize_cloud_table(cloud_parts)
        cloud.to_parquet(paths.cloud_parquet, index=False)
        manifest.log("cloud_cmg", str(paths.cloud_parquet), "pyhdf",
                     detail=f"{len(cloud)} rows; kinds="
                            f"{sorted(cloud['period_kind'].unique())}")
        build_cloud_climatology_fallback(cloud).to_parquet(
            Path(paths.cloud_parquet).with_name("cloud_climatology.parquet"),
            index=False)
    else:
        manifest.critical("cloud_cmg", "no MOD14C8Q/MYD14C8Q/Mx14CMQ grid "
                          "reachable via UMD or CMR (zero-cloud assumptions "
                          "are never fabricated)")
    if burn_parts:
        pd.concat(burn_parts, ignore_index=True).to_parquet(
            paths.burned_area_parquet, index=False)
        manifest.log("burned_area_cmq", str(paths.burned_area_parquet), "pyhdf")
    else:
        manifest.log("burned_area_cmq", "none", "pyhdf",
                     detail="MCD64CMQ unavailable: validation-only product",
                     status="WARN")
    manifest.save()
    summary["manifest"] = str(manifest.path)
    return summary
