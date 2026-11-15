"""Schema normalization, polygon clipping, and Science-tier deduplication.

Contract (readme Sections 4.5, 6.3):
* Parse heterogeneous source schemas into the canonical schema:
  ``country_iso, sensor_family, platform, data_tier, latitude, longitude,
  acq_date, acq_hhmm, t_mir_k, t_tir_k, scan_km, track_km, sample_idx,
  pixarea_km2, frp_mw, conf_raw, conf_num, type_raw, version_raw, ingest_ts``.
* Clip points strictly inside the 6 Natural Earth Admin-0 polygons
  (``shapely.contains_xy``).
* Enforce ``(platform, acq_date)`` Science-tier dominance
  (``ML_SCIENCE > SP_SCIENCE > NRT``) and resolve intra-tier duplicates on
  ``(platform, acq_date, acq_hhmm, ROUND(lat,3), ROUND(lon,3))`` keeping
  ``tier_rank ASC, ingest_ts DESC``.
* Compute empirical ``platform_active_day[P, T]`` (>= 5 detections/day across
  the 6 countries) and run the mandatory coverage audit.
* Write Zstd Parquet partitioned by ``(country_iso, year)`` into
  ``active_fires/``.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely import contains_xy

from ..config import settings as S

logger = logging.getLogger("fire_harmonizer.ingest.normalize")

CANONICAL_COLUMNS = [
    "country_iso", "sensor_family", "platform", "data_tier",
    "latitude", "longitude", "acq_date", "acq_hhmm",
    "t_mir_k", "t_tir_k", "scan_km", "track_km", "sample_idx",
    "pixarea_km2", "frp_mw", "conf_raw", "conf_num", "type_raw",
    "version_raw", "ingest_ts",
]

_CONF_LETTER = {"l": 20.0, "n": 65.0, "h": 95.0, "low": 20.0, "nominal": 65.0,
                "high": 95.0}
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def normalize_platform(raw) -> str | None:
    """Map any satellite alias onto the canonical platform name."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    key = str(raw).strip().upper()
    if not key:
        return None
    return S.SATELLITE_ALIASES.get(key)


def normalize_confidence(raw):
    """Return (conf_raw, conf_num): letters preserved natively, numerics parsed."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "n", 65.0
    s = str(raw).strip()
    if not s:
        return "n", 65.0
    low = s.lower()
    if low in _CONF_LETTER:
        return low[0], _CONF_LETTER[low]
    m = _NUM_RE.search(s)
    if m:
        val = float(m.group())
        return s, val
    return low, 65.0


def sensor_family_of(platform: str) -> str:
    return S.SENSOR_FAMILY.get(platform, "MODIS" if platform in ("Terra", "Aqua") else "VIIRS")


def _ingest_ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_canonical(n: int = 0) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS})
    if n:
        df = pd.concat([df] * 0, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# MCD14ML: space-delimited .txt.gz  (YYYYMMDD HHMM sat lat lon T21 T31 sample FRP conf type dn)
# ---------------------------------------------------------------------------
# Column aliases of the UMD text archives, matched case-insensitively against
# the header line (MCD14ML C6 / C6.1: "YYYYMMDD HHMM sat lat lon T21 T31
# sample FRP conf type dn"; VNP14IMGML / VJ114IMGML / VJ214IMGML C1 / C2:
# comma-separated with Sat/Lat/Lon/T_I4|T4/T_I5|T5/Sample/[Line]/[Pixarea]/
# FRP/Conf/Type...).  Unknown header columns are ignored.
UMD_COLUMN_ALIASES = {
    "ymd": ("yyyymmdd", "date", "acq_date", "yyyy-mm-dd"),
    "hhmm": ("hhmm", "time", "acq_time"),
    "sat": ("sat", "satellite"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude"),
    "t_mir": ("t21", "t_i4", "t4", "ti4", "bright_ti4", "brightness",
              "t_i4_k", "fire_t4"),
    "t_tir": ("t31", "t_i5", "t5", "ti5", "bright_ti5", "bright_t31",
              "t_i5_k"),
    "sample": ("sample", "samp"),
    "pixarea": ("pixarea", "pix_area", "area", "pixelarea"),
    "frp": ("frp", "frp_mw"),
    "conf": ("conf", "confidence"),
    "type": ("type",),
    "dn": ("dn", "daynight", "day_night"),
}
_MCD14ML_POSITIONAL = ["ymd", "hhmm", "sat", "lat", "lon", "t_mir", "t_tir",
                       "sample", "frp", "conf", "type", "dn"]
_VXX_POSITIONAL = ["ymd", "hhmm", "sat", "lat", "lon", "t_mir", "t_tir",
                   "sample", "pixarea", "frp", "conf", "type"]


def _header_mapping(tokens: list[str]) -> dict[str, int] | None:
    """Canonical field -> column index from a header line, or None when the
    line is data (no recognisable date + lat + lon header names)."""
    norm = [t.strip().strip('"').lower().replace(" ", "_") for t in tokens]
    mapping: dict[str, int] = {}
    for field, aliases in UMD_COLUMN_ALIASES.items():
        for i, name in enumerate(norm):
            if name in aliases:
                mapping[field] = i
                break
    if {"ymd", "lat", "lon"} <= set(mapping):
        return mapping
    return None


def parse_umd_text(rows: list[str], product: str, data_tier: str = "ML_SCIENCE",
                   platform_hint: str | None = None,
                   version_raw: str | None = None) -> pd.DataFrame:
    """Vectorized parser of UMD ``MCD14ML`` / ``Vxx14IMGML`` text archives.

    * The column layout is read from the HEADER line when present (any
      collection: C6, C6.1, VIIRS C1, C2); header-less files fall back to the
      documented positional layout (MCD14ML version 1 without ``dn`` is
      accepted).  Whitespace or comma delimiters are detected.
    * Malformed lines are skipped and COUNTED (``df.attrs['n_bad_lines']``,
      ``df.attrs['n_lines']``) instead of discarding the whole file.
    * Platform from the ``sat`` column, else ``platform_hint`` (VIIRS file
      name: VNP -> SNPP, VJ1 -> NOAA-20, VJ2 -> NOAA-21).
    """
    import io
    product = product.upper()
    lines = [ln for ln in (r.rstrip("\r\n") for r in rows) if ln.strip()]
    empty = pd.DataFrame(columns=CANONICAL_COLUMNS + ["dn_source"])
    empty.attrs.update(n_lines=0, n_bad_lines=0, header=None)
    if not lines:
        return empty
    sep = "," if "," in lines[0] else None
    first = [t for t in (lines[0].split(",") if sep else lines[0].split())]
    mapping = _header_mapping(first)
    body = lines[1:] if mapping is not None else lines
    if mapping is None:
        names = _MCD14ML_POSITIONAL if product.startswith("MCD14ML") \
            else _VXX_POSITIONAL
        mapping = {f: i for i, f in enumerate(names)}
    width = max(mapping.values()) + 1
    # generous spare columns + index_col=False: rows with extra trailing
    # fields (or trailing delimiters) keep their leading fields aligned
    # instead of being shifted into an implicit index
    raw = pd.read_csv(io.StringIO("\n".join(body)),
                      sep="," if sep else r"\s+", header=None, dtype=str,
                      index_col=False, on_bad_lines="skip",
                      skipinitialspace=True,
                      names=range(max(width, len(first)) + 16))
    return _umd_finish(raw, mapping, first, len(body), product, data_tier,
                       platform_hint, version_raw)


def _umd_finish(raw: pd.DataFrame, mapping: dict, first: list[str],
                n_lines: int, product: str, data_tier: str,
                platform_hint: str | None, version_raw: str | None,
                extra_bad: int = 0) -> pd.DataFrame:
    """Shared tail of the UMD parsers: string table with integer column
    labels -> canonical frame (bad rows dropped and counted)."""
    product = product.upper()
    empty = pd.DataFrame(columns=CANONICAL_COLUMNS + ["dn_source"])
    empty.attrs.update(n_lines=n_lines, n_bad_lines=extra_bad, header=None)

    def col(field):
        i = mapping.get(field)
        if i is None or i >= raw.shape[1]:
            return pd.Series(np.nan, index=raw.index, dtype=object)
        v = raw[i].astype(str).str.strip()
        return v.where(~v.isin(["", "nan", "None", "NaN"]), None)

    num = lambda f: pd.to_numeric(col(f).str.rstrip("."), errors="coerce")  # noqa: E731
    ymd = col("ymd").str.replace("-", "", regex=False)
    lat, lon, frp = num("lat"), num("lon"), num("frp")
    hhmm = pd.to_numeric(col("hhmm"), errors="coerce")
    good = (ymd.str.fullmatch(r"\d{8}", na=False) & lat.between(-90, 90)
            & lon.between(-180, 180) & hhmm.between(0, 2359) & frp.notna())
    plat = col("sat").map(lambda v: normalize_platform(v) if isinstance(v, str)
                          else None)
    if platform_hint:
        plat = plat.where(plat.notna(), platform_hint)
    good &= plat.notna()
    n_bad = int(extra_bad + len(raw) - good.sum())
    r = good.to_numpy()
    if not r.any():
        empty.attrs.update(n_lines=n_lines, n_bad_lines=n_bad,
                           header=first if _header_mapping(first) else None)
        return empty
    conf = [normalize_confidence(v) for v in col("conf")[r].tolist()]
    typ = pd.to_numeric(col("type"), errors="coerce")[r]
    plat_r = plat[r]
    out = pd.DataFrame({
        "country_iso": None,
        "sensor_family": plat_r.map(sensor_family_of).to_numpy(),
        "platform": plat_r.to_numpy(), "data_tier": data_tier,
        "latitude": lat[r].to_numpy(dtype=float),
        "longitude": lon[r].to_numpy(dtype=float),
        "acq_date": (ymd[r].str[:4] + "-" + ymd[r].str[4:6] + "-"
                     + ymd[r].str[6:8]).to_numpy(),
        "acq_hhmm": hhmm[r].astype(int).to_numpy(),
        "t_mir_k": num("t_mir")[r].to_numpy(dtype=float),
        "t_tir_k": num("t_tir")[r].to_numpy(dtype=float),
        "scan_km": np.nan, "track_km": np.nan,
        "sample_idx": num("sample")[r].to_numpy(dtype=float),
        "pixarea_km2": num("pixarea")[r].to_numpy(dtype=float),
        "frp_mw": frp[r].to_numpy(dtype=float),
        "conf_raw": [c[0] for c in conf],
        "conf_num": np.array([c[1] for c in conf], dtype=float),
        "type_raw": typ.fillna(-1).astype(int).to_numpy(),
        "version_raw": version_raw or ("C6" if product.startswith("MCD14ML")
                                       else "C2"),
        "ingest_ts": _ingest_ts_now(),
        "dn_source": col("dn")[r].to_numpy(),
    })
    out = out[CANONICAL_COLUMNS + ["dn_source"]].reset_index(drop=True)
    out.attrs.update(n_lines=n_lines, n_bad_lines=n_bad,
                     header=first if _header_mapping(first) else None)
    return out


def _boxes_mask(lat, lon, bboxes) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if bboxes is None:
        return np.isfinite(lat) & np.isfinite(lon)
    keep = np.zeros(len(lat), dtype=bool)
    for (w, so, e, n) in bboxes:
        keep |= (lon >= w) & (lon <= e) & (lat >= so) & (lat <= n)
    return keep


def _concat_parsed(parts: list[pd.DataFrame], n_lines: int, n_bad: int,
                   header) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    out = (pd.concat(parts, ignore_index=True) if parts else
           pd.DataFrame(columns=CANONICAL_COLUMNS + ["dn_source"]))
    out.attrs.update(n_lines=int(n_lines), n_bad_lines=int(n_bad),
                     header=header)
    return out


def _parse_umd_csv_arrow(path: Path, first: list[str], mapping: dict,
                         product: str, data_tier: str, platform_hint,
                         version_raw, bboxes) -> pd.DataFrame | None:
    """Fast path for comma-delimited, headed UMD files (VIIRS C1/C2):
    multithreaded pyarrow streaming CSV reader, lat/lon typed at parse time,
    rows outside ``bboxes`` dropped per block BEFORE any Python-object
    conversion (2.6 M-line monthly VIIRS file: ~60 s -> a few seconds,
    bounded memory).  Returns None when the file has malformed rows (e.g.
    extra trailing fields) so the caller can use the tolerant pandas path,
    which keeps such rows aligned instead of dropping them."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    names = [str(i) for i in range(len(first))]
    invalid = [0]

    def _on_invalid(_row):
        invalid[0] += 1
        return "skip"

    types = {str(i): pa.string() for i in range(len(first))}
    types[str(mapping["lat"])] = pa.float64()
    types[str(mapping["lon"])] = pa.float64()
    try:
        reader = pacsv.open_csv(
            str(path),
            read_options=pacsv.ReadOptions(skip_rows=1, column_names=names,
                                           block_size=16 << 20),
            parse_options=pacsv.ParseOptions(delimiter=",",
                                             invalid_row_handler=_on_invalid),
            convert_options=pacsv.ConvertOptions(column_types=types,
                                                 strings_can_be_null=False))
        parts, n_rows, n_nolatlon = [], 0, 0
        for batch in reader:
            n_rows += batch.num_rows
            lat = batch.column(str(mapping["lat"])).to_numpy(zero_copy_only=False)
            lon = batch.column(str(mapping["lon"])).to_numpy(zero_copy_only=False)
            lat = np.asarray(lat, dtype=float)
            lon = np.asarray(lon, dtype=float)
            n_nolatlon += int((~(np.isfinite(lat) & np.isfinite(lon))).sum())
            keep = _boxes_mask(lat, lon, bboxes)
            if keep.any():
                sub = batch.filter(pa.array(keep)).to_pandas()
                sub.columns = [int(c) for c in sub.columns]
                parts.append(sub)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        logger.info("%s: arrow fast path not applicable (%s); pandas path",
                    Path(path).name, str(exc)[:200])
        return None
    if invalid[0]:
        logger.info("%s: %d malformed rows; re-parsing with the tolerant "
                    "pandas path", Path(path).name, invalid[0])
        return None
    raw = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=list(range(len(first)))))
    out = _umd_finish(raw, mapping, first, n_rows, product, data_tier,
                      platform_hint, version_raw, extra_bad=n_nolatlon)
    out.attrs["header"] = first
    return out


def parse_umd_file(path, product: str, data_tier: str = "ML_SCIENCE",
                   platform_hint: str | None = None,
                   version_raw: str | None = None, bboxes=None,
                   batch_lines: int = 400_000) -> pd.DataFrame:
    """Parse a UMD ``MCD14ML`` / ``Vxx14IMGML`` file (``.gz`` or plain) with
    BOUNDED memory, keeping only rows inside ``bboxes`` (all rows if None).

    * Comma-delimited headed files (VIIRS) -> :func:`_parse_umd_csv_arrow`.
    * Otherwise (whitespace MCD14ML, header-less, or malformed rows) the
      file is streamed in ``batch_lines`` batches through
      :func:`parse_umd_text` (the header is re-applied to every batch).
    ``attrs``: ``n_lines`` / ``n_bad_lines`` over the WHOLE file, ``header``.
    """
    import gzip
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        first_line = ""
        for ln in fh:
            if ln.strip():
                first_line = ln.rstrip("\r\n")
                break
    if not first_line:
        return _concat_parsed([], 0, 0, None)
    sep = "," if "," in first_line else None
    first = first_line.split(",") if sep else first_line.split()
    mapping = _header_mapping(first)
    if sep and mapping is not None:
        out = _parse_umd_csv_arrow(path, first, mapping, product, data_tier,
                                   platform_hint, version_raw, bboxes)
        if out is not None:
            return out
    parts, n_lines, n_bad = [], 0, 0
    header = [first_line] if mapping is not None else []

    def _flush(batch):
        nonlocal n_lines, n_bad
        df = parse_umd_text(header + batch, product, data_tier, platform_hint,
                            version_raw)
        n_lines += df.attrs.get("n_lines", 0)
        n_bad += df.attrs.get("n_bad_lines", 0)
        if len(df):
            parts.append(df[_boxes_mask(df["latitude"], df["longitude"],
                                        bboxes)])

    with opener(path, "rt", errors="replace") as fh:
        batch: list[str] = []
        skip_header = mapping is not None
        for ln in fh:
            if skip_header:
                if ln.strip():
                    skip_header = False
                continue
            batch.append(ln)
            if len(batch) >= batch_lines:
                _flush(batch)
                batch = []
        if batch:
            _flush(batch)
    return _concat_parsed(parts, n_lines, n_bad,
                          first if mapping is not None else None)


def parse_mcd14ml_rows(rows: list[str], data_tier: str = "ML_SCIENCE",
                       version_raw: str = "C6") -> pd.DataFrame:
    """MCD14ML space-delimited archive (header-aware; v1 without ``dn``;
    trailing-dot ``sample`` such as ``681.``)."""
    return parse_umd_text(rows, "MCD14ML", data_tier, None, version_raw)


# ---------------------------------------------------------------------------
# Vxx14IMGML: comma-separated archives (header-aware; C1 and C2 layouts)
# ---------------------------------------------------------------------------
def parse_vxx14imgml_rows(rows: list[str], data_tier: str = "ML_SCIENCE",
                          platform_hint: str | None = None,
                          version_raw: str = "C2") -> pd.DataFrame:
    return parse_umd_text(rows, "VNP14IMGML", data_tier, platform_hint,
                          version_raw)


def platform_from_vxx_filename(filename: str) -> str | None:
    """VNP->SNPP, VJ1->NOAA-20, VJ2->NOAA-21."""
    name = filename.upper()
    if "VNP14IMGML" in name:
        return "SNPP"
    if "VJ114IMGML" in name:
        return "NOAA-20"
    if "VJ214IMGML" in name:
        return "NOAA-21"
    return None


# ---------------------------------------------------------------------------
# FIRMS API CSV (NRT)  --  no `type` column; daynight by orbit node (untrusted)
# ---------------------------------------------------------------------------
FIRMS_API_FIELDS = {
    "latitude": "latitude", "longitude": "longitude",
    "bright_ti4": "t_mir_k", "brightness": "t_mir_k",
    "bright_t31": "t_tir_k", "bright_ti5": "t_tir_k",
    "scan": "scan_km", "track": "track_km",
    "acq_date": "acq_date", "acq_time": "acq_time",
    "satellite": "satellite", "instrument": "instrument",
    "confidence": "confidence", "version": "version",
    "frp": "frp_mw", "daynight": "daynight", "type": "type",
}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _num_series(sr: pd.Series) -> pd.Series:
    return pd.to_numeric(sr, errors="coerce")


def _platforms_from(df: pd.DataFrame) -> pd.Series:
    """Canonical platform per row from ``satellite`` (+ instrument/sensor)."""
    sat = _col(df, "satellite").astype(object)
    inst = _col(df, "instrument").fillna(_col(df, "sensor")).astype(str) \
        .str.upper()
    key = sat.astype(str).str.strip().str.upper()
    plat = key.map(S.SATELLITE_ALIASES)
    # bare VIIRS satellite codes from the FIRMS API ('N', '1', '2')
    viirs_code = {"N": "SNPP", "1": "NOAA-20", "2": "NOAA-21"}
    is_v = inst.str.startswith("VIIRS")
    plat = plat.where(~(plat.isna() & is_v), key.map(viirs_code))
    return plat


def _firms_like_to_canonical(df: pd.DataFrame, data_tier, lat, lon,
                             acq_date: pd.Series, acq_hhmm: pd.Series,
                             version_default: str) -> pd.DataFrame:
    """Shared FIRMS-schema -> canonical mapping (API CSV and Portolan).

    Brightness/thermal columns are chosen PER SENSOR (VIIRS: bright_ti4 /
    bright_ti5; MODIS: brightness / bright_t31) so a present-but-NaN column
    of the other sensor never masks the real value (INV-06: the two sensors'
    temperatures are never mixed).  FIRMS ``scan`` / ``track`` are the pixel
    dimensions in km.
    """
    plat = _platforms_from(df)
    keep = plat.notna().to_numpy()
    fam = plat.map(lambda p: sensor_family_of(p) if isinstance(p, str)
                   else None)
    is_v = (fam == "VIIRS").to_numpy()
    t_mir = np.where(is_v, _num_series(_col(df, "bright_ti4")),
                     _num_series(_col(df, "brightness")))
    t_tir = np.where(is_v, _num_series(_col(df, "bright_ti5")),
                     _num_series(_col(df, "bright_t31")))
    conf_src = _col(df, "confidence_pct").where(
        _col(df, "confidence_pct").notna(), _col(df, "confidence"))
    conf = [normalize_confidence(v) for v in conf_src.tolist()]
    typ = _num_series(_col(df, "type"))
    frp = _num_series(_col(df, "frp")).fillna(0.0)
    ver = _col(df, "version").astype(object).where(
        _col(df, "version").notna(), version_default).astype(str)
    out = pd.DataFrame({
        "country_iso": None, "sensor_family": fam.to_numpy(),
        "platform": plat.to_numpy(),
        "data_tier": (data_tier.to_numpy() if isinstance(data_tier, pd.Series)
                      else data_tier),
        "latitude": np.asarray(lat, dtype=np.float64),
        "longitude": np.asarray(lon, dtype=np.float64),
        "acq_date": acq_date.to_numpy(), "acq_hhmm": acq_hhmm.to_numpy(),
        "t_mir_k": t_mir.astype(float), "t_tir_k": t_tir.astype(float),
        "scan_km": _num_series(_col(df, "scan")).to_numpy(dtype=float),
        "track_km": _num_series(_col(df, "track")).to_numpy(dtype=float),
        "sample_idx": np.nan, "pixarea_km2": np.nan,
        "frp_mw": frp.to_numpy(dtype=float),
        "conf_raw": [c[0] for c in conf],
        "conf_num": np.array([c[1] for c in conf], dtype=float),
        "type_raw": typ.fillna(-1).astype(int).to_numpy(),
        "version_raw": ver.to_numpy(),
        "ingest_ts": _ingest_ts_now(),
        "dn_source": _col(df, "daynight").to_numpy(),
    }, index=df.index)
    return out[keep].reset_index(drop=True)[CANONICAL_COLUMNS + ["dn_source"]]


def parse_firms_api_frame(df: pd.DataFrame, data_tier: str = "NRT"
                          ) -> pd.DataFrame:
    """FIRMS REST API rows (as a frame) -> canonical schema.

    NRT feeds carry no ``type`` column (-> -1) and an orbit-node
    ``daynight`` (kept only as provenance ``dn_source``; the pipeline
    recomputes day/night from local solar time, readme 5.1.1).  FIRMS API
    columns: ``bright_ti4``/``bright_ti5`` (VIIRS) or ``brightness``/
    ``bright_t31`` (MODIS), ``scan``, ``track``, ``acq_time`` (HHMM).
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not len(df):
        return _empty_canonical().assign(dn_source=pd.Series(dtype=object))
    hhmm = _num_series(_col(df, "acq_time")).fillna(0).astype(int)
    return _firms_like_to_canonical(
        df, data_tier, _num_series(df["latitude"]),
        _num_series(df["longitude"]), df["acq_date"].astype(str).str[:10],
        hhmm, "NRT")


def parse_firms_api_csv(text: str, data_tier: str = "NRT") -> pd.DataFrame:
    """Parse FIRMS REST API CSV text (see :func:`parse_firms_api_frame`)."""
    from io import StringIO
    return parse_firms_api_frame(pd.read_csv(StringIO(text), dtype=str),
                                 data_tier)


# ---------------------------------------------------------------------------
# Portolan GeoParquet mirror (FIRMS schema, 2000-present)
# ---------------------------------------------------------------------------
def portolan_data_tier(df: pd.DataFrame) -> pd.Series:
    """Processing tier per Portolan row from ``quality`` / ``version``:
    any 'NRT' marker -> ``NRT``; otherwise the FIRMS archive's standard
    (science) processing -> ``SP_SCIENCE`` (UMD ``*ML`` files are the only
    ``ML_SCIENCE`` source)."""
    q = _col(df, "quality").astype(str).str.upper()
    v = _col(df, "version").astype(str).str.upper()
    nrt = q.str.contains("NRT") | v.str.contains("NRT") | (q == "N")
    return pd.Series(np.where(nrt, "NRT", "SP_SCIENCE"), index=df.index)


def parse_portolan_frame(df: pd.DataFrame, data_tier: str | None = None
                         ) -> pd.DataFrame:
    """Parse a Portolan detections frame (geometry ST_Point / WKB bytes or
    pre-extracted longitude/latitude).  ``data_tier=None`` derives the tier
    per row from ``quality`` / ``version``."""
    if not len(df):
        return _empty_canonical().assign(dn_source=pd.Series(dtype=object))
    lon = _portolan_lonlat(df, 0)
    lat = _portolan_lonlat(df, 1)
    acq = _col(df, "acq_datetime").where(_col(df, "acq_datetime").notna(),
                                         _col(df, "acq_date"))
    parts = [_split_datetime(v) for v in acq.astype(str).tolist()]
    tier = portolan_data_tier(df) if data_tier is None else data_tier
    return _firms_like_to_canonical(
        df, tier, lat, lon, pd.Series([p[0] for p in parts], index=df.index),
        pd.Series([p[1] for p in parts], index=df.index), "portolan")


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _split_datetime(s: str) -> tuple[str, int]:
    s = str(s).strip()
    if "T" in s:
        s = s.replace("T", " ")
    if " " in s:
        d, t = s.split(" ", 1)
        hhmm = int(t[:2]) * 100 + int(t[3:5])
    else:
        d, hhmm = s, 0
    return d[:10], hhmm


def _portolan_lonlat(df: pd.DataFrame, axis: int) -> np.ndarray:
    """Extract lon/lat from a geometry column (ST_X/ST_Y applied upstream, or
    raw GeoParquet WKB point bytes)."""
    if "longitude" in df.columns and "latitude" in df.columns:
        return df["longitude"].to_numpy() if axis == 0 else df["latitude"].to_numpy()
    geom = df["geometry"].to_numpy()
    out = np.empty(len(df), dtype=np.float64)
    for i, g in enumerate(geom):
        if isinstance(g, (bytes, bytearray, memoryview)):
            out[i] = _wkb_point(g)[axis]
        elif isinstance(g, (tuple, list)):
            out[i] = float(g[axis])
        else:  # WKT "POINT (x y)" or a shapely geometry
            coords = str(g).split("(", 1)[1].rsplit(")", 1)[0].replace(",", " ").split()
            out[i] = float(coords[axis])
    return out


def _wkb_point(wkb: bytes) -> tuple[float, float]:
    """Minimal GeoParquet/OGC WKB POINT reader (little/big endian)."""
    b = bytes(wkb)
    if len(b) < 21:
        raise ValueError("invalid WKB point")
    import struct
    byte_order = b[0]  # 0 = big endian, 1 = little endian
    fmt = "<" if byte_order == 1 else ">"
    geom_type = struct.unpack(fmt + "I", b[1:5])[0]
    if geom_type not in (1, 0x80000001):  # POINT / POINT(Z) with 3 dims
        raise ValueError(f"WKB is not a POINT (type={geom_type})")
    x, y = struct.unpack(fmt + "dd", b[5:21])
    return x, y


# ---------------------------------------------------------------------------
# Polygon clipping (strict Admin-0 containment)
# ---------------------------------------------------------------------------
def load_country_polygons(parquet_path) -> dict[str, "shape"]:
    """Load ``country_polygons.parquet`` -> {iso3: shapely geometry}."""
    df = pd.read_parquet(parquet_path)
    out = {}
    for _, r in df.iterrows():
        out[r["country_iso"]] = shape(json.loads(r["geometry"]))
    return out


def clip_to_countries(df: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """Keep rows strictly inside the target country polygons; tag country_iso.

    Prefilter boxes cut the O(n_polygon) work, then exact ``contains_xy``.
    """
    if df.empty:
        df = df.copy()
        df["country_iso"] = pd.Series(dtype="object")
        return df
    out_parts = []
    for iso, geom in polygons.items():
        boxes = [b for c in S.TARGET_COUNTRIES if c.iso3 == iso
                 for b in c.prefilter_boxes]
        # OR over EVERY prefilter box (USA = CONUS box + Alaska box)
        if boxes:
            m = pd.Series(False, index=df.index)
            for min_lon, min_lat, max_lon, max_lat in boxes:
                m |= ((df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
                      & (df["latitude"] >= min_lat)
                      & (df["latitude"] <= max_lat))
        else:
            m = pd.Series(True, index=df.index)
        if not m.any():
            continue
        sub = df[m]
        inside = contains_xy(geom, sub["longitude"].to_numpy(),
                             sub["latitude"].to_numpy())
        keep = sub[np.asarray(inside)]
        if len(keep):
            keep = keep.copy()
            keep["country_iso"] = iso
            out_parts.append(keep)
    if not out_parts:
        out = df.iloc[0:0].copy()
        out["country_iso"] = pd.Series(dtype="object")
        return out
    out = pd.concat(out_parts, ignore_index=True)
    # Multipolygon countries (USA) handled by shapely geometry union above.
    return out


# ---------------------------------------------------------------------------
# Tier-dominance deduplication
# ---------------------------------------------------------------------------
def science_tier_dominance_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NRT rows for any (platform, acq_date) covered by Science data;
    resolve intra-tier duplicates keeping ``tier_rank ASC, ingest_ts DESC``."""
    if df.empty:
        return df
    df = df.copy()
    df["_tier_rank"] = df["data_tier"].map(S.TIER_RANK).fillna(9)
    science_days = set(
        map(tuple, df.loc[df["_tier_rank"] <= 1, ["platform", "acq_date"]]
            .drop_duplicates().to_numpy()))
    is_nrt = df["data_tier"] == "NRT"
    nrt_keys = list(map(tuple, df.loc[is_nrt, ["platform", "acq_date"]].to_numpy()))
    drop = pd.Series(False, index=df.index)
    if nrt_keys:
        drop.loc[is_nrt] = [k in science_days for k in nrt_keys]
    drop_mask = is_nrt & drop
    df = df[~drop_mask].copy()
    # Intra-tier exact duplicates on (platform, acq_date, acq_hhmm,
    # ROUND(lat,3), ROUND(lon,3)); keep tier_rank ASC then ingest_ts DESC.
    df["_lat3"] = df["latitude"].round(3)
    df["_lon3"] = df["longitude"].round(3)
    df = df.sort_values(
        ["_tier_rank", "ingest_ts"],
        ascending=[True, False], kind="mergesort")
    df = df.drop_duplicates(
        subset=["platform", "acq_date", "acq_hhmm", "_lat3", "_lon3"],
        keep="first")
    return df.drop(columns=["_tier_rank", "_lat3", "_lon3"], errors="ignore")


# ---------------------------------------------------------------------------
# platform_active_day[P, T]  (empirical spacecraft outage masking, INV-04)
# ---------------------------------------------------------------------------
def date_to_t_index(dates, date_start: date = date.fromisoformat(S.DATE_START)) -> np.ndarray:
    ds = pd.to_datetime(pd.Series(list(dates)))
    return ((ds - pd.Timestamp(date_start)).dt.days).to_numpy(dtype=np.int64)


def compute_platform_active_day(df: pd.DataFrame) -> dict:
    """Mark platform p inactive on UTC date t if total detections across all 6
    countries < 5 (masking safe-mode outages).  Returns nested dict of lists
    keyed by platform -> sorted list of active UTC dates."""
    if df.empty:
        return {p: [] for p in S.PLATFORM_INDEX}
    d = df.assign(acq_date=df["acq_date"].astype(str).str[:10])
    g = d.groupby(["platform", "acq_date"]).size()
    g = g[g >= 5]
    active = {p: [] for p in S.PLATFORM_INDEX}
    for p, dates in g.reset_index().groupby("platform")["acq_date"]:
        if p in active:
            active[p] = sorted(dates.tolist())
    return active


def platform_active_matrix(active_days: dict, t_days: int = S.T_DAYS,
                           date_start: date = date.fromisoformat(S.DATE_START)
                           ) -> np.ndarray:
    """Boolean (P, T) matrix from compute_platform_active_day output."""
    mat = np.zeros((S.N_PLATFORMS, t_days), dtype=bool)
    for p, days in active_days.items():
        p_idx = S.PLATFORM_INDEX[p]
        for d in days:
            t = (date.fromisoformat(d) - date_start).days
            if 0 <= t < t_days:
                mat[p_idx, t] = True
    return mat


def platform_operational_matrix(t_days: int = S.T_DAYS) -> np.ndarray:
    """Static operational calendar from PLATFORM_TIMELINE (launch/retirement)."""
    start = date.fromisoformat(S.DATE_START)
    mat = np.zeros((S.N_PLATFORMS, t_days), dtype=bool)
    for p, (d0, d1) in S.PLATFORM_TIMELINE.items():
        i0 = max(0, (date.fromisoformat(d0) - start).days)
        i1 = min(t_days - 1, (date.fromisoformat(d1) - start).days)
        if i1 >= i0:
            mat[S.PLATFORM_INDEX[p], i0:i1 + 1] = True
    return mat


# ---------------------------------------------------------------------------
# Coverage audit (readme 6.3.6)
# ---------------------------------------------------------------------------
def coverage_audit(df: pd.DataFrame) -> dict:
    """Mandatory checks: Chilecito 2026 present, SNPP ends 2026-11-02,
    NOAA-20/21 continue through 2026-11-14, DZA type=2 flares preserved."""
    audit: dict = {}
    sub = df[(df["latitude"].between(-29.6, -28.7)) &
             (df["longitude"].between(-68.0, -67.1))]
    audit["chilecito_jul_aug_2026_detections"] = int(
        ((pd.to_datetime(sub["acq_date"]).dt.year == 2026) &
         (pd.to_datetime(sub["acq_date"]).dt.month.isin([7, 8]))).sum())
    snpp = df[df["platform"] == "SNPP"]
    if len(snpp):
        last = pd.to_datetime(snpp["acq_date"]).max()
        audit["snpp_last_acq_date"] = str(last.date())
        audit["snpp_ended_by_2026_11_02"] = bool(last <= pd.Timestamp("2026-11-02"))
    else:
        audit["snpp_last_acq_date"] = None
        audit["snpp_ended_by_2026_11_02"] = True
    for p in ("NOAA-20", "NOAA-21"):
        sub_p = df[df["platform"] == p]
        audit[f"{p}_through_2026_11_14"] = bool(
            len(sub_p) and pd.to_datetime(sub_p["acq_date"]).max()
            >= pd.Timestamp(S.DATE_END) - pd.Timedelta(days=3))
    dza = df[(df["country_iso"] == "DZA") & (df["type_raw"] == 2)]
    audit["dza_type2_flares_preserved"] = int(len(dza))
    audit["n_rows"] = int(len(df))
    audit["platforms_present"] = sorted(df["platform"].dropna().unique().tolist())
    return audit


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _merge_audits(parts: list[dict]) -> dict:
    """Combine per-month :func:`coverage_audit` results into the audit of
    the union (sums for counts, max for last dates, OR for coverage)."""
    if not parts:
        return coverage_audit(_empty_canonical())
    out = {"chilecito_jul_aug_2026_detections":
           sum(p["chilecito_jul_aug_2026_detections"] for p in parts)}
    lasts = [p["snpp_last_acq_date"] for p in parts if p["snpp_last_acq_date"]]
    out["snpp_last_acq_date"] = max(lasts) if lasts else None
    out["snpp_ended_by_2026_11_02"] = (
        out["snpp_last_acq_date"] is None
        or out["snpp_last_acq_date"] <= "2026-11-02")
    for p in ("NOAA-20", "NOAA-21"):
        out[f"{p}_through_2026_11_14"] = any(x[f"{p}_through_2026_11_14"]
                                             for x in parts)
    out["dza_type2_flares_preserved"] = sum(p["dza_type2_flares_preserved"]
                                            for p in parts)
    out["n_rows"] = sum(p["n_rows"] for p in parts)
    out["platforms_present"] = sorted(set().union(
        *[set(p["platforms_present"]) for p in parts]))
    return out


def _iter_staged_frames(raw_dir: Path):
    """Yield (name, canonical frame) for every Env 1 staged chunk, one file
    at a time (bounded memory)."""
    import gzip as _gzip
    for p in sorted(raw_dir.glob("*.parquet")):
        if p.name.startswith("umd_"):
            yield p.name, pd.read_parquet(p)              # canonical (Env 1)
        elif p.name.startswith("portolan_"):
            yield p.name, parse_portolan_frame(pd.read_parquet(p))
        elif p.name.startswith("firms_nrt_"):
            nrt = pd.read_parquet(p)
            yield p.name, (nrt if set(CANONICAL_COLUMNS) <= set(nrt.columns)
                           else parse_firms_api_frame(nrt, "NRT"))
    for p in sorted(raw_dir.glob("firms_nrt_*.csv")):
        yield p.name, parse_firms_api_csv(p.read_text(), "NRT")
    for p in sorted(raw_dir.glob("*.gz")):
        with _gzip.open(p, "rt", errors="replace") as fh:
            text = fh.read().splitlines()
        if p.name.startswith("MCD14ML"):
            yield p.name, parse_mcd14ml_rows(text)
        else:
            yield p.name, parse_vxx14imgml_rows(
                text, platform_hint=platform_from_vxx_filename(p.name))


def run_normalization(paths, staged_frames: list[pd.DataFrame] | None = None
                      ) -> dict:
    """Parse -> clip -> dedup -> platform_active_day -> partitioned Parquet.

    Streams with bounded memory (real 6-country archives are tens of
    millions of rows): pass 1 parses ONE staged file at a time, clips it to
    the country polygons and spools it by acquisition month; pass 2 runs the
    Science-tier dominance dedup per month (its keys are
    ``(platform, acq_date)``, so a month holds every competing row), writes
    the ``(country_iso, year)`` partitions, and accumulates
    ``platform_active_day`` and the coverage audit.

    ``staged_frames`` allows tests / the synthetic twin to inject canonical
    frames directly; production reads ``staged_raw/`` chunks from Env 1.
    """
    import shutil
    import tempfile

    polygons = load_country_polygons(paths.country_polygons_parquet)
    out_root = Path(paths.active_fires_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for old in out_root.glob("country_iso=*"):
        shutil.rmtree(old)                  # idempotent re-normalization
    if staged_frames is None:
        raw_dir = Path(paths.raw_stage_dir)
        source = (_iter_staged_frames(raw_dir) if raw_dir.is_dir() else iter(()))
    else:
        source = ((f"frame_{i}", f) for i, f in enumerate(staged_frames))

    spool = Path(tempfile.mkdtemp(prefix="fh_norm_spool_",
                                  dir=str(out_root.parent)))
    n_in = 0
    try:
        for k, (name, df) in enumerate(source):
            n_in += len(df)
            if not len(df):
                continue
            df = clip_to_countries(df, polygons)
            if not len(df):
                continue
            df = df.drop(columns=["dn_source"], errors="ignore")
            ym = df["acq_date"].astype(str).str[:7]
            for key, grp in df.groupby(ym):
                d = spool / f"ym={key}"
                d.mkdir(exist_ok=True)
                grp.to_parquet(d / f"part_{k:06d}.parquet", index=False)
        audits, active = [], {p: [] for p in S.PLATFORM_INDEX}
        n_out = 0
        countries = set()
        for d in sorted(spool.glob("ym=*")):
            df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                           ignore_index=True)
            df = science_tier_dominance_dedup(df)
            audits.append(coverage_audit(df))
            for p, days in compute_platform_active_day(df).items():
                active.setdefault(p, []).extend(days)
            n_out += len(df)
            countries |= set(df["country_iso"].dropna().unique())
            df["year"] = pd.to_datetime(df["acq_date"]).dt.year.astype(int)
            month = d.name.split("=")[1]
            for (iso, year), grp in df.groupby(["country_iso", "year"]):
                part = out_root / f"country_iso={iso}" / f"year={year}"
                part.mkdir(parents=True, exist_ok=True)
                (grp.drop(columns=["year"])
                    .to_parquet(part / f"part_{month}.parquet", index=False,
                                compression="zstd"))
    finally:
        shutil.rmtree(spool, ignore_errors=True)
    for p in active:
        active[p] = sorted(set(active[p]))
    audit = _merge_audits(audits)
    report = {"audit": audit, "platform_active_day": active,
              "n_rows": int(n_out), "n_rows_parsed": int(n_in),
              "n_countries": int(len(countries))}
    (out_root / "ingestion_audit.json").write_text(json.dumps(
        {"audit": audit, "n_rows": int(n_out), "n_rows_parsed": int(n_in)},
        indent=2, default=str))
    return report
