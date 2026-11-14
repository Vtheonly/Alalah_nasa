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
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely import contains_xy

from ..config import settings as S

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
def parse_mcd14ml_rows(rows: list[str], data_tier: str = "ML_SCIENCE",
                       version_raw: str = "C6") -> pd.DataFrame:
    """Parse MCD14ML space-delimited lines (v1 omits trailing ``dn``; the
    ``sample`` field may carry a trailing dot, e.g. ``681.``)."""
    recs = []
    for line in rows:
        parts = line.split()
        if len(parts) < 12:
            if len(parts) == 11:      # version 1: no trailing dn flag
                parts = parts + ["-1"]
            else:
                continue
        ymd, hhmm, sat = parts[0], parts[1], parts[2]
        lat, lon, t21, t31 = (float(x) for x in parts[3:7])
        sample = parts[7].rstrip(".")
        frp, conf = float(parts[8]), parts[9]
        type_raw = int(parts[10]) if parts[10] not in (".", "") else -1
        dn = parts[11]
        platform = normalize_platform(sat)
        if platform is None:
            continue
        conf_r, conf_n = normalize_confidence(conf)
        recs.append({
            "country_iso": None, "sensor_family": sensor_family_of(platform),
            "platform": platform, "data_tier": data_tier,
            "latitude": lat, "longitude": lon,
            "acq_date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}", "acq_hhmm": int(hhmm),
            "t_mir_k": t21, "t_tir_k": t31, "scan_km": np.nan, "track_km": np.nan,
            "sample_idx": float(sample), "pixarea_km2": np.nan,
            "frp_mw": frp, "conf_raw": conf_r, "conf_num": conf_n,
            "type_raw": type_raw, "version_raw": version_raw,
            "ingest_ts": _ingest_ts_now(), "dn_source": dn,
        })
    return pd.DataFrame(recs, columns=CANONICAL_COLUMNS + ["dn_source"])


# ---------------------------------------------------------------------------
# Vxx14IMGML: comma-separated .csv.gz
# (YYYYMMDD, HHMM, Sat, Lat, Lon, T_I4, T_I5, Sample, Pixarea, FRP, Conf, Type)
# ---------------------------------------------------------------------------
def parse_vxx14imgml_rows(rows: list[str], data_tier: str = "ML_SCIENCE",
                          platform_hint: str | None = None,
                          version_raw: str = "C2") -> pd.DataFrame:
    recs = []
    for line in rows:
        line = line.strip()
        if not line or line.upper().startswith(("YYYYMMDD", "SAT",)):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        ymd, hhmm, sat = parts[0], parts[1], parts[2]
        lat, lon, ti4, ti5 = (float(x) for x in parts[3:7])
        sample = parts[7].rstrip(".")
        pixarea, frp, conf, typ = parts[8], parts[9], parts[10], parts[11]
        platform = normalize_platform(sat)
        if platform is None and platform_hint:
            platform = platform_hint
        if platform is None:
            continue
        conf_r, conf_n = normalize_confidence(conf)
        recs.append({
            "country_iso": None, "sensor_family": sensor_family_of(platform),
            "platform": platform, "data_tier": data_tier,
            "latitude": lat, "longitude": lon,
            "acq_date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}", "acq_hhmm": int(hhmm),
            "t_mir_k": ti4, "t_tir_k": ti5, "scan_km": np.nan, "track_km": np.nan,
            "sample_idx": float(sample) if sample else np.nan,
            "pixarea_km2": float(pixarea) if pixarea else np.nan,
            "frp_mw": float(frp), "conf_raw": conf_r, "conf_num": conf_n,
            "type_raw": int(typ) if typ not in ("", ".") else -1,
            "version_raw": version_raw, "ingest_ts": _ingest_ts_now(),
            "dn_source": None,
        })
    return pd.DataFrame(recs, columns=CANONICAL_COLUMNS + ["dn_source"])


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


def parse_firms_api_csv(text: str, data_tier: str = "NRT") -> pd.DataFrame:
    """Parse FIRMS REST API CSV (NRT: type missing -> -1, orbit-node daynight)."""
    df = pd.read_csv(pd.io.common.StringIO(text), dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    recs = []
    for _, r in df.iterrows():
        sat = r.get("satellite")
        platform = normalize_platform(sat)
        if platform is None and r.get("instrument", "").upper().startswith("VIIRS"):
            platform = {"N": "SNPP", "1": "NOAA-20", "2": "NOAA-21"}.get(str(sat))
        if platform is None:
            continue
        conf_r, conf_n = normalize_confidence(r.get("confidence"))
        hhmm_raw = str(r.get("acq_time", "")).strip() or "0"
        hhmm = int(float(hhmm_raw))
        type_raw = r.get("type")
        recs.append({
            "country_iso": None, "sensor_family": sensor_family_of(platform),
            "platform": platform, "data_tier": data_tier,
            "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
            "acq_date": str(r["acq_date"]), "acq_hhmm": hhmm,
            "t_mir_k": float(r["t_mir_k"]) if r.get("t_mir_k") else np.nan,
            "t_tir_k": float(r["t_tir_k"]) if r.get("t_tir_k") else np.nan,
            "scan_km": float(r["scan_km"]) if r.get("scan_km") else np.nan,
            "track_km": float(r["track_km"]) if r.get("track_km") else np.nan,
            "sample_idx": np.nan, "pixarea_km2": np.nan,
            "frp_mw": float(r["frp"]) if r.get("frp") not in (None, "", "nan") else 0.0,
            "conf_raw": conf_r, "conf_num": conf_n,
            "type_raw": int(float(type_raw)) if type_raw not in (None, "", "nan")
                       and str(type_raw).strip() != "" else -1,
            "version_raw": str(r.get("version", "NRT")),
            "ingest_ts": _ingest_ts_now(), "dn_source": r.get("daynight"),
        })
    return pd.DataFrame(recs, columns=CANONICAL_COLUMNS + ["dn_source"])


# ---------------------------------------------------------------------------
# Portolan GeoParquet mirror (FIRMS schema, 2000-present)
# ---------------------------------------------------------------------------
def parse_portolan_frame(df: pd.DataFrame, data_tier: str = "ML_SCIENCE") -> pd.DataFrame:
    """Parse a Portolan detections frame (geometry ST_Point or WKB bytes)."""
    lon = _portolan_lonlat(df, 0)
    lat = _portolan_lonlat(df, 1)
    sat_col = df["satellite"] if "satellite" in df.columns else pd.Series([None] * len(df))
    recs = []
    for i in range(len(df)):
        r = df.iloc[i]
        platform = normalize_platform(sat_col.iloc[i])
        if platform is None and "sensor" in df.columns:
            platform = {"MODIS": "Terra", "VIIRS": "SNPP"}.get(
                str(r.get("sensor", "")).upper())
        if platform is None:
            continue
        conf_r, conf_n = normalize_confidence(
            r.get("confidence_pct", r.get("confidence")))
        acq_dt = str(r.get("acq_datetime") or r.get("acq_date"))
        # acq_datetime like '2026-08-14 13:42:00' (UTC)
        date_part, time_part = _split_datetime(acq_dt)
        recs.append({
            "country_iso": None, "sensor_family": sensor_family_of(platform),
            "platform": platform, "data_tier": data_tier,
            "latitude": float(lat[i]), "longitude": float(lon[i]),
            "acq_date": date_part, "acq_hhmm": time_part,
            "t_mir_k": _num(r.get("bright_ti4", r.get("brightness"))),
            "t_tir_k": _num(r.get("bright_ti5", r.get("bright_t31"))),
            "scan_km": _num(r.get("scan")), "track_km": _num(r.get("track")),
            "sample_idx": np.nan,
            "pixarea_km2": np.nan,
            "frp_mw": _num(r.get("frp")) or 0.0,
            "conf_raw": conf_r, "conf_num": conf_n,
            "type_raw": int(float(r["type"])) if _num(r.get("type")) is not None
                       else -1,
            "version_raw": str(r.get("version", "portolan")),
            "ingest_ts": _ingest_ts_now(), "dn_source": r.get("daynight"),
        })
    return pd.DataFrame(recs, columns=CANONICAL_COLUMNS + ["dn_source"])


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
        box = next((b for c in S.TARGET_COUNTRIES if c.iso3 == iso
                    for b in c.prefilter_boxes), None)
        m = pd.Series(True, index=df.index)
        if box:
            min_lon, min_lat, max_lon, max_lat = box
            m &= (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
            m &= (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
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
    g = (df.groupby(["platform", "acq_date"]).size()
           .reset_index(name="n"))
    active = {p: [] for p in S.PLATFORM_INDEX}
    for _, r in g.iterrows():
        if r["platform"] in active and r["n"] >= 5:
            active[r["platform"]].append(r["acq_date"])
    for p in active:
        active[p] = sorted(active[p])
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
def run_normalization(paths, staged_frames: list[pd.DataFrame] | None = None
                      ) -> dict:
    """Parse -> clip -> dedup -> platform_active_day -> partitioned Parquet.

    ``staged_frames`` allows tests / the synthetic twin to inject canonical
    frames directly; production reads ``staged_raw/`` chunks from Env 1.
    """
    if staged_frames is None:
        frames = []
        raw_dir = Path(paths.raw_stage_dir)
        if raw_dir and raw_dir.is_dir():
            for p in sorted(raw_dir.glob("*.parquet")):
                if p.name.startswith("portolan_"):
                    frames.append(parse_portolan_frame(pd.read_parquet(p)))
                elif p.name.startswith("firms_nrt_"):
                    frames.append(parse_firms_api_csv(p.read_text()))
            # .txt.gz / .csv.gz chunks
            for p in sorted(raw_dir.glob("*.gz")):
                import gzip as _gzip
                with _gzip.open(p, "rt", errors="replace") as fh:
                    text = fh.read().splitlines()
                if p.name.startswith("MCD14ML"):
                    frames.append(parse_mcd14ml_rows(text))
                else:
                    frames.append(parse_vxx14imgml_rows(
                        text, platform_hint=platform_from_vxx_filename(p.name)))
        staged_frames = frames

    df = (pd.concat(staged_frames, ignore_index=True)
          if staged_frames else _empty_canonical())
    polygons = load_country_polygons(paths.country_polygons_parquet)
    df = clip_to_countries(df, polygons)
    df = science_tier_dominance_dedup(df)
    audit = coverage_audit(df)
    active_days = compute_platform_active_day(df)

    out_root = Path(paths.active_fires_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if len(df):
        df = df.drop(columns=["dn_source"], errors="ignore")
        df["year"] = pd.to_datetime(df["acq_date"]).dt.year.astype(int)
        for (iso, year), grp in df.groupby(["country_iso", "year"]):
            part = out_root / f"country_iso={iso}" / f"year={year}"
            part.mkdir(parents=True, exist_ok=True)
            (grp.drop(columns=["year"])
                 .to_parquet(part / "part.parquet", index=False,
                             compression="zstd"))
    report = {"audit": audit,
              "platform_active_day": active_days,
              "n_rows": int(len(df)),
              "n_countries": int(df["country_iso"].nunique())}
    (out_root / "ingestion_audit.json").write_text(json.dumps(
        {"audit": audit, "n_rows": int(len(df))}, indent=2, default=str))
    return report
