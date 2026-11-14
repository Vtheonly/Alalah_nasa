"""Environment 1 normalization, polygon clipping & Science-tier dedup.

Contract (readme.md Sections 4 & 6.3):
  * Parse heterogeneous source schemas (MCD14ML space-delimited .txt,
    Vxx14IMGML CSV, FIRMS API CSV, Portolan GeoParquet) into the canonical
    schema ``country_iso, sensor_family, platform, data_tier, latitude,
    longitude, acq_date, acq_hhmm, t_mir_k, t_tir_k, scan_km, track_km,
    sample_idx, pixarea_km2, frp_mw, conf_raw, conf_num, type_raw,
    version_raw, ingest_ts``.
  * Clip points strictly inside the 6 Natural Earth Admin-0 polygons
    (``shapely.contains_xy``).
  * Enforce ``(platform, acq_date)`` Science-tier dominance
    (``ML_SCIENCE > SP_SCIENCE > NRT``) and intra-tier dedup on
    ``(platform, acq_date, acq_hhmm, ROUND(lat,3), ROUND(lon,3))`` keeping
    ``tier_rank ASC, ingest_ts DESC``.
  * Compute empirical ``platform_active_day[P, T]`` (>= 5 detections/day
    across the six countries) and emit the structured coverage audit.
  * Write Zstd Parquet partitioned by ``(country_iso, year)``.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import settings as S

CANONICAL_COLUMNS: Tuple[str, ...] = (
    "country_iso",
    "sensor_family",
    "platform",
    "data_tier",
    "latitude",
    "longitude",
    "acq_date",
    "acq_hhmm",
    "t_mir_k",
    "t_tir_k",
    "scan_km",
    "track_km",
    "sample_idx",
    "pixarea_km2",
    "frp_mw",
    "conf_raw",
    "conf_num",
    "type_raw",
    "version_raw",
    "ingest_ts",
)


# --------------------------------------------------------------------------- #
# Satellite / confidence normalization helpers
# --------------------------------------------------------------------------- #
def normalize_satellite(raw: object) -> Optional[str]:
    """Map every documented satellite alias to its canonical platform name."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    key = str(raw).strip().upper()
    if key in S.SATELLITE_ALIASES:
        return S.SATELLITE_ALIASES[key]
    # pandas may coerce integer alias columns to floats ('2' -> '2.0').
    if key.endswith(".0"):
        return S.SATELLITE_ALIASES.get(key[:-2])
    return None


def platform_from_viirs_filename(filename: str) -> Optional[str]:
    """Infer the VIIRS platform from a Vxx14IMGML filename prefix."""
    base = filename.upper()
    for prefix, platform in S.VIIRS_FILE_PREFIX_PLATFORM.items():
        if prefix in base:
            return platform
    return None


def parse_conf(conf_raw: object) -> Tuple[str, Optional[float]]:
    """Return ``(conf_raw, conf_num)`` for textual or numeric confidence.

    Textual labels map to ``low=20, nominal=65, high=95``; numeric values in
    [0, 100] pass through directly.
    """
    if conf_raw is None or (isinstance(conf_raw, float) and np.isnan(conf_raw)):
        return "n", None
    text = str(conf_raw).strip()
    if text in S.CONF_LABEL_VALUES:
        return text, S.CONF_LABEL_VALUES[text]
    try:
        value = float(text)
    except ValueError:
        return text, None
    return text, value


def hhmm_to_hours(acq_hhmm: object) -> float:
    """Base-60 conversion: HHMM -> fractional hours (e.g. 51 -> 0.85 h)."""
    hhmm = int(acq_hhmm)
    return (hhmm // 100) + (hhmm % 100) / 60.0


# --------------------------------------------------------------------------- #
# Source parsers
# --------------------------------------------------------------------------- #
def parse_mcd14ml(lines: Iterable[str], ingest_ts: Optional[str] = None,
                  version_raw: str = "C61") -> pd.DataFrame:
    """Parse MCD14ML space-delimited .txt rows.

    Format: ``YYYYMMDD HHMM sat lat lon T21 T31 sample FRP conf type [dn]``
    (version 1 omits ``dn``; ``sample`` may carry a trailing dot, e.g. '681.').
    """
    ts = ingest_ts or datetime.utcnow().isoformat() + "Z"
    records: List[dict] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 12:  # v2: ... conf type dn
            (ymd, hhmm, sat, lat, lon, t21, t31, sample, frp, conf, ftype,
             dn) = parts
        elif len(parts) == 11:  # version 1 omits dn
            (ymd, hhmm, sat, lat, lon, t21, t31, sample, frp, conf,
             ftype) = parts
            dn = None
        else:
            raise ValueError(
                f"MCD14ML row has {len(parts)} fields; expected 11 or 12: {line!r}"
            )
        platform = normalize_satellite(sat)
        conf_txt, conf_num = parse_conf(conf)
        records.append(
            {
                "country_iso": None,
                "sensor_family": "MODIS",
                "platform": platform,
                "data_tier": "ML_SCIENCE",
                "latitude": float(lat),
                "longitude": float(lon),
                "acq_date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}",
                "acq_hhmm": int(hhmm),
                "t_mir_k": float(t21),
                "t_tir_k": float(t31),
                "scan_km": np.nan,
                "track_km": np.nan,
                "sample_idx": int(round(float(sample.rstrip(".")))),
                "pixarea_km2": np.nan,
                "frp_mw": float(frp),
                "conf_raw": conf_txt,
                "conf_num": conf_num,
                "type_raw": int(float(ftype)) if ftype not in ("", "-") else -1,
                "version_raw": version_raw,
                "ingest_ts": ts,
                "dn_raw": dn,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(
        ("country_iso", "sensor_family", "platform", "data_tier", "latitude",
         "longitude", "acq_date", "acq_hhmm", "t_mir_k", "t_tir_k", "scan_km",
         "track_km", "sample_idx", "pixarea_km2", "frp_mw", "conf_raw",
         "conf_num", "type_raw", "version_raw", "ingest_ts", "dn_raw"))
    )


def parse_vxx14imgml(
    csv_text: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    filename: str = "",
    ingest_ts: Optional[str] = None,
    data_tier: str = "ML_SCIENCE",
) -> pd.DataFrame:
    """Parse VNP14IMGML / VJ114IMGML / VJ214IMGML comma-separated rows.

    Header: ``YYYYMMDD, HHMM, Sat, Lat, Lon, T_I4, T_I5, Sample, Pixarea,
    FRP, Conf, Type``; the platform is inferred from the filename prefix when
    ``Sat`` is missing or non-standard.
    """
    if df is None:
        df = pd.read_csv(io.StringIO(csv_text))
    ts = ingest_ts or datetime.utcnow().isoformat() + "Z"
    file_platform = platform_from_viirs_filename(filename)
    out = pd.DataFrame(
        {
            "country_iso": None,
            "sensor_family": "VIIRS",
            "platform": [normalize_satellite(v) or file_platform
                         for v in df.get("Sat", pd.Series([None] * len(df)))],
            "data_tier": data_tier,
            "latitude": df["Lat"].astype(float),
            "longitude": df["Lon"].astype(float),
            "acq_date": [
                f"{str(v)[:4]}-{str(v)[4:6]}-{str(v)[6:8]}"
                for v in df["YYYYMMDD"]
            ],
            "acq_hhmm": df["HHMM"].astype(int),
            "t_mir_k": df["T_I4"].astype(float),
            "t_tir_k": df["T_I5"].astype(float),
            "scan_km": np.nan,
            "track_km": np.nan,
            "sample_idx": df["Sample"].astype(float).astype("Int64"),
            "pixarea_km2": df["Pixarea"].astype(float),
            "frp_mw": df["FRP"].astype(float),
            "conf_raw": df["Conf"].astype(str),
            "conf_num": [parse_conf(c)[1] for c in df["Conf"]],
            "type_raw": [
                int(float(t)) if str(t).strip() not in ("", "nan") else -1
                for t in df["Type"]
            ],
            "version_raw": filename or "Vxx14IMGML",
            "ingest_ts": ts,
        }
    )
    return out


def parse_firms_api_csv(
    df: pd.DataFrame, ingest_ts: Optional[str] = None
) -> pd.DataFrame:
    """Parse NASA FIRMS /api/area/csv rows (NRT tier, type missing -> -1)."""
    ts = ingest_ts or datetime.utcnow().isoformat() + "Z"
    viirs = "bright_ti4" in df.columns
    t_mir = df["bright_ti4"] if viirs else df["brightness"]
    t_tir = df["bright_ti5"] if viirs else df["bright_t31"]
    conf_raw = df["confidence"].astype(str)
    return pd.DataFrame(
        {
            "country_iso": None,
            "sensor_family": "VIIRS" if viirs else "MODIS",
            "platform": [normalize_satellite(v) for v in df["satellite"]],
            "data_tier": "NRT",
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
            "acq_date": df["acq_date"].astype(str),
            "acq_hhmm": [
                int(float(str(v).replace(":", "")))
                for v in df["acq_time"]
            ],
            "t_mir_k": t_mir.astype(float),
            "t_tir_k": t_tir.astype(float),
            "scan_km": df["scan"].astype(float),
            "track_km": df["track"].astype(float),
            "sample_idx": pd.array([pd.NA] * len(df), dtype="Int64"),
            "pixarea_km2": np.nan,
            "frp_mw": df["frp"].astype(float),
            "conf_raw": conf_raw,
            "conf_num": [parse_conf(c)[1] for c in conf_raw],
            "type_raw": -1,  # NRT feeds drop the type column entirely
            "version_raw": df.get("version", pd.Series(["NRT"] * len(df)))
            .astype(str),
            "ingest_ts": ts,
        }
    )


def parse_portolan(
    df: pd.DataFrame, ingest_ts: Optional[str] = None
) -> pd.DataFrame:
    """Parse Source Cooperative Portolan FIRMS-catalog GeoParquet rows."""
    ts = ingest_ts or datetime.utcnow().isoformat() + "Z"
    quality = df.get("quality", pd.Series(["science"] * len(df))).astype(str)
    tier = np.where(quality.str.lower() == "nrt", "NRT", "SP_SCIENCE")
    conf_raw = df.get("confidence", pd.Series(["n"] * len(df))).astype(str)
    conf_pct = df.get("confidence_pct", pd.Series([np.nan] * len(df)))
    conf_num = [
        parse_conf(c)[1] for c in conf_raw
    ]
    # confidence_pct (numeric) is authoritative when present.
    conf_num = [
        float(p) if (pd.notna(p) and str(p) != "None" and float(p) > 0) else cn
        for p, cn in zip(conf_pct, conf_num)
    ]
    t_mir = df["bright_ti4"] if "bright_ti4" in df.columns else df["brightness"]
    t_tir = df["bright_ti5"] if "bright_ti5" in df.columns else df["bright_t31"]
    type_raw = [
        int(float(t)) if str(t).strip() not in ("", "None", "nan") else -1
        for t in df.get("type", pd.Series([-1] * len(df)))
    ]
    return pd.DataFrame(
        {
            "country_iso": None,
            "sensor_family": df["sensor"].astype(str).str.upper(),
            "platform": [normalize_satellite(v) for v in df["satellite"]],
            "data_tier": tier,
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
            "acq_date": df["acq_date"].astype(str),
            "acq_hhmm": [
                int(float(str(v).replace(":", "")))
                for v in df["acq_datetime"].astype(str).str.slice(11, 16)
            ],
            "t_mir_k": t_mir.astype(float),
            "t_tir_k": t_tir.astype(float),
            "scan_km": df["scan"].astype(float),
            "track_km": df["track"].astype(float),
            "sample_idx": pd.array([pd.NA] * len(df), dtype="Int64"),
            "pixarea_km2": np.nan,
            "frp_mw": df["frp"].astype(float),
            "conf_raw": conf_raw,
            "conf_num": conf_num,
            "type_raw": type_raw,
            "version_raw": df.get("version", pd.Series(["2.0"] * len(df)))
            .astype(str),
            "ingest_ts": ts,
        }
    )


# --------------------------------------------------------------------------- #
# Polygon clipping (exact Natural Earth Admin-0 containment)
# --------------------------------------------------------------------------- #
def clip_to_countries(
    df: pd.DataFrame, country_polygons: pd.DataFrame
) -> pd.DataFrame:
    """Assign ``country_iso`` and keep only points strictly inside polygons.

    ``country_polygons`` must contain ``country_iso`` and ``geometry_wkb``
    columns (as produced by download_data.download_country_polygons).
    Uses ``shapely.contains_xy`` for exact Admin-0 containment, excluding
    neighbouring countries that share the prefilter bounding boxes.
    """
    from shapely import contains_xy
    from shapely import prepare as shapely_prepare
    from shapely import wkb as shapely_wkb

    if df.empty:
        df = df.copy()
        df["country_iso"] = pd.Series(dtype="object")
        return df

    lons = df["longitude"].to_numpy(dtype=float)
    lats = df["latitude"].to_numpy(dtype=float)
    assigned = np.full(len(df), "", dtype=object)
    for _, row in country_polygons.iterrows():
        geom = shapely_wkb.loads(row["geometry_wkb"])
        shapely_prepare(geom)  # in-place spatial index acceleration
        minx, miny, maxx, maxy = geom.bounds
        cand = (
            (lons >= minx) & (lons <= maxx) & (lats >= miny) & (lats <= maxy)
            & (assigned == "")
        )
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        inside = contains_xy(geom, lons[idx], lats[idx])
        assigned[idx[inside]] = row["country_iso"]
    keep = assigned != ""
    out = df.loc[keep].copy()
    out["country_iso"] = assigned[keep]
    return out.reset_index(drop=True)


def prefilter_boxes_mask(df: pd.DataFrame) -> np.ndarray:
    """Cheap rectangular prefilter over the union of country boxes."""
    mask = np.zeros(len(df), dtype=bool)
    lons = df["longitude"].to_numpy(dtype=float)
    lats = df["latitude"].to_numpy(dtype=float)
    for spec in S.TARGET_COUNTRIES.values():
        for (min_lon, min_lat, max_lon, max_lat) in spec.prefilter_boxes:
            mask |= (
                (lons >= min_lon) & (lons <= max_lon)
                & (lats >= min_lat) & (lats <= max_lat)
            )
    return mask


# --------------------------------------------------------------------------- #
# Tier dominance & deduplication
# --------------------------------------------------------------------------- #
def apply_tier_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """Science-tier dominance + intra-tier exact-duplicate resolution.

    1. Drop every NRT row whose ``(platform, acq_date)`` is covered by an
       ML_SCIENCE or SP_SCIENCE row for the same platform (removing the
       40-300 m geolocation-LUT-shift duplicates).
    2. Resolve intra-tier duplicates keyed on ``(platform, acq_date, acq_hhmm,
       ROUND(lat,3), ROUND(lon,3))`` keeping ``tier_rank ASC, ingest_ts DESC``
       (deterministic row order as final tie-break).
    """
    if df.empty:
        return df.copy()
    df = df.copy()
    df["_tier_rank"] = df["data_tier"].map(S.TIER_RANK)
    science_keys = set(
        map(
            tuple,
            df.loc[df["_tier_rank"] < S.TIER_RANK["NRT"], ["platform", "acq_date"]]
            .to_numpy(),
        )
    )
    if science_keys:
        key_pairs = list(zip(df["platform"], df["acq_date"]))
        is_nrt = df["data_tier"] == "NRT"
        drop_mask = np.array(
            [
                nrt and (p, d) in science_keys
                for nrt, (p, d) in zip(is_nrt, key_pairs)
            ]
        )
        df = df.loc[~drop_mask]
    df["_lat3"] = df["latitude"].round(3)
    df["_lon3"] = df["longitude"].round(3)
    df["_row_id"] = np.arange(len(df))
    df = df.sort_values(
        ["_tier_rank", "ingest_ts", "_row_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    dedup_keys = ["platform", "acq_date", "acq_hhmm", "_lat3", "_lon3"]
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    return df.drop(columns=["_tier_rank", "_lat3", "_lon3", "_row_id"]) \
        .reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Platform operational activity (spacecraft safe-mode masking)
# --------------------------------------------------------------------------- #
def compute_platform_active_day(df: pd.DataFrame) -> pd.DataFrame:
    """``platform_active_day[p, d]``: active iff >= 5 detections on that UTC
    day across all six target countries."""
    counts = (
        df.groupby(["platform", "acq_date"]).size().reset_index(name="n_det")
    )
    counts["platform_active"] = (
        counts["n_det"] >= S.PLATFORM_ACTIVE_DAY_MIN_DETECTIONS
    ).astype(np.int8)
    return counts


# --------------------------------------------------------------------------- #
# Coverage audit (readme.md 6.3 item 6)
# --------------------------------------------------------------------------- #
CHILECITO_BBOX = S.SPOTLIGHT_PRESETS["Chilecito ARG 2026"]["bbox"]


def coverage_audit(df: pd.DataFrame) -> Dict[str, object]:
    """Mandatory coverage checks over the normalized archive."""
    audit: Dict[str, object] = {}
    arg = df[df["country_iso"] == "ARG"] if "country_iso" in df else df.iloc[0:0]
    chilecito = arg[
        (arg["latitude"] >= CHILECITO_BBOX[1])
        & (arg["latitude"] <= CHILECITO_BBOX[3])
        & (arg["longitude"] >= CHILECITO_BBOX[0])
        & (arg["longitude"] <= CHILECITO_BBOX[2])
        & arg["acq_date"].between("2026-07-01", "2026-08-31")
    ]
    audit["chilecito_jul_aug_2026_detections"] = int(len(chilecito))
    snpp = df[df["platform"] == "SNPP"]
    audit["snpp_last_date"] = (
        str(snpp["acq_date"].max()) if len(snpp) else None
    )
    audit["snpp_ends_2026_11_02"] = bool(
        len(snpp) and str(snpp["acq_date"].max()) <= "2026-11-02"
    )
    for plat in ("NOAA-20", "NOAA-21"):
        sub = df[df["platform"] == plat]
        audit[f"{plat.lower().replace('-', '_')}_through_2026_11_14"] = bool(
            len(sub) and str(sub["acq_date"].max()) >= "2026-11-14"
        )
    dza_flare = df[
        (df["country_iso"] == "DZA") & (df["type_raw"] == 2)
    ]
    audit["dza_static_flares_preserved"] = int(len(dza_flare))
    audit["total_rows"] = int(len(df))
    return audit


# --------------------------------------------------------------------------- #
# Parquet output (country, year) partitions
# --------------------------------------------------------------------------- #
def write_active_fires(
    df: pd.DataFrame, out_dir, compression: str = "zstd"
) -> int:
    """Write Zstd Parquet partitioned by ``(country_iso, year)``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path_wrap(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["year"] = df["acq_date"].str.slice(0, 4).astype(int)
    n = 0
    for (iso, year), part in df.groupby(["country_iso", "year"]):
        part_out = out_dir / f"country_iso={iso}" / f"year={year}"
        part_out.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(part, preserve_index=False)
        pq.write_table(
            table, part_out / "part-0.parquet", compression=compression
        )
        n += len(part)
    return n


def Path_wrap(path):
    from pathlib import Path

    return path if isinstance(path, Path) else Path(path)


def run_normalization(
    raw_frames: Sequence[pd.DataFrame],
    country_polygons: pd.DataFrame,
    out_dir=None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Normalize + clip + dedup a batch of already-parsed canonical frames."""
    if raw_frames:
        df = pd.concat(list(raw_frames), ignore_index=True)
    else:
        df = pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = df[prefilter_boxes_mask(df).astype(bool)] if len(df) else df
    df = clip_to_countries(df, country_polygons)
    df = apply_tier_dominance(df)
    audit = coverage_audit(df)
    active = compute_platform_active_day(df)
    audit["platforms_active_days"] = int(active["platform_active"].sum())
    if out_dir is not None:
        audit["written_rows"] = write_active_fires(df, out_dir)
    return df, audit
