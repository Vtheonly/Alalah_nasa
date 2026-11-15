"""Detection hygiene, bow-tie weighting, event clustering, pass aggregation.

Contract (readme Sections 4.6, 5.1.3):
* ``type_raw IN (1, 3)`` drop (volcano/offshore), SAMA nighttime low-conf
  filter, ``H3-Res8`` persistent industrial gas-flare mask
  (``is_static_industrial``).
* Single-platform overpass segmentation (Delta t <= 5 min), ECEF ``cKDTree``
  bow-tie overlap weighting (``omega_bt = 1/(1+deg)`` for MODIS pixels with
  ``A > 1.5 km^2``) and physical fire-event clustering
  (``d_ij <= 1.4 max(sqrt(A_i), sqrt(A_j))``) via
  ``csgraph.connected_components``.
* ``i4_m13_fallback`` vs Tobit ``sat_flag`` classification.
* Outputs **Product 1 base** (``obs_provenance_base.parquet``) and
  **``pass_observations.parquet``** keyed strictly by
  ``(h3_index, solar_date, platform, dn_flag)`` -- never summing platforms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..config import settings as S
from .h3_utils import latlon_to_h3_int
from .solar_geometry import (compute_solar_time_and_sza, hhmm_to_hours,
                             latlon_to_ecef, pixel_geometry_from_row)

PASS_COLUMNS = [
    "h3_index", "solar_date", "platform", "dn_flag", "sensor_family",
    "country_iso", "y_frp_mw", "raw_event_count", "det_footprint_km2",
    "mean_area_km2", "mean_vza_deg", "mean_solar_hour", "mean_conf_num",
    "has_i4_m13_fallback", "sat_flag", "w_mean", "n_pixels",
]

PROV_COLUMNS = [
    "platform", "data_tier", "acq_date", "acq_hhmm", "frp_mw", "t_mir_k",
    "t_tir_k", "scan_km", "track_km", "conf_raw", "type_raw", "ingest_ts",
    "latitude", "longitude", "sensor_family", "country_iso", "conf_num",
    "version_raw", "dn_source",
    "h3_index", "sza_deg", "solar_hour", "solar_date", "dn_flag",
    "pixel_area_km2", "vza_deg", "is_static_industrial", "is_sama_filtered",
    "bowtie_weight", "event_cluster_id", "pass_id", "i4_m13_fallback",
    "sat_flag", "is_biomass", "w_rel",
]


# ---------------------------------------------------------------------------
# Geometry & solar enrichment
# ---------------------------------------------------------------------------
def enrich_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add solar time/SZA/dn_flag, pixel geometry, and H3 index columns."""
    out = df.copy()
    solar_hour, solar_date, dn_flag, sza_deg = compute_solar_time_and_sza(
        out["acq_date"].to_numpy(dtype="datetime64[D]"),
        out["acq_hhmm"].to_numpy(dtype=np.int64),
        out["longitude"].to_numpy(dtype=np.float64),
        out["latitude"].to_numpy(dtype=np.float64))
    out["solar_hour"] = solar_hour
    out["solar_date"] = solar_date
    out["dn_flag"] = dn_flag
    out["sza_deg"] = sza_deg
    area, vza, scan, track = pixel_geometry_from_row(
        out.get("sample_idx"), out.get("pixarea_km2"),
        out.get("scan_km"), out.get("track_km"), out["sensor_family"])
    out["pixel_area_km2"] = area
    out["vza_deg"] = vza
    out["scan_km_resolved"] = scan
    out["track_km_resolved"] = track
    out["h3_index"] = latlon_to_h3_int(out["latitude"].to_numpy(),
                                       out["longitude"].to_numpy(),
                                       S.H3_MAIN_RES)
    return out


# ---------------------------------------------------------------------------
# SAMA + static flare hygiene (INV-08)
# ---------------------------------------------------------------------------
def sama_filter_mask(df: pd.DataFrame) -> np.ndarray:
    """VIIRS nighttime low-confidence detections inside the SAMA box."""
    is_viirs = df["sensor_family"].to_numpy() == "VIIRS"
    is_night = df["dn_flag"].to_numpy() == "N"
    conf_raw = df["conf_raw"].astype(str).str.lower().to_numpy()
    conf_num = df["conf_num"].fillna(0.0).to_numpy()
    low_conf = (conf_raw == "l") | (conf_num < 30.0)
    lon = df["longitude"].to_numpy()
    lat = df["latitude"].to_numpy()
    in_box = ((lon >= S.SAMA_BOX[0]) & (lon <= S.SAMA_BOX[2])
              & (lat >= S.SAMA_BOX[1]) & (lat <= S.SAMA_BOX[3]))
    return is_viirs & is_night & low_conf & in_box


def static_flare_mask(df: pd.DataFrame, res: int = S.H3_FLARE_RES) -> np.ndarray:
    """H3-Res8 persistent industrial flare mask.

    Robust to 50-300 m MODIS geolocation jitter: Res-8 micro-cells
    (~0.74 km^2) absorb off-nadir geolocation drift.  ``is_static_industrial``
    is True for type_raw == 2 or (Y_m >= 3 years AND mean active days/year
    >= 15 AND day/night symmetry R_DN > 0.30).

    Fully vectorized (factorize + bincount): the former per-cell Python
    lambdas and per-row ``strftime`` took hours at tens of millions of
    detections; this is O(n log n) with integer keys only.
    """
    n = len(df)
    type2 = df["type_raw"].to_numpy() == 2
    if n == 0:
        return type2
    h3r8 = latlon_to_h3_int(df["latitude"].to_numpy(),
                            df["longitude"].to_numpy(), res)
    day = (pd.to_datetime(df["solar_date"]).to_numpy()
           .astype("datetime64[D]").astype(np.int64))
    dn = df["dn_flag"].to_numpy().astype(str)
    static_ids = static_flare_cells(h3r8, day, dn)
    return type2 | np.isin(h3r8, static_ids)


def static_flare_cells(h3r8: np.ndarray, day: np.ndarray,
                       dn: np.ndarray) -> np.ndarray:
    """Res-8 cell ids meeting the static-industrial rule from per-detection
    arrays (``day``: integer solar day, ``dn``: 'D'/'N').  Separate so the
    rule can be evaluated over ALL years from lean arrays, then applied to
    one partition at a time (:func:`preprocess_partitioned`)."""
    if len(h3r8) == 0:
        return np.array([], dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    year = (day.astype("datetime64[D]").astype("datetime64[Y]")
            .astype(np.int64))
    dn = np.asarray(dn).astype(str)
    cell, ucell = pd.factorize(np.asarray(h3r8), sort=False)
    n_cells = len(ucell)
    # distinct (cell, day) and (cell, year) pairs -> per-cell counts
    cd = np.unique(cell.astype(np.int64) * (1 << 32) + (day - day.min()))
    active_days = np.bincount((cd >> 32).astype(np.int64), minlength=n_cells)
    cy = np.unique(cell.astype(np.int64) * (1 << 16) + (year - year.min()))
    active_years = np.bincount((cy >> 16).astype(np.int64), minlength=n_cells)
    n_d = np.bincount(cell, weights=(dn == "D").astype(float),
                      minlength=n_cells)
    n_n = np.bincount(cell, weights=(dn == "N").astype(float),
                      minlength=n_cells)
    mean_days_per_year = active_days / np.maximum(1, active_years)
    r_dn = np.minimum(n_d, n_n) / (np.maximum(n_d, n_n) + 1.0)
    static = ((active_years >= S.FLARE_MIN_YEARS)
              & (mean_days_per_year >= S.FLARE_MIN_DAYS_PER_YEAR)
              & (r_dn > S.FLARE_MIN_DN_SYMMETRY))
    return np.asarray(ucell, dtype=np.int64)[static]


def _static_flare_mask_reference(df: pd.DataFrame,
                                 res: int = S.H3_FLARE_RES) -> np.ndarray:
    """Original groupby/lambda implementation (kept for equivalence tests)."""
    h3r8 = latlon_to_h3_int(df["latitude"].to_numpy(),
                            df["longitude"].to_numpy(), res)
    dates = pd.to_datetime(df["solar_date"])
    grp = pd.DataFrame({"m": h3r8, "year": dates.dt.year,
                        "day": dates.dt.strftime("%Y-%m-%d"),
                        "dn": df["dn_flag"].to_numpy()})
    per_cell = grp.groupby("m").agg(
        active_days=("day", "nunique"),
        active_years=("year", "nunique"),
        n_D=("dn", lambda s: int((s == "D").sum())),
        n_N=("dn", lambda s: int((s == "N").sum())))
    per_cell["mean_days_per_year"] = per_cell["active_days"] / np.maximum(
        1, per_cell["active_years"])
    per_cell["r_dn"] = (per_cell[["n_D", "n_N"]].min(axis=1)
                        / (per_cell[["n_D", "n_N"]].max(axis=1) + 1.0))
    static_cells = set(
        per_cell.index[(per_cell["active_years"] >= S.FLARE_MIN_YEARS)
                       & (per_cell["mean_days_per_year"]
                          >= S.FLARE_MIN_DAYS_PER_YEAR)
                       & (per_cell["r_dn"] > S.FLARE_MIN_DN_SYMMETRY)])
    type2 = df["type_raw"].to_numpy() == 2
    return type2 | np.isin(h3r8, list(static_cells) if static_cells else [-1])


# ---------------------------------------------------------------------------
# Pass segmentation + intra-pass graphs
# ---------------------------------------------------------------------------
def pass_epoch_minutes(df: pd.DataFrame) -> np.ndarray:
    """UTC epoch minutes from acq_date + base-60 parsed acq_hhmm."""
    days = (pd.to_datetime(df["acq_date"]) -
            pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=np.float64)
    hours = np.asarray(hhmm_to_hours(df["acq_hhmm"].to_numpy(dtype=np.int64)),
                       dtype=np.float64)
    return days * 1440.0 + hours * 60.0


def segment_passes(df: pd.DataFrame) -> pd.ndarray:
    """Assign ``pass_id``: sorted per platform, split where consecutive
    timestamps differ by more than 5 minutes."""
    order = np.lexsort((pass_epoch_minutes(df), df["platform"].to_numpy()))
    epoch = pass_epoch_minutes(df)[order]
    plat_sorted = df["platform"].to_numpy()[order]
    gap = np.diff(epoch)
    new_pass = np.ones(len(df), dtype=bool)
    new_pass[1:] = (gap > S.FIVE_MINUTES) | (plat_sorted[1:] != plat_sorted[:-1])
    pass_ids_sorted = np.cumsum(new_pass) - 1
    pass_ids = np.empty(len(df), dtype=np.int64)
    pass_ids[order] = pass_ids_sorted
    return pass_ids


def cluster_pass(ecef: np.ndarray, areas: np.ndarray, tracks: np.ndarray,
                 sensor_family: str):
    """Bow-tie weights + connected-component event clustering for one pass.

    * Bow-tie graph (MODIS pixels with A_i > 1.5 km^2): connect i, j if
      d_ij < 0.7 * (track_i + track_j)/2; omega_i = 1 / (1 + deg_i).
    * Event graph: connect i, j if d_ij <= 1.4 * max(sqrt(A_i), sqrt(A_j)).
    Returns (omega_bt array, event labels array).
    """
    n = len(ecef)
    omega = np.ones(n, dtype=np.float64)
    labels = np.arange(n, dtype=np.int64)
    if n < 2:
        return omega, labels
    r_event = float(1.4 * np.sqrt(max(areas.max(), 1e-6)))
    tree = cKDTree(ecef)
    pairs = tree.query_pairs(r=max(r_event, 1.0 * S.BOWTIE_AREA_KM2), output_type="ndarray")
    # Vectorized over all candidate pairs (the former per-pair Python loop
    # was the hot spot at tens of millions of pixels); same edge rules.
    pi, pj = pairs[:, 0], pairs[:, 1]
    diff = ecef[pi] - ecef[pj]
    dij = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    sq = np.sqrt(areas)
    # ---- event clustering graph -----------------------------------------
    ev = dij <= S.CLUSTER_D_FACTOR * np.maximum(sq[pi], sq[pj])
    if ev.any():
        g = coo_matrix((np.ones(int(ev.sum())), (pi[ev], pj[ev])), shape=(n, n))
        ncomp, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(n, dtype=np.int64)
    # ---- bow-tie graph (MODIS, A > 1.5 km^2) -----------------------------
    if str(sensor_family).upper().startswith("MOD"):
        big = areas > S.BOWTIE_AREA_KM2
        if big.any():
            thr = S.BOWTIE_TRACK_FACTOR * 0.5 * (tracks[pi] + tracks[pj])
            e = big[pi] & big[pj] & (dij < thr)
            deg = (np.bincount(pi[e], minlength=n)
                   + np.bincount(pj[e], minlength=n))
            omega = 1.0 / (1.0 + deg.astype(np.float64))
    return omega, labels


# ---------------------------------------------------------------------------
# Saturation classification (readme 5.1.3.3)
# ---------------------------------------------------------------------------
def _cluster_pass_reference(ecef, areas, tracks, sensor_family):
    """Original per-pair loop implementation (equivalence tests only)."""
    n = len(ecef)
    omega = np.ones(n, dtype=np.float64)
    labels = np.arange(n, dtype=np.int64)
    if n < 2:
        return omega, labels
    r_event = float(1.4 * np.sqrt(max(areas.max(), 1e-6)))
    tree = cKDTree(ecef)
    pairs = tree.query_pairs(r=max(r_event, 1.0 * S.BOWTIE_AREA_KM2), output_type="ndarray")
    # ---- event clustering graph -----------------------------------------
    ev_src, ev_dst = [], []
    for i, j in pairs:
        dij = float(np.linalg.norm(ecef[i] - ecef[j]))
        if dij <= S.CLUSTER_D_FACTOR * max(np.sqrt(areas[i]), np.sqrt(areas[j])):
            ev_src.append(i)
            ev_dst.append(j)
    if ev_src:
        g = coo_matrix((np.ones(len(ev_src)), (ev_src, ev_dst)), shape=(n, n))
        ncomp, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(n, dtype=np.int64)
    # ---- bow-tie graph (MODIS, A > 1.5 km^2) -----------------------------
    if str(sensor_family).upper().startswith("MOD"):
        big = areas > S.BOWTIE_AREA_KM2
        if big.any():
            deg = np.zeros(n, dtype=np.int64)
            for i, j in pairs:
                if big[i] and big[j]:
                    dij = float(np.linalg.norm(ecef[i] - ecef[j]))
                    thr = S.BOWTIE_TRACK_FACTOR * 0.5 * (tracks[i] + tracks[j])
                    if dij < thr:
                        deg[i] += 1
                        deg[j] += 1
            omega = 1.0 / (1.0 + deg.astype(np.float64))
    return omega, labels


def saturation_flags(sensor_family, t_mir_k, frp_mw):
    """``i4_m13_fallback`` (VIIRS M13 retrieval; sigma x 1.35) and Tobit
    right-censored ``sat_flag``."""
    fam = np.asarray(sensor_family)
    t = np.asarray(t_mir_k, dtype=np.float64)
    frp = np.asarray(frp_mw, dtype=np.float64)
    is_viirs = fam == "VIIRS"
    is_modis = fam == "MODIS"
    m13fb = (is_viirs & (t >= S.VIIRS_I4_M13FB_LO) & (frp > 0.0)).astype(np.int8)
    sat = ((is_modis & (t >= S.MODIS_SAT_K) & (frp > 0.0))
           | (is_viirs & (t >= S.VIIRS_I4_M13FB_HI)
              & (frp >= S.VIIRS_TOBIT_FRP_MIN))).astype(np.int8)
    return m13fb, sat


# ---------------------------------------------------------------------------
# Full preprocessing
# ---------------------------------------------------------------------------
def preprocess_observations(df: pd.DataFrame, static_cells=None
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical detections -> (Product 1 base, pass_observations).

    ``static_cells``: Res-8 static-industrial cell ids computed over the
    whole archive (partitioned runs); None -> computed from ``df``."""
    if df.empty:
        prov = pd.DataFrame(columns=PROV_COLUMNS)
        pas = pd.DataFrame(columns=PASS_COLUMNS)
        return prov, pas
    out = enrich_geometry(df)

    # ---- hygiene flags (kept in Product 1, excluded from biomass fusion)
    out["is_sama_filtered"] = sama_filter_mask(out)
    if static_cells is None:
        out["is_static_industrial"] = static_flare_mask(out)
    else:
        h3r8 = latlon_to_h3_int(out["latitude"].to_numpy(),
                                out["longitude"].to_numpy(), S.H3_FLARE_RES)
        out["is_static_industrial"] = ((out["type_raw"].to_numpy() == 2)
                                       | np.isin(h3r8, static_cells))
    type_excluded = out["type_raw"].isin([1, 3]).to_numpy()
    out["is_biomass"] = (~(type_excluded | out["is_sama_filtered"].to_numpy()
                           | out["is_static_industrial"].to_numpy()))

    # ---- saturation flags
    m13fb, sat = saturation_flags(out["sensor_family"], out["t_mir_k"],
                                  out["frp_mw"])
    out["i4_m13_fallback"] = m13fb
    out["sat_flag"] = sat
    out["w_rel"] = 1.0                 # reliability weight (set by operators)
    out["bowtie_weight"] = 1.0
    out["event_cluster_id"] = -1
    out["pass_id"] = -1

    bio = out[out["is_biomass"]].copy()
    if len(bio):
        bio["pass_id"] = segment_passes(bio)
        bio = bio.sort_values(["platform", "pass_id"], kind="mergesort")
        ecef = latlon_to_ecef(bio["latitude"].to_numpy(),
                              bio["longitude"].to_numpy())
        areas = bio["pixel_area_km2"].fillna(1.0).to_numpy()
        tracks = bio["track_km_resolved"].fillna(1.0).to_numpy()
        omega_all = np.ones(len(bio))
        labels_all = np.arange(len(bio))
        for (_plat, _pid), idx in bio.groupby(["platform", "pass_id"],
                                              sort=False).indices.items():
            ii = np.asarray(idx)
            om, lab = cluster_pass(ecef[ii], areas[ii], tracks[ii],
                                   bio["sensor_family"].iloc[ii[0]]
                                   if len(ii) else "MODIS")
            omega_all[ii] = om
            labels_all[ii] = lab
        bio["bowtie_weight"] = omega_all
        # globally unique event ids
        bio["event_cluster_id"] = (bio["pass_id"].to_numpy() * (len(bio) + 1)
                                   + labels_all)
        out.loc[bio.index, "bowtie_weight"] = bio["bowtie_weight"]
        out.loc[bio.index, "event_cluster_id"] = bio["event_cluster_id"]
        out.loc[bio.index, "pass_id"] = bio["pass_id"]

    prov = out[[c for c in PROV_COLUMNS if c in out.columns]].copy()
    prov["solar_date"] = pd.to_datetime(prov["solar_date"])

    pas = _aggregate_passes(bio)
    return prov, pas


def _active_fire_files_by_year(root) -> dict[int, list]:
    """``active_fires/country_iso=X/year=Y/*.parquet`` grouped by year."""
    import re
    from pathlib import Path
    groups: dict[int, list] = {}
    for f in sorted(Path(root).rglob("*.parquet")):
        m = re.search(r"year=(\d{4})", str(f))
        groups.setdefault(int(m.group(1)) if m else -1, []).append(f)
    return groups


def _read_parquets(files, columns=None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not files:
        return pd.DataFrame(columns=columns or [])
    tables = [pq.read_table(f, columns=columns, partitioning=None) for f in files]
    t = pa.concat_tables(tables, promote_options="permissive")
    del tables
    return t.to_pandas(self_destruct=True, split_blocks=True,
                       deduplicate_objects=True)


def preprocess_partitioned(active_fires_dir, prov_dir) -> tuple[pd.DataFrame, dict]:
    """Bounded-memory :func:`preprocess_observations` over the whole
    ``active_fires/`` archive (tens of millions of detections).

    1. Lean pass over every partition (5 columns) -> the Res-8
       static-industrial cell set over ALL years (the rule needs >= 3 years).
    2. One UTC year at a time (all countries): full preprocess with that
       global cell set; Product-1 base rows written to
       ``prov_dir/year=YYYY.parquet``; pass ids / event cluster ids offset so
       they stay globally unique.  Passes last minutes, so a year boundary
       can only split a pass that straddles 00:00 UTC on 1 January.
    Returns ``(pass_obs, summary)``; the provenance is NOT held in memory.
    """
    import shutil
    from pathlib import Path
    groups = _active_fire_files_by_year(active_fires_dir)
    lean_cols = ["latitude", "longitude", "acq_date", "acq_hhmm", "type_raw"]
    cells, days, dns = [], [], []
    for year in sorted(groups):
        d = _read_parquets(groups[year], lean_cols)
        if not len(d):
            continue
        _sh, sdate, dn, _sza = compute_solar_time_and_sza(
            d["acq_date"].to_numpy(dtype="datetime64[D]"),
            d["acq_hhmm"].to_numpy(dtype=np.int64),
            d["longitude"].to_numpy(dtype=np.float64),
            d["latitude"].to_numpy(dtype=np.float64))
        cells.append(latlon_to_h3_int(d["latitude"].to_numpy(),
                                      d["longitude"].to_numpy(),
                                      S.H3_FLARE_RES))
        days.append(np.asarray(sdate, dtype="datetime64[D]").astype(np.int64))
        dns.append(np.asarray(dn).astype("U1"))
        del d
    static_ids = (static_flare_cells(np.concatenate(cells),
                                     np.concatenate(days),
                                     np.concatenate(dns))
                  if cells else np.array([], dtype=np.int64))
    del cells, days, dns
    prov_dir = Path(prov_dir)
    if prov_dir.exists():
        shutil.rmtree(prov_dir) if prov_dir.is_dir() else prov_dir.unlink()
    prov_dir.mkdir(parents=True)
    pass_parts, n_rows, pass_off, event_off = [], 0, 0, 0
    for year in sorted(groups):
        df = _read_parquets(groups[year])
        df = df.drop(columns=[c for c in ("year",) if c in df.columns])
        if not len(df):
            continue
        prov, pas = preprocess_observations(df, static_cells=static_ids)
        del df
        bio = prov["pass_id"].to_numpy() >= 0
        if bio.any():
            pid = prov["pass_id"].to_numpy(np.int64)
            eid = prov["event_cluster_id"].to_numpy(np.int64)
            prov.loc[bio, "pass_id"] = pid[bio] + pass_off
            prov.loc[bio, "event_cluster_id"] = eid[bio] + event_off
            pass_off = int(prov.loc[bio, "pass_id"].max()) + 1
            event_off = int(prov.loc[bio, "event_cluster_id"].max()) + 1
        prov.to_parquet(prov_dir / f"year={year}.parquet", index=False,
                        compression="zstd")
        n_rows += len(prov)
        pass_parts.append(pas)
        del prov
    pass_obs = (pd.concat(pass_parts, ignore_index=True) if pass_parts
                else pd.DataFrame(columns=PASS_COLUMNS))
    key = ["h3_index", "solar_date", "platform", "dn_flag"]
    n_dup = int(pass_obs.duplicated(key).sum()) if len(pass_obs) else 0
    return pass_obs, {"provenance_rows": n_rows, "years": sorted(groups),
                      "static_flare_cells": int(len(static_ids)),
                      "duplicate_pass_keys_across_years": n_dup}


def _aggregate_passes(bio: pd.DataFrame) -> pd.DataFrame:
    """Aggregate biomass rows strictly per (h3_index, solar_date, platform,
    dn_flag) -- never summing platforms (INV-02)."""
    if not len(bio):
        return pd.DataFrame(columns=PASS_COLUMNS)
    b = bio.copy()
    b["w_frp"] = b["bowtie_weight"] * b["w_rel"] * b["frp_mw"].fillna(0.0)
    b["w_area"] = b["bowtie_weight"] * b["pixel_area_km2"].fillna(0.0)
    agg = b.groupby(["h3_index", "solar_date", "platform", "dn_flag"],
                    as_index=False, sort=True).agg(
        sensor_family=("sensor_family", "first"),
        country_iso=("country_iso", "first"),
        y_frp_mw=("w_frp", "sum"),
        raw_event_count=("event_cluster_id", "nunique"),
        det_footprint_km2=("w_area", "sum"),
        mean_area_km2=("pixel_area_km2", "mean"),
        mean_vza_deg=("vza_deg", "mean"),
        mean_solar_hour=("solar_hour", "mean"),
        mean_conf_num=("conf_num", "mean"),
        has_i4_m13_fallback=("i4_m13_fallback", "max"),
        sat_flag=("sat_flag", "max"),
        w_mean=("w_rel", "mean"),
        n_pixels=("frp_mw", "size"))
    agg["h3_index"] = agg["h3_index"].astype(np.int64)
    agg["solar_date"] = pd.to_datetime(agg["solar_date"])
    return agg


def run_preprocess(df: pd.DataFrame, paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess + persist Product 1 base and pass observations."""
    prov, pas = preprocess_observations(df)
    prov.to_parquet(paths.provenance_base_parquet, index=False)
    pas.to_parquet(paths.pass_obs_parquet, index=False)
    return prov, pas
