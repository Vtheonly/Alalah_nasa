"""Geometry preprocessing: hygiene masks, bow-tie weighting, clustering.

Contract (readme.md Section 4 & 5.1.3):
  * ``type_raw IN (1, 3)`` drop (volcano/offshore), SAMA nighttime
    low-confidence filter, and the ``H3-Res8`` persistent industrial /
    gas-flare mask (``is_static_industrial``) prior to biomass fusion
    (records retained in the Product 1 provenance base).
  * Single-platform overpass segmentation (Delta t <= 5 min), ECEF
    ``cKDTree`` bow-tie weighting (omega_bt = 1/(1+deg) for MODIS
    A_i > 1.5 km^2) and physical fire-event clustering
    (``event_cluster_id``) via ``csgraph.connected_components``.
  * ``i4_m13_fallback`` vs Tobit ``sat_flag`` classification.
  * Outputs ``obs_provenance_base.parquet`` and ``pass_observations.parquet``
    keyed strictly by ``(h3_index, solar_date, platform, dn_flag)`` —
    never summing same-orbit platforms (INV-02).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.config import settings as S
from src.geometry import h3_utils
from src.geometry.solar_geometry import (
    area_to_vza_deg,
    compute_solar_time_and_sza,
    modis_sample_to_geometry,
    viirs_sample_to_geometry,
)


# --------------------------------------------------------------------------- #
# ECEF coordinates
# --------------------------------------------------------------------------- #
def latlon_to_ecef(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """3D ECEF chords on a sphere of radius R_E = 6371 km."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    phi = np.deg2rad(lats)
    lam = np.deg2rad(lons)
    re = S.EARTH_RADIUS_KM
    return np.column_stack(
        (
            re * np.cos(phi) * np.cos(lam),
            re * np.cos(phi) * np.sin(lam),
            re * np.sin(phi),
        )
    )


# --------------------------------------------------------------------------- #
# Hygiene masks
# --------------------------------------------------------------------------- #
def volcano_offshore_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows to drop from biomass fusion: type_raw in (1, 3)."""
    return (df["type_raw"] == 1) | (df["type_raw"] == 3)


def sama_nighttime_lowconf_mask(df: pd.DataFrame) -> np.ndarray:
    """SAMA nighttime low-confidence VIIRS artifacts (readme.md 5.1.3).

    Sensor VIIRS AND dn_flag 'N' AND (conf_raw 'l' OR conf_num < 30) AND
    -110 <= lon <= 11 AND -55 <= lat <= 7.
    """
    lon_min, lat_min, lon_max, lat_max = S.SAMA_BOX
    in_box = (
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        & (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
    )
    low_conf = (df["conf_raw"].astype(str) == "l") | (
        pd.to_numeric(df["conf_num"], errors="coerce") < 30
    )
    return (
        (df["sensor_family"] == "VIIRS")
        & (df["dn_flag"] == "N")
        & low_conf
        & in_box
    ).to_numpy()


def flare_microcell_mask(df: pd.DataFrame) -> np.ndarray:
    """H3-Res8 persistent industrial/gas-flare detection (INV-08).

    Groups biomass rows into ~0.74 km^2 micro-cells; a cell is
    ``is_static_industrial`` when ``type_raw == 2`` anywhere inside it OR
    (distinct active years >= 3 AND mean active days/year >= 15 AND
    day/night symmetry R_DN > 0.30).
    """
    if "h3_flare8" in df.columns:
        flare_ids = df["h3_flare8"].to_numpy()
    else:
        # Standalone use: derive the H3-Res8 micro-cell ids directly.
        flare_ids = h3_utils.latlon_to_h3_int(
            df["latitude"].to_numpy(dtype=float),
            df["longitude"].to_numpy(dtype=float),
            S.H3_FLARE_RES,
        )
    years = df["solar_date"].str.slice(0, 4).astype(int).to_numpy()
    dn = (df["dn_flag"] == "D").to_numpy()
    type2 = (df["type_raw"] == 2).to_numpy()

    group = pd.DataFrame(
        {"flare_id": flare_ids, "year": years, "dn": dn, "type2": type2,
         "date": df["solar_date"].to_numpy()}
    )
    stats = group.groupby("flare_id").agg(
        n_years=("year", lambda s: s.nunique()),
        active_days=("date", "nunique"),
        n_d=("dn", "sum"),
        n=("dn", "size"),
        any_type2=("type2", "max"),
    )
    # Mean active days per year uses DISTINCT active years Y_m (readme.md
    # 5.1.3: D_bar_m = active_days_m / max(1, Y_m)).
    stats["days_per_year"] = stats["active_days"] / np.maximum(
        stats["n_years"], 1
    )
    stats["n_n"] = stats["n"] - stats["n_d"]
    stats["r_dn"] = np.minimum(stats["n_d"], stats["n_n"]) / (
        np.maximum(stats["n_d"], stats["n_n"]) + 1.0
    )
    stats["is_static"] = (
        stats["any_type2"].astype(bool)
        | (
            (stats["n_years"] >= S.FLARE_MIN_YEARS)
            & (stats["days_per_year"] >= S.FLARE_MIN_ACTIVE_DAYS_PER_YEAR)
            & (stats["r_dn"] > S.FLARE_MIN_DAY_NIGHT_RATIO)
        )
    )
    static_ids = set(stats.index[stats["is_static"]])
    return np.array([fid in static_ids for fid in flare_ids], dtype=bool)


# --------------------------------------------------------------------------- #
# Overpass segmentation
# --------------------------------------------------------------------------- #
def utc_epoch_minutes(acq_date: pd.Series, acq_hhmm: pd.Series) -> np.ndarray:
    """Absolute minutes since epoch for pass segmentation."""
    dates = pd.to_datetime(acq_date, format="%Y-%m-%d")
    hh = (acq_hhmm.to_numpy(dtype=int) // 100).astype(float)
    mm = (acq_hhmm.to_numpy(dtype=int) % 100).astype(float)
    return (
        dates.to_numpy().astype("datetime64[s]").astype(np.int64) / 60.0
        + hh * 60.0
        + mm
    )


def segment_passes(df: pd.DataFrame) -> pd.Series:
    """Split rows into overpasses: > 5 minutes gap starts a new pass.

    Rows are sorted by ``(platform, utc_epoch_minutes)``; ``pass_id``
    enumerates consecutive segments per platform.
    """
    minutes = utc_epoch_minutes(df["acq_date"], df["acq_hhmm"])
    order = np.lexsort((minutes, df["platform"].to_numpy()))
    sorted_minutes = minutes[order]
    sorted_platforms = df["platform"].to_numpy()[order]
    new_pass = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        gap = np.diff(sorted_minutes)
        same_platform = sorted_platforms[1:] == sorted_platforms[:-1]
        new_pass[1:] = ~(same_platform & (gap <= S.PASS_SPLIT_MINUTES))
    pass_sorted = np.cumsum(new_pass) - 1
    pass_id = np.empty(len(df), dtype=np.int64)
    pass_id[order] = pass_sorted
    return pd.Series(pass_id, index=df.index, name="pass_id")


# --------------------------------------------------------------------------- #
# Bow-tie weighting & event clustering (within one platform pass)
# --------------------------------------------------------------------------- #
def bowtie_weights(
    ecef: np.ndarray,
    areas: np.ndarray,
    tracks: np.ndarray,
    sensor_family: str,
) -> np.ndarray:
    """omega_bt = 1 / (1 + deg_bt) for overlapping MODIS off-nadir pixels.

    Only MODIS pixels with A_i > 1.5 km^2 participate; pixels i, j are
    connected when their ECEF chord distance is below
    ``0.7 * (track_i + track_j) / 2``. Two overlapping 40 MW edge pixels each
    receive weight 0.5, conserving total FRP (readme.md 5.1.3 item 1).
    """
    n = len(ecef)
    weights = np.ones(n, dtype=float)
    if n < 2 or sensor_family != "MODIS":
        return weights
    eligible = areas > S.BOWTIE_AREA_THRESHOLD_KM2
    if eligible.sum() < 2:
        return weights
    idx = np.flatnonzero(eligible)
    tree = cKDTree(ecef[idx])
    max_track = float(np.max(tracks[idx]))
    pairs = tree.query_pairs(r=S.BOWTIE_TRACK_FACTOR * max_track,
                             output_type="ndarray")
    deg = np.zeros(len(idx), dtype=np.int64)
    keep = []
    for a, b in pairs:
        d = float(np.linalg.norm(ecef[idx[a]] - ecef[idx[b]]))
        if d < S.BOWTIE_TRACK_FACTOR * 0.5 * (
            tracks[idx[a]] + tracks[idx[b]]
        ):
            keep.append((a, b))
    for a, b in keep:
        deg[a] += 1
        deg[b] += 1
    weights[idx] = 1.0 / (1.0 + deg)
    return weights


def cluster_events(
    ecef: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    """Connected-component physical fire-event clustering within a pass.

    Pixels i, j are connected when their ECEF chord distance is at most
    ``1.4 * max(sqrt(A_i), sqrt(A_j))``; components become
    ``event_cluster_id`` labels.
    """
    n = len(ecef)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    tree = cKDTree(ecef)
    max_reach = S.CLUSTER_DISTANCE_FACTOR * float(np.max(np.sqrt(areas)))
    pairs = tree.query_pairs(r=max_reach, output_type="ndarray")
    rows: List[int] = []
    cols: List[int] = []
    side = np.sqrt(areas)
    for a, b in pairs:
        reach = S.CLUSTER_DISTANCE_FACTOR * max(side[a], side[b])
        if float(np.linalg.norm(ecef[a] - ecef[b])) <= reach:
            rows.append(a)
            cols.append(b)
    if rows:
        g = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        )
        g = g + g.T
        n_comp, labels = connected_components(
            g, directed=False, connection="weak"
        )
    else:
        labels = np.arange(n, dtype=np.int64)
    return labels.astype(np.int64)


# --------------------------------------------------------------------------- #
# Saturation classification
# --------------------------------------------------------------------------- #
def classify_saturation(
    sensor_family: str, t_mir_k: float, frp_mw: float
) -> Tuple[int, int]:
    """Return ``(i4_m13_fallback, sat_flag)`` for one detection.

    ``i4_m13_fallback = 1`` when VIIRS and t_mir_k >= 366.0 and frp > 0
    (M13-retrieved FRP; observation noise sigma inflated by 1.35x).
    ``sat_flag = 1`` (Tobit right-censored) when MODIS and t_mir_k >= 498.0
    and frp > 0, or VIIRS and t_mir_k >= 366.9 and frp >= 300 MW.
    """
    fam = str(sensor_family).upper()
    i4_fallback = 0
    sat_flag = 0
    if fam == "VIIRS":
        if t_mir_k >= S.VIIRS_I4_SATURATION_K and frp_mw > 0.0:
            i4_fallback = 1
        if (
            t_mir_k >= S.VIIRS_I4_TOBIT_K and frp_mw >= S.VIIRS_TOBIT_MIN_FRP_MW
        ):
            sat_flag = 1
    elif fam == "MODIS":
        if t_mir_k >= S.MODIS_CH21_SATURATION_K and frp_mw > 0.0:
            sat_flag = 1
    return i4_fallback, sat_flag


# --------------------------------------------------------------------------- #
# Full preprocessing pipeline
# --------------------------------------------------------------------------- #
PROVENANCE_DERIVED_COLUMNS = (
    "h3_index", "h3_flare8", "sza_deg", "solar_hour", "solar_date", "dn_flag",
    "pixel_area_km2", "vza_deg", "pass_id", "bowtie_weight",
    "event_cluster_id", "i4_m13_fallback", "sat_flag",
    "is_static_industrial", "is_sama_filtered", "is_excluded_thermal",
    "reliability_prob", "biomass_eligible",
)


def enrich_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived geometry / solar columns to a canonical detection frame."""
    out = df.reset_index(drop=True).copy()
    lats = out["latitude"].to_numpy(dtype=float)
    lons = out["longitude"].to_numpy(dtype=float)

    solar_hour, solar_date, dn_flag, sza_deg = compute_solar_time_and_sza(
        out["acq_date"].to_numpy(), out["acq_hhmm"].to_numpy(), lons, lats
    )
    out["solar_hour"] = np.asarray(solar_hour, dtype=float)
    out["solar_date"] = np.asarray(solar_date, dtype=object)
    out["dn_flag"] = np.asarray(dn_flag, dtype=object)
    out["sza_deg"] = np.asarray(sza_deg, dtype=float)

    # Pixel geometry per row via the dispatching resolver.
    n = len(out)
    scan = np.full(n, np.nan)
    track = np.full(n, np.nan)
    vza = np.full(n, np.nan)
    area = np.full(n, np.nan)
    for i in range(n):
        sample = out["sample_idx"].iloc[i]
        sample_val = None if pd.isna(sample) else int(sample)
        pixarea = out["pixarea_km2"].iloc[i]
        pixarea_val = None if pd.isna(pixarea) else float(pixarea)
        s_km = out["scan_km"].iloc[i]
        t_km = out["track_km"].iloc[i]
        s_val = None if pd.isna(s_km) else float(s_km)
        t_val = None if pd.isna(t_km) else float(t_km)
        sc, tr, vz, ar = _resolve_row_geometry(
            out["sensor_family"].iloc[i], sample_val, pixarea_val, s_val, t_val
        )
        scan[i], track[i], vza[i], area[i] = sc, tr, vz, ar
    out["scan_km"] = scan
    out["track_km"] = track
    out["vza_deg"] = vza
    out["pixel_area_km2"] = area

    h3_main = h3_utils.latlon_to_h3_int(lats, lons, S.H3_MAIN_RES)
    h3_flare = h3_utils.latlon_to_h3_int(lats, lons, S.H3_FLARE_RES)
    out["h3_index"] = h3_main
    out["h3_flare8"] = h3_flare

    out["pass_id"] = segment_passes(out)
    return out


def _resolve_row_geometry(sensor_family, sample, pixarea, scan_km, track_km):
    from src.geometry.solar_geometry import resolve_pixel_geometry

    return resolve_pixel_geometry(
        sensor_family,
        sample_idx=sample,
        pixarea_km2=pixarea,
        scan_km=scan_km,
        track_km=track_km,
    )


def preprocess_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full geometry preprocessing for one canonical detection frame.

    Returns ``(provenance_base, pass_observations)``:

    * ``provenance_base`` keeps every raw record with derived diagnostics and
      exclusion flags (Product 1 base, INV-09);
    * ``pass_observations`` aggregates biomass-eligible rows strictly per
      ``(h3_index, solar_date, platform, dn_flag)``.
    """
    out = enrich_geometry(df)

    # --- hygiene masks (rows remain in provenance, excluded from fusion) ---
    excluded_thermal = volcano_offshore_mask(out)
    sama_filtered = sama_nighttime_lowconf_mask(out)
    out["is_excluded_thermal"] = excluded_thermal
    out["is_sama_filtered"] = sama_filtered

    biomass = out[~(excluded_thermal | sama_filtered)].copy()
    if len(biomass):
        biomass["is_static_industrial"] = flare_microcell_mask(biomass)
    else:
        biomass["is_static_industrial"] = pd.Series(dtype=bool)
    static_by_index = set(
        biomass.loc[biomass["is_static_industrial"], "h3_flare8"]
    )
    out["is_static_industrial"] = out["h3_flare8"].isin(static_by_index)
    out["reliability_prob"] = 1.0  # calibrated in operators.py
    out["biomass_eligible"] = (
        ~(out["is_excluded_thermal"] | out["is_sama_filtered"]
          | out["is_static_industrial"])
    )

    # --- pass-level geometry: bow-tie weights + event clusters ---
    bowtie = np.ones(len(out), dtype=float)
    cluster_id = np.full(len(out), -1, dtype=np.int64)
    i4_flag = np.zeros(len(out), dtype=np.int8)
    sat_flag = np.zeros(len(out), dtype=np.int8)
    global_cluster = 0
    for (platform, pass_id), part in out.groupby(
        ["platform", "pass_id"], sort=True
    ):
        idx = part.index.to_numpy()
        ecef = latlon_to_ecef(
            part["latitude"].to_numpy(dtype=float),
            part["longitude"].to_numpy(dtype=float),
        )
        areas = part["pixel_area_km2"].to_numpy(dtype=float)
        tracks = part["track_km"].to_numpy(dtype=float)
        fam = part["sensor_family"].iloc[0]
        bowtie[idx] = bowtie_weights(ecef, areas, tracks, fam)
        labels = cluster_events(ecef, areas)
        # unique ids across passes: (platform, pass, component)
        cluster_id[idx] = labels + global_cluster
        global_cluster += int(labels.max()) + 1 if len(labels) else 0
        for row_i in idx:
            i4f, sf = classify_saturation(
                out.at[row_i, "sensor_family"],
                float(out.at[row_i, "t_mir_k"]),
                float(out.at[row_i, "frp_mw"]),
            )
            i4_flag[row_i] = i4f
            sat_flag[row_i] = sf
    out["bowtie_weight"] = bowtie
    out["event_cluster_id"] = cluster_id
    out["i4_m13_fallback"] = i4_flag
    out["sat_flag"] = sat_flag

    provenance = out[
        [
            "country_iso", "sensor_family", "platform", "data_tier",
            "latitude", "longitude", "acq_date", "acq_hhmm", "t_mir_k",
            "t_tir_k", "scan_km", "track_km", "sample_idx", "pixarea_km2",
            "frp_mw", "conf_raw", "conf_num", "type_raw", "version_raw",
            "ingest_ts",
        ]
        + list(PROVENANCE_DERIVED_COLUMNS)
    ].copy()

    passes = aggregate_passes(out[out["biomass_eligible"]])
    return provenance, passes


def aggregate_passes(biomass: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eligible detections per (h3, solar_date, platform, dn).

    * ``y_frp_mw = sum(omega_bt * rel_weight * frp_mw)`` (rel_weight enters
      after calibration; initialized 1.0 here);
    * ``raw_event_count = count_distinct(event_cluster_id)``;
    * ``det_footprint_km2 = sum(omega_bt * A_i)`` plus means and flags.
    """
    if len(biomass) == 0:
        return pd.DataFrame(
            columns=[
                "h3_index", "solar_date", "platform", "dn_flag", "y_frp_mw",
                "raw_event_count", "det_footprint_km2", "mean_area_km2",
                "mean_vza_deg", "mean_solar_hour", "mean_conf_num",
                "has_i4_m13_fallback", "sat_flag", "rel_weight",
                "sensor_family", "n_pixels", "mean_latitude", "mean_longitude",
            ]
        )
    rel = 1.0  # reliability weight applied later by operators.py
    w = biomass["bowtie_weight"].to_numpy(dtype=float)
    frp = biomass["frp_mw"].to_numpy(dtype=float)
    contrib = w * rel * frp
    foot = w * biomass["pixel_area_km2"].to_numpy(dtype=float)
    tmp = biomass.assign(_frp_w=contrib, _foot_w=foot, _rel=rel)
    grouped = tmp.groupby(
        ["h3_index", "solar_date", "platform", "dn_flag"], sort=True
    )
    agg = grouped.agg(
        y_frp_mw=("_frp_w", "sum"),
        raw_event_count=("event_cluster_id", "nunique"),
        det_footprint_km2=("_foot_w", "sum"),
        mean_area_km2=("pixel_area_km2", "mean"),
        mean_vza_deg=("vza_deg", "mean"),
        mean_solar_hour=("solar_hour", "mean"),
        mean_conf_num=("conf_num", "mean"),
        has_i4_m13_fallback=("i4_m13_fallback", "max"),
        sat_flag=("sat_flag", "max"),
        sensor_family=("sensor_family", "first"),
        n_pixels=("frp_mw", "size"),
        mean_latitude=("latitude", "mean"),
        mean_longitude=("longitude", "mean"),
    ).reset_index()
    agg["rel_weight"] = 1.0
    return agg
