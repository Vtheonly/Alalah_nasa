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
    """
    h3r8 = latlon_to_h3_int(df["latitude"].to_numpy(),
                            df["longitude"].to_numpy(), res)
    dates = pd.to_datetime(df["solar_date"])
    years = dates.dt.year
    doy = dates.dt.dayofyear
    days = dates.dt.strftime("%Y-%m-%d")
    dn = df["dn_flag"].to_numpy()

    grp = pd.DataFrame({"m": h3r8, "year": years, "doy": doy, "day": days,
                        "dn": dn})
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


# ---------------------------------------------------------------------------
# Saturation classification (readme 5.1.3.3)
# ---------------------------------------------------------------------------
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
def preprocess_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical detections -> (Product 1 base, pass_observations)."""
    if df.empty:
        prov = pd.DataFrame(columns=PROV_COLUMNS)
        pas = pd.DataFrame(columns=PASS_COLUMNS)
        return prov, pas
    out = enrich_geometry(df)

    # ---- hygiene flags (kept in Product 1, excluded from biomass fusion)
    out["is_sama_filtered"] = sama_filter_mask(out)
    out["is_static_industrial"] = static_flare_mask(out)
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
