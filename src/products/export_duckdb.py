"""DuckDB + Zstd Parquet product export (readme Sections 4.9, 5.4.3).

* **Product 1** ``obs_harmonized_provenance``: untouched raw sensor fields +
  derived diagnostics (h3_index, sza_deg, solar_hour, dn_flag,
  pixel_area_km2, vza_deg, is_static_industrial, is_sama_filtered,
  bowtie_weight, reliability_prob, event_cluster_id, forward_gain_c,
  obs_sigma).
* **Product 2** ``state_reconstructed_monthly`` (dense) +
  ``state_reconstructed_daily`` (**sparse**: ``phi_q50_mw > 0.25`` OR within
  +-3 days of a detection OR inside a spotlight ROI).
* **Product 3** ``uncertainty_diagnostics`` (1-to-1 with Product 2 monthly):
  conformal_c_factor, rel_uncertainty_width, measured_fraction,
  inferred_fraction, active_platforms_bitmask, mean_cloud_frac,
  has_nrt_gapfill, attribution_flag, quality_flag, revision_count.
* All products loaded as indexed tables/views into ``fire_harmonized.duckdb``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..config import settings as S

PRODUCT1_COLUMNS = [
    "platform", "data_tier", "acq_date", "acq_hhmm", "frp_mw", "t_mir_k",
    "t_tir_k", "scan_km", "track_km", "conf_raw", "type_raw", "ingest_ts",
    "latitude", "longitude", "sensor_family", "country_iso", "conf_num",
    "version_raw", "dn_source",
    "h3_index", "sza_deg", "solar_hour", "solar_date", "dn_flag",
    "pixel_area_km2", "vza_deg", "is_static_industrial", "is_sama_filtered",
    "bowtie_weight", "reliability_prob", "event_cluster_id", "pass_id",
    "i4_m13_fallback", "sat_flag", "is_biomass", "forward_gain_c",
    "obs_sigma",
]

PRODUCT2_MONTHLY_COLUMNS = [
    "h3_index", "country_iso", "pyrome_id", "year", "month",
    "raw_count_sum", "raw_frp_sum_mw", "raw_clustered_events",
    "phi_day_mw", "phi_night_mw", "phi_q05_mw", "phi_q50_mw", "phi_q95_mw",
    "phi_mean_mw", "phi_flux_mw_km2", "fre_tj_q50",
    "energy_equiv_events_q50", "hurdle_adj_events", "est_active_area_km2",
    "night_day_ratio", "hist_median_2003_2020", "anomaly_ratio",
    "anomaly_zscore", "climatology_pctile",
]

PRODUCT3_COLUMNS = [
    "h3_index", "year", "month", "conformal_c_factor",
    "rel_uncertainty_width", "measured_fraction", "inferred_fraction",
    "active_platforms_bitmask", "mean_cloud_frac", "has_nrt_gapfill",
    "attribution_flag", "quality_flag", "revision_count",
]


def build_product1(prov: pd.DataFrame, pass_cal: pd.DataFrame) -> pd.DataFrame:
    """Product 1: raw provenance + derived diagnostics."""
    p1 = prov.copy()
    # reliability + forward gain + obs sigma joined from the calibrated pass
    # table (per detection, via h3/date/platform/dn groups)
    if len(pass_cal):
        key = pass_cal.copy()
        key["solar_date"] = pd.to_datetime(key["solar_date"]).dt.strftime(
            "%Y-%m-%d")
        gain = key.groupby(["h3_index", "solar_date", "platform", "dn_flag"],
                           as_index=False).agg(
            forward_gain_c=("mu_gain_det", "mean"),
            obs_sigma=("sigma2", lambda s: float(np.sqrt(np.mean(s)))),
            reliability_prob=("w_mean", "mean"))
        p1["solar_date_str"] = pd.to_datetime(
            p1["solar_date"]).dt.strftime("%Y-%m-%d")
        p1 = p1.merge(gain, left_on=["h3_index", "solar_date_str", "platform",
                                     "dn_flag"],
                      right_on=["h3_index", "solar_date", "platform",
                                "dn_flag"],
                      how="left", suffixes=("", "_g"))
        p1 = p1.drop(columns=["solar_date_str"], errors="ignore")
    for col, default in (("forward_gain_c", 0.0), ("obs_sigma", 0.5),
                         ("reliability_prob", 1.0)):
        if col not in p1.columns:
            p1[col] = default
        p1[col] = p1[col].fillna(default)
    keep = [c for c in PRODUCT1_COLUMNS if c in p1.columns]
    return p1[keep]


def sparse_daily_filter(daily: pd.DataFrame, pass_obs: pd.DataFrame,
                        t_days: int) -> pd.Series:
    """Sparse episode filter (readme 5.4.4): keep a daily row iff
    ``phi_q50_mw > 0.25`` OR the SAME cell had a detection within +-3 days
    OR the cell centre lies inside a spotlight ROI.  Vectorized."""
    if not len(daily):
        return pd.Series(False, index=daily.index)
    from ..engine.solver import _cell_centers, spotlight_mask
    t0 = pd.Timestamp(S.DATE_START)
    h = daily["h3_index"].to_numpy(np.int64)
    t = daily["t"].to_numpy(np.int64)
    keep = (daily["phi_q50_mw"] > S.DAILY_SPARSE_PHI_MIN).to_numpy()
    if pass_obs is not None and len(pass_obs):
        po = pass_obs
        if "y_frp_mw" in po.columns:
            po = po[po["y_frp_mw"] > 0]
        dt = (pd.to_datetime(po["solar_date"]) - t0).dt.days.to_numpy(np.int64)
        dh = po["h3_index"].to_numpy(np.int64)
        pad = S.DAILY_SPARSE_DETECTION_PAD_DAYS
        offs = np.arange(-pad, pad + 1)
        kh = np.repeat(dh, len(offs))
        kt = (dt[:, None] + offs[None, :]).ravel()
        det_keys = np.unique(kh * (t_days + 2 * pad + 1) + (kt + pad))
        row_keys = h * (t_days + 2 * pad + 1) + (t + pad)
        keep |= np.isin(row_keys, det_keys)
    uc, inv = np.unique(h, return_inverse=True)
    lat, lon = _cell_centers(uc)
    keep |= spotlight_mask(lat, lon)[inv]
    return pd.Series(keep, index=daily.index)


REQUIRED_FINAL_COLUMNS = ("conformal_c_factor", "quality_flag",
                          "attribution_flag", "measured_fraction",
                          "mean_cloud_frac", "active_platforms_bitmask")


def build_products(monthly: pd.DataFrame, daily: pd.DataFrame,
                   pass_obs: pd.DataFrame, prov: pd.DataFrame,
                   pass_cal: pd.DataFrame, priors: dict | None = None,
                   artifacts: dict | None = None, cloud_df=None,
                   t_days: int = S.T_DAYS) -> dict:
    """Assemble Products 1-3 from the FINALIZED monthly frame
    (:func:`src.engine.conformal.finalize_monthly`) and the sparse daily
    posterior rows.  No defaults are fabricated: missing diagnostics raise.
    """
    missing = [c for c in REQUIRED_FINAL_COLUMNS if c not in monthly.columns]
    if missing:
        raise ValueError("monthly frame is not finalized (missing "
                         f"{missing}); run conformal.finalize_monthly first")
    product1 = build_product1(prov, pass_cal)
    m = monthly
    product2_monthly = m[[c for c in PRODUCT2_MONTHLY_COLUMNS
                          if c in m.columns]].reset_index(drop=True)
    daily2 = daily
    if len(daily2):
        keep = sparse_daily_filter(daily2, pass_obs, t_days)
        daily2 = daily2.loc[keep.to_numpy()].copy()
        dates = pd.Timestamp(S.DATE_START) + pd.to_timedelta(
            daily2["t"].to_numpy(), unit="D")
        daily2["solar_date"] = dates.strftime("%Y-%m-%d")
        daily2["year"] = dates.year
        daily2["month"] = dates.month
    product2_daily = daily2.reset_index(drop=True)
    product3 = m[[c for c in PRODUCT3_COLUMNS if c in m.columns]] \
        .reset_index(drop=True)
    return {"product1": product1, "product2_monthly": product2_monthly,
            "product2_daily": product2_daily, "product3": product3}


PRODUCT_TABLES = {"product1": "obs_harmonized_provenance",
                  "product2_monthly": "state_reconstructed_monthly",
                  "product2_daily": "state_reconstructed_daily",
                  "product3": "uncertainty_diagnostics"}


def hex_centers_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per active hexagon: centre lat/lon + country (web index)."""
    from ..engine.solver import _cell_centers
    cells = np.unique(monthly["h3_index"].to_numpy(np.int64))
    lat, lon = _cell_centers(cells)
    country = (monthly.drop_duplicates("h3_index").set_index("h3_index")
               ["country_iso"].reindex(cells).to_numpy()
               if "country_iso" in monthly.columns else None)
    return pd.DataFrame({"h3_index": cells, "latitude": lat,
                         "longitude": lon, "country_iso": country})


def export_duckdb(products: dict, duckdb_path, parquet_dir=None,
                  diagnostics: dict | None = None) -> dict:
    """Write indexed Zstd Parquet (spec filenames) + load every product into
    ``fire_harmonized.duckdb`` with the ``hex_centers`` index table, the
    ``diagnostics`` table and the ``v_calendar`` view.  Returns the
    {table: parquet_path} map."""
    import json

    import duckdb

    duckdb_path = str(duckdb_path)
    os.makedirs(os.path.dirname(duckdb_path) or ".", exist_ok=True)
    out_dir = parquet_dir or os.path.dirname(duckdb_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    parquet_paths = {}
    for name, df in products.items():
        table = PRODUCT_TABLES[name]
        path = os.path.join(out_dir, f"{table}.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        parquet_paths[table] = path
    centers = hex_centers_frame(products["product2_monthly"])
    con = duckdb.connect(duckdb_path)
    try:
        for table, path in parquet_paths.items():
            con.execute(f"DROP TABLE IF EXISTS {table}")
            if table in ("state_reconstructed_monthly",
                         "uncertainty_diagnostics"):
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM "
                            "read_parquet(?)", [path])
                # (h3_index, year, month) is UNIQUE -> safe ART index
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_cell_ym "
                            f"ON {table}(h3_index, year, month)")
            else:
                # Non-unique h3_index: clustered by ORDER BY so DuckDB's
                # min/max zone maps prune scans.  No ART index here: DuckDB
                # 1.3.x (the Kaggle image) fails building ART indexes on
                # heavily duplicated keys ("unrecognized enum value for
                # NType") above ~1e5 rows.
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM "
                            "read_parquet(?) ORDER BY h3_index", [path])
        con.execute("DROP TABLE IF EXISTS hex_centers")
        con.execute("CREATE TABLE hex_centers (h3_index BIGINT, latitude "
                    "DOUBLE, longitude DOUBLE, country_iso VARCHAR)")
        centers = centers.assign(country_iso=centers["country_iso"].map(
            lambda v: None if v is None or (isinstance(v, float) and v != v)
            else str(v)).astype(object))
        con.register("_centers", centers)
        con.execute("INSERT INTO hex_centers SELECT CAST(h3_index AS BIGINT), "
                    "latitude, longitude, CAST(country_iso AS VARCHAR) "
                    "FROM _centers")
        con.unregister("_centers")
        con.execute("CREATE INDEX IF NOT EXISTS idx_hex_centers_ll ON "
                    "hex_centers(longitude, latitude)")
        con.execute("DROP TABLE IF EXISTS diagnostics")
        con.execute("CREATE TABLE diagnostics (diagnostics_json VARCHAR)")
        if diagnostics is not None:
            con.execute("INSERT INTO diagnostics VALUES (?)",
                        [json.dumps(diagnostics, default=float)])
        con.execute("""
            CREATE OR REPLACE VIEW v_calendar AS
            SELECT m.h3_index, m.country_iso, m.pyrome_id, m.year, m.month,
                   m.phi_q50_mw AS value, m.phi_q05_mw AS q05,
                   m.phi_q95_mw AS q95, m.hist_median_2003_2020 AS hist_median,
                   m.anomaly_ratio, m.raw_count_sum AS raw_count,
                   m.raw_frp_sum_mw AS raw_frp, u.measured_fraction,
                   u.quality_flag, u.attribution_flag
            FROM state_reconstructed_monthly m
            JOIN uncertainty_diagnostics u
              ON m.h3_index = u.h3_index AND m.year = u.year
             AND m.month = u.month
        """)
    finally:
        con.close()
    return parquet_paths


def store_diagnostics(duckdb_path, diagnostics: dict) -> None:
    """(Re)write the diagnostics table after validation."""
    import json

    import duckdb
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute("CREATE TABLE IF NOT EXISTS diagnostics "
                    "(diagnostics_json VARCHAR)")
        con.execute("DELETE FROM diagnostics")
        con.execute("INSERT INTO diagnostics VALUES (?)",
                    [json.dumps(diagnostics, default=float)])
    finally:
        con.close()
