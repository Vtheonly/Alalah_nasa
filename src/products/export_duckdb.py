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
from ..engine import conformal as conf
from ..geometry.h3_utils import h3_boundary_geojson

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
    """Sparse episode filter: phi_q50 > 0.25 OR within +-3 days of a
    detection OR inside a spotlight ROI."""
    if not len(daily):
        return pd.Series(False, index=daily.index)
    t0 = pd.Timestamp(S.DATE_START)
    det_t = set()
    if len(pass_obs):
        det_t = set(((pd.to_datetime(pass_obs["solar_date"]) - t0).dt.days)
                    .astype(int).tolist())
    t = daily["t"].to_numpy(dtype=int)
    near_det = np.zeros(len(daily), dtype=bool)
    for pad in range(-S.DAILY_SPARSE_DETECTION_PAD_DAYS,
                     S.DAILY_SPARSE_DETECTION_PAD_DAYS + 1):
        near_det |= np.isin(t, np.array(sorted(det_t)) + pad) if det_t else False
    keep = (daily["phi_q50_mw"] > S.DAILY_SPARSE_PHI_MIN).to_numpy() | near_det
    # spotlight ROIs (per-cell lat/lon inside any preset bbox)
    from ..geometry.h3_utils import h3_cell_center
    in_spot = np.zeros(len(daily), dtype=bool)
    cells = {int(c): h3_cell_center(int(c)) for c in daily["h3_index"].unique()}
    for i, c in enumerate(daily["h3_index"].to_numpy()):
        lat, lon = cells.get(int(c), (0.0, 0.0))
        for sp in S.SPOTLIGHT_PRESETS:
            if (sp.bbox[0] <= lon <= sp.bbox[2]
                    and sp.bbox[1] <= lat <= sp.bbox[3]):
                in_spot[i] = True
                break
    return pd.Series(keep | in_spot, index=daily.index)


def build_products(monthly: pd.DataFrame, daily: pd.DataFrame,
                   pass_obs: pd.DataFrame, prov: pd.DataFrame,
                   pass_cal: pd.DataFrame, priors: dict,
                   artifacts: dict, cloud_df: pd.DataFrame | None,
                   t_days: int = S.T_DAYS) -> dict:
    """Assemble Products 1-3 (monthly dense, daily sparse, diagnostics)."""
    product1 = build_product1(prov, pass_cal)

    m = monthly.copy()
    if "country_iso" not in m.columns and len(prov):
        cc = prov.groupby("h3_index")["country_iso"].first()
        m["country_iso"] = m["h3_index"].map(
            lambda h: cc.get(int(h)) if pd.notna(cc.get(int(h))) else "ARG")
    elif "country_iso" not in m.columns:
        m["country_iso"] = "ARG"
    # burning footprint + flux density + event indices
    area_model = artifacts.get("area_model")
    from ..calibration.operators import predict_burn_area
    hex_area = 36.13
    m["est_active_area_km2"] = predict_burn_area(
        area_model, np.log(m["phi_q50_mw"].clip(lower=1e-3) + 0.01),
        (m["phi_night_mw"] / m["phi_day_mw"].clip(lower=1e-6)).fillna(1.0),
        m["pyrome_id"], hex_area)
    m["phi_flux_mw_km2"] = m["phi_q50_mw"] / hex_area
    f_event = priors["f_event"]
    m["energy_equiv_events_q50"] = conf.energy_equiv_events(
        m["phi_q50_mw"], [f_event.get(int(b), 1.0) for b in m["pyrome_id"]])
    # hurdle_adj_events: mean over detecting platforms per half-day, scaled
    # by 1/max(measured_fraction, 0.25), capped at 5x raw clustered events
    po_all = pass_cal if (pass_cal is not None and len(pass_cal)) else pass_obs
    if po_all is not None and len(po_all) and "pyrome_id" in po_all.columns:
        po = po_all.copy()
        t0 = pd.Timestamp(S.DATE_START)
        po["t"] = ((pd.to_datetime(po["solar_date"]) - t0).dt.days).astype(int)
        lognorm = artifacts["lognorm_params"]
        from ..calibration.operators import psi_cnt
        po["psi_cnt_pass"] = [
            float(psi_cnt(S.SENSOR_FAMILY.get(str(p), "VIIRS"), int(b),
                          float(a), str(dn), lognorm))
            for p, b, a, dn in zip(po["platform"], po["pyrome_id"],
                                   po["mean_area_km2"], po["dn_flag"])]
        po["e_half"] = po["raw_event_count"] / np.maximum(
            po["psi_cnt_pass"] * 0.9, 1e-6)
        hd = po.groupby(["h3_index", "t", "dn_flag"], as_index=False).agg(
            e_half=("e_half", "mean"), platforms=("platform", lambda s:
                                                  conf.active_platforms_bitmask(s)))
        dd = daily.copy()
        dd["year"] = pd.Timestamp(S.DATE_START).year + (
            dd["t"] // 365).clip(upper=26)
        dates = pd.DatetimeIndex([pd.Timestamp(S.DATE_START)
                                  + pd.Timedelta(days=int(t))
                                  for t in dd["t"]])
        dd["year"] = dates.year
        dd["month"] = dates.month
        mm = dd.merge(hd, on=["h3_index", "t"], how="left")
        mm["e_half"] = mm["e_half"].fillna(0.0)
        agg = mm.groupby(["h3_index", "year", "month"], as_index=False).agg(
            e_sum=("e_half", "sum"),
            bitmask=("platforms", "max"),
            meas=("e_half", lambda s: float((s > 0).mean())))
        m = m.merge(agg, on=["h3_index", "year", "month"], how="left")
        m["hurdle_adj_events"] = np.minimum(
            m["e_sum"] / np.maximum(m["meas"].fillna(1.0),
                                    S.MEASURED_FRACTION_FLOOR),
            S.HURDLE_EVENTS_CAP_FACTOR * m["raw_clustered_events"]
        ).fillna(m["raw_clustered_events"])
        m["hurdle_adj_events"] = m["hurdle_adj_events"].fillna(0.0)
    else:
        m["hurdle_adj_events"] = 0.0
        m["bitmask"] = 0
        m["meas"] = 1.0
    m["night_day_ratio"] = (m["phi_night_mw"]
                            / m["phi_day_mw"].clip(lower=1e-6)).fillna(0.0)

    # conformal era + flags already applied upstream (calibrated intervals)
    product2_monthly = m[[c for c in PRODUCT2_MONTHLY_COLUMNS
                          if c in m.columns]].copy()

    # daily sparse
    daily2 = daily.copy()
    keep = sparse_daily_filter(daily2, pass_obs, t_days)
    product2_daily = daily2.loc[keep].copy()

    # Product 3: diagnostics 1-to-1 with monthly
    def _col(name, default):
        if name in m.columns:
            return m[name]
        return pd.Series(np.full(len(m), default), index=m.index)

    p3 = pd.DataFrame({
        "h3_index": m["h3_index"].to_numpy(),
        "year": m["year"].to_numpy(),
        "month": m["month"].to_numpy(),
        "conformal_c_factor": _col("conformal_c_factor", 1.0).to_numpy(),
        "rel_uncertainty_width": ((m["phi_q95_mw"] - m["phi_q05_mw"])
                                  / m["phi_q50_mw"].clip(lower=1e-6)).to_numpy(),
        "measured_fraction": pd.Series(_col("meas", 1.0)).fillna(1.0).to_numpy(),
        "inferred_fraction": (1.0 - pd.Series(_col("meas", 1.0))
                              .fillna(1.0)).clip(0, 1).to_numpy(),
        "active_platforms_bitmask": pd.Series(_col("bitmask", 0))
            .fillna(0).astype(int).to_numpy(),
        "mean_cloud_frac": pd.Series(_col("mean_cloud_frac", 0.0))
            .fillna(0.0).to_numpy(),
        "has_nrt_gapfill": 0,
        "attribution_flag": _col("attribution_flag",
                                 "WITHIN_NORMAL_VARIABILITY").to_numpy(),
        "quality_flag": _col("quality_flag", "NOMINAL").to_numpy(),
        "revision_count": 1,
    })
    product3 = p3[[c for c in PRODUCT3_COLUMNS if c in p3.columns]]
    return {"product1": product1, "product2_monthly": product2_monthly,
            "product2_daily": product2_daily, "product3": product3}


def export_duckdb(products: dict, duckdb_path, parquet_dir=None) -> None:
    """Write indexed Zstd Parquet + load all tables into fire_harmonized.duckdb."""
    import duckdb

    duckdb_path = str(duckdb_path)
    os.makedirs(os.path.dirname(duckdb_path) or ".", exist_ok=True)
    if parquet_dir:
        os.makedirs(parquet_dir, exist_ok=True)
    parquet_paths = {}
    for name, df in products.items():
        path = os.path.join(parquet_dir or (os.path.dirname(duckdb_path)),
                            f"{name}.parquet")
        df.to_parquet(path, index=False, compression="zstd")
        parquet_paths[name] = path

    con = duckdb.connect(duckdb_path)
    try:
        for name, path in parquet_paths.items():
            table = {"product1": "obs_harmonized_provenance",
                     "product2_monthly": "state_reconstructed_monthly",
                     "product2_daily": "state_reconstructed_daily",
                     "product3": "uncertainty_diagnostics"}[name]
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{path}')")
            if table in ("state_reconstructed_monthly",
                         "uncertainty_diagnostics"):
                con.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_cell_ym ON "
                    f"{table}(h3_index, year, month)")
            elif table == "obs_harmonized_provenance":
                con.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_h3 ON "
                    f"{table}(h3_index)")
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
