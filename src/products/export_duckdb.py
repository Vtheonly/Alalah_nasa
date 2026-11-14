"""Three complementary output products + indexed DuckDB export.

Contract (readme.md Sections 4 & 5.4 item 3):
  * **Product 1** ``obs_harmonized_provenance``: untouched raw sensor fields
    plus derived diagnostics (h3_index, sza_deg, solar_hour, dn_flag,
    pixel_area_km2, vza_deg, is_static_industrial, is_sama_filtered,
    bowtie_weight, reliability_prob, event_cluster_id, forward_gain_c,
    obs_sigma).
  * **Product 2** ``state_reconstructed_monthly`` (dense) and
    ``state_reconstructed_daily`` (sparse: phi_q50_mw > 0.25 OR within
    +/-3 days of a detection OR inside a spotlight ROI).
  * **Product 3** ``uncertainty_diagnostics`` (1-to-1 with Product 2
    monthly): conformal_c_factor, rel_uncertainty_width, measured_fraction,
    inferred_fraction, active_platforms_bitmask, mean_cloud_frac,
    has_nrt_gapfill, attribution_flag, quality_flag, revision_count.
  * Zstd Parquet + ``fire_harmonized.duckdb`` tables and summary views.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import paths as paths_mod
from src.config import settings as S
from src.engine import conformal as cf

logger = logging.getLogger("fire_harmonizer.products.export_duckdb")

PRODUCT1_COLUMNS = [
    "platform", "data_tier", "acq_date", "acq_hhmm", "frp_mw", "t_mir_k",
    "t_tir_k", "scan_km", "track_km", "conf_raw", "type_raw", "ingest_ts",
    "h3_index", "sza_deg", "solar_hour", "dn_flag", "pixel_area_km2",
    "vza_deg", "is_static_industrial", "is_sama_filtered", "bowtie_weight",
    "reliability_prob", "event_cluster_id", "forward_gain_c", "obs_sigma",
]

PRODUCT2_MONTHLY_COLUMNS = [
    "h3_index", "country_iso", "pyrome_id", "year", "month",
    "raw_count_sum", "raw_frp_sum_mw", "raw_clustered_events",
    "phi_day_mw", "phi_night_mw", "phi_q05_mw", "phi_q50_mw",
    "phi_q95_mw", "phi_mean_mw", "phi_flux_mw_km2", "fre_tj_q50",
    "energy_equiv_events_q50", "hurdle_adj_events", "est_active_area_km2",
    "night_day_ratio",
    "hist_median_2003_2020", "anomaly_ratio", "anomaly_zscore",
    "climatology_pctile",
]

PRODUCT3_COLUMNS = [
    "h3_index", "year", "month", "conformal_c_factor",
    "rel_uncertainty_width", "measured_fraction", "inferred_fraction",
    "active_platforms_bitmask", "mean_cloud_frac", "has_nrt_gapfill",
    "attribution_flag", "quality_flag", "revision_count",
]


# --------------------------------------------------------------------------- #
# Product frame builders
# --------------------------------------------------------------------------- #
def build_product1(
    provenance_base: pd.DataFrame,
    pass_calibrated: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Product 1: raw provenance + derived diagnostics.

    ``forward_gain_c`` and ``obs_sigma`` join from the calibrated pass
    table on (h3_index, solar_date, platform, dn_flag); rows without a
    calibrated pass keep 0.0 / NaN (excluded from biomass fusion).
    """
    p1 = provenance_base.copy()
    if pass_calibrated is not None and len(pass_calibrated):
        gains = pass_calibrated[
            ["h3_index", "solar_date", "platform", "dn_flag",
             "mu_gain", "obs_sigma"]
        ].rename(
            columns={"mu_gain": "forward_gain_c", "solar_date": "acq_date"}
        )
        p1 = p1.merge(
            gains, on=["h3_index", "acq_date", "platform", "dn_flag"],
            how="left",
        )
    else:
        p1["forward_gain_c"] = 0.0
        p1["obs_sigma"] = np.nan
    for col in PRODUCT1_COLUMNS:
        if col not in p1.columns:
            p1[col] = np.nan
    return p1[PRODUCT1_COLUMNS]


def _spotlight_mask(
    lat: float, lon: float
) -> bool:
    for preset in S.SPOTLIGHT_PRESETS.values():
        bbox = preset["bbox"]
        if (
            bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]
        ):
            return True
    return False


def build_product2_monthly(
    solved: Dict[str, object],
    cell_meta: pd.DataFrame,
    month_keys: List[Tuple[int, int]],
    pass_calibrated: pd.DataFrame,
    f_event: np.ndarray,
    cell_area_km2: float = 36.13,
    cqr_factor: float = 1.0,
) -> pd.DataFrame:
    """Dense monthly reconstructed states + raw baselines + indices."""
    n_cells = len(cell_meta)
    n_months = len(month_keys)
    month_ids = solved.get("month_ids")
    if month_ids is None:
        month_ids = np.repeat(
            np.arange(n_months),
            [28, 31, 30][:1] * 0,
        )  # placeholder never used when month_ids provided
    phi_day = np.exp(np.asarray(solved["z_day"]))
    phi_night = np.exp(np.asarray(solved["z_night"]))
    q05 = solved["monthly_q05"]
    q50 = solved["monthly_q50"]
    q95 = solved["monthly_q95"]
    mean = solved["monthly_mean"]
    q05_cal, q95_cal = cf.rescale_intervals(q05, q50, q95, cqr_factor)

    # raw baselines per cell-month
    po = pass_calibrated.copy()
    po["year"] = po["solar_date"].str.slice(0, 4).astype(int)
    po["month"] = po["solar_date"].str.slice(5, 7).astype(int)
    raw = po.groupby(["cell_pos", "year", "month"]).agg(
        raw_count_sum=("n_pixels", "sum"),
        raw_frp_sum_mw=("y_frp_mw", "sum"),
        raw_clustered_events=("raw_event_count", "sum"),
    ).reset_index()

    rows = []
    for b in range(n_cells):
        meta = cell_meta.iloc[b]
        f_event_b = float(f_event[int(meta["pyrome_id"])])
        phi_series = {}
        for m, (year, month) in enumerate(month_keys):
            phi_series[(year, month)] = float(q50[b, m])
        phi_q50_lookup = pd.Series(
            phi_series,
            index=pd.MultiIndex.from_tuples(
                phi_series.keys(), names=["year", "month"]
            ),
        )
        for m, (year, month) in enumerate(month_keys):
            phi_d = float(phi_day[b][month_ids == m].mean())
            phi_n = float(phi_night[b][month_ids == m].mean())
            q50_m = float(q50[b, m])
            ndr = phi_n / max(phi_d, 1e-9)
            idx = cf.climatology_indices(phi_q50_lookup, year, month)
            sub = raw[
                (raw["cell_pos"] == b) & (raw["year"] == year)
                & (raw["month"] == month)
            ]
            raw_events = float(
                sub["raw_clustered_events"].sum()
            ) if len(sub) else 0.0
            rows.append(
                {
                    "h3_index": int(meta["h3_index"]),
                    "country_iso": meta.get("country_iso", "ARG"),
                    "pyrome_id": int(meta["pyrome_id"]),
                    "year": year,
                    "month": month,
                    "raw_count_sum": int(
                        sub["raw_count_sum"].sum()
                    ) if len(sub) else 0,
                    "raw_frp_sum_mw": float(
                        sub["raw_frp_sum_mw"].sum()
                    ) if len(sub) else 0.0,
                    "raw_clustered_events": raw_events,
                    "phi_day_mw": phi_d,
                    "phi_night_mw": phi_n,
                    "phi_q05_mw": float(q05_cal[b, m]),
                    "phi_q50_mw": q50_m,
                    "phi_q95_mw": float(q95_cal[b, m]),
                    "phi_mean_mw": float(mean[b, m]),
                    "phi_flux_mw_km2": q50_m / cell_area_km2,
                    "fre_tj_q50": S.FRE_TJ_PER_MW_DAY * q50_m,
                    "energy_equiv_events_q50": cf.energy_equiv_events_q50(
                        q50_m, f_event_b
                    ),
                    "hurdle_adj_events": raw_events,  # refined by caller
                    "est_active_area_km2": min(
                        0.5 * (phi_d + phi_n) / max(f_event_b, 1.0),
                        cell_area_km2,
                    ),
                    "night_day_ratio": ndr,
                    **idx,
                }
            )
    return pd.DataFrame(rows, columns=PRODUCT2_MONTHLY_COLUMNS)


def build_product2_daily(
    solved: Dict[str, object],
    cell_meta: pd.DataFrame,
    month_keys: List[Tuple[int, int]],
    pass_calibrated: pd.DataFrame,
) -> pd.DataFrame:
    """Sparse daily episodes (readme.md 5.4 item 4 filter)."""
    dq05 = solved["daily_q05"]
    dq50 = solved["daily_q50"]
    dq95 = solved["daily_q95"]
    month_ids = solved["month_ids"]
    n_cells, T = dq50.shape
    date_index = pd.date_range(S.DATE_START, periods=T, freq="D")

    detection_days: Dict[int, set] = {b: set() for b in range(n_cells)}
    for row in pass_calibrated.itertuples():
        detection_days[int(row.cell_pos)].add(int(row.t_index))

    rows = []
    for b in range(n_cells):
        meta = cell_meta.iloc[b]
        lat = float(meta.get("latitude", 0.0))
        lon = float(meta.get("longitude", 0.0))
        in_spotlight = _spotlight_mask(lat, lon)
        dets = detection_days[b]
        for t in range(T):
            q50 = float(dq50[b, t])
            near_det = any(
                (t - S.DAILY_EXPORT_DETECTION_WINDOW_DAYS)
                <= d
                <= (t + S.DAILY_EXPORT_DETECTION_WINDOW_DAYS)
                for d in dets
            )
            if q50 > S.DAILY_EXPORT_PHI_MIN_MW or near_det or in_spotlight:
                year = date_index[t].year
                month = date_index[t].month
                rows.append(
                    {
                        "h3_index": int(meta["h3_index"]),
                        "country_iso": meta.get("country_iso", "ARG"),
                        "pyrome_id": int(meta["pyrome_id"]),
                        "date": date_index[t].strftime("%Y-%m-%d"),
                        "year": year,
                        "month": month,
                        "phi_q05_mw": float(dq05[b, t]),
                        "phi_q50_mw": q50,
                        "phi_q95_mw": float(dq95[b, t]),
                    }
                )
    return pd.DataFrame(rows)


def build_product3(
    solved: Dict[str, object],
    cell_meta: pd.DataFrame,
    month_keys: List[Tuple[int, int]],
    pass_calibrated: pd.DataFrame,
    product2_monthly: pd.DataFrame,
    cqr_factor: float,
    cloud_monthly: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """1-to-1 uncertainty diagnostics for the monthly product."""
    q05 = solved["monthly_q05"]
    q50 = solved["monthly_q50"]
    q95 = solved["monthly_q95"]
    n_cells = len(cell_meta)
    month_ids = solved["month_ids"]
    platform_calendar = solved.get("platform_calendar")
    rows = []
    po = pass_calibrated
    for b in range(n_cells):
        meta = cell_meta.iloc[b]
        for m, (year, month) in enumerate(month_keys):
            in_month = month_ids == m
            days = int(in_month.sum())
            det_days = len(
                set(
                    po.loc[po["cell_pos"] == b, "t_index"]
                ).intersection(set(np.flatnonzero(in_month)))
            )
            measured_fraction = det_days / max(days, 1)
            bitmask = 0
            if platform_calendar is not None:
                for p in range(S.N_PLATFORMS):
                    if platform_calendar[p, np.flatnonzero(in_month)].any():
                        bitmask |= S.PLATFORM_BITMASK[p]
            mean_cloud = float("nan")
            if cloud_monthly is not None and len(cloud_monthly):
                sub = cloud_monthly[
                    (cloud_monthly["cell_pos"] == b)
                    & (cloud_monthly["month_id"] == m)
                ]
                if len(sub):
                    mean_cloud = float(sub["cloud_frac"].mean())
            p2 = product2_monthly[
                (product2_monthly["h3_index"] == int(meta["h3_index"]))
                & (product2_monthly["year"] == year)
                & (product2_monthly["month"] == month)
            ]
            hist_med = (
                float(p2["hist_median_2003_2020"].iloc[0])
                if len(p2) else 0.0
            )
            quality = "NOMINAL"
            rows.append(
                {
                    "h3_index": int(meta["h3_index"]),
                    "year": year,
                    "month": month,
                    "conformal_c_factor": cqr_factor,
                    "rel_uncertainty_width": float(
                        (q95[b, m] - q05[b, m]) / max(q50[b, m], 1e-6)
                    ),
                    "measured_fraction": measured_fraction,
                    "inferred_fraction": 1.0 - measured_fraction,
                    "active_platforms_bitmask": bitmask,
                    "mean_cloud_frac": mean_cloud,
                    "has_nrt_gapfill": bool(
                        po.loc[
                            (po["cell_pos"] == b)
                            & po["t_index"].isin(np.flatnonzero(in_month)),
                            "data_tier",
                        ].eq("NRT").any()
                    ) if "data_tier" in po.columns else False,
                    "attribution_flag": cf.attribution_flag(
                        quality,
                        float(p2["phi_q05_mw"].iloc[0]) if len(p2) else 0.0,
                        float(p2["phi_q95_mw"].iloc[0]) if len(p2) else 0.0,
                        hist_med,
                        float(p2["raw_frp_sum_mw"].iloc[0]) if len(p2) else 0.0,
                        max(hist_med, 1.0),
                    ),
                    "quality_flag": quality,
                    "revision_count": 0,
                }
            )
    return pd.DataFrame(rows, columns=PRODUCT3_COLUMNS)


# --------------------------------------------------------------------------- #
# Parquet + DuckDB export
# --------------------------------------------------------------------------- #
def export_all(
    product1: pd.DataFrame,
    product2_monthly: pd.DataFrame,
    product2_daily: pd.DataFrame,
    product3: pd.DataFrame,
    duckdb_path,
    product_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, int]:
    """Write Zstd Parquet products and load indexed DuckDB tables/views."""
    import duckdb

    dp = product_paths or {}
    p1_path = dp.get("product1") or Path("obs_harmonized_provenance.parquet")
    p2m_path = dp.get(
        "product2_monthly"
    ) or Path("state_reconstructed_monthly.parquet")
    p2d_path = dp.get(
        "product2_daily"
    ) or Path("state_reconstructed_daily.parquet")
    p3_path = dp.get("product3") or Path("uncertainty_diagnostics.parquet")

    for path, frame in (
        (p1_path, product1), (p2m_path, product2_monthly),
        (p2d_path, product2_daily), (p3_path, product3),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE obs_harmonized_provenance AS "
            f"SELECT * FROM read_parquet('{p1_path}')"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE state_reconstructed_monthly AS "
            f"SELECT * FROM read_parquet('{p2m_path}')"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE state_reconstructed_daily AS "
            f"SELECT * FROM read_parquet('{p2d_path}')"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE uncertainty_diagnostics AS "
            f"SELECT * FROM read_parquet('{p3_path}')"
        )
        for table, cols in (
            ("obs_harmonized_provenance", ["h3_index", "acq_date"]),
            ("state_reconstructed_monthly", ["h3_index", "year", "month"]),
            ("state_reconstructed_daily", ["h3_index", "date"]),
            ("uncertainty_diagnostics", ["h3_index", "year", "month"]),
        ):
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table} ON {table}"
                f" ({', '.join(cols)})"
            )
        con.execute(
            "CREATE OR REPLACE VIEW v_monthly_summary AS SELECT "
            "country_iso, year, month, COUNT(DISTINCT h3_index) AS n_cells, "
            "SUM(raw_count_sum) AS raw_counts, "
            "SUM(phi_q50_mw) AS harmonized_phi_mw, "
            "SUM(fre_tj_q50) AS harmonized_fre_tj, "
            "AVG(rel_uncertainty_width) AS avg_rel_width "
            "FROM state_reconstructed_monthly m "
            "JOIN uncertainty_diagnostics u "
            "ON m.h3_index = u.h3_index AND m.year = u.year "
            "AND m.month = u.month "
            "GROUP BY country_iso, year, month ORDER BY year, month"
        )
        counts = {
            "product1": len(product1),
            "product2_monthly": len(product2_monthly),
            "product2_daily": len(product2_daily),
            "product3": len(product3),
        }
    finally:
        con.close()
    return counts


def export_from_paths(
    data_paths: Optional[paths_mod.DataPaths] = None,
    **product_frames,
) -> Dict[str, int]:
    dp = data_paths or paths_mod.DataPaths.from_env()
    dp.ensure_dirs()
    return export_all(
        product_frames["product1"],
        product_frames["product2_monthly"],
        product_frames["product2_daily"],
        product_frames["product3"],
        duckdb_path=dp.duckdb_path,
        product_paths={
            "product1": dp.product1_parquet,
            "product2_monthly": dp.product2_monthly_parquet,
            "product2_daily": dp.product2_daily_parquet,
            "product3": dp.product3_parquet,
        },
    )
