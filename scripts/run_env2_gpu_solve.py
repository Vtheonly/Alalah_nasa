#!/usr/bin/env python3
"""Environment 2 GPU solve: multi-GPU float64 MAP-Laplace + MC + conformal +
DuckDB/Parquet export."""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.config import settings as S
from src.config.paths import DataPaths
from src.engine import conformal as conf
from src.engine import solver as slv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_gpu_solve")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 2 GPU solve + export")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--sweeps", type=int, default=S.SOLVER_SWEEPS)
    ap.add_argument("--batch-size", type=int, default=S.BATCH_SIZE)
    args = ap.parse_args(argv)
    paths = DataPaths.from_env(args.base_dir)

    with open(paths.priors_pkl, "rb") as fh:
        blob = pickle.load(fh)
    priors = blob["priors"]
    platform_active = blob.get("platform_active_day")
    hex_pyromes = pd.read_parquet(paths.pyromes_parquet)
    pass_cal = pd.read_parquet(paths.pass_calibrated_parquet)
    with open(paths.calibration_pkl, "rb") as fh:
        artifacts = pickle.load(fh)
    cloud_df = pd.read_parquet(paths.cloud_parquet) \
        if Path(paths.cloud_parquet).exists() else None

    active_cells = sorted(set(int(c) for c in pass_cal["h3_index"].unique()))
    logger.info("solving %d active cells (batch %d, sweeps %d)",
                len(active_cells), args.batch_size, args.sweeps)
    res = slv.solve_all_chunks(pass_cal, active_cells, hex_pyromes, priors,
                               artifacts, cloud_df, t_days=S.T_DAYS,
                               platform_active=platform_active,
                               batch_size=args.batch_size, paths=paths)
    monthly_state, daily_state = slv.aggregate_trajectories(
        res["cells"], S.T_DAYS, {})
    monthly, daily = slv.combine_states_to_products(monthly_state,
                                                    daily_state)
    monthly = monthly.merge(hex_pyromes[["h3_index", "pyrome_id"]],
                            on="h3_index", how="left")
    monthly["pyrome_id"] = monthly["pyrome_id"].fillna(0).astype(int)

    # conformal folds (VIIRS-masked 2018-2020 + leave-one-VIIRS-out)
    t0 = pd.Timestamp(S.DATE_START)
    t_2018 = (pd.Timestamp("2018-01-01") - t0).days
    t_2021 = (pd.Timestamp("2021-12-31") - t0).days
    viirs_idx = [S.PLATFORM_INDEX[p] for p in S.VIIRS_PLATFORMS]

    def pmask(masked, lo, hi):
        return lambda pi, t: not (pi in masked and lo <= t <= hi)

    fold_m = slv.solve_all_chunks(pass_cal, active_cells, hex_pyromes,
                                  priors, artifacts, cloud_df, t_days=S.T_DAYS,
                                  platform_active=platform_active,
                                  platform_mask=pmask(set(viirs_idx), t_2018,
                                                      t_2021))
    fm, _ = slv.combine_states_to_products(fold_m["monthly"], fold_m["daily"])
    fm = fm.merge(hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index")
    phi_ref = {(int(h), int(y), int(m)): v for h, y, m, v in
               zip(monthly["h3_index"], monthly["year"], monthly["month"],
                   monthly["phi_q50_mw"])}
    c_modis = conf.conformal_multipliers(fm[(fm["year"] >= 2018)
                                            & (fm["year"] <= 2020)], phi_ref)
    fold_l = slv.solve_all_chunks(pass_cal, active_cells, hex_pyromes,
                                  priors, artifacts, cloud_df, t_days=S.T_DAYS,
                                  platform_active=platform_active,
                                  platform_mask=pmask({S.PLATFORM_INDEX["SNPP"]},
                                                      t_2018, t_2021))
    fl, _ = slv.combine_states_to_products(fold_l["monthly"], fold_l["daily"])
    fl = fl.merge(hex_pyromes[["h3_index", "pyrome_id"]], on="h3_index")
    c_multi = conf.conformal_multipliers(fl[(fl["year"] >= 2018)
                                            & (fl["year"] <= 2021)], phi_ref)

    monthly = conf.apply_conformal(monthly, c_modis, c_multi,
                                   monthly["year"] < 2012)
    monthly = conf.climatology_indices(monthly)

    from src.products import export_duckdb as ex
    prov = pd.read_parquet(paths.provenance_base_parquet)
    pass_obs = pd.read_parquet(paths.pass_obs_parquet)
    products = ex.build_products(monthly, daily, pass_obs, prov, pass_cal,
                                 priors, artifacts, cloud_df)
    ex.export_duckdb(products, paths.duckdb_path,
                     parquet_dir=str(paths.duckdb_path.parent))
    logger.info("exported fire_harmonized.duckdb + %d products",
                len(products))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
