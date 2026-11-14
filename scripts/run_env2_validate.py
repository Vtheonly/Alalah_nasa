#!/usr/bin/env python3
"""Environment 2 scientific validation -> diagnostics.json (+ DuckDB table).

* Chow / Pettitt structural breaks (raw counts vs harmonized flux, 2012).
* 2019-2021 VIIRS-masked ablation: a REAL re-solve with every VIIRS platform
  masked over 2019-01-01..2021-12-31 (multi-GPU, same engine), scored
  against the full multi-sensor product (the only reference available on
  real data) and against the raw MODIS baseline -- never the model compared
  with itself.
* MCD64CMQ burned-area correlation (exact 0.25 deg CMG bin join).
* Post-harmonization residual slopes vs VZA / solar hour.
* Spotlight reports (Chilecito ARG 2026, Kabylie, Hassi Messaoud, Alaska,
  Black Summer, Kalimantan) and the Nov 2, 2026 S-NPP end.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.config import settings as S  # noqa: E402
from src.config.paths import DataPaths  # noqa: E402
from src.products.export_duckdb import store_diagnostics  # noqa: E402
from src.validation.diagnostics import (run_diagnostics,  # noqa: E402
                                        save_diagnostics)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_validate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 2 validation")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=S.BATCH_SIZE)
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--skip-ablation", action="store_true",
                    help="skip the VIIRS-masked re-solve (diagnostics then "
                         "omit the ablation section; nothing is fabricated)")
    args = ap.parse_args(argv)
    paths = DataPaths.from_env(args.base_dir)
    t0 = time.time()
    if not Path(paths.product2_monthly_parquet).exists():
        logger.error("%s missing: run scripts/run_env2_gpu_solve.py first",
                     paths.product2_monthly_parquet)
        return 1
    monthly = pd.read_parquet(paths.product2_monthly_parquet)
    daily = (pd.read_parquet(paths.product2_daily_parquet)
             if Path(paths.product2_daily_parquet).exists() else None)
    pass_obs = pd.read_parquet(paths.pass_obs_parquet)
    pass_cal = pd.read_parquet(paths.pass_calibrated_parquet)
    burned = (pd.read_parquet(paths.burned_area_parquet)
              if Path(paths.burned_area_parquet).exists() else None)
    cf_path = Path(paths.duckdb_path).parent / "conformal_factors.pkl"
    c_modis = None
    if cf_path.exists():
        with open(cf_path, "rb") as fh:
            c_modis = pickle.load(fh).get("c_modis")

    ablation = None
    if not args.skip_ablation:
        with open(paths.priors_pkl, "rb") as fh:
            blob = pickle.load(fh)
        with open(paths.calibration_pkl, "rb") as fh:
            artifacts = pickle.load(fh)
        hex_pyromes = pd.read_parquet(paths.pyromes_parquet)
        cloud_df = (pd.read_parquet(paths.cloud_parquet)
                    if Path(paths.cloud_parquet).exists() else None)
        from src.validation.twin_pipeline import viirs_masked_ablation_fold
        cells = sorted(set(int(c) for c in pass_cal["h3_index"].unique()))
        ablation = viirs_masked_ablation_fold(
            pass_cal, cells, hex_pyromes, blob["priors"], artifacts, cloud_df,
            blob.get("platform_active_day"), S.T_DAYS, args.batch_size)
        logger.info("VIIRS-masked 2019-2021 ablation re-solve done (%.0fs)",
                    time.time() - t0)
    diag = run_diagnostics(monthly, truth=None, burned_area=burned,
                           pass_obs=pass_obs, pass_cal=pass_cal,
                           ablation_fold=ablation, daily=daily,
                           c_modis=c_modis)
    if Path(paths.priors_pkl).exists():
        with open(paths.priors_pkl, "rb") as fh:
            diag["coverage_audit"] = pickle.load(fh).get("coverage_audit")
    save_diagnostics(diag, paths.diagnostics_json)
    if Path(paths.duckdb_path).exists():
        store_diagnostics(paths.duckdb_path, diag)
    logger.info("diagnostics.json written (%d sections, %.0fs)", len(diag),
                time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
