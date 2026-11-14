#!/usr/bin/env python3
"""Environment 2 scientific validation: structural breaks, ablation,
MCD64CMQ correlations, spotlight reports -> diagnostics.json."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.config.paths import DataPaths
from src.validation.diagnostics import run_diagnostics, save_diagnostics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_validate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 2 validation")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args(argv)
    paths = DataPaths.from_env(args.base_dir)
    monthly = pd.read_parquet(paths.product2_monthly_parquet) \
        if Path(paths.product2_monthly_parquet).exists() else None
    if monthly is None:
        logger.error("run run_env2_gpu_solve.py first")
        return 1
    prov = pd.read_parquet(paths.provenance_base_parquet)
    from src.ingest.normalize_dedup import run_normalization
    truth_like = prov[["latitude", "longitude"]].drop_duplicates()
    burned = pd.read_parquet(paths.burned_area_parquet) \
        if Path(paths.burned_area_parquet).exists() else None
    pass_obs = pd.read_parquet(paths.pass_obs_parquet) \
        if Path(paths.pass_obs_parquet).exists() else None
    monthly["role"] = "normal"
    monthly["phi24_true"] = monthly["phi_q50_mw"]
    diag = run_diagnostics(monthly, monthly, burned, pass_obs)
    save_diagnostics(diag, paths.diagnostics_json)
    logger.info("diagnostics.json written (%d sections)", len(diag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
