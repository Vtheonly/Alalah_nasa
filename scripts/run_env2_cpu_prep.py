#!/usr/bin/env python3
"""Environment 2 CPU pre-processing (0 GPU-hours): geometry, clustering,
pyromes, operator calibration (2012-2021 freeze), priors.

Inputs come from Environment 1: ``active_fires/`` (already normalized,
polygon-clipped and Science-over-NRT deduplicated, ``(country_iso, year)``
partitioned).  Normalization is only re-run here when ``active_fires/`` is
absent but raw ``staged_raw/`` chunks exist (local development); on Kaggle
the staged dataset is read-only and ``staged_raw/`` is not shipped, so
``platform_active_day`` is recomputed from ``active_fires/`` itself (never
from an empty raw directory, which would zero the INV-04 calendar).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.calibration import operators as ops  # noqa: E402
from src.calibration.pyromes import fit_pyromes  # noqa: E402
from src.config import settings as S  # noqa: E402
from src.config.paths import DataPaths  # noqa: E402
from src.engine import priors as pr  # noqa: E402
from src.geometry.preprocess_geometry import preprocess_observations  # noqa: E402
from src.ingest.normalize_dedup import (compute_platform_active_day,  # noqa: E402
                                        coverage_audit, run_normalization)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_cpu_prep")


def read_active_fires(root) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(root) / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _has_parquet(root) -> bool:
    return bool(glob.glob(str(Path(root) / "**" / "*.parquet"),
                          recursive=True))


def _read_if_exists(path):
    path = Path(path)
    return pd.read_parquet(path) if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 2 CPU prep (0 GPU-hours)")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--force-normalize", action="store_true",
                    help="re-run normalization from staged_raw/ even when "
                         "active_fires/ already exists")
    args = ap.parse_args(argv)
    paths = DataPaths.from_env(args.base_dir)
    Path(paths.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    have_active = _has_parquet(paths.active_fires_dir)
    have_raw = _has_parquet(paths.raw_stage_dir) or bool(
        glob.glob(str(Path(paths.raw_stage_dir) / "*.gz")))
    if (not have_active or args.force_normalize) and have_raw:
        logger.info("normalizing staged_raw/ -> active_fires/")
        run_normalization(paths)
    elif not have_active:
        logger.error("no active_fires/ and no staged_raw/ under %s; run "
                     "Env 1 (make stage) first", paths.active_fires_dir)
        return 1
    det = read_active_fires(paths.active_fires_dir)
    det = det.drop(columns=[c for c in ("dn_source", "year") if c in det],
                   errors="ignore")
    # INV-04 empirical outage calendar from the CLIPPED, deduplicated set
    platform_active = compute_platform_active_day(det)
    audit = coverage_audit(det)
    logger.info("active fires: %d rows; audit=%s", len(det),
                json.dumps(audit, default=str)[:400])

    prov, pass_obs = preprocess_observations(det)
    prov.to_parquet(paths.provenance_base_parquet, index=False)
    pass_obs.to_parquet(paths.pass_obs_parquet, index=False)
    prov_bio = prov[prov["is_biomass"]].copy()
    logger.info("preprocess: %d provenance rows, %d pass rows (%.0fs)",
                len(prov), len(pass_obs), time.time() - t0)

    hex_pyromes, gmm, scaler = fit_pyromes(prov_bio)
    hex_pyromes.to_parquet(paths.pyromes_parquet, index=False)
    with open(str(paths.pyromes_parquet).replace("hex_pyromes.parquet",
                                                 "pyrome_gmm.pkl"), "wb") as fh:
        pickle.dump({"gmm": gmm, "scaler": scaler}, fh)

    cloud_df = _read_if_exists(paths.cloud_parquet)
    if cloud_df is None:
        logger.warning("cloud_visibility_025deg.parquet missing: v = p_swath "
                       "* clip(1 - 0.5) fallback will be used")
    artifacts, pass_cal = ops.calibrate(prov_bio, pass_obs, hex_pyromes,
                                        cloud_df)
    ops.save_artifacts(artifacts, paths.calibration_pkl)
    pass_cal.to_parquet(paths.pass_calibrated_parquet, index=False)
    logger.info("calibration done (%.0fs)", time.time() - t0)

    active_cells = sorted(set(int(c) for c in pass_cal["h3_index"].unique()))
    prior = pr.estimate_priors(pass_cal, hex_pyromes, active_cells, cloud_df,
                               artifacts.get("cloud_climatology"),
                               t_days=S.T_DAYS,
                               psi_cnt_fn=ops.psi_cnt_row_fn(artifacts),
                               platform_active=platform_active)
    with open(paths.priors_pkl, "wb") as fh:
        pickle.dump({"priors": prior,
                     "platform_active_day": platform_active,
                     "coverage_audit": audit}, fh)
    logger.info("CPU prep complete: %d active cells (%.0fs)",
                len(active_cells), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
