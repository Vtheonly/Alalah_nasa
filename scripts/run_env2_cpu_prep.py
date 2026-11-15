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
from src.geometry.preprocess_geometry import preprocess_partitioned  # noqa: E402
from src.ingest.normalize_dedup import (compute_platform_active_day,  # noqa: E402
                                        coverage_audit, run_normalization)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_cpu_prep")


def read_active_fires(root) -> pd.DataFrame:
    """All ``active_fires/`` partitions as one frame with bounded peak RAM:
    Arrow tables are concatenated (zero-copy) and converted to pandas ONCE
    (``self_destruct`` frees Arrow buffers as columns convert; repeated
    strings are deduplicated), instead of per-file frames + ``pd.concat``
    which briefly needs twice the final size."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(root) / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        return pd.DataFrame()
    # partitioning=None: the hive dirs (country_iso=X/year=Y) must not add
    # dictionary-typed duplicates of columns already stored in the files
    tables = [pq.read_table(f, partitioning=None) for f in files]
    # permissive: an all-null column in one partition (type null) unifies
    # with the typed column of the others; missing columns become null
    table = pa.concat_tables(tables, promote_options="permissive")
    del tables
    return table.to_pandas(self_destruct=True, split_blocks=True,
                           deduplicate_objects=True)


# Columns of the provenance table read by pyromes + operator calibration
# (the full 35-column table is never loaded: tens of millions of rows).
PROV_BIO_COLUMNS = ["h3_index", "solar_date", "platform", "dn_flag",
                    "sensor_family", "country_iso", "latitude", "longitude",
                    "frp_mw", "conf_num", "vza_deg", "bowtie_weight",
                    "pass_id", "event_cluster_id"]
# Columns for the INV-04 calendar + coverage audit.
AUDIT_COLUMNS = ["platform", "acq_date", "country_iso", "latitude",
                 "longitude", "type_raw"]


def read_prov_biomass(prov_dir, columns=PROV_BIO_COLUMNS) -> pd.DataFrame:
    """Biomass rows of the partitioned provenance, needed columns only."""
    import pyarrow.dataset as ds
    dset = ds.dataset(str(prov_dir), format="parquet")
    cols = [c for c in columns if c in dset.schema.names]
    t = dset.to_table(columns=cols, filter=ds.field("is_biomass") == True)  # noqa: E712
    return t.to_pandas(self_destruct=True, split_blocks=True)


def read_active_fire_columns(root, columns) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(root) / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        return pd.DataFrame(columns=columns)
    t = pa.concat_tables([pq.read_table(f, columns=columns, partitioning=None)
                          for f in files],
                         promote_options="permissive")
    return t.to_pandas(self_destruct=True)


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
    det = read_active_fire_columns(paths.active_fires_dir, AUDIT_COLUMNS)
    # INV-04 empirical outage calendar from the CLIPPED, deduplicated set
    platform_active = compute_platform_active_day(det)
    audit = coverage_audit(det)
    logger.info("active fires: %d rows; audit=%s", len(det),
                json.dumps(audit, default=str)[:400])
    del det

    # Partitioned preprocess (one year at a time; provenance written as
    # provenance_base.parquet/year=YYYY.parquet, never held whole in RAM).
    pass_obs, pre = preprocess_partitioned(paths.active_fires_dir,
                                           paths.provenance_base_parquet)
    pass_obs.to_parquet(paths.pass_obs_parquet, index=False)
    logger.info("preprocess: %s, %d pass rows (%.0fs)", pre, len(pass_obs),
                time.time() - t0)
    if pre["duplicate_pass_keys_across_years"]:
        logger.warning("%d pass keys split across year partitions",
                       pre["duplicate_pass_keys_across_years"])
    prov_bio = read_prov_biomass(paths.provenance_base_parquet)
    logger.info("biomass provenance rows loaded: %d (%d columns)",
                len(prov_bio), prov_bio.shape[1])

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
