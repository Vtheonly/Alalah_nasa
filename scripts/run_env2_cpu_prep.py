#!/usr/bin/env python3
"""Environment 2 CPU pre-processing (0 GPU-hours): geometry, clustering,
pyromes, operator calibration, priors."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.config.paths import DataPaths
from src.engine import priors as pr
from src.calibration import operators as ops
from src.calibration.pyromes import fit_pyromes
from src.geometry.preprocess_geometry import preprocess_observations
from src.ingest.normalize_dedup import (clip_to_countries,
                                        compute_platform_active_day,
                                        load_country_polygons,
                                        run_normalization)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_cpu_prep")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 2 CPU prep (0 GPU-hours)")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args(argv)
    paths = DataPaths.from_env(args.base_dir)

    report = run_normalization(paths)
    det = pd.read_parquet(paths.active_fires_dir, filters=None) \
        if False else _read_partitioned(paths.active_fires_dir)
    polygons = load_country_polygons(paths.country_polygons_parquet)
    det = clip_to_countries(det, polygons)
    det = det.drop(columns=[c for c in ("dn_source",) if c in det.columns])
    prov, pass_obs = preprocess_observations(det)
    prov.to_parquet(paths.provenance_base_parquet, index=False)
    pass_obs.to_parquet(paths.pass_obs_parquet, index=False)
    prov_bio = prov[prov["is_biomass"]].copy()

    hex_pyromes, _, _ = fit_pyromes(prov_bio)
    hex_pyromes.to_parquet(paths.pyromes_parquet, index=False)

    cloud_df = _read_if_exists(paths.cloud_parquet)
    artifacts, pass_cal = ops.calibrate(prov_bio, pass_obs, hex_pyromes,
                                        cloud_df)
    ops.save_artifacts(artifacts, paths.calibration_pkl)
    pass_cal.to_parquet(paths.pass_calibrated_parquet, index=False)

    active_cells = sorted(set(int(c) for c in pass_cal["h3_index"].unique()))

    def psi_cnt_fn(b, g):
        import src.calibration.operators as O
        from src.config import settings as S
        n = len(g)
        fams = [S.SENSOR_FAMILY.get(str(p), "VIIRS")
                for p in g.get("platform", pd.Series(["SNPP"] * n))]
        dns = g["dn_flag"].to_numpy()
        out = np.empty(n)
        for fam in ("MODIS", "VIIRS"):
            m = np.array([f == fam for f in fams])
            if m.any():
                out[m] = O.psi_cnt(fam, b, S.PASS_GEOMETRY_AVG[fam][0],
                                   dns[m], artifacts["lognorm_params"])
        return out

    platform_active = report["platform_active_day"]
    prior = pr.estimate_priors(pass_cal, hex_pyromes, active_cells, cloud_df,
                               artifacts.get("cloud_climatology"),
                               t_days=S_T_DAYS(), psi_cnt_fn=psi_cnt_fn)
    import pickle
    with open(paths.priors_pkl, "wb") as fh:
        pickle.dump({"priors": prior,
                     "platform_active_day": platform_active}, fh)
    logger.info("CPU prep complete: %d cells active", len(active_cells))
    return 0


def S_T_DAYS():
    from src.config import settings as S
    return S.T_DAYS


def _read_partitioned(root):
    import glob
    files = sorted(glob.glob(str(Path(root) / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _read_if_exists(path):
    path = Path(path)
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
