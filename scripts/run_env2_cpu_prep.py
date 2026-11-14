"""Environment 2 CPU preparation (internet OFF, Kaggle GPU notebook).

Pipeline (readme.md Section 8, Phase 5 first half):
  1. Load staged active-fire parquet + ancillary grids.
  2. Geometry preprocessing (hygiene, bow-tie, clustering, saturation).
  3. Pyrome discovery (K=7 behavioral GMM).
  4. Observation-operator calibration (2012-01-20 .. 2021-12-31).
  5. Empirical-Bayes prior estimation + 3-color neighbor packing.

All steps run on CPU (zero GPU quota usage).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import paths as paths_mod
from src.engine import priors as priors_mod
from src.engine import solver as sv

logger = logging.getLogger("fire_harmonizer.scripts.run_env2_cpu_prep")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Environment 2 CPU preparation stage"
    )
    parser.add_argument(
        "--max-cells", type=int, default=0,
        help="optional cap on active cells (smoke runs)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    dp = paths_mod.DataPaths.from_env()
    dp.ensure_dirs()

    # 1) Load the staged, normalized archive.
    frames = []
    for parquet in sorted(dp.active_fires_dir.rglob("*.parquet")):
        frames.append(pd.read_parquet(parquet))
    if not frames:
        logger.error("no staged active-fire parquet under %s",
                     dp.active_fires_dir)
        return 1
    detections = pd.concat(frames, ignore_index=True)
    logger.info("loaded %d normalized detections", len(detections))
    polygons = pd.read_parquet(dp.country_polygons_parquet)

    # 2) Geometry preprocessing.
    from src.geometry import preprocess_geometry as pg

    provenance, pass_obs = pg.preprocess_frame(detections)
    provenance.to_parquet(dp.provenance_base_parquet, index=False)
    pass_obs.to_parquet(dp.pass_obs_parquet, index=False)
    logger.info("geometry: %d provenance rows, %d pass rows",
                len(provenance), len(pass_obs))

    # 3) Pyromes.
    from src.calibration import pyromes

    hex_pyromes = pyromes.run_pyromes(
        pass_obs, dp.pyromes_parquet,
        gmm_pkl=str(dp.pyromes_parquet).replace(
            "hex_pyromes", "pyrome_gmm"
        ).replace(".parquet", ".pkl"),
    )
    logger.info("pyromes assigned to %d hexagons", len(hex_pyromes))

    # 4) Observation operators.
    from src.calibration import operators as op

    cloud_df = pd.read_parquet(dp.cloud_parquet)
    provider = op.CloudVisibilityProvider(cloud_df)
    pass_calibrated, artifacts = op.calibrate_passes(
        provenance, pass_obs, hex_pyromes, provider,
        out_parquet=dp.pass_calibrated_parquet,
        artifacts_pkl=dp.calibration_pkl,
    )
    logger.info("calibrated %d pass rows", len(pass_calibrated))

    # 5) Priors + active-cell graph (3-coloring).
    from src.geometry import h3_utils

    active_cells = pass_calibrated["h3_index"].unique()
    if args.max_cells:
        active_cells = sorted(active_cells)[: args.max_cells]
    active_cells = sorted(active_cells)
    colors, adjacency = h3_utils.three_color_hex_graph(
        __import__("numpy").array(active_cells, dtype="int64")
    )
    cell_pos = {int(c): i for i, c in enumerate(active_cells)}
    pass_calibrated["cell_pos"] = pass_calibrated["h3_index"].map(cell_pos)
    hex_pyromes["cell_pos"] = hex_pyromes["h3_index"].map(cell_pos)
    hex_sub = hex_pyromes.dropna(subset=["cell_pos"])
    priors = priors_mod.estimate_priors(
        pass_calibrated.dropna(subset=["cell_pos"]), hex_sub, adjacency,
        len(active_cells),
    )
    priors["cell_ids"] = active_cells
    priors["colors"] = colors
    priors["adjacency"] = adjacency
    with open(dp.priors_pkl, "wb") as fh:
        pickle.dump(priors, fh)
    logger.info("priors estimated for %d active cells", len(active_cells))

    # Stash the platform activity table next to the priors.
    from src.ingest import normalize_dedup as nd

    active_day = nd.compute_platform_active_day(detections)
    active_day.to_parquet(
        dp.provenance_base_parquet.parent / "platform_active_day.parquet",
        index=False,
    )
    logger.info("Environment 2 CPU preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
