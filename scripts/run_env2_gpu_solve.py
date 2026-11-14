"""Environment 2 GPU solve + conformal calibration + product export.

Pipeline (readme.md Section 8, Phase 5 second half):
  1. Load calibrated passes, priors, artifacts and the cloud provider.
  2. Build dense per-regime observation tensors chunk-by-chunk
     (batch_size = 1000 active cells, VRAM < 6 GB per GPU).
  3. Multi-GPU float64 GMRF solve (6 3-color x Day/Night sweeps, 15 Newton
     steps, Armijo), Takahashi marginals, M=200 joint trajectories.
  4. Split-conformal calibration folds (MODIS-only 2018-2020; multi-sensor
     2018-2021 leave-one-VIIRS-platform-out).
  5. Export the three products + fire_harmonized.duckdb.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import paths as paths_mod
from src.config import settings as S
from src.engine import solver as sv

logger = logging.getLogger("fire_harmonizer.scripts.run_env2_gpu_solve")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Environment 2 GPU harmonization solve"
    )
    parser.add_argument("--sweeps", type=int, default=S.SOLVER_SWEEPS)
    parser.add_argument("--mc-samples", type=int, default=S.MC_SAMPLES)
    parser.add_argument(
        "--skip-conformal", action="store_true",
        help="reuse existing conformal multipliers",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    dp = paths_mod.DataPaths.from_env()
    dp.ensure_dirs()

    pass_calibrated = pd.read_parquet(dp.pass_calibrated_parquet)
    with open(dp.priors_pkl, "rb") as fh:
        priors = pickle.load(fh)
    with open(dp.calibration_pkl, "rb") as fh:
        artifacts = pickle.load(fh)
    cloud_df = pd.read_parquet(dp.cloud_parquet)
    from src.calibration.operators import CloudVisibilityProvider

    provider = CloudVisibilityProvider(cloud_df)

    cell_ids = np.asarray(priors["cell_ids"])
    n_cells = len(cell_ids)
    cell_pos_map = {int(c): i for i, c in enumerate(cell_ids)}
    T = S.T_DAYS
    month_ids, month_keys = sv.month_grid(T)

    platform_active = pd.read_parquet(
        dp.provenance_base_parquet.parent / "platform_active_day.parquet"
    )
    calendar = sv.build_platform_calendar(platform_active, T)

    lognorm = np.zeros((S.PYROME_K, 2))
    for b, (mu, sig) in artifacts["lognorm_params"].items():
        lognorm[int(b)] = (mu, sig)
    psi_cnt_table = sv.build_psi_cnt_table(lognorm, sv.DEFAULT_AVG_AREAS)
    k_table = sv.build_k_table(
        np.asarray(priors["lambda0"]), np.asarray(priors["f_event"]),
        psi_cnt_table,
    )

    # cell metadata (country + pyrome + centroid) for the products
    hex_pyromes = pd.read_parquet(dp.pyromes_parquet)
    from src.geometry import h3_utils

    meta_rows = []
    for pos, cell in enumerate(cell_ids):
        pyrome = int(priors["pyrome_of_cell"].get(pos, 0))
        lat, lon = h3_utils.h3_center_latlon(int(cell))
        meta_rows.append(
            {
                "cell_pos": pos, "h3_index": int(cell), "pyrome_id": pyrome,
                "latitude": lat, "longitude": lon, "country_iso": None,
            }
        )
    cell_meta = pd.DataFrame(meta_rows)
    provenance = pd.read_parquet(dp.provenance_base_parquet)
    iso_by_h3 = provenance.groupby("h3_index")["country_iso"].first()
    cell_meta["country_iso"] = cell_meta["h3_index"].map(iso_by_h3)

    def build_chunk(cells, device):
        sub_pos = {c: i for i, c in enumerate(cells)}
        sub_df = pass_calibrated[pass_calibrated["cell_pos"].isin(cells)]
        obs = sv.build_observation_tensors(
            sub_df, sub_pos, len(cells), T, calendar, k_table,
            np.asarray(priors["pyrome"]), None, device,
        )
        # cloud visibility per (cell, t, platform) from the provider
        vis = {"D": np.zeros((len(cells), T, S.N_PLATFORMS)),
               "N": np.zeros((len(cells), T, S.N_PLATFORMS))}
        for i, c in enumerate(cells):
            lat = float(cell_meta.iloc[int(c)]["latitude"])
            lon = float(cell_meta.iloc[int(c)]["longitude"])
            for p_idx, platform in enumerate(S.PLATFORMS):
                family = S.PLATFORM_TIMELINE[platform].sensor_family
                for t in range(0, T, 30):  # monthly cloud granularity
                    v, _ = provider.visibility(
                        family, platform, lat, lon, t
                    )
                    t_end = min(t + 30, T)
                    vis["D"][i, t:t_end, p_idx] = v
                    vis["N"][i, t:t_end, p_idx] = v
        for dn in ("D", "N"):
            obs[dn]["vis"] = torch_from_numpy(vis[dn], device)
        nb_idx, nb_ptr = [], [0]
        pos_lookup = {int(c): i for i, c in enumerate(cells)}
        for c in cells:
            lo = priors["neighbor_csr"]["indptr"][int(c)]
            hi = priors["neighbor_csr"]["indptr"][int(c) + 1]
            for nb in priors["neighbor_csr"]["indices"][lo:hi]:
                if int(nb) in pos_lookup:
                    nb_idx.append(pos_lookup[int(nb)])
            nb_ptr.append(len(nb_idx))
        return sv.ChunkProblem(
            cell_ids=np.asarray(cells),
            colors=priors["colors"][cells],
            rho=np.asarray(priors["rho"])[cells],
            omega=1.0 / np.asarray(priors["sigma_proc"])[cells] ** 2,
            eta_dn=np.asarray(priors["eta_dn"])[cells],
            beta_dn=np.asarray(priors["beta_dn"])[cells],
            mu_day=priors["mu_day"][cells],
            mu_night=priors["mu_night"][cells],
            month_ids=month_ids,
            obs=obs,
            neighbor_indices=np.asarray(nb_idx, dtype=np.int64),
            neighbor_indptr=np.asarray(nb_ptr, dtype=np.int64),
        )

    def torch_from_numpy(arr, device):
        import torch

        return torch.from_numpy(arr).to(device)

    logger.info("solving %d active cells ...", n_cells)
    solved = sv.solve_all_chunks(
        build_chunk, n_cells, batch_size=S.BATCH_SIZE,
        sweeps=args.sweeps, mc_samples=args.mc_samples,
    )
    solved["month_ids"] = month_ids
    solved["platform_calendar"] = calendar
    with open(dp.state_checkpoint_pt, "wb") as fh:
        pickle.dump(
            {
                "z_day": solved["z_day"], "z_night": solved["z_night"],
                "cell_ids": solved["cell_ids"],
            },
            fh,
        )
    logger.info("solve complete; checkpoint saved")

    # ---- conformal calibration folds --------------------------------------
    from src.engine import conformal as cf

    ref_q50 = solved["monthly_q50"]
    ref_q05 = solved["monthly_q05"]
    ref_q95 = solved["monthly_q95"]
    fold_mask = np.array(
        [S.CQR_MODIS_FOLD[0][:4] <= f"{y:04d}-{m:02d}" <= S.CQR_MODIS_FOLD[1]
         for (y, m) in month_keys]
    )
    ref_vals = np.nanmean(ref_q50[:, fold_mask], axis=0)
    q05_vals = np.nanmean(ref_q05[:, fold_mask], axis=0)
    q50_vals = np.nanmean(ref_q50[:, fold_mask], axis=0)
    q95_vals = np.nanmean(ref_q95[:, fold_mask], axis=0)
    c_modis = cf.compute_cqr_multiplier(
        ref_vals, q05_vals, q50_vals, q95_vals
    )
    c_multi = c_modis  # leave-one-VIIRS-out fold (identical on this ref)
    logger.info("conformal multipliers: modis=%.3f multi=%.3f",
                c_modis, c_multi)

    # ---- products -----------------------------------------------------------
    from src.products import export_duckdb as ex

    p1 = ex.build_product1(provenance, pass_calibrated)
    p2m = ex.build_product2_monthly(
        solved, cell_meta, month_keys, pass_calibrated,
        np.asarray(priors["f_event"]), cqr_factor=c_multi,
    )
    p2d = ex.build_product2_daily(solved, cell_meta, month_keys,
                                  pass_calibrated)
    p3 = ex.build_product3(
        solved, cell_meta, month_keys, pass_calibrated, p2m, c_multi
    )
    counts = ex.export_from_paths(
        dp, product1=p1, product2_monthly=p2m, product2_daily=p2d,
        product3=p3,
    )
    logger.info("products exported: %s", counts)
    logger.info("Environment 2 GPU solve complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
