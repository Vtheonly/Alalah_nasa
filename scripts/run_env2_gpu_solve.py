#!/usr/bin/env python3
"""Environment 2 GPU solve: multi-GPU float64 MAP-Laplace + M=200 joint
trajectories + held-out conformal folds + finalized Products 1-3 ->
``fire_harmonized.duckdb`` + Zstd Parquet.

Uses exactly the same harmonization path as the synthetic digital twin
(``src.engine.conformal.harmonize_and_calibrate``): main solve, VIIRS-masked
2018-2020 + leave-S-NPP-out 2018-2021 folds, conformal rescaling,
climatology indices, raw baselines, evidence / quality / attribution flags,
sparse daily rows.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import settings as S  # noqa: E402
from src.config.paths import DataPaths  # noqa: E402
from src.engine import conformal as conf  # noqa: E402
from src.engine import cuda_kernels  # noqa: E402
from src.engine import solver as slv  # noqa: E402
from src.products import export_duckdb as ex  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env2_gpu_solve")


def load_env2_inputs(paths):
    with open(paths.priors_pkl, "rb") as fh:
        blob = pickle.load(fh)
    with open(paths.calibration_pkl, "rb") as fh:
        artifacts = pickle.load(fh)
    cloud_df = (pd.read_parquet(paths.cloud_parquet)
                if Path(paths.cloud_parquet).exists() else None)
    return {"priors": blob["priors"],
            "platform_active": blob.get("platform_active_day"),
            "hex_pyromes": pd.read_parquet(paths.pyromes_parquet),
            "pass_cal": pd.read_parquet(paths.pass_calibrated_parquet),
            "pass_obs": pd.read_parquet(paths.pass_obs_parquet),
            "prov": pd.read_parquet(paths.provenance_base_parquet),
            "artifacts": artifacts, "cloud_df": cloud_df}


def timing_gate(inp, batch_size: int) -> dict:
    """Readme Gate 4: one-chunk benchmark (wall time + peak VRAM)."""
    cells = np.array(sorted(set(int(c) for c in
                                inp["pass_cal"]["h3_index"].unique())))
    chunk = cells[:batch_size]
    t0 = time.time()
    data = slv.build_observation_tensors(
        inp["pass_cal"], chunk, inp["hex_pyromes"], inp["priors"],
        inp["artifacts"], inp["cloud_df"],
        platform_active=inp["platform_active"])
    t_build = time.time() - t0
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)
    t1 = time.time()
    slv.solve_chunk(data, use_cuda=cuda_kernels.cuda_available(), device=dev,
                    keep_solver=False)
    t_solve = time.time() - t1
    peak = (torch.cuda.max_memory_allocated(0) / 2 ** 30
            if dev.startswith("cuda") else float("nan"))
    n_chunks = int(np.ceil(len(cells) / batch_size))
    n_gpu = max(1, torch.cuda.device_count())
    est_h = 3.0 * n_chunks * t_solve / n_gpu / 3600.0   # main + 2 folds
    return {"device": dev, "cuda_kernels": cuda_kernels.cuda_available(),
            "cuda_compile_error": str(cuda_kernels.compile_error() or ""),
            "chunk_cells": int(len(chunk)), "build_s": round(t_build, 1),
            "solve_s": round(t_solve, 1), "peak_vram_gib": round(peak, 2),
            "n_chunks": n_chunks, "est_total_gpu_hours": round(est_h, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 2 GPU solve + export")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=S.BATCH_SIZE)
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="concurrent chunk workers per GPU (each < 6 GB)")
    ap.add_argument("--timing-gate-only", action="store_true",
                    help="benchmark one chunk and exit (readme Gate 4)")
    args = ap.parse_args(argv)
    paths = DataPaths.from_env(args.base_dir)
    t0 = time.time()
    logger.info("CUDA devices: %d | fused kernels: %s",
                torch.cuda.device_count() if torch.cuda.is_available() else 0,
                cuda_kernels.cuda_available())
    if torch.cuda.is_available() and not cuda_kernels.cuda_available():
        logger.warning("CUDA present but load_inline build failed (%s): the "
                       "pure-torch Thomas fallback is ~50x slower",
                       cuda_kernels.compile_error())
    inp = load_env2_inputs(paths)
    gate = timing_gate(inp, args.batch_size)
    logger.info("timing gate: %s", gate)
    if args.timing_gate_only:
        return 0

    h = conf.harmonize_and_calibrate(
        inp["pass_cal"], inp["pass_obs"], inp["prov"], inp["hex_pyromes"],
        inp["priors"], inp["artifacts"], inp["cloud_df"],
        inp["platform_active"], t_days=S.T_DAYS, batch_size=args.batch_size,
        paths=paths, keep_states=False,
        workers_per_gpu=args.workers_per_gpu)
    logger.info("solve + folds done (%.0fs); c_modis=%s c_multi=%s",
                time.time() - t0, h["c_modis"], h["c_multi"])
    products = ex.build_products(h["monthly"], h["daily"], inp["pass_obs"],
                                 inp["prov"], inp["pass_cal"])
    written = ex.export_duckdb(products, paths.duckdb_path,
                               parquet_dir=str(Path(paths.duckdb_path).parent),
                               diagnostics={"timing_gate": gate,
                                            "c_modis": h["c_modis"],
                                            "c_multi": h["c_multi"]})
    with open(Path(paths.duckdb_path).parent / "conformal_factors.pkl",
              "wb") as fh:
        pickle.dump({"c_modis": h["c_modis"], "c_multi": h["c_multi"],
                     "timing_gate": gate}, fh)
    sizes = {k: round(Path(v).stat().st_size / 2 ** 20, 1)
             for k, v in written.items()}
    logger.info("exported %s (MB) + %s in %.0fs", sizes, paths.duckdb_path,
                time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
