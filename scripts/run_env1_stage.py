#!/usr/bin/env python3
"""Environment 1 entrypoint: stage data + offline wheels (Internet ON, CPU).

Readme Section 8 Phase 1: download active fires / cloud / burned-area /
country polygons (with autonomous failover), normalize + clip + dedup into
``active_fires/`` Zstd Parquet, and build the offline wheel bundle for
Environment 2 (duckdb, polars, h3 only).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import DataPaths  # noqa: E402
from src.ingest import download_data, normalize_dedup  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_env1_stage")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Env 1 data + wheel stager")
    ap.add_argument("--base-dir", default=None,
                    help="Optional base dir (defaults to Kaggle/local autodetect)")
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="Years to stage (default 2000..2026)")
    ap.add_argument("--firms-map-key", default=None,
                    help="NASA FIRMS MAP_KEY for trailing-2026 NRT gap-fill")
    ap.add_argument("--skip-wheels", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip network work; only verify the layout")
    args = ap.parse_args(argv)

    paths = DataPaths.from_env(args.base_dir)
    paths.ensure_output_dirs()
    logger.info("Env 1 staging root: %s", paths.raw_stage_dir)

    if args.dry_run:
        logger.info("dry-run: layout OK (%s)", paths.root)
        return 0

    years = args.years or list(range(2000, 2027))
    download_data.download_all(paths, years=years,
                               firms_map_key=args.firms_map_key)
    report = normalize_dedup.run_normalization(paths)
    logger.info("Ingestion audit: %s", report["audit"])

    if not args.skip_wheels:
        wheels_script = REPO_ROOT / "src" / "ingest" / "build_wheels.sh"
        subprocess.run(["bash", str(wheels_script), str(paths.wheels_dir)],
                       check=True)
    logger.info("Env 1 staging complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
