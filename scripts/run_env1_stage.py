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
                    help="NASA FIRMS MAP_KEY for trailing-2026 NRT gap-fill "
                         "(default: $FIRMS_MAP_KEY; never hardcode keys)")
    ap.add_argument("--wheel-python", default=None,
                    help="Python minor version of the Env 2 runtime the "
                         "offline wheels target (default 3.13; Kaggle 4x L4 "
                         "image runs 3.13)")
    ap.add_argument("--originals-dir", default=None,
                    help="keep every original downloaded UMD file here "
                         "(otherwise deleted after parsing; size + SHA-256 "
                         "are always logged in data_provenance_manifest.json)")
    ap.add_argument("--probe-only", action="store_true",
                    help="only run the source connectivity probe and exit")
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
    if args.probe_only:
        import json as _json
        import os as _os
        res = download_data.probe_sources(_os.environ.get("FIRMS_MAP_KEY"))
        print(_json.dumps(res, indent=2))
        return 0

    years = args.years or list(range(2000, 2027))
    import os
    map_key = args.firms_map_key or os.environ.get("FIRMS_MAP_KEY")
    download_data.download_all(paths, years=years, firms_map_key=map_key,
                               originals_dir=args.originals_dir)
    report = normalize_dedup.run_normalization(paths)
    logger.info("Ingestion audit: %s", report["audit"])

    if not args.skip_wheels:
        wheels_script = REPO_ROOT / "src" / "ingest" / "build_wheels.sh"
        cmd = ["bash", str(wheels_script), str(paths.wheels_dir)]
        if args.wheel_python:
            cmd.append(args.wheel_python)
        subprocess.run(cmd, check=True)
    logger.info("Env 1 staging complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
