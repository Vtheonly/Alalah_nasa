"""Environment 1 staging entry point (internet ON, CPU).

Orchestrates the complete Env-1 pipeline (readme.md Section 8, Phase 1):
  1. Download + normalize Science archives (UMD primary, Portolan failover),
     ancillary HDF4 cloud / burned-area grids, Natural Earth polygons and the
     trailing-2026 FIRMS NRT backfill.
  2. Emit ``staged_data/`` Zstd Parquet + ``offline_wheels/`` bundle.
  3. Print the mandatory coverage audit (Gate 1).

Usage:
  python scripts/run_env1_stage.py [--start-year 2000] [--end-year 2026]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import paths as paths_mod  # noqa: E402
from src.ingest import download_data, normalize_dedup  # noqa: E402

logger = logging.getLogger("fire_harmonizer.scripts.run_env1_stage")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Environment 1 data & wheel stager"
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--skip-umd", action="store_true",
        help="force Source Cooperative Portolan mirror failover",
    )
    parser.add_argument("--skip-nrt", action="store_true")
    parser.add_argument(
        "--wheels-only", action="store_true",
        help="only refresh the offline wheel bundle",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    dp = paths_mod.DataPaths.from_env()
    dp.ensure_dirs()

    if not args.wheels_only:
        summary = download_data.run_env1_download(
            data_paths=dp,
            start_year=args.start_year,
            end_year=args.end_year,
            include_umd=not args.skip_umd,
            include_nrt=not args.skip_nrt,
        )
        logger.info("Env 1 download summary: %s", summary)

        # Normalize whatever landed in staged_raw/ through the canonical
        # schema, polygon clipping and tier-dominance dedup pipeline.
        frames = []
        for parquet in sorted(dp.raw_stage_dir.rglob("*.parquet")):
            raw = download_data.pd.read_parquet(parquet)
            name = parquet.name
            if "line" in raw.columns:  # UMD verbatim-line chunks
                fname = str(raw["source_file"].iloc[0])
                if "MCD14ML" in fname.upper():
                    frames.append(
                        normalize_dedup.parse_mcd14ml(raw["line"].tolist())
                    )
                else:
                    frames.append(
                        normalize_dedup.parse_vxx14imgml(
                            csv_text="\n".join(raw["line"].tolist()),
                            filename=fname,
                        )
                    )
            elif "firms_" in name:
                frames.append(normalize_dedup.parse_firms_api_csv(raw))
            elif "portolan" in name:
                frames.append(normalize_dedup.parse_portolan(raw))
        polygons = download_data.pd.read_parquet(dp.country_polygons_parquet)
        df, audit = normalize_dedup.run_normalization(
            frames, polygons, out_dir=dp.active_fires_dir
        )
        logger.info("coverage audit: %s", json.dumps(audit, indent=2))

    # Offline wheel bundle (duckdb, polars, h3 ONLY).
    import subprocess

    rc = subprocess.call(
        [
            "bash",
            str(Path(__file__).resolve().parents[1]
                / "src" / "ingest" / "build_wheels.sh"),
            str(dp.offline_wheels_dir),
        ]
    )
    if rc != 0:
        logger.error("build_wheels.sh failed with rc=%d", rc)
        return rc
    logger.info("Environment 1 staging complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
