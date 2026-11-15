#!/usr/bin/env python3
"""Build the SYNTHETIC demonstration store used by the web app when the real
harmonization output is unavailable or not validated.

The store has the exact schema of the real engine export (written by the
engine's own ``export_duckdb``) plus a ``dataset_provenance`` table marking
it as SYNTHETIC_DEMO.  It is never mixed with real results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.webapp.providers.synthetic import build_synthetic_store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/synthetic_demo.duckdb")
    ap.add_argument("--seed", type=int, default=20261115)
    args = ap.parse_args(argv)
    print(json.dumps(build_synthetic_store(args.out, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
