#!/usr/bin/env bash
# ===========================================================================
# Environment 1 wheel stager (readme Section 4, item 4).
# Verifies Python 3.12 and downloads manylinux2014_x86_64 wheels strictly for
# duckdb, polars, h3 (offline install in Environment 2).  Must NOT bundle
# PyTorch, CUDA, or pyhdf.
# Usage: bash src/ingest/build_wheels.sh [output_dir]
# ===========================================================================
set -euo pipefail

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PY_VER" != "3.12" ]; then
    echo "ERROR: Environment 1 requires Python 3.12 (found ${PY_VER})." >&2
    echo "Kaggle GPU images (v173, late 2026) run Python 3.12; refusing to" >&2
    "build an incompatible offline wheel bundle." >&2
    exit 1
fi

OUT_DIR="${1:-offline_wheels}"
mkdir -p "$OUT_DIR"

python3 -m pip download \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version "${PY_VER}" \
    -d "${OUT_DIR}" \
    duckdb polars h3

SIZE_MB=$(du -sm "${OUT_DIR}" | cut -f1)
echo "Offline wheel bundle staged in ${OUT_DIR} (${SIZE_MB} MB)."
if [ "${SIZE_MB}" -gt 80 ]; then
    echo "WARNING: wheel bundle exceeds the 80 MB budget." >&2
fi
# Guard: no torch / cuda / pyhdf artifacts may enter the offline bundle.
if ls "${OUT_DIR}" | grep -Ei 'torch|cuda|pyhdf'; then
    echo "ERROR: forbidden package in offline wheel bundle." >&2
    exit 1
fi
