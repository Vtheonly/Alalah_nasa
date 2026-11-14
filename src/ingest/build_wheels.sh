#!/usr/bin/env bash
# Environment 1 wheel stager (readme.md Section 4, src/ingest/build_wheels.sh).
#
# Downloads Python-3.12 manylinux2014_x86_64 prebuilt wheels STRICTLY for
# duckdb, polars, and h3 so that Environment 2 (internet-OFF Kaggle GPU) can
# `pip install --no-index --find-links=<dir> duckdb polars h3`.
#
# MUST NOT bundle PyTorch, CUDA, or pyhdf (those are either preinstalled on
# Kaggle or compiled locally in Environment 1).
set -euo pipefail

OUT_DIR="${1:-offline_wheels}"
PY_VER="${2:-}"

# --- Detect the active Python minor version (must be 3.12 on Env 1) --------
if [ -z "${PY_VER}" ]; then
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
fi
echo "[build_wheels] active Python version: ${PY_VER}"
if [ "${PY_VER}" != "3.12" ]; then
  echo "[build_wheels] WARNING: Environment 1 targets Python 3.12; wheels built" \
       "for ${PY_VER} will not install on the Kaggle Python 3.12 image." >&2
fi

mkdir -p "${OUT_DIR}"

# --- Download the three allowed wheels --------------------------------------
python3 -m pip download \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version "${PY_VER}" \
  --no-deps \
  -d "${OUT_DIR}" \
  duckdb polars h3

# --- Guard: refuse forbidden wheels (torch / CUDA / pyhdf) ------------------
FORBIDDEN="$(find "${OUT_DIR}" -maxdepth 1 -type f \( -name 'torch*' -o -name 'cuda*' -o -name 'pyhdf*' -o -name 'nvidia*' \) || true)"
if [ -n "${FORBIDDEN}" ]; then
  echo "[build_wheels] ERROR: forbidden wheels present in ${OUT_DIR}:" >&2
  echo "${FORBIDDEN}" >&2
  exit 1
fi

TOTAL_MB="$(du -sm "${OUT_DIR}" | cut -f1)"
echo "[build_wheels] bundle size: ${TOTAL_MB} MB (budget <= 80 MB)"
if [ "${TOTAL_MB}" -gt 80 ]; then
  echo "[build_wheels] ERROR: wheel bundle exceeds 80 MB budget." >&2
  exit 1
fi

echo "[build_wheels] OK -> ${OUT_DIR}"
