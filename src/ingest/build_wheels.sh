#!/usr/bin/env bash
# ===========================================================================
# Environment 1 wheel stager (readme Section 4, item 4).
# Downloads prebuilt manylinux x86_64 wheels strictly for duckdb, polars, h3
# for offline (--no-index) installation in Environment 2.  Must NOT bundle
# PyTorch, CUDA, or pyhdf.
#
# The TARGET Python minor version is the one Environment 2 runs, NOT
# necessarily the stager's own interpreter (h3 ships C extensions, so a
# cp312 wheel will not install on a cp313 runtime).  The readme assumed
# Kaggle image v173 = Python 3.12; the verified Kaggle 4x L4 runtime used by
# this project reports Python 3.13.16, so the default target is 3.13.
#
# Usage: bash src/ingest/build_wheels.sh [output_dir] [target_py_ver]
#        TARGET_PY_VER=3.12 bash src/ingest/build_wheels.sh offline_wheels
# ===========================================================================
set -euo pipefail

OUT_DIR="${1:-offline_wheels}"
PY_VER="${2:-${TARGET_PY_VER:-3.13}}"

case "${PY_VER}" in
    3.12|3.13) ;;
    *)
        echo "ERROR: unsupported Environment 2 target Python ${PY_VER}" >&2
        echo "       (supported: 3.12, 3.13)." >&2
        exit 1
        ;;
esac

HOST_PY=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Stager interpreter: Python ${HOST_PY}; target Env 2 runtime: Python ${PY_VER}"

mkdir -p "${OUT_DIR}"

# --python-version + --only-binary lets pip resolve wheels for the TARGET
# interpreter regardless of the host interpreter.  Several manylinux tags are
# accepted because recent h3/polars/duckdb wheels are tagged
# manylinux_2_17 / manylinux_2_28 rather than the legacy manylinux2014 alias.
python3 -m pip download \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_28_x86_64 \
    --implementation cp \
    --python-version "${PY_VER}" \
    -d "${OUT_DIR}" \
    duckdb polars h3

# Every downloaded wheel must be ABI-compatible with the target runtime.
PY_TAG="cp${PY_VER/./}"
for whl in "${OUT_DIR}"/*.whl; do
    name=$(basename "${whl}")
    if [[ "${name}" != *"${PY_TAG}"* && "${name}" != *"abi3"* && "${name}" != *"py3-none"* ]]; then
        echo "ERROR: ${name} is not compatible with ${PY_TAG}." >&2
        exit 1
    fi
done

SIZE_MB=$(du -sm "${OUT_DIR}" | cut -f1)
echo "Offline wheel bundle staged in ${OUT_DIR} (${SIZE_MB} MB) for ${PY_TAG}."
if [ "${SIZE_MB}" -gt 80 ]; then
    echo "WARNING: wheel bundle exceeds the 80 MB budget." >&2
fi
# Guard: no torch / cuda / pyhdf artifacts may enter the offline bundle.
if ls "${OUT_DIR}" | grep -Ei 'torch|cuda|pyhdf'; then
    echo "ERROR: forbidden package in offline wheel bundle." >&2
    exit 1
fi
echo "Install in Environment 2 with:"
echo "  pip install --no-index --find-links=${OUT_DIR} duckdb polars h3"
