"""Shared pytest fixtures for the Fire Harmonizer test suite.

Deterministic, CPU-only fixtures: tiny synthetic country polygons, canonical
detection frames, and the 26-year synthetic digital twin world used by the
integration suites (test_06..test_10).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings as S  # noqa: E402
from src.config.paths import DataPaths  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic small country polygons (squares; ~1 deg) around anchor points
# ---------------------------------------------------------------------------
POLY_ANCHORS = {
    "ARG": (-67.5, -30.0),   # Chilecito region
    "DZA": (4.5, 36.6),      # Kabylie
    "BRA": (-55.0, -10.0),
    "IDN": (113.5, -2.0),
    "AUS": (149.0, -36.0),
    "USA": (-150.0, 65.0),   # Alaska-ish
}


def _square_geojson(cx: float, cy: float, half: float = 1.5) -> dict:
    ring = [[cx - half, cy - half], [cx + half, cy - half],
            [cx + half, cy + half], [cx - half, cy + half],
            [cx - half, cy - half]]
    return {"type": "Polygon", "coordinates": [ring]}


@pytest.fixture(scope="session")
def country_polygons_parquet(tmp_path_factory) -> Path:
    """Tiny square polygons for the 6 target countries (Natural Earth stand-in
    with identical geometry semantics)."""
    rows = [{"country_iso": iso, "geometry": json.dumps(_square_geojson(cx, cy))}
            for iso, (cx, cy) in POLY_ANCHORS.items()]
    p = tmp_path_factory.mktemp("polygons") / "country_polygons.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


@pytest.fixture()
def paths(tmp_path, country_polygons_parquet) -> DataPaths:
    """DataPaths rooted in a temp dir for each test."""
    import shutil
    p = DataPaths.from_env(str(tmp_path))
    p.ensure_output_dirs()
    staged = Path(p.raw_stage_dir).parent
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy(country_polygons_parquet, Path(p.country_polygons_parquet))
    return p


# ---------------------------------------------------------------------------
# Canonical-frame builders
# ---------------------------------------------------------------------------
def canonical_row(**over) -> dict:
    """One canonical detection row with sensible defaults (ARG area)."""
    row = dict(
        country_iso="ARG", sensor_family="MODIS", platform="Terra",
        data_tier="ML_SCIENCE", latitude=-29.16, longitude=-67.50,
        acq_date="2026-08-14", acq_hhmm=1330,
        t_mir_k=330.0, t_tir_k=295.0, scan_km=1.0, track_km=1.0,
        sample_idx=676.0, pixarea_km2=np.nan, frp_mw=25.0,
        conf_raw="n", conf_num=65.0, type_raw=0, version_raw="C6",
        ingest_ts="2026-11-14T00:00:00Z", dn_source="D",
    )
    row.update(over)
    return row


def canonical_frame(rows: list[dict]) -> pd.DataFrame:
    from src.ingest.normalize_dedup import CANONICAL_COLUMNS
    df = pd.DataFrame([canonical_row(**r) for r in rows])
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CANONICAL_COLUMNS + ["dn_source"]]


@pytest.fixture()
def row_builder():
    return canonical_row


@pytest.fixture()
def frame_builder():
    return canonical_frame


# ---------------------------------------------------------------------------
# Time grid helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def date_grid():
    start = date.fromisoformat(S.DATE_START)
    dates = [start + timedelta(days=t) for t in range(S.T_DAYS)]
    return np.array(dates, dtype="datetime64[D]")


def month_of_t(t: int) -> int:
    d = date.fromisoformat(S.DATE_START) + timedelta(days=int(t))
    return d.month


def year_of_t(t: int) -> int:
    d = date.fromisoformat(S.DATE_START) + timedelta(days=int(t))
    return d.year


@pytest.fixture(scope="session")
def month_year_arrays():
    ts = np.arange(S.T_DAYS)
    dates = np.array([date.fromisoformat(S.DATE_START) + timedelta(days=int(t))
                      for t in ts], dtype="datetime64[D]")
    months = pd.DatetimeIndex(dates).month.to_numpy()
    years = pd.DatetimeIndex(dates).year.to_numpy()
    return months, years


# ---------------------------------------------------------------------------
# Deterministic RNG shared by numerically sensitive suites
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: hardware-gated CUDA parity test")


def pytest_collection_modifyitems(config, items):
    if torch_cuda_available():
        return
    skip = pytest.mark.skip(reason="CUDA hardware not available on this machine "
                                   "(hardware-gated test; see final report)")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


def torch_cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Session-scoped synthetic twin fixtures (shared by test_04/05/07/08/09)
# ---------------------------------------------------------------------------
_TWIN_CACHE: dict = {}


def _short_twin():
    """2-year twin (2012-2014) pipeline result (fast, cached per session)."""
    if "short" not in _TWIN_CACHE:
        from src.validation import synthetic_twin as twin
        params = twin.twin_lognorm_params(42)
        t_end = twin.t_of("2014-12-31") + 1
        cfg = twin.TwinConfig(t_days=t_end, outages=(), events=(),
                              lognorm=params)
        _TWIN_CACHE["short"] = twin.run_twin_pipeline(cfg)
    return _TWIN_CACHE["short"]


def _full_twin():
    """Full 26-year twin pipeline result (cached per session, ~4 min)."""
    if "full" not in _TWIN_CACHE:
        from src.validation import synthetic_twin as twin
        _TWIN_CACHE["full"] = twin.run_twin_pipeline(twin.TwinConfig())
    return _TWIN_CACHE["full"]


@pytest.fixture(scope="session")
def short_twin():
    return _short_twin()


@pytest.fixture(scope="session")
def full_twin():
    return _full_twin()


@pytest.fixture(scope="session")
def twin_diagnostics(full_twin):
    from src.validation.diagnostics import run_diagnostics
    return run_diagnostics(full_twin["monthly"], full_twin["truth"],
                           None, full_twin["pass_obs"],
                           full_twin["pass_cal"])
