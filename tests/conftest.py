"""Shared pytest fixtures for the fire_harmonizer test suites.

Fixtures are deterministic and network-free: every dataset used by the 166+
tests is synthesized locally (tiny fixtures), never downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Repository root on sys.path so `import src.*` works under plain pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Synthetic country polygons (WKB, shapely-compatible)
# --------------------------------------------------------------------------- #
_SYNTHETIC_COUNTRY_POLYGONS = {
    # Argentina: western boundary bulges west near La Rioja (~-68.4 at
    # lat -31) so Chilecito (-67.5, -29.16) is inside while Chilean points
    # (lon <= -70) stay outside.
    "ARG": [(-68.5, -54.9), (-63.5, -55.0), (-53.6, -34.0), (-53.6, -21.5),
            (-62.0, -22.0), (-67.8, -27.0), (-68.4, -31.0), (-70.3, -38.0),
            (-71.9, -45.0)],
    # Algeria: coastline keeps Algiers (3.05, 36.75) and Hassi Messaoud
    # (6.05, 31.75) inside while Tunisia (9.8, 36.9) stays outside.
    "DZA": [(-8.7, 19.0), (-8.7, 28.0), (-2.0, 33.0), (2.0, 37.1),
            (7.8, 37.0), (8.2, 34.5), (6.5, 31.0), (4.0, 26.0),
            (-4.0, 22.0)],
    "BRA": [(-74.0, -34.0), (-35.0, -34.0), (-35.0, 5.4), (-60.0, 5.0),
            (-70.0, -10.0), (-74.0, -20.0)],
    "IDN": [(95.2, -11.0), (141.1, -11.0), (141.1, 6.3), (120.0, 6.0),
            (95.2, 0.0)],
    "AUS": [(112.9, -43.7), (153.7, -43.7), (153.7, -10.6), (130.0, -11.0),
            (112.9, -25.0)],
    # USA: CONUS polygon; Alaska is added as a second MultiPolygon part
    # inside the fixture builder below.
    "USA": [(-124.9, 24.4), (-66.9, 24.4), (-66.9, 49.5), (-95.0, 49.0),
            (-124.9, 45.0)],
}


@pytest.fixture(scope="session")
def country_polygons() -> pd.DataFrame:
    """WKB polygon table mimicking the Natural Earth Admin-0 parquet."""
    from shapely import wkb as shapely_wkb
    from shapely.geometry import MultiPolygon, Polygon

    rows = []
    for iso, ring in _SYNTHETIC_COUNTRY_POLYGONS.items():
        geom = Polygon(ring)
        if iso == "USA":
            alaska = Polygon(
                [(-168.0, 63.0), (-142.0, 63.0), (-142.0, 71.0),
                 (-166.0, 71.0)]
            )
            geom = MultiPolygon([geom, alaska])
        rows.append(
            {
                "country_iso": iso,
                "admin_name": iso,
                "geometry_wkb": shapely_wkb.dumps(geom),
                "source": "synthetic-test-fixture",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


# --------------------------------------------------------------------------- #
# Session-scoped synthetic digital twin + solved results (shared by
# test_07 / test_08 / test_09; one ~1-2 minute solve for the whole suite).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def twin():
    """Deterministic 26-year digital twin (known ground truth)."""
    from src.validation import synthetic_twin as st

    return st.generate_twin(n_cells=5, T=9510, seed=42)


@pytest.fixture(scope="session")
def twin_solved(twin):
    """Full-batch MAP-Laplace solve of the twin with joint sampling."""
    from src.engine import solver as sv

    return sv.solve_all_chunks(
        lambda cells, dev: twin_helpers_build(twin, cells, dev),
        len(twin["cells"]),
        batch_size=1000,
        sweeps=6,
        newton_steps=15,
        mc_samples=100,
        sample_subbatch=5,
    )


@pytest.fixture(scope="session")
def twin_ablation_solved(twin):
    """2019-2021 VIIRS-masked ablation solve (MODIS-only reconstruction).

    Masked VIIRS platforms are marked non-operational inside the window so
    their removed detections never become false clear-sky non-detections.
    """
    from src.engine import solver as sv
    from src.validation.synthetic_twin import build_chunk_problem

    t_lo = (pd.Timestamp("2019-01-01") - pd.Timestamp("2000-11-01")).days
    t_hi = (pd.Timestamp("2021-12-31") - pd.Timestamp("2000-11-01")).days
    viirs = ("SNPP", "NOAA-20", "NOAA-21")

    def build(cells, dev):
        return build_chunk_problem(
            twin, cells, dev, mask_platforms=viirs,
            mask_window=(t_lo, t_hi),
        )

    return sv.solve_all_chunks(
        build, len(twin["cells"]), batch_size=1000, sweeps=6,
        newton_steps=15, sample=False,
    )


def twin_helpers_build(twin, cells, dev):
    from src.validation.synthetic_twin import build_chunk_problem

    return build_chunk_problem(twin, cells, dev)


@pytest.fixture(scope="session")
def twin_products(twin, twin_solved):
    """The four product frames built from the twin solve (shared by
    test_09 schema invariants and test_10 API fixtures)."""
    from src.products import export_duckdb as ex

    cells = twin["cells"]
    latlon = np.asarray(twin["latlon"], dtype=float)
    cell_meta = pd.DataFrame(
        {
            "cell_pos": np.arange(len(cells)),
            "h3_index": np.asarray(cells, dtype=np.int64),
            "pyrome_id": np.zeros(len(cells), dtype=int),
            "latitude": latlon[:, 0],
            "longitude": latlon[:, 1],
            "country_iso": ["ARG"] * len(cells),
        }
    )
    pass_df = twin["pass_df"].copy()
    pass_df["data_tier"] = "ML_SCIENCE"
    solved = dict(twin_solved)
    solved["month_ids"] = twin["month_ids"]
    solved["platform_calendar"] = twin["platform_calendar"]
    f_event = np.full(7, 5.0)
    p1 = ex.build_product1(
        _twin_provenance_frame(twin), pass_df
    )
    p2m = ex.build_product2_monthly(
        solved, cell_meta, twin["month_keys"], pass_df, f_event,
        cqr_factor=1.5,
    )
    p2d = ex.build_product2_daily(solved, cell_meta, twin["month_keys"],
                                  pass_df)
    p3 = ex.build_product3(
        solved, cell_meta, twin["month_keys"], pass_df, p2m, 1.5
    )
    return {"p1": p1, "p2m": p2m, "p2d": p2d, "p3": p3,
            "cell_meta": cell_meta, "pass_df": pass_df}


def _twin_provenance_frame(twin):
    """Minimal provenance-base frame for Product 1 schema verification."""
    df = twin["pass_df"].head(50).copy()
    df["acq_date"] = df["solar_date"]
    df["acq_hhmm"] = 1345
    df["t_mir_k"] = 340.0
    df["t_tir_k"] = 300.0
    df["conf_raw"] = "n"
    df["type_raw"] = 0
    df["data_tier"] = "ML_SCIENCE"
    df["scan_km"] = np.nan
    df["track_km"] = np.nan
    df["sza_deg"] = 40.0
    df["pixel_area_km2"] = 0.2
    df["vza_deg"] = 30.0
    df["is_static_industrial"] = False
    df["is_sama_filtered"] = False
    df["bowtie_weight"] = 1.0
    df["reliability_prob"] = 0.9
    df["event_cluster_id"] = 0
    df["frp_mw"] = df["y_frp_mw"]
    df["ingest_ts"] = "2026-11-14T00:00:00Z"
    return df
