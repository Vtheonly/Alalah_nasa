"""Shared pytest fixtures for the fire_harmonizer test suites.

Fixtures are deterministic and network-free: every dataset used by the 166+
tests is synthesized locally (tiny fixtures), never downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
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
