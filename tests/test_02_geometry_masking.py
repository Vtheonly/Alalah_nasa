"""test_02: spherical geometry, bow-tie conservation, masking, H3 3-coloring.

18 tests (readme.md Section 7 item 2): exact spherical Sample -> scan/track/
VZA (MODIS nadir 1.00 km^2, edge ~9.72 km^2), monotone spline VZA inversion
in [0, 65], MODIS bow-tie overlap conservation (2 x 40 MW -> 40 MW), SAMA
nighttime low-conf filter, H3-Res8 gas-flare isolation absorbing 200 m
geolocation jitter while preserving seasonal fires, h3 v3/v4 int64 <-> str
compatibility, and strict 3-coloring validity (color[u] != color[v] across
all adjacent edges).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.geometry import h3_utils
from src.geometry import preprocess_geometry as pg
from src.geometry.solar_geometry import (
    area_to_vza_deg,
    compute_solar_time_and_sza,
    modis_sample_to_geometry,
    resolve_pixel_geometry,
    viirs_sample_to_geometry,
)


# --------------------------------------------------------------------------- #
# Exact spherical pixel geometry
# --------------------------------------------------------------------------- #

def test_02_modis_nadir_geometry():
    scan, track, vza, area = modis_sample_to_geometry(676.5)
    assert scan == pytest.approx(1.00, abs=1e-3)
    assert track == pytest.approx(1.00, abs=1e-3)
    assert vza == pytest.approx(0.0, abs=1e-6)
    assert area == pytest.approx(1.00, abs=2e-3)


def test_02_modis_edge_geometry():
    for edge in (0, 1353):
        scan, track, vza, area = modis_sample_to_geometry(edge)
        assert scan == pytest.approx(4.84, abs=0.02)
        assert track == pytest.approx(2.01, abs=0.02)
        assert area == pytest.approx(9.72, abs=0.06)
        assert vza == pytest.approx(65.5, abs=0.3)


def test_02_modis_geometry_monotone_off_nadir():
    samples = np.linspace(676.5, 0, 40)
    _, _, vza, area = modis_sample_to_geometry(samples)
    assert np.all(np.diff(vza) > 0)
    assert np.all(np.diff(area) > 0)


def test_02_viirs_area_range_nadir_to_edge():
    _, _, vza_n, area_n = viirs_sample_to_geometry(3199)
    _, _, vza_e, area_e = viirs_sample_to_geometry(0)
    assert area_n == pytest.approx(0.141, abs=5e-4)
    assert area_e == pytest.approx(0.620, abs=5e-4)
    assert vza_n < 0.1
    assert vza_e == pytest.approx(69.5, abs=0.3)
    # Areas across the full swath remain inside the spec range.
    areas = viirs_sample_to_geometry(np.arange(0, 6400, 64))[3]
    assert areas.min() >= 0.141 - 1e-9
    assert areas.max() <= 0.620 + 1e-9


def test_02_viirs_pixarea_direct_with_clip():
    _, _, vza, area = resolve_pixel_geometry(
        "VIIRS", sample_idx=3200, pixarea_km2=0.90
    )
    assert area == pytest.approx(0.75)  # clipped to [0.12, 0.75]
    _, _, vza2, area2 = resolve_pixel_geometry(
        "VIIRS", sample_idx=3200, pixarea_km2=0.05
    )
    assert area2 == pytest.approx(0.12)
    # scan = track = sqrt(A) proxy; VZA recovered from the knot spline.
    scan, track, vza3, area3 = resolve_pixel_geometry(
        "VIIRS", sample_idx=None, pixarea_km2=0.620
    )
    assert scan == pytest.approx(np.sqrt(0.620), abs=1e-9)
    assert vza3 == pytest.approx(65.0, abs=1e-6)


def test_02_scan_track_area_and_vza_inversion():
    # MODIS path: A = clip(scan*track, 0.12, 12), VZA via knot interp.
    _, _, vza, area = resolve_pixel_geometry(
        "MODIS", sample_idx=None, scan_km=4.84, track_km=2.01
    )
    assert area == pytest.approx(9.7284, abs=1e-3)
    assert vza == pytest.approx(65.0, abs=1e-6)
    _, _, vza_v, area_v = resolve_pixel_geometry(
        "VIIRS", sample_idx=None, pixarea_km2=None, scan_km=0.43, track_km=0.43
    )
    assert area_v == pytest.approx(0.1849, abs=1e-9)
    assert vza_v == pytest.approx(area_to_vza_deg(0.1849, "VIIRS"), abs=1e-9)


def test_02_area_to_vza_knot_monotonicity():
    areas_m = np.array([k[0] for k in S.MODIS_AREA_VZA_KNOTS])
    vzas_m = np.array([k[1] for k in S.MODIS_AREA_VZA_KNOTS])
    assert np.allclose(area_to_vza_deg(areas_m, "MODIS"), vzas_m, atol=1e-9)
    areas_v = np.array([k[0] for k in S.VIIRS_AREA_VZA_KNOTS])
    vzas_v = np.array([k[1] for k in S.VIIRS_AREA_VZA_KNOTS])
    assert np.allclose(area_to_vza_deg(areas_v, "VIIRS"), vzas_v, atol=1e-9)
    # Monotone on a dense grid, clamped to [0, 65].
    dense = np.linspace(0.1, 15.0, 200)
    assert np.all(np.diff(area_to_vza_deg(dense, "MODIS")) >= -1e-12)
    assert area_to_vza_deg(99.0, "MODIS") == pytest.approx(65.0)


def test_02_vza_inversion_roundtrip_vs_sample_path():
    samples = np.linspace(676.5, 10, 30)
    _, _, vza_exact, area = modis_sample_to_geometry(samples)
    vza_inv = area_to_vza_deg(area, "MODIS")
    # The spec defines the forward path by exact spherical equations and the
    # inversion path by the coarse knot table; the two agree at the knots and
    # stay monotone-consistent within ~10 deg between them (largest gap sits
    # in the 1.00 -> 1.35 km^2 first segment spanning 0 -> 20 deg).
    assert np.max(np.abs(vza_exact - vza_inv)) < 10.0
    # Inversion is monotone in the exact area.
    assert np.all(np.diff(vza_inv) >= -1e-12)
    # Exact agreement at the knot areas themselves.
    for knot_area, knot_vza in S.MODIS_AREA_VZA_KNOTS:
        assert area_to_vza_deg(knot_area, "MODIS") == pytest.approx(
            knot_vza, abs=1e-9
        )


# --------------------------------------------------------------------------- #
# Bow-tie conservation
# --------------------------------------------------------------------------- #

def _modis_offnadir_pair():
    """Two overlapping off-nadir MODIS pixels (A ~ 9.7 km^2, track ~ 2 km)."""
    scan, track, vza, area = modis_sample_to_geometry(30)
    return scan, track, area


def test_02_bowtie_overlap_conserves_frp():
    scan, track, area = _modis_offnadir_pair()
    assert area > S.BOWTIE_AREA_THRESHOLD_KM2
    # Two pixels 1 km apart (within 0.7 * mean track ~ 1.4 km).
    lats = np.array([-29.10, -29.109])  # ~1 km apart along track
    lons = np.array([-67.45, -67.45])
    ecef = pg.latlon_to_ecef(lats, lons)
    w = pg.bowtie_weights(
        ecef, np.array([area, area]), np.array([track, track]), "MODIS"
    )
    assert area * 2 > 2 * S.BOWTIE_AREA_THRESHOLD_KM2
    # Each pixel has one bow-tie neighbour -> weight 0.5 each.
    assert w[0] == pytest.approx(0.5) and w[1] == pytest.approx(0.5)
    # FRP conservation: 2 x 40 MW overlapping = 40 MW effective.
    total = float(np.sum(w * np.array([40.0, 40.0])))
    assert total == pytest.approx(40.0)


def test_02_bowtie_weights_unity_for_small_pixels():
    # VIIRS pixels never participate; small MODIS pixels stay at 1.0.
    ecef = pg.latlon_to_ecef(
        np.array([-29.10, -29.1009]), np.array([-67.45, -67.45])
    )
    w_v = pg.bowtie_weights(
        ecef, np.array([0.2, 0.2]), np.array([0.45, 0.45]), "VIIRS"
    )
    assert np.all(w_v == 1.0)
    w_m = pg.bowtie_weights(
        ecef, np.array([1.0, 1.0]), np.array([1.0, 1.0]), "MODIS"
    )
    assert np.all(w_m == 1.0)


# --------------------------------------------------------------------------- #
# SAMA & gas-flare masking
# --------------------------------------------------------------------------- #

def _viirs_frame(lats, lons, dns, confs, types=None, dates=None, hhmm=1342):
    n = len(lats)
    return pd.DataFrame(
        {
            "sensor_family": ["VIIRS"] * n,
            "platform": ["SNPP"] * n,
            "latitude": lats,
            "longitude": lons,
            "acq_date": dates or ["2021-08-01"] * n,
            "acq_hhmm": [hhmm] * n,
            "t_mir_k": [340.0] * n,
            "t_tir_k": [300.0] * n,
            "scan_km": [np.nan] * n,
            "track_km": [np.nan] * n,
            "sample_idx": pd.array([3200] * n, dtype="Int64"),
            "pixarea_km2": [0.2] * n,
            "frp_mw": [5.0] * n,
            "conf_raw": confs,
            "conf_num": [
                S.CONF_LABEL_VALUES[c] if c in S.CONF_LABEL_VALUES else 50.0
                for c in confs
            ],
            "type_raw": types if types is not None else [0] * n,
            "dn_flag": dns,
            "solar_date": dates or ["2021-08-01"] * n,
        }
    )


def test_02_sama_nighttime_lowconf_filter():
    df = _viirs_frame(
        lats=[-30.0, -30.0, -30.0, -30.0, 10.0],
        lons=[-45.0, -45.0, -45.0, -45.0, -60.0],
        dns=["N", "N", "D", "N", "N"],
        confs=["l", "l", "l", "h", "l"],
    )
    mask = pg.sama_nighttime_lowconf_mask(df)
    # SAMA box: lon in [-110, 11], lat in [-55, 7].
    assert mask.tolist() == [True, True, False, False, False]
    # numeric low confidence (< 30) is filtered too
    df2 = df.copy()
    df2.loc[0, "conf_num"] = 20.0
    assert pg.sama_nighttime_lowconf_mask(df2)[0]


def test_02_gas_flare_jitter_absorption_keeps_seasonal_fires():
    rng = np.random.default_rng(7)
    # Anchor the flare at the exact center of its H3-Res8 micro-cell so the
    # +/-200 m geolocation jitter stays inside the ~0.74 km^2 cell (the
    # absorption property the H3-Res8 resolution choice guarantees).
    from src.geometry import h3_utils
    cell8 = int(h3_utils.latlon_to_h3_int(31.75, 6.05, S.H3_FLARE_RES)[0])
    c_lat, c_lon = h3_utils.h3_center_latlon(cell8)
    rows = []
    # Persistent Hassi Messaoud flare: 4 years x 20 days/yr, day+night
    # symmetric, with 200 m geolocation jitter.
    jitter = rng.uniform(-0.0018, 0.0018, size=(4 * 40, 2))
    for k in range(4):
        for d in range(20):
            for dn in ("D", "N"):
                j = jitter[k * 40 + d * 2 + (dn == "N")]
                rows.append(
                    (
                        f"20{18 + k}-05-{1 + d % 28:02d}",
                        c_lat + j[0], c_lon + j[1], dn,
                    )
                )
    # Seasonal savanna wildfire: 3 years but only August, daytime only.
    for k in range(3):
        for d in range(20):
            rows.append(
                (f"20{15 + k}-08-{1 + d % 28:02d}", -12.60, -55.95, "D")
            )
    dates = [r[0] for r in rows]
    lats = np.array([r[1] for r in rows])
    lons = np.array([r[2] for r in rows])
    dns = [r[3] for r in rows]
    df = _viirs_frame(lats, lons, dns, ["n"] * len(rows), dates=dates)
    mask = pg.flare_microcell_mask(df)
    flare_rows = mask[: len(rows) - 60]
    wildfire_rows = mask[len(rows) - 60:]
    assert flare_rows.all()          # persistent jittered flare captured
    assert not wildfire_rows.any()   # seasonal daytime wildfire preserved


def test_02_flare_type2_directly_flagged():
    df = _viirs_frame(
        lats=[31.75], lons=[6.05], dns=["D"], confs=["n"], types=[2],
        dates=["2021-08-01"],
    )
    assert pg.flare_microcell_mask(df).tolist() == [True]


# --------------------------------------------------------------------------- #
# h3 v3/v4 compatibility & 3-coloring
# --------------------------------------------------------------------------- #

def test_02_h3_int_str_roundtrip_and_resolution():
    cell = h3_utils.latlon_to_h3_int(-29.16, -67.50, S.H3_MAIN_RES)[0]
    cell_str = h3_utils.h3_int_to_str(cell)
    assert isinstance(cell_str, str) and len(cell_str) == 15
    assert h3_utils.h3_str_to_int(cell_str) == int(cell)
    assert h3_utils.h3_resolution(cell) == S.H3_MAIN_RES
    cell8 = h3_utils.latlon_to_h3_int(-29.16, -67.50, S.H3_FLARE_RES)[0]
    assert h3_utils.h3_resolution(cell8) == S.H3_FLARE_RES


def test_02_h3_area_res6_and_res8():
    cell6 = h3_utils.latlon_to_h3_int(-29.16, -67.50, 6)[0]
    cell8 = h3_utils.latlon_to_h3_int(-29.16, -67.50, 8)[0]
    assert h3_utils.h3_area_km2(cell6) == pytest.approx(36.13, abs=0.05)
    assert h3_utils.h3_area_km2(cell8) == pytest.approx(0.74, abs=0.05)
    assert len(h3_utils.h3_boundary_geojson(cell6)) == 6


def _hex_patch(center_lat, center_lon, radius_km, res):
    """Active cells covering a disc so interior cells have 6 neighbors."""
    center = int(
        h3_utils.latlon_to_h3_int(center_lat, center_lon, res)[0]
    )
    cells = set(h3_utils.h3_neighbors(center)) | {center}
    # two rings for a real planar patch with interior nodes
    ring2 = set()
    for c in list(cells):
        ring2 |= set(h3_utils.h3_neighbors(c))
    cells |= ring2
    return np.array(sorted(cells), dtype=np.int64)


def test_02_three_coloring_validity():
    active = _hex_patch(-29.16, -67.50, radius_km=50, res=6)
    colors, adjacency = h3_utils.three_color_hex_graph(active)
    assert set(np.unique(colors).tolist()) == {0, 1, 2}  # all colors used
    assert (colors >= 0).all()
    # Zero adjacent same-color neighbors across every edge.
    for i, neighbors in adjacency.items():
        for j in neighbors:
            assert colors[i] != colors[j], (i, j)
    # interior cells have full 6-neighbor adjacency
    degrees = [len(v) for v in adjacency.values()]
    assert max(degrees) == 6


def test_02_three_coloring_deterministic():
    active = _hex_patch(-12.60, -55.95, radius_km=50, res=6)
    c1, _ = h3_utils.three_color_hex_graph(active)
    c2, _ = h3_utils.three_color_hex_graph(active)
    assert np.array_equal(c1, c2)
    # A second, geographically separate patch also 3-colors deterministically.
    active2 = _hex_patch(31.75, 6.05, radius_km=50, res=6)
    d1, _ = h3_utils.three_color_hex_graph(active2)
    d2, _ = h3_utils.three_color_hex_graph(active2)
    assert np.array_equal(d1, d2)


# --------------------------------------------------------------------------- #
# Spencer SZA astronomy sanity
# --------------------------------------------------------------------------- #

def test_02_spencer_sza_known_values():
    # Equator, equinox (Mar 20), solar noon: SZA ~ 0-3 deg (Spencer's
    # series gives declination ~ -0.5 deg at doy 79 and Et ~ -8 min).
    tau, sdate, dn, sza = compute_solar_time_and_sza(
        "2026-03-20", 1200, 0.0, 0.0
    )
    assert sza < 3.0
    # 40N, June solstice noon: SZA ~ |40 - 23.44| ~ 16.6 deg.
    _, _, _, sza_jun = compute_solar_time_and_sza(
        "2026-06-21", 1200, 0.0, 40.0
    )
    assert sza_jun == pytest.approx(16.6, abs=1.5)
    # 40N, December solstice noon: SZA ~ 63.4 deg.
    _, _, _, sza_dec = compute_solar_time_and_sza(
        "2026-12-21", 1200, 0.0, 40.0
    )
    assert sza_dec == pytest.approx(63.4, abs=1.5)
    # Polar night: 75N in December at noon, SZA > 85.
    _, _, _, sza_polar = compute_solar_time_and_sza(
        "2026-12-21", 1200, 0.0, 75.0
    )
    assert sza_polar > 85.0
