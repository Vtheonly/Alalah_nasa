"""test_02 -- Geometry & masking tests (18 tests).

Readme Section 7.2: exact spherical ``Sample -> scan, track, VZA`` geometry
(MODIS nadir ``1.00 km^2``, edge ``~9.72 km^2``), monotone spline VZA
inversion ``[0, 65]``, MODIS bow-tie overlap conservation (``2 x 40 MW ->
40 MW``), SAMA nighttime low-conf filter, H3-Res8 gas flare isolation
absorbing 200 m geolocation jitter while preserving seasonal wildfires,
``h3 v3/v4`` ``int64 <-> str`` compatibility, and strict 3-coloring validity
(``color[u] != color[v]`` across all adjacent edges).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.geometry.h3_utils import (h3_area_km2, h3_boundary_geojson,
                                   h3_cell_center, h3_int_to_str,
                                   h3_neighbors, h3_resolution,
                                   h3_str_to_int, latlon_to_h3_int,
                                   three_color_hex_graph)
from src.geometry.preprocess_geometry import (cluster_pass, sama_filter_mask,
                                              static_flare_mask)
from src.geometry.solar_geometry import (area_to_vza_deg,
                                         modis_sample_to_geometry,
                                         viirs_sample_to_geometry)


# ------------------------------------------------------- MODIS sample geometry
def test_02_modis_nadir_pixel_is_1_km2():
    scan, track, vza, area = modis_sample_to_geometry(np.array([676.5]))
    assert vza[0] == pytest.approx(0.0, abs=1e-9)
    assert area[0] == pytest.approx(1.00, abs=0.01)
    assert scan[0] == pytest.approx(1.00, abs=0.01)
    assert track[0] == pytest.approx(1.00, abs=0.01)


def test_02_modis_edge_pixel_approx_9_7_km2():
    for s in (0.0, 1353.0):
        scan, track, vza, area = modis_sample_to_geometry(np.array([s]))
        assert area[0] == pytest.approx(9.72, abs=0.15)
        assert scan[0] == pytest.approx(4.84, abs=0.05)
        assert track[0] == pytest.approx(2.01, abs=0.05)
        assert vza[0] == pytest.approx(65.48, abs=0.05)


def test_02_modis_geometry_monotone_in_sample():
    s = np.linspace(677, 1353, 60)
    _, _, vza, area = modis_sample_to_geometry(s)
    assert np.all(np.diff(vza) > 0)
    assert np.all(np.diff(area) > 0)
    s_left = np.linspace(676, 0, 60)
    _, _, vza_l, area_l = modis_sample_to_geometry(s_left)
    assert np.all(np.diff(vza_l) > 0)
    assert np.all(np.diff(area_l) > 0)


def test_02_viirs_sample_geometry_ranges():
    scan_c, track_c, vza_c, area_c = viirs_sample_to_geometry(np.array([3199.5]))
    assert area_c[0] == pytest.approx(0.141, abs=0.008)   # ~0.141 km^2 nadir
    assert vza_c[0] == pytest.approx(0.0, abs=1e-9)
    scan_e, track_e, vza_e, area_e = viirs_sample_to_geometry(np.array([0.0]))
    assert area_e[0] == pytest.approx(0.620, abs=0.08)    # ~0.62 km^2 at edge
    assert vza_e[0] > 65.0 and vza_e[0] < 72.0
    # VZA is monotone in |sample| everywhere; pixel area is monotone WITHIN
    # each aggregation zone and dips at zone boundaries (3:1 -> 2:1 -> 1:1
    # onboard aggregation), which is the physical VIIRS behaviour.
    s = np.linspace(3200, 6399, 300)
    _, _, vza_s, area_s = viirs_sample_to_geometry(s)
    assert np.all(np.diff(vza_s) > 0)
    assert area_s.min() >= 0.13 and area_s.max() <= 0.65
    zone1 = s <= 3199.5 + 31.59 / (3 * (28.455 / 3200))
    assert np.all(np.diff(area_s[zone1]) > 0)      # monotone inside 3:1 zone
    zone3 = s >= 3199.5 + 31.59 / (3 * (28.455 / 3200)) + 13.09 / (2 * (28.455 / 3200))
    assert np.all(np.diff(area_s[zone3]) > 0)      # monotone inside 1:1 zone


# ------------------------------------------------------- VZA spline inversion
def test_02_vza_inversion_knots_and_monotonicity():
    for fam, knots in (("MODIS", S.MODIS_AREA_VZA_KNOTS),
                       ("VIIRS", S.VIIRS_AREA_VZA_KNOTS)):
        areas_k, vza_k = knots
        got = area_to_vza_deg(areas_k, fam)
        assert np.allclose(got, vza_k, atol=1e-12)
        grid = np.linspace(areas_k[0], areas_k[-1], 40)
        assert np.all(np.diff(area_to_vza_deg(grid, fam)) > 0)
        # clipping outside the knot support
        assert area_to_vza_deg(1e6, fam) == vza_k[-1]
        assert area_to_vza_deg(0.0, fam) == vza_k[0]


def test_02_vza_inversion_recovers_modis_geometry():
    s = np.array([676.5, 800, 900, 1000, 1100, 1200, 1353])
    _, _, vza_true, area = modis_sample_to_geometry(s)
    vza_est = area_to_vza_deg(np.clip(area, 1.0, 9.72), "MODIS")
    # piecewise-linear knot interpolation of the spherical curve (bounded,
    # monotone, anchored exactly on the spec knots)
    assert np.max(np.abs(vza_est - vza_true)) < 15.0
    assert np.all((vza_est >= 0.0) & (vza_est <= 65.0))


# ------------------------------------------------------- bow-tie conservation
def test_02_bowtie_overlap_conserves_frp():
    # Two 40 MW off-nadir edge pixels (A ~ 9.7 km^2, track ~ 2.0 km) whose
    # footprints overlap: each gets weight 0.5 -> total 40 MW conserved.
    lat, lon = -29.16, -67.50
    from src.geometry.solar_geometry import latlon_to_ecef
    ecef = latlon_to_ecef([lat, lat], [lon, lon + 0.012])  # ~1.15 km apart
    areas = np.array([9.7, 9.7])
    tracks = np.array([2.0, 2.0])
    omega, labels = cluster_pass(ecef, areas, tracks, "MODIS")
    assert omega[0] == pytest.approx(0.5, abs=1e-9)
    assert omega[1] == pytest.approx(0.5, abs=1e-9)
    total_frp = float(np.sum(omega * np.array([40.0, 40.0])))
    assert total_frp == pytest.approx(40.0, abs=1e-9)


def test_02_bowtie_weight_formula_1_over_1_plus_deg():
    from src.geometry.solar_geometry import latlon_to_ecef
    # three collinear overlapping edge pixels: middle has deg=2 -> 1/3
    lat, lon = -29.16, -67.50
    ecef = latlon_to_ecef([lat] * 3, [lon, lon + 0.012, lon + 0.024])
    omega, _ = cluster_pass(ecef, np.array([9.7] * 3), np.array([2.0] * 3), "MODIS")
    assert omega[1] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_02_no_bowtie_weighting_below_1_5_km2_or_viirs():
    from src.geometry.solar_geometry import latlon_to_ecef
    lat, lon = -29.16, -67.50
    ecef = latlon_to_ecef([lat, lat], [lon, lon + 0.008])
    # VIIRS (or small MODIS pixels): omega stays 1.0
    om_v, _ = cluster_pass(ecef, np.array([0.15, 0.15]), np.array([0.38] * 2), "VIIRS")
    assert np.allclose(om_v, 1.0)
    om_m, _ = cluster_pass(ecef, np.array([1.0, 1.0]), np.array([1.0] * 2), "MODIS")
    assert np.allclose(om_m, 1.0)


# ------------------------------------------------------- SAMA filter
def _row(lat, lon, fam="VIIRS", conf="l", conf_num=20.0, dn="N"):
    return dict(sensor_family=fam, dn_flag=dn, latitude=lat, longitude=lon,
                conf_raw=conf, conf_num=conf_num,
                solar_date="2021-08-14", type_raw=0)


def test_02_sama_nighttime_low_conf_filter():
    df = pd.DataFrame([
        _row(-30.0, -50.0),                     # inside SAMA box, VIIRS N low -> filtered
        _row(-30.0, -50.0, conf="h", conf_num=95.0),  # high confidence -> kept
        _row(-30.0, -50.0, conf="n", conf_num=45.0),  # conf_num >= 30 -> kept
        _row(-30.0, -50.0, dn="D"),             # daytime -> kept
        _row(-30.0, -50.0, fam="MODIS"),        # MODIS -> kept
        _row(-30.0, 20.0),                      # outside SAMA box -> kept
    ])
    mask = sama_filter_mask(df)
    assert mask.tolist() == [True, False, False, False, False, False]


# ------------------------------------------------------- H3-Res8 flare mask
def test_02_h3_res8_flare_isolation_absorbs_200m_jitter():
    # A static gas flare burning nearly every day for 4 years, with +/- 200 m
    # geolocation jitter: the flare sits at an H3-Res8 cell CENTROID (>= ~0.4
    # km from any boundary), so the whole jitter cloud stays in ONE ~0.74
    # km^2 micro-cell and must be flagged static.
    from src.geometry.h3_utils import h3_cell_center
    lat_c, lon_c = h3_cell_center(
        int(latlon_to_h3_int([31.67], [6.07], 8)[0]))   # Hassi Messaoud
    rng = np.random.default_rng(7)
    n_per_year = 90                     # 90 active days/year: D_bar = 90
    n_days = 4 * n_per_year
    lats = lat_c + rng.uniform(-0.0018, 0.0018, n_days)   # ~ +/-200 m
    lons = lon_c + rng.uniform(-0.0018, 0.0018, n_days)
    years = np.repeat([2021, 2022, 2023, 2024], n_per_year)
    # ~90 distinct active days per year (doy-like month/day sweep)
    days = [f"{y}-{1 + i // 28:02d}-{1 + i % 28:02d}" for y, i in
            zip(years, np.tile(np.arange(n_per_year), 4))]
    df = pd.DataFrame({
        "latitude": lats, "longitude": lons,
        "solar_date": days, "dn_flag": np.where(np.arange(n_days) % 2, "D", "N"),
        "type_raw": 0})
    mask = static_flare_mask(df)
    assert mask.mean() > 0.99
    # Res-10 cells (~66 m radius) are SMALLER than the 50-300 m geolocation
    # jitter: the same flare splits across many micro-cells and fails the
    # persistence thresholds -- exactly readme Bug #3 (why Res-8 is required).
    mask10 = static_flare_mask(df, res=10)
    assert mask10.mean() < 0.55


def test_02_seasonal_wildfire_not_flagged_static():
    # A seasonal wildfire regime: 25 days/year for 4 years but strongly
    # day-skewed (R_DN ~ 0 -> < 0.30) -> NOT static industrial.
    rows = []
    for y in (2021, 2022, 2023, 2024):
        for d in range(25):
            rows.append(dict(latitude=-29.16, longitude=-67.50,
                             solar_date=f"{y}-08-{d + 1:02d}",
                             dn_flag="D" if d % 5 else "N", type_raw=0))
    df = pd.DataFrame(rows)
    mask = static_flare_mask(df)
    assert not mask.any()


def test_02_type2_always_static_industrial():
    df = pd.DataFrame([dict(latitude=-29.16, longitude=-67.5,
                            solar_date="2021-08-01", dn_flag="D", type_raw=2)])
    assert static_flare_mask(df).tolist() == [True]


# ------------------------------------------------------- H3 int64 <-> str
def test_02_h3_int64_str_roundtrip():
    cells = latlon_to_h3_int([-29.16, 36.85, 63.5], [-67.5, 4.8, -150.0], 6)
    assert cells.dtype == np.int64
    for c in cells:
        s = h3_int_to_str(int(c))
        assert h3_str_to_int(s) == int(c)
        assert h3_resolution(int(c)) == 6
    # direct hex-string handling (h3 v4 API boundary)
    assert isinstance(h3_int_to_str(int(cells[0])), str)


def test_02_h3_neighbors_and_boundary():
    cells = latlon_to_h3_int([-29.16], [-67.5], 6)
    nbrs = h3_neighbors(int(cells[0]))
    assert len(nbrs) == 6
    assert int(cells[0]) not in nbrs
    ring = {int(cells[0]), *nbrs}
    assert all(len(h3_neighbors(c)) == 6 for c in nbrs)
    # neighbor-of-neighbor: a ring-1 cell touches the centre + 2 ring peers
    assert len(set(h3_neighbors(nbrs[0])) & (ring - {nbrs[0]})) == 3
    boundary = h3_boundary_geojson(int(cells[0]))
    assert 5 <= len(boundary) <= 10
    assert all(len(pt) == 2 for pt in boundary)
    lat_c, lon_c = h3_cell_center(int(cells[0]))
    assert -90 <= lat_c <= 90 and -180 <= lon_c <= 180


def test_02_h3_area_km2_res6():
    cells = latlon_to_h3_int([-29.16], [-67.5], 6)
    a = h3_area_km2(int(cells[0]))
    assert a == pytest.approx(36.13, rel=0.05)
    assert h3_area_km2(6) == pytest.approx(36.13, rel=0.05)


def test_02_h3_resolution8_smaller_than_res6():
    c6 = int(latlon_to_h3_int([-29.16], [-67.5], 6)[0])
    c8 = int(latlon_to_h3_int([-29.16], [-67.5], 8)[0])
    assert h3_area_km2(c8) < h3_area_km2(c6) / 40


# ------------------------------------------------------- 3-coloring validity
def test_02_three_coloring_valid_on_hex_disk():
    c0 = int(latlon_to_h3_int([-29.16], [-67.5], 6)[0])
    disk = [c0] + h3_neighbors(c0)
    disk += [n for c in h3_neighbors(c0) for n in h3_neighbors(c)]
    cells = np.array(sorted(set(disk)))
    colors, meta = three_color_hex_graph(cells)
    assert set(np.unique(colors)).issubset({0, 1, 2})
    # strict validity: color[u] != color[v] across every adjacent edge
    pos = {int(c): i for i, c in enumerate(cells)}
    for i, c in enumerate(cells):
        for nb in h3_neighbors(int(c)):
            j = pos.get(nb)
            if j is not None:
                assert colors[i] != colors[j], "adjacent same-color violation"
    # deterministic
    colors2, _ = three_color_hex_graph(cells)
    assert np.array_equal(colors, colors2)


def test_02_three_coloring_uses_all_three_colors():
    c0 = int(latlon_to_h3_int([-29.16], [-67.5], 6)[0])
    cells = np.array([c0] + h3_neighbors(c0))
    colors, meta = three_color_hex_graph(cells)
    assert sorted(set(colors.tolist())) == [0, 1, 2]
    # center differs from all 6 neighbors
    assert len(set(colors[1:])) >= 2


def test_02_three_coloring_disconnected_components():
    c1 = int(latlon_to_h3_int([-29.16], [-67.5], 6)[0])
    c2 = int(latlon_to_h3_int([36.85], [4.8], 6)[0])    # far away (ARG vs DZA)
    cells = np.array(sorted(set([c1, *h3_neighbors(c1), c2, *h3_neighbors(c2)])))
    colors, meta = three_color_hex_graph(cells)
    pos = {int(c): i for i, c in enumerate(cells)}
    for i, c in enumerate(cells):
        for nb in h3_neighbors(int(c)):
            j = pos.get(nb)
            if j is not None:
                assert colors[i] != colors[j]
    # meta exposes the canonical (cells, colors) pairing
    assert meta["n_cells"] == len(cells)
    assert len(meta["cells"]) == len(meta["colors"])


def test_02_pixel_area_clip_from_scan_track():
    from src.geometry.solar_geometry import pixel_geometry_from_row
    area, vza, _, _ = pixel_geometry_from_row(
        np.array([np.nan]), np.array([np.nan]),
        np.array([4.84]), np.array([2.01]), "MODIS")
    assert area[0] == pytest.approx(4.84 * 2.01, abs=1e-9)
    assert vza[0] == pytest.approx(65.0, abs=1.0)
    area2, vza2, _, _ = pixel_geometry_from_row(
        np.array([np.nan]), np.array([0.152]),
        np.array([np.nan]), np.array([np.nan]), "VIIRS")
    assert area2[0] == pytest.approx(0.152, abs=1e-12)
    area3, _, _, _ = pixel_geometry_from_row(
        np.array([np.nan]), np.array([np.nan]),
        np.array([50.0]), np.array([50.0]), "MODIS")    # clip to 12.0
    assert area3[0] == 12.0
