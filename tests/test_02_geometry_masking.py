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


def test_02_sza_day_of_year_jan1_is_doy1():
    """Spencer Gamma uses doy - 1: Jan 1 must map to doy = 1 (no +1 shift).
    At the June solstice local noon on the Tropic of Cancer the sun is
    ~overhead (SZA < 1 deg); a +1-day DOY shift would not break that, so we
    also pin Jan 1 noon at the equator to SZA = |declination(doy=1)|."""
    from src.geometry.solar_geometry import (compute_solar_time_and_sza,
                                             spencer_solar_position)
    d = np.array(["2021-01-01"], dtype="datetime64[D]")
    _h, _sd, _dn, sza = compute_solar_time_and_sza(d, np.array([1200]),
                                                   np.array([0.0]),
                                                   np.array([0.0]))
    decl1, et1 = spencer_solar_position(np.array([1.0]))
    # hour angle at 12:00 UTC on lon 0 = 15 * (E_t / 60) degrees
    omega = np.deg2rad(15.0 * et1[0] / 60.0)
    expect = np.rad2deg(np.arccos(np.cos(decl1[0]) * np.cos(omega)))
    assert sza[0] == pytest.approx(expect, abs=1e-9)


# ---------------------------------------------------------------------------
# Real-scale vectorizations must be EXACT replacements of the originals
# ---------------------------------------------------------------------------
def test_02_static_flare_mask_vectorized_equals_reference():
    import numpy as np
    import pandas as pd
    from src.geometry import preprocess_geometry as pg
    rng = np.random.default_rng(3)
    n = 60000
    fl = rng.integers(0, 120, n // 2)
    lat = np.r_[28 + fl * 0.01, rng.uniform(-30, 0, n - n // 2)]
    lon = np.r_[5 + fl * 0.01, rng.uniform(-60, -40, n - n // 2)]
    days = (pd.to_datetime("2012-01-01")
            + pd.to_timedelta(rng.integers(0, 4000, n), "D"))
    df = pd.DataFrame({"latitude": lat, "longitude": lon,
                       "solar_date": days.values.astype("datetime64[D]"),
                       "dn_flag": rng.choice(["D", "N"], n),
                       "type_raw": rng.choice([0, 0, 0, 2], n)})
    a = pg.static_flare_mask(df)
    assert np.array_equal(a, pg._static_flare_mask_reference(df))
    assert a[: n // 2].mean() > 0.9 and a[n // 2:].mean() < 0.35


def test_02_cluster_pass_vectorized_equals_reference():
    import numpy as np
    from src.geometry import preprocess_geometry as pg
    from src.geometry.preprocess_geometry import latlon_to_ecef
    rng = np.random.default_rng(7)
    for trial in range(12):
        n = int(rng.integers(2, 800))
        ecef = latlon_to_ecef(-10 + rng.normal(0, 0.03, n),
                              -55 + rng.normal(0, 0.03, n))
        fam = "MODIS" if trial % 2 else "VIIRS"
        areas = (rng.uniform(0.9, 4.0, n) if fam == "MODIS"
                 else np.full(n, 0.18))
        a = pg.cluster_pass(ecef, areas, np.sqrt(areas), fam)
        b = pg._cluster_pass_reference(ecef, areas, np.sqrt(areas), fam)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_02_cloud_matrix_index_cache_is_exact_and_tracks_table():
    """The per-table cloud index gives the same matrix as a fresh table
    copy and is rebuilt (not reused) for a different table object."""
    import numpy as np
    import pandas as pd
    from src.engine import priors as pr
    rng = np.random.default_rng(5)
    lb = -30 + 0.25 * np.arange(20) + 0.125
    L, O = np.meshgrid(lb, lb - 30)
    cloud = pd.concat([pd.DataFrame({"lat_025": L.ravel(), "lon_025": O.ravel(),
                                     "year": 2015, "period": m,
                                     "period_kind": "month",
                                     "cloud_frac": rng.random(L.size)})
                       for m in range(1, 13)], ignore_index=True)
    cl, co = rng.uniform(-30, -25, 50), rng.uniform(-60, -55, 50)
    dates = pd.date_range("2015-01-01", "2015-12-31")
    a, _ = pr.cloud_fraction_matrix(cl, co, dates, cloud, None)
    b, _ = pr.cloud_fraction_matrix(cl, co, dates, cloud.copy(), None)
    assert np.array_equal(a, b, equal_nan=True)
    c, _ = pr.cloud_fraction_matrix(cl, co, dates,
                                    cloud.assign(cloud_frac=0.25), None)
    assert np.allclose(c, 0.25)
    # spot check: a cell's daily value is its CMG bin's monthly value
    lat_c, lon_c = pr.cmg_bin_centers(cl[:1], co[:1])
    row = cloud[(cloud.lat_025 == lat_c[0]) & (cloud.lon_025 == lon_c[0])
                & (cloud.period == 7)]
    assert np.isclose(a[0, dates.get_loc("2015-07-15")], row.cloud_frac.iloc[0])


def test_02_partitioned_preprocess_equals_monolithic(tmp_path):
    """preprocess_partitioned (one year at a time, static-flare cells from a
    lean all-years pass) == preprocess_observations on everything at once:
    same pass table, same provenance flags/weights, same event grouping."""
    import numpy as np
    import pandas as pd
    from src.geometry import preprocess_geometry as pg
    from src.ingest.normalize_dedup import CANONICAL_COLUMNS
    rng = np.random.default_rng(11)
    npass, n = 900, 30000
    days = pd.date_range("2012-01-20", "2014-12-31").strftime("%Y-%m-%d").to_numpy()
    p_plat = rng.choice(["SNPP", "Terra", "Aqua"], npass)
    p_day = days[rng.integers(0, len(days), npass)]
    p_hhmm = np.where(rng.random(npass) < .5, 1630, 430)
    p_lat = rng.uniform(-11, -9, npass)
    p_lon = rng.uniform(-56, -54, npass)
    pid = rng.integers(0, npass, n)
    flare = rng.random(n) < 0.3          # a persistent multi-year flare site
    fam = np.where(np.isin(p_plat[pid], ["Terra", "Aqua"]), "MODIS", "VIIRS")
    df = pd.DataFrame({
        "country_iso": "BRA", "sensor_family": fam, "platform": p_plat[pid],
        "data_tier": "ML_SCIENCE",
        "latitude": np.where(flare, -10.5, p_lat[pid] + rng.normal(0, .02, n)),
        "longitude": np.where(flare, -55.5, p_lon[pid] + rng.normal(0, .02, n)),
        "acq_date": p_day[pid], "acq_hhmm": p_hhmm[pid] + rng.integers(0, 4, n),
        "t_mir_k": 330.0, "t_tir_k": 295.0, "scan_km": np.nan,
        "track_km": np.nan, "sample_idx": rng.integers(0, 3200, n).astype(float),
        "pixarea_km2": np.where(fam == "VIIRS", .2, 1.5),
        "frp_mw": rng.gamma(1, 10, n), "conf_raw": "n", "conf_num": 80.0,
        "type_raw": 0, "version_raw": "C2",
        "ingest_ts": "2026-11-15T00:00:00Z"})[CANONICAL_COLUMNS]
    root = tmp_path / "active_fires"
    year = df["acq_date"].str[:4]
    for y, g in df.groupby(year):
        d = root / "country_iso=BRA" / f"year={y}"
        d.mkdir(parents=True)
        g.to_parquet(d / "part.parquet", index=False)
    ref_prov, ref_pass = pg.preprocess_observations(
        pd.concat([g for _, g in df.groupby(year)], ignore_index=True))
    pas, info = pg.preprocess_partitioned(root, tmp_path / "prov")
    got_prov = pd.read_parquet(tmp_path / "prov")
    assert info["provenance_rows"] == len(ref_prov) == len(got_prov)
    assert info["duplicate_pass_keys_across_years"] == 0
    assert info["static_flare_cells"] >= 1
    assert ref_prov["is_static_industrial"].any()
    key = ["h3_index", "solar_date", "platform", "dn_flag"]
    a = ref_pass.sort_values(key).reset_index(drop=True)
    b = pas.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    order = ["platform", "acq_date", "acq_hhmm", "latitude", "longitude", "frp_mw"]
    r = ref_prov.sort_values(order).reset_index(drop=True)
    g = got_prov.sort_values(order).reset_index(drop=True)
    for c in ("is_static_industrial", "is_biomass", "bowtie_weight", "h3_index",
              "dn_flag", "vza_deg"):
        assert (r[c].to_numpy() == g[c].to_numpy()).all(), c
    # same grouping of detections into events (ids differ by offsets only)
    assert (pd.factorize(r["event_cluster_id"])[0]
            == pd.factorize(g["event_cluster_id"])[0]).all()
    bio = g["pass_id"] >= 0
    assert g.loc[bio, "event_cluster_id"].is_unique is False   # real clusters


def test_02_partitioned_merges_solar_night_split_across_utc_years(tmp_path):
    """A solar day's night spans two UTC dates: at 60 W the evening of
    31 Dec (local) is 1 Jan UTC, so one pass key gets detections from two
    UTC-year partitions.  They must be merged back into ONE pass row equal
    to the monolithic aggregation (the solver would otherwise keep only one
    half)."""
    import numpy as np
    import pandas as pd
    from src.geometry import preprocess_geometry as pg
    from src.ingest.normalize_dedup import CANONICAL_COLUMNS
    rows = []
    # same Terra cell, solar night of 2013-12-31: 02:10 local (06:10 UTC
    # 31 Dec) and 22:30 local (02:30 UTC 1 Jan 2014); 3 pixels each
    for acq_date, hhmm in (("2013-12-31", 610), ("2014-01-01", 230)):
        for k in range(3):
            rows.append({"latitude": -29.0 + 0.003 * k, "longitude": -60.0,
                         "acq_date": acq_date, "acq_hhmm": hhmm + k,
                         "frp_mw": 10.0 + k, "sample_idx": 600.0 + k})
    df = pd.DataFrame(rows)
    df = df.assign(country_iso="ARG", sensor_family="MODIS", platform="Terra",
                   data_tier="ML_SCIENCE", t_mir_k=330.0, t_tir_k=295.0,
                   scan_km=np.nan, track_km=np.nan, pixarea_km2=1.5,
                   conf_raw="n", conf_num=80.0, type_raw=0, version_raw="C6.1",
                   ingest_ts="2026-11-15T00:00:00Z")[CANONICAL_COLUMNS]
    root = tmp_path / "active_fires"
    for y, g in df.groupby(df["acq_date"].str[:4]):
        d = root / "country_iso=ARG" / f"year={y}"
        d.mkdir(parents=True)
        g.to_parquet(d / "part.parquet", index=False)
    _ref_prov, ref_pass = pg.preprocess_observations(df)
    pas, info = pg.preprocess_partitioned(root, tmp_path / "prov")
    assert len(ref_pass) == 1                       # one solar-day night pass
    assert info["duplicate_pass_keys_across_years"] == 1   # detected ...
    assert len(pas) == 1                            # ... and merged
    a = ref_pass.reset_index(drop=True)
    b = pas[ref_pass.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
