"""test_03 -- Event clustering tests (14 tests).

Readme Section 7.3: 3x3 contiguous VIIRS cluster -> 1 event, separated
clusters -> 2 events, ``SNPP + N20 + N21`` within 50 minutes kept as 3
distinct pass rows (never summed, INV-02), 5-minute pass-gap split,
``i4_m13_fallback`` vs Tobit ``sat_flag`` classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.geometry.preprocess_geometry import (PASS_COLUMNS,
                                              preprocess_observations,
                                              saturation_flags,
                                              segment_passes)
from src.geometry.solar_geometry import latlon_to_ecef
from tests.conftest import canonical_frame


def _detections(rows):
    return canonical_frame(rows)


# ---------------------------------------------------------------- clustering
def test_03_contiguous_3x3_viirs_cluster_is_one_event():
    rows = []
    for i in range(3):
        for j in range(3):
            rows.append(dict(sensor_family="VIIRS", platform="SNPP",
                             latitude=-29.10 + 0.003 * i,
                             longitude=-67.40 + 0.003 * j,
                             pixarea_km2=0.15, sample_idx=np.nan,
                             scan_km=np.nan, track_km=np.nan,
                             frp_mw=5.0, acq_hhmm=1330, type_raw=0))
    prov, pas = preprocess_observations(_detections(rows))
    bio = prov[prov["is_biomass"]]
    assert bio["event_cluster_id"].nunique() == 1
    row = pas.iloc[0]
    assert row["raw_event_count"] == 1
    assert row["y_frp_mw"] == pytest.approx(45.0, abs=1e-6)


def test_03_separated_fires_are_two_events():
    # two 2x2 clusters INSIDE one H3-Res6 cell, ~1.2 km apart: 2 distinct
    # events (gap > 1.4 * sqrt(A) ~ 0.54 km) within the same pass row.
    rows = []
    for i in range(2):
        for j in range(2):
            rows.append(dict(sensor_family="VIIRS", platform="SNPP",
                             latitude=-29.10 + 0.002 * i,
                             longitude=-67.40 - 0.008 + 0.002 * j,
                             pixarea_km2=0.15, frp_mw=4.0,
                             acq_hhmm=1330, type_raw=0))
            rows.append(dict(sensor_family="VIIRS", platform="SNPP",
                             latitude=-29.10 + 0.002 * i,
                             longitude=-67.40 + 0.008 + 0.002 * j,
                             pixarea_km2=0.15, frp_mw=4.0,
                             acq_hhmm=1330, type_raw=0))
    prov, pas = preprocess_observations(_detections(rows))
    bio = prov[prov["is_biomass"]]
    assert bio["event_cluster_id"].nunique() == 2
    assert len(pas) == 1
    assert pas.iloc[0]["raw_event_count"] == 2


def test_03_adjacent_modis_edge_pixels_one_event_bowtie_weighted():
    # two overlapping MODIS edge pixels (A ~ 9.7 km^2): same event, bow-tie
    # weight 0.5 each -> FRP conserved at 40 MW total.
    rows = [
        dict(sensor_family="MODIS", platform="Terra", sample_idx=1300.0,
             latitude=-29.10, longitude=-67.40, frp_mw=40.0, acq_hhmm=1330,
             scan_km=np.nan, track_km=np.nan, pixarea_km2=np.nan),
        dict(sensor_family="MODIS", platform="Terra", sample_idx=1310.0,
             latitude=-29.10, longitude=-67.40 - 0.010, frp_mw=40.0,
             acq_hhmm=1331, scan_km=np.nan, track_km=np.nan,
             pixarea_km2=np.nan),
    ]
    prov, pas = preprocess_observations(_detections(rows))
    bio = prov[prov["is_biomass"]]
    assert bio["event_cluster_id"].nunique() == 1
    assert bio["bowtie_weight"].tolist() == [0.5, 0.5]
    assert pas.iloc[0]["y_frp_mw"] == pytest.approx(40.0, abs=1e-6)


# ---------------------------------------------------------------- INV-02
def test_03_same_orbit_constellation_never_summed():
    base = dict(sensor_family="VIIRS", pixarea_km2=0.15, frp_mw=10.0,
                type_raw=0, sample_idx=np.nan, scan_km=np.nan,
                track_km=np.nan, latitude=-29.10, longitude=-67.40)
    rows = []
    for platform, hhmm in (("SNPP", 1305), ("NOAA-20", 1335), ("NOAA-21", 1400)):
        rows.append(dict(base, platform=platform, acq_hhmm=hhmm))
    prov, pas = preprocess_observations(_detections(rows))
    # three distinct platforms -> three distinct pass rows (INV-02)
    assert len(pas) == 3
    assert set(pas["platform"]) == {"SNPP", "NOAA-20", "NOAA-21"}
    assert (pas["y_frp_mw"] == 10.0).all()


def test_03_same_platform_same_day_two_passes():
    base = dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
                frp_mw=8.0, type_raw=0, sample_idx=np.nan, scan_km=np.nan,
                track_km=np.nan, latitude=-29.10, longitude=-67.40)
    rows = [dict(base, acq_hhmm=105),                 # nighttime pass
            dict(base, acq_hhmm=1335)]                # daytime pass
    prov, pas = preprocess_observations(_detections(rows))
    # 13:35 local solar is Day, 01:05 is Night -> 2 rows (dn split)
    assert len(pas) == 2
    assert set(pas["dn_flag"]) == {"D", "N"}


# ---------------------------------------------------------------- pass split
def test_03_five_minute_pass_boundary_split():
    rows = []
    for i, hhmm in enumerate([1300, 1302, 1303, 1310, 1312]):
        rows.append(dict(sensor_family="VIIRS", platform="SNPP",
                         latitude=-29.10 + 0.01 * i, longitude=-67.40,
                         pixarea_km2=0.15, frp_mw=5.0, acq_hhmm=hhmm,
                         type_raw=0, sample_idx=np.nan, scan_km=np.nan,
                         track_km=np.nan))
    prov, _ = preprocess_observations(_detections(rows))
    bio = prov[prov["is_biomass"]]
    # 13:03 -> 13:10 gap is 7 minutes (> 5) -> two passes
    assert bio["pass_id"].nunique() == 2


def test_03_within_five_minutes_same_pass():
    rows = []
    for i, hhmm in enumerate([1300, 1302, 1303, 1305]):
        rows.append(dict(sensor_family="VIIRS", platform="SNPP",
                         latitude=-29.10 + 0.01 * i, longitude=-67.40,
                         pixarea_km2=0.15, frp_mw=5.0, acq_hhmm=hhmm,
                         type_raw=0, sample_idx=np.nan, scan_km=np.nan,
                         track_km=np.nan))
    prov, _ = preprocess_observations(_detections(rows))
    assert prov[prov["is_biomass"]]["pass_id"].nunique() == 1


def test_03_pass_segmentation_unit():
    df = pd.DataFrame({
        "platform": ["SNPP"] * 4 + ["Terra"] * 2,
        "acq_date": ["2026-08-14"] * 6,
        "acq_hhmm": [1300, 1304, 1305, 1500, 1320, 1326]})
    ids = segment_passes(df)
    # SNPP: 13:00, 13:04, 13:05 one pass (gaps 4 and 1 min), 15:00 new pass;
    # Terra: platform change starts a new pass and the 13:20 -> 13:26 gap of
    # 6 minutes (> 5) splits it again -> 4 passes total.
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[0] + 1
    assert ids[4] != ids[5]
    assert len(set(ids.tolist())) == 4


# ---------------------------------------------------------------- saturation
def test_03_i4_m13_fallback_classification():
    m13fb, sat = saturation_flags(
        ["VIIRS", "VIIRS", "VIIRS", "MODIS"],
        [366.0, 300.0, 367.5, 320.0],
        [50.0, 10.0, 5.0, 20.0])
    # 366.0 & frp>0 -> fallback; 300K -> none; 367.5 & frp 5 -> fallback only
    assert m13fb.tolist() == [1, 0, 1, 0]
    assert sat.tolist() == [0, 0, 0, 0]     # VIIRS sat needs frp >= 300 MW


def test_03_tobit_sat_flag_modis_and_viirs():
    m13fb, sat = saturation_flags(
        ["MODIS", "MODIS", "VIIRS", "VIIRS", "MODIS"],
        [499.0, 499.0, 367.0, 367.0, 497.0],
        [120.0, 0.0, 350.0, 100.0, 5.0])
    # MODIS sat: t >= 498 & frp > 0
    assert sat.tolist() == [1, 0, 1, 0, 0]
    # i4_m13_fallback applies to the VIIRS rows with t >= 366 & frp > 0
    assert m13fb.tolist() == [0, 0, 1, 1, 0]


def test_03_m13_fallback_inflates_sigma_downstream():
    # The 1.35x sigma inflation is applied in operators.sigma2; the flag must
    # survive aggregation into pass_observations.
    rows = [
        dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
             frp_mw=12.0, acq_hhmm=1330, t_mir_k=366.5, type_raw=0,
             sample_idx=np.nan, scan_km=np.nan, track_km=np.nan),
        dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
             frp_mw=3.0, acq_hhmm=1330, t_mir_k=340.0, type_raw=0,
             sample_idx=np.nan, scan_km=np.nan, track_km=np.nan),
    ]
    prov, pas = preprocess_observations(_detections(rows))
    assert pas.iloc[0]["has_i4_m13_fallback"] == 1
    assert prov[prov["t_mir_k"] >= 366.0]["i4_m13_fallback"].tolist() == [1]


def test_03_aggregation_key_never_sums_platforms():
    base = dict(sensor_family="MODIS", sample_idx=676.0, frp_mw=30.0,
                type_raw=0, scan_km=np.nan, track_km=np.nan,
                pixarea_km2=np.nan, latitude=-29.10, longitude=-67.40)
    rows = [dict(base, platform="Terra", acq_hhmm=1030),
            dict(base, platform="Aqua", acq_hhmm=1330)]
    prov, pas = preprocess_observations(_detections(rows))
    assert len(pas) == 2
    assert list(pas.columns) == PASS_COLUMNS or set(PASS_COLUMNS) == set(pas.columns)
    assert (pas["y_frp_mw"] == 30.0).all()


def test_03_static_and_type_rows_excluded_from_biomass():
    rows = [
        dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
             frp_mw=5.0, type_raw=2, sample_idx=np.nan, scan_km=np.nan,
             track_km=np.nan),                       # static gas flare
        dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
             frp_mw=5.0, type_raw=1, sample_idx=np.nan, scan_km=np.nan,
             track_km=np.nan),                       # volcano
        dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
             frp_mw=5.0, type_raw=3, sample_idx=np.nan, scan_km=np.nan,
             track_km=np.nan),                       # offshore
        dict(sensor_family="VIIRS", platform="SNPP", pixarea_km2=0.15,
             frp_mw=5.0, type_raw=0, sample_idx=np.nan, scan_km=np.nan,
             track_km=np.nan),                       # biomass
    ]
    prov, pas = preprocess_observations(_detections(rows))
    assert len(pas) == 1                             # only biomass aggregated
    assert prov["is_biomass"].sum() == 1
    assert len(prov) == 4                            # all kept in Product 1
    assert prov[prov["type_raw"] == 2]["is_static_industrial"].all()
