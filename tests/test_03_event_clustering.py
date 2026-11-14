"""test_03: overpass segmentation, event clustering, saturation flags.

14 tests (readme.md Section 7 item 3): 3x3 contiguous VIIRS cluster -> 1
event, separated clusters -> 2 events, same-afternoon SNPP + N20 + N21 kept
as 3 distinct pass rows (never summed, INV-02), 5-minute pass boundary split,
and i4_m13_fallback vs Tobit sat_flag classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.geometry import preprocess_geometry as pg
from src.ingest import normalize_dedup as nd


def _viirs_csv(rows):
    """Build a Vxx14IMGML CSV text block from (date, hhmm, sat, lat, lon, frp)."""
    header = "YYYYMMDD,HHMM,Sat,Lat,Lon,T_I4,T_I5,Sample,Pixarea,FRP,Conf,Type"
    lines = [header]
    for date, hhmm, sat, lat, lon, frp, t_mir in rows:
        ymd = date.replace("-", "")
        lines.append(
            f"{ymd},{hhmm},{sat},{lat},{lon},{t_mir},300.5,3200,0.20,"
            f"{frp},n,0"
        )
    return "\n".join(lines) + "\n"


def _frame(rows):
    return nd.parse_vxx14imgml(
        csv_text=_viirs_csv(rows),
        filename="VNP14IMGML_2021_08.txt.gz",
        ingest_ts="2026-11-14T00:00:00Z",
    )


def test_03_contiguous_3x3_viirs_cluster_single_event():
    rows = []
    for i in range(3):
        for j in range(3):
            rows.append(
                ("2021-08-01", 1342, "N", -29.10 + 0.004 * i,
                 -67.45 + 0.004 * j, 5.0, 340.0)
            )
    prov, passes = pg.preprocess_frame(_frame(rows))
    # All 9 pixels fall into one connected physical fire event.
    assert prov["event_cluster_id"].nunique() == 1
    row = passes.iloc[0]
    assert row["raw_event_count"] == 1
    assert row["y_frp_mw"] == pytest.approx(45.0)


def test_03_separated_fires_two_events():
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 5.0, 340.0),
        ("2021-08-01", 1342, "N", -29.104, -67.45, 5.0, 340.0),
        ("2021-08-01", 1342, "N", -29.35, -67.70, 8.0, 340.0),
    ]
    prov, passes = pg.preprocess_frame(_frame(rows))
    assert prov["event_cluster_id"].nunique() == 2
    assert len(passes) == 2
    assert set(passes["raw_event_count"]) == {1}


def test_03_same_afternoon_three_satellites_three_rows():
    """SNPP + NOAA-20 + NOAA-21 within ~50 min stay 3 distinct rows (INV-02)."""
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 10.0, 340.0),
        ("2021-08-01", 1432, "1", -29.10, -67.45, 10.0, 340.0),
        ("2021-08-01", 1522, "2", -29.10, -67.45, 10.0, 340.0),
    ]
    prov, passes = pg.preprocess_frame(_frame(rows))
    # same (h3, solar_date, dn) but three platforms -> three rows, never summed
    assert len(passes) == 3
    assert set(passes["platform"]) == {"SNPP", "NOAA-20", "NOAA-21"}
    assert (passes["y_frp_mw"] == 10.0).all()
    # 3 distinct passes as well
    assert prov["pass_id"].nunique() == 3


def test_03_pass_split_on_5_minute_gap():
    rows = [
        # 13:42 and 13:46.9 -> same pass (gap <= 5 min)
        ("2021-08-01", 1342, "N", -29.10, -67.45, 5.0, 340.0),
        ("2021-08-01", 1346, "N", -29.35, -67.70, 5.0, 340.0),
        # 13:42 and 14:20 -> different passes (gap > 5 min)
        ("2021-08-02", 1342, "N", -29.10, -67.45, 5.0, 340.0),
        ("2021-08-02", 1420, "N", -29.35, -67.70, 5.0, 340.0),
    ]
    df = _frame(rows)
    prov, _ = pg.preprocess_frame(df)
    passes_day1 = prov[prov["acq_date"] == "2021-08-01"]["pass_id"].nunique()
    passes_day2 = prov[prov["acq_date"] == "2021-08-02"]["pass_id"].nunique()
    assert passes_day1 == 1  # 4.9-minute gap stays one overpass
    assert passes_day2 == 2  # 38-minute gap splits


def test_03_i4_m13_fallback_classification():
    i4, sat = pg.classify_saturation("VIIRS", 366.0, 5.0)
    assert (i4, sat) == (1, 0)  # M13 retrieved FRP below Tobit censoring
    i4, sat = pg.classify_saturation("VIIRS", 365.9, 5.0)
    assert (i4, sat) == (0, 0)
    # zero FRP under background failure: no fallback flag
    i4, sat = pg.classify_saturation("VIIRS", 367.5, 0.0)
    assert (i4, sat) == (0, 0)


def test_03_tobit_sat_flag_modis_ch21():
    sat_ok = pg.classify_saturation("MODIS", 498.0, 12.0)[1]
    assert sat_ok == 1  # Channel 21 saturates near 500 K
    sat_no = pg.classify_saturation("MODIS", 497.9, 12.0)[1]
    assert sat_no == 0
    sat_zero = pg.classify_saturation("MODIS", 499.0, 0.0)[1]
    assert sat_zero == 0  # frp > 0 required


def test_03_tobit_sat_flag_viirs_high_frp():
    i4, sat = pg.classify_saturation("VIIRS", 366.9, 300.0)
    assert sat == 1  # saturated I4 + extreme FRP -> right-censored
    i4, sat = pg.classify_saturation("VIIRS", 366.9, 299.9)
    assert sat == 0
    i4, sat = pg.classify_saturation("VIIRS", 367.5, 350.0)
    assert (i4, sat) == (1, 1)  # both flags coexist


def test_03_normal_detections_unflagged():
    assert pg.classify_saturation("MODIS", 400.0, 30.0) == (0, 0)
    assert pg.classify_saturation("VIIRS", 350.0, 30.0) == (0, 0)


def test_03_pass_rows_keyed_by_platform_and_dn():
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 5.0, 340.0),  # SNPP day
        ("2021-08-01", 1342, "N", -12.60, -55.95, 5.0, 340.0),  # other cell
        ("2021-08-01", 134, "N", -29.10, -67.45, 5.0, 340.0),   # SNPP night
    ]
    # move second row far enough to differ in H3 cell; third row: local night
    rows[1] = ("2021-08-01", 1342, "N", -12.60, -55.95, 5.0, 340.0)
    df = _frame(rows)
    prov, passes = pg.preprocess_frame(df)
    keys = set(
        zip(passes["h3_index"], passes["solar_date"], passes["platform"],
            passes["dn_flag"])
    )
    assert len(keys) == len(passes)  # keys strictly unique
    assert passes["dn_flag"].tolist().count("N") >= 1
    assert passes["dn_flag"].tolist().count("D") >= 1


def test_03_pass_aggregation_weighted_frp_sum():
    # two non-overlapping VIIRS pixels in one pass, same cell/half-day
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 7.5, 340.0),
        ("2021-08-01", 1342, "N", -29.101, -67.449, 12.5, 340.0),
    ]
    prov, passes = pg.preprocess_frame(_frame(rows))
    assert len(passes) == 1
    assert passes.iloc[0]["y_frp_mw"] == pytest.approx(20.0)
    assert passes.iloc[0]["n_pixels"] == 2


def test_03_raw_event_count_distinct_clusters():
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 5.0, 340.0),
        ("2021-08-01", 1342, "N", -29.103, -67.45, 5.0, 340.0),
        ("2021-08-01", 1342, "N", -29.35, -67.70, 5.0, 340.0),
    ]
    prov, passes = pg.preprocess_frame(_frame(rows))
    # Two separate clusters, potentially the same H3 Res-6 cell aggregation:
    total_events = passes["raw_event_count"].sum()
    assert total_events == 2


def test_03_detection_footprint_weighted_sum():
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 5.0, 340.0),
        ("2021-08-01", 1342, "N", -29.101, -67.449, 5.0, 340.0),
    ]
    prov, passes = pg.preprocess_frame(_frame(rows))
    expected = 2 * 0.20  # Pixarea 0.20 km^2 per pixel, omega = 1
    assert passes.iloc[0]["det_footprint_km2"] == pytest.approx(
        expected, abs=1e-9
    )


def test_03_volcano_offshore_excluded_from_passes():
    rows = [
        ("2021-08-01", 1342, "N", -29.10, -67.45, 5.0, 340.0),
        ("2021-08-01", 1342, "N", -29.12, -67.47, 5.0, 340.0),  # volcano
        ("2021-08-01", 1342, "N", -29.14, -67.49, 5.0, 340.0),  # offshore
    ]
    df = _frame(rows)
    df.loc[1, "type_raw"] = 1
    df.loc[2, "type_raw"] = 3
    prov, passes = pg.preprocess_frame(df)
    # Volcano/offshore rows stay in the provenance base with their flags set
    assert prov["is_excluded_thermal"].tolist() == [False, True, True]
    assert len(passes) == 1  # excluded from biomass fusion
    assert passes.iloc[0]["n_pixels"] == 1


def test_03_sama_rows_excluded_but_retained_in_provenance():
    # A SAMA nighttime low-confidence VIIRS point inside the box.
    sama_frame = pd.DataFrame(
        {
            "sensor_family": ["VIIRS"], "platform": ["SNPP"],
            "latitude": [-30.0], "longitude": [-45.0],
            "acq_date": ["2021-08-01"], "acq_hhmm": [230],
            "t_mir_k": [340.0], "t_tir_k": [300.0],
            "scan_km": [np.nan], "track_km": [np.nan],
            "sample_idx": pd.array([3200], dtype="Int64"),
            "pixarea_km2": [0.2], "frp_mw": [5.0],
            "conf_raw": ["l"], "conf_num": [20.0], "type_raw": [0],
            "data_tier": ["ML_SCIENCE"], "country_iso": [None],
            "version_raw": ["V"], "ingest_ts": ["2026-11-14T00:00:00Z"],
        }
    )
    prov, passes = pg.preprocess_frame(sama_frame)
    assert prov["is_sama_filtered"].tolist() == [True]
    assert len(passes) == 0  # excluded from fusion
    assert len(prov) == 1    # retained in Product 1 provenance (INV-08)
