"""test_01 -- Ingestion schema tests (18 tests).

Readme Section 7.1: multi-schema parsing (MCD14ML space .txt with trailing-dot
sample and v1 rows missing ``dn``; Vxx14IMGML CSV; Portolan GeoParquet
geometry; FIRMS API CSV), satellite alias normalization, mixed confidence
parsing, missing ``type -> -1``, (platform, acq_date) Science-tier dominance
removing shifted NRT duplicates, polygon clipping excluding neighbors
(Bolivia/Tunisia/Chile), base-60 ``acq_hhmm`` conversion (``51 -> 0.85 h``),
and solar-date midnight rollover across +-180 degrees.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.geometry.solar_geometry import hhmm_to_hours
from src.ingest.normalize_dedup import (
    clip_to_countries,
    compute_platform_active_day,
    load_country_polygons,
    normalize_confidence,
    normalize_platform,
    parse_firms_api_csv,
    parse_mcd14ml_rows,
    parse_portolan_frame,
    parse_vxx14imgml_rows,
    platform_from_vxx_filename,
    science_tier_dominance_dedup,
)


# ---------------------------------------------------------------- satellites
def test_01_satellite_alias_normalization():
    assert normalize_platform("T") == "Terra"
    assert normalize_platform("TERRA") == "Terra"
    assert normalize_platform("A") == "Aqua"
    assert normalize_platform("AQUA") == "Aqua"
    assert normalize_platform("N") == "SNPP"
    assert normalize_platform("NPP") == "SNPP"
    assert normalize_platform("SUOMI-NPP") == "SNPP"
    assert normalize_platform("1") == "NOAA-20"
    assert normalize_platform("J1") == "NOAA-20"
    assert normalize_platform("N20") == "NOAA-20"
    assert normalize_platform("2") == "NOAA-21"
    assert normalize_platform("N21") == "NOAA-21"
    assert normalize_platform(None) is None
    assert normalize_platform(float("nan")) is None


def test_01_vxx_filename_platform_inference():
    assert platform_from_vxx_filename("VNP14IMGML.2021_07.csv.gz") == "SNPP"
    assert platform_from_vxx_filename("VJ114IMGML.2020_01.csv.gz") == "NOAA-20"
    assert platform_from_vxx_filename("VJ214IMGML.2024_06.csv.gz") == "NOAA-21"


# ---------------------------------------------------------------- MCD14ML
def test_01_mcd14ml_space_parse_and_trailing_dot_sample():
    rows = [
        "20210814 1330 T -29.160 -67.500 330.5 295.1 681. 25.3 n 0 D",
        "20210814 1331 A -29.170 -67.480 335.2 294.8 690 40.1 h 0 D",
    ]
    df = parse_mcd14ml_rows(rows)
    assert len(df) == 2
    r = df.iloc[0]
    assert r["acq_date"] == "2021-08-14" and r["acq_hhmm"] == 1330
    assert r["platform"] == "Terra" and r["sensor_family"] == "MODIS"
    assert r["t_mir_k"] == 330.5 and r["t_tir_k"] == 295.1
    assert r["sample_idx"] == 681.0          # trailing dot stripped
    assert r["frp_mw"] == 25.3 and r["type_raw"] == 0
    assert df.iloc[1]["platform"] == "Aqua"


def test_01_mcd14ml_v1_missing_dn_parses():
    # version 1 omits the trailing dn column (11 fields)
    rows = ["20100520 0210 T -29.100 -67.400 321.0 290.0 700. 12.0 l 2"]
    df = parse_mcd14ml_rows(rows)
    assert len(df) == 1
    assert df.iloc[0]["type_raw"] == 2
    assert df.iloc[0]["conf_num"] == 20.0


def test_01_vxx14imgml_csv_parse_with_pixarea():
    rows = [
        "YYYYMMDD, HHMM, Sat, Lat, Lon, T_I4, T_I5, Sample, Pixarea, FRP, Conf, Type",
        "20210814, 1335, N, -29.155, -67.495, 312.7, 294.1, 3200, 0.152, 4.8, n, 0",
        "20210814, 0140, 1, -29.150, -67.490, 340.9, 293.7, 3100, 0.148, 9.1, h, 2",
    ]
    df = parse_vxx14imgml_rows(rows, platform_hint="SNPP")
    assert len(df) == 2
    r = df.iloc[0]
    assert r["sensor_family"] == "VIIRS" and r["t_mir_k"] == 312.7
    assert r["pixarea_km2"] == 0.152 and r["frp_mw"] == 4.8
    assert df.iloc[1]["platform"] == "NOAA-20"   # from Sat alias '1'


# ---------------------------------------------------------------- Portolan
def test_01_portolan_wkb_geometry_extraction():
    # little-endian WKB POINT(lon lat)
    import struct

    def wkb_point(x: float, y: float) -> bytes:
        return (b"\x01" + struct.pack("<I", 1) + struct.pack("<dd", x, y))

    df = pd.DataFrame({
        "geometry": [wkb_point(-67.50, -29.16), wkb_point(4.65, 36.85)],
        "acq_datetime": ["2021-08-14 13:35:00", "2021-08-14 01:40:00"],
        "sensor": ["VIIRS", "VIIRS"],
        "satellite": ["N", "1"],
        "quality": ["p", "p"],
        "version": ["2.0nrt", "2.0nrt"],
        "brightness": [np.nan, np.nan], "bright_t31": [290.0, 291.0],
        "bright_ti4": [315.2, 320.0], "bright_ti5": [294.0, 295.0],
        "scan": [0.38, 0.37], "track": [0.38, 0.37],
        "frp": [6.2, 8.8], "daynight": ["D", "N"],
        "confidence": ["n", "h"], "confidence_pct": [68.0, 95.0],
        "type": [0, 2], "_month": [8, 8], "_hilbert": [11, 12],
    })
    out = parse_portolan_frame(df)
    assert len(out) == 2
    r = out.iloc[0]
    assert r["longitude"] == pytest.approx(-67.50, abs=1e-12)
    assert r["latitude"] == pytest.approx(-29.16, abs=1e-12)
    assert r["platform"] == "SNPP" and r["t_mir_k"] == 315.2
    assert r["acq_hhmm"] == 1335 and r["acq_date"] == "2021-08-14"
    assert out.iloc[1]["platform"] == "NOAA-20" and out.iloc[1]["type_raw"] == 2


def test_01_firms_api_csv_parse_missing_type():
    csv_text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_t31,frp,daynight\n"
        "-29.161,-67.502,314.1,1.0,1.0,2026-11-10,1332,N,VIIRS,n,3.0,293.5,7.4,D\n"
        "-29.150,-67.480,335.0,1.1,1.0,2026-11-10,1333,A,MODIS,l,6.03,295.0,0.0,N\n"
    )
    df = parse_firms_api_csv(csv_text, data_tier="NRT")
    assert len(df) == 2
    r = df.iloc[0]
    assert r["platform"] == "SNPP" and r["data_tier"] == "NRT"
    assert r["type_raw"] == -1                       # NRT drops `type`
    assert r["frp_mw"] == 7.4 and r["acq_hhmm"] == 1332
    assert df.iloc[1]["platform"] == "Aqua" and df.iloc[1]["conf_num"] == 20.0


# ---------------------------------------------------------------- confidence
def test_01_mixed_confidence_parsing():
    assert normalize_confidence("l") == ("l", 20.0)
    assert normalize_confidence("n") == ("n", 65.0)
    assert normalize_confidence("h") == ("h", 95.0)
    assert normalize_confidence("82") == ("82", 82.0)
    assert normalize_confidence(74.0) == ("74.0", 74.0)
    assert normalize_confidence(None)[1] == 65.0


# ---------------------------------------------------------------- dedup
def test_01_science_tier_dominance_drops_shifted_nrt():
    from tests.conftest import canonical_frame
    rows = [
        # Science ML row (Terra 2026-08-14)
        dict(platform="Terra", data_tier="ML_SCIENCE", latitude=-29.160,
             longitude=-67.500, acq_hhmm=1330, frp_mw=25.0),
        # NRT row same platform/date shifted by ~150 m (LUT shift) -> dropped
        dict(platform="Terra", data_tier="NRT", latitude=-29.1612,
             longitude=-67.5018, acq_hhmm=1330, frp_mw=24.8),
        # NRT row on a date without Science coverage -> kept
        dict(platform="Terra", data_tier="NRT", latitude=-29.160,
             longitude=-67.500, acq_date="2026-11-10", acq_hhmm=1330,
             frp_mw=11.0),
        # NRT row for a different platform/date -> kept
        dict(platform="SNPP", data_tier="NRT", frp_mw=7.4),
    ]
    df = canonical_frame(rows)
    out = science_tier_dominance_dedup(df)
    assert len(out) == 3
    kept_pairs = set(zip(out["platform"], out["acq_date"]))
    assert ("Terra", "2026-08-14") in kept_pairs
    assert ("Terra", "2026-11-10") in kept_pairs
    assert ("SNPP", "2026-08-14") in kept_pairs
    # the surviving Terra 2026-08-14 row must be the Science one
    t = out[(out["platform"] == "Terra") & (out["acq_date"] == "2026-08-14")]
    assert len(t) == 1 and t.iloc[0]["data_tier"] == "ML_SCIENCE"


def test_01_intra_tier_dedup_keeps_lowest_rank_newest_ingest():
    from tests.conftest import canonical_frame
    rows = [
        dict(platform="SNPP", data_tier="NRT", latitude=-29.1600,
             longitude=-67.5000, acq_hhmm=1335, ingest_ts="2026-11-13T00:00:00Z"),
        dict(platform="SNPP", data_tier="NRT", latitude=-29.1604,
             longitude=-67.5003, acq_hhmm=1335, ingest_ts="2026-11-14T00:00:00Z"),
        dict(platform="SNPP", data_tier="SP_SCIENCE", latitude=-29.1604,
             longitude=-67.5003, acq_hhmm=1335,
             ingest_ts="2026-11-12T00:00:00Z"),
    ]
    out = science_tier_dominance_dedup(canonical_frame(rows))
    snpp = out[(out["platform"] == "SNPP") & (out["acq_date"] == "2026-08-14")]
    assert len(snpp) == 1
    # SP_SCIENCE (rank 1) beats NRT (rank 2) even with older ingest_ts
    assert snpp.iloc[0]["data_tier"] == "SP_SCIENCE"


# ---------------------------------------------------------------- clipping
def test_01_polygon_clipping_excludes_neighbors(paths):
    from tests.conftest import canonical_frame
    rows = [
        dict(country_iso=None, latitude=-29.16, longitude=-67.50),  # ARG inside
        dict(country_iso=None, latitude=-22.2, longitude=-66.5),    # Bolivia-like -> out
        dict(country_iso=None, latitude=-29.16, longitude=-67.50 - 3.0),  # Chile-like -> out
        dict(country_iso=None, latitude=36.9, longitude=4.8),       # DZA inside
        dict(country_iso=None, latitude=37.2, longitude=9.9),       # Tunisia-like -> out
        dict(country_iso=None, latitude=-29.16, longitude=-67.50 + 3.0),  # edge just outside
    ]
    polygons = load_country_polygons(paths.country_polygons_parquet)
    out = clip_to_countries(canonical_frame(rows), polygons)
    isos = sorted(out["country_iso"].dropna().unique())
    assert isos == ["ARG", "DZA"]
    assert len(out) == 2


# ---------------------------------------------------------------- time parse
def test_01_base60_hhmm_conversion():
    # 51 means 00:51 -> 0.85 hours
    assert hhmm_to_hours(51) == pytest.approx(0.85, abs=1e-12)
    assert hhmm_to_hours(1330) == pytest.approx(13.5, abs=1e-12)
    assert hhmm_to_hours(1) == pytest.approx(1.0 / 60.0, abs=1e-12)
    assert hhmm_to_hours(2359) == pytest.approx(23.983333333, abs=1e-6)


def test_01_solar_date_midnight_rollover():
    from src.geometry.solar_geometry import compute_solar_time_and_sza
    # UTC 2026-08-14 23:50 at lon +150 deg -> local 09:50 next calendar day
    solar_hour, solar_date, dn_flag, _ = compute_solar_time_and_sza(
        np.array(["2026-08-14"], dtype="datetime64[D]"), np.array([2350]),
        np.array([150.0]), np.array([-20.0]))
    assert solar_hour[0] == pytest.approx(9.8333, abs=1e-3)
    assert np.datetime64(solar_date[0]) == np.datetime64("2026-08-15")
    assert dn_flag[0] == "D"
    # UTC 2026-08-14 00:10 at lon -150 deg -> local 14:10 of the PREVIOUS
    # solar day (tau_raw = -9.833 < 0 shifts the solar date by -1 day).
    solar_hour2, solar_date2, dn2, _ = compute_solar_time_and_sza(
        np.array(["2026-08-14"], dtype="datetime64[D]"), np.array([10]),
        np.array([-150.0]), np.array([-20.0]))
    assert solar_hour2[0] == pytest.approx(14.1666667, abs=1e-3)
    assert np.datetime64(solar_date2[0]) == np.datetime64("2026-08-13")
    assert dn2[0] == "D"
    # UTC 00:10 at lon +170 -> local 11:30 same solar day
    sh3, sd3, _, _ = compute_solar_time_and_sza(
        np.array(["2026-08-14"], dtype="datetime64[D]"), np.array([10]),
        np.array([170.0]), np.array([-20.0]))
    assert sh3[0] == pytest.approx(11.5, abs=1e-3)
    assert np.datetime64(sd3[0]) == np.datetime64("2026-08-14")


# ---------------------------------------------------------------- SZA day/night
def test_01_sza_daynight_replaces_orbit_node_at_high_latitude():
    from src.geometry.solar_geometry import compute_solar_time_and_sza
    # Alaska mid-summer: an NRT 'N' (descending, 04:30 local) pass at 65N is
    # physically Daylight (SZA < 85) in June -- SZA rule must label it 'D'
    # in the stored provenance flag while the latent window stays Night.
    _, _, dn_window, sza = compute_solar_time_and_sza(
        np.array(["2026-06-21"], dtype="datetime64[D]"), np.array([430]),
        np.array([-150.0]), np.array([65.0]))
    assert sza[0] < 85.0                     # physically daytime (sun up)
    assert dn_window[0] == "N"               # latent window by solar hour 2.5
    # Equator noon vs midnight
    _, _, dn_eq_noon, sza_noon = compute_solar_time_and_sza(
        np.array(["2026-03-20"], dtype="datetime64[D]"), np.array([1200]),
        np.array([0.0]), np.array([0.0]))
    assert sza_noon[0] < 10.0 and dn_eq_noon[0] == "D"
    _, _, dn_eq_night, sza_night = compute_solar_time_and_sza(
        np.array(["2026-03-20"], dtype="datetime64[D]"), np.array([0]),
        np.array([0.0]), np.array([0.0]))
    assert sza_night[0] > 85.0 and dn_eq_night[0] == "N"


# ---------------------------------------------------------------- schema gate
def test_01_schema_validation_gate():
    from src.ingest.download_data import schema_validation_gate
    from tests.conftest import canonical_frame
    good = canonical_frame([dict()] * 50)
    assert schema_validation_gate(good, "unit")
    bad = canonical_frame([dict()] * 50)
    bad.loc[0, "latitude"] = 200.0            # out of range
    assert not schema_validation_gate(bad, "unit")
    bad2 = canonical_frame([dict()] * 50)
    bad2.loc[0:10, "frp_mw"] = -5.0           # negative FRP
    assert not schema_validation_gate(bad2, "unit")
    bad3 = good.copy()
    bad3.loc[0:5, "acq_date"] = None          # > 1% null dates
    assert not schema_validation_gate(bad3, "unit")


# ---------------------------------------------------------------- platform_active_day
def test_01_platform_active_day_outage_masking():
    from tests.conftest import canonical_frame
    rows = []
    # SNPP healthy on 2026-10-30 (10 detections), outage on 2026-10-31 (2)
    for i in range(10):
        rows.append(dict(platform="SNPP", acq_date="2026-10-30", frp_mw=5.0))
    for i in range(2):
        rows.append(dict(platform="SNPP", acq_date="2026-10-31", frp_mw=5.0))
    for i in range(6):
        rows.append(dict(platform="NOAA-20", acq_date="2026-10-31", frp_mw=4.0))
    active = compute_platform_active_day(canonical_frame(rows))
    assert "2026-10-30" in active["SNPP"]
    assert "2026-10-31" not in active["SNPP"]     # < 5 detections -> outage
    assert "2026-10-31" in active["NOAA-20"]


def test_01_coverage_audit_fields(paths):
    from src.ingest.normalize_dedup import coverage_audit
    from tests.conftest import canonical_frame
    rows = [dict(platform="Terra", acq_date="2026-08-01", frp_mw=20.0,
                 latitude=-29.16, longitude=-67.50, country_iso="ARG")]
    audit = coverage_audit(canonical_frame(rows))
    assert audit["chilecito_jul_aug_2026_detections"] == 1
    assert "snpp_last_acq_date" in audit
    assert "dza_type2_flares_preserved" in audit
    assert audit["n_rows"] == 1


def test_01_canonical_schema_columns_complete():
    from src.ingest.normalize_dedup import CANONICAL_COLUMNS
    need = {"country_iso", "sensor_family", "platform", "data_tier", "latitude",
            "longitude", "acq_date", "acq_hhmm", "t_mir_k", "t_tir_k",
            "scan_km", "track_km", "sample_idx", "pixarea_km2", "frp_mw",
            "conf_raw", "conf_num", "type_raw", "version_raw", "ingest_ts"}
    assert need.issubset(set(CANONICAL_COLUMNS))


def test_01_missing_type_defaults_to_minus_one():
    csv_text = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_t31,frp,daynight\n"
        "-29.161,-67.502,314.1,1.0,1.0,2026-11-10,1332,N,VIIRS,n,3.0,293.5,7.4,D\n")
    df = parse_firms_api_csv(csv_text)
    assert (df["type_raw"] == -1).all()


def test_01_gz_chunk_roundtrip(tmp_path, paths):
    # MCD14ML .txt.gz chunk -> parsed via run_normalization staged path
    import gzip
    raw_dir = paths.raw_stage_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "20260814 1330 T -29.160 -67.500 330.5 295.1 681. 25.3 n 0 D",
        "20260814 1335 N -29.155 -67.495 312.7 294.1 3200 4.8 n 0 D",
    ]
    (raw_dir / "MCD14ML.2026_08.txt.gz").write_bytes(
        gzip.compress("\n".join(lines).encode()))
    from src.ingest.normalize_dedup import run_normalization
    report = run_normalization(paths)
    assert report["n_rows"] == 2
    assert report["audit"]["platforms_present"] == ["SNPP", "Terra"]


def test_01_alaska_second_prefilter_box_kept(paths):
    """USA has TWO prefilter boxes (CONUS + Alaska); clipping ORs them so
    the Alaska Boreal spotlight survives (the fixture USA polygon is an
    Alaska square around (-150, 65))."""
    from src.ingest.normalize_dedup import clip_to_countries, load_country_polygons
    polys = load_country_polygons(paths.country_polygons_parquet)
    df = pd.DataFrame({"latitude": [65.2, 10.0], "longitude": [-150.3, -150.0],
                       "frp_mw": [3.0, 3.0]})
    out = clip_to_countries(df, polys)
    assert len(out) == 1 and out.iloc[0]["country_iso"] == "USA"


def test_01_firms_api_sensor_specific_columns():
    """FIRMS API: VIIRS rows map bright_ti4/ti5, MODIS rows brightness /
    bright_t31; scan/track carried as pixel km (never NaN)."""
    csv_text = (
        "latitude,longitude,brightness,bright_ti4,scan,track,acq_date,acq_time,"
        "satellite,instrument,confidence,version,bright_t31,bright_ti5,frp,daynight\n"
        "-29.16,-67.50,,331.5,0.39,0.36,2026-11-10,1332,N20,VIIRS,h,2.0NRT,,290.1,7.4,D\n"
        "-29.15,-67.48,345.2,,1.20,1.10,2026-11-10,1333,T,MODIS,77,6.1NRT,296.4,,30.0,D\n")
    df = parse_firms_api_csv(csv_text)
    v, m = df.iloc[0], df.iloc[1]
    assert v["platform"] == "NOAA-20" and v["t_mir_k"] == 331.5
    assert v["t_tir_k"] == 290.1 and v["scan_km"] == 0.39
    assert v["track_km"] == 0.36
    assert m["platform"] == "Terra" and m["t_mir_k"] == 345.2
    assert m["t_tir_k"] == 296.4 and m["scan_km"] == 1.20
    assert m["conf_num"] == 77.0 and (df["type_raw"] == -1).all()


def test_01_portolan_modis_temperature_and_tier():
    """A MODIS Portolan row with a present-but-NaN bright_ti4 column keeps
    its MODIS brightness (MODIS saturation detection intact) and the tier
    comes from quality/version (NRT vs standard processing)."""
    df = pd.DataFrame({
        "longitude": [-67.5, -67.4], "latitude": [-29.1, -29.2],
        "acq_datetime": ["2015-08-14 13:35:00", "2026-11-12 14:00:00"],
        "sensor": ["MODIS", "MODIS"], "satellite": ["Aqua", "Terra"],
        "quality": ["standard", "nrt"], "version": ["6.1", "6.1NRT"],
        "brightness": [499.0, 330.0], "bright_t31": [300.0, 299.0],
        "bright_ti4": [np.nan, np.nan], "bright_ti5": [np.nan, np.nan],
        "scan": [1.0, 1.1], "track": [1.0, 1.0], "frp": [250.0, 10.0],
        "daynight": ["D", "D"], "confidence": [90, 60], "type": [0, np.nan]})
    out = parse_portolan_frame(df)
    assert out.iloc[0]["t_mir_k"] == 499.0 and out.iloc[0]["t_tir_k"] == 300.0
    assert out.iloc[0]["data_tier"] == "SP_SCIENCE"
    assert out.iloc[1]["data_tier"] == "NRT" and out.iloc[1]["type_raw"] == -1


def test_01_nrt_parquet_chunk_ingested(paths):
    """download_all stores NRT gap-fill as firms_nrt_*.parquet; normalization
    reads it as Parquet (never as text) and tags it NRT."""
    from src.ingest.normalize_dedup import run_normalization
    raw_dir = paths.raw_stage_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"latitude": [f"{-29.16 - 0.01 * i:.3f}" for i in range(6)],
                  "longitude": ["-67.50"] * 6,
                  "bright_ti4": ["320"] * 6, "scan": ["0.4"] * 6,
                  "track": ["0.4"] * 6, "acq_date": ["2026-11-12"] * 6,
                  "acq_time": ["1730"] * 6, "satellite": ["N21"] * 6,
                  "instrument": ["VIIRS"] * 6, "confidence": ["n"] * 6,
                  "version": ["2.0NRT"] * 6, "bright_ti5": ["290"] * 6,
                  "frp": ["3.5"] * 6, "daynight": ["D"] * 6}).to_parquet(
        raw_dir / "firms_nrt_VIIRS_NOAA21_NRT_ARG_2026-11-12.parquet")
    rep = run_normalization(paths)
    assert rep["n_rows"] == 6
    assert rep["platform_active_day"]["NOAA-21"] == ["2026-11-12"]


def test_01_cmg_filename_periods_and_nrt_windows():
    """8-day vs monthly CMG periods never collide (period_kind), and NRT
    gap-fill steps 5-day windows from the last Science date to the
    reference date (DAY_RANGE <= 5)."""
    from src.ingest import download_data as d
    m8 = d.parse_cmg_filename("MOD14C8Q.A2015009.061.2018123.hdf")
    mm = d.parse_cmg_filename("MYD14CMQ.A2015213.061.2019.hdf")
    assert (m8["period"], m8["period_kind"]) == (2, "8d")
    assert (mm["period"], mm["period_kind"]) == (8, "month")
    w = d.nrt_windows("2026-10-31", "2026-11-14")
    assert w == [("2026-11-01", 5), ("2026-11-06", 5), ("2026-11-11", 4)]
    assert all(1 <= r <= 5 for _, r in d.nrt_windows("2026-07-01"))


def test_01_cloud_cmg_parse_scales_percent_without_nanmax_initial():
    """parse_cloud_cmg normalizes 0..100 MeanCloudFraction with dynamic
    scale/fill attributes (no unsupported np.nanmax(initial=...))."""
    from src.ingest import download_data as d
    raw = np.full((720, 1440), 255.0)
    raw[240, 450] = 40.0                     # one valid 40% cell
    attrs = {"_FillValue": 255.0, "scale_factor": 1.0}
    df = d.parse_cloud_cmg({"MeanCloudFraction": (attrs, raw)}, 2015, 2, "8d")
    assert len(df) == 1
    assert df.iloc[0]["cloud_frac"] == pytest.approx(0.40)
    assert df.iloc[0]["period_kind"] == "8d"


def test_01_streaming_normalization_dedups_across_files(paths, frame_builder):
    """Bounded-memory normalization: Science and NRT rows for the same
    (platform, acq_date) arriving in DIFFERENT staged files are still
    deduplicated (NRT dropped), and months are partitioned separately."""
    from src.ingest.normalize_dedup import run_normalization
    sci = frame_builder([dict(platform="SNPP", sensor_family="VIIRS",
                              data_tier="SP_SCIENCE", acq_date="2021-08-14",
                              latitude=-29.16 - 0.01 * i) for i in range(5)]
                        + [dict(platform="SNPP", sensor_family="VIIRS",
                                data_tier="SP_SCIENCE", acq_date="2021-09-02",
                                latitude=-29.20)])
    nrt = frame_builder([dict(platform="SNPP", sensor_family="VIIRS",
                              data_tier="NRT", acq_date="2021-08-14",
                              latitude=-29.1601 - 0.01 * i) for i in range(5)])
    rep = run_normalization(paths, staged_frames=[sci, nrt])
    assert rep["n_rows_parsed"] == 11 and rep["n_rows"] == 6
    parts = sorted(p.name for p in paths.active_fires_dir.rglob("*.parquet"))
    assert parts == ["part_2021-08.parquet", "part_2021-09.parquet"]
    assert rep["platform_active_day"]["SNPP"] == ["2021-08-14"]


def test_01_portolan_wkb_vectorized_decoder():
    """The PyArrow Portolan failover decodes GeoParquet WKB POINTs
    vectorially (little- and big-endian) identically to the scalar reader."""
    import struct

    import pyarrow as pa
    from src.ingest.download_data import _wkb_points_xy
    from src.ingest.normalize_dedup import _wkb_point
    pts = [(-67.5, -29.16), (4.65, 36.85), (-150.25, 65.5)]
    wkb = [b"\x01" + struct.pack("<I", 1) + struct.pack("<dd", *pts[0]),
           b"\x00" + struct.pack(">I", 1) + struct.pack(">dd", *pts[1]),
           b"\x01" + struct.pack("<I", 1) + struct.pack("<dd", *pts[2])]
    x, y = _wkb_points_xy(pa.chunked_array([pa.array(wkb[:2], pa.binary()),
                                            pa.array(wkb[2:], pa.binary())]))
    for i, (px, py) in enumerate(pts):
        assert (x[i], y[i]) == (px, py) == _wkb_point(wkb[i])
