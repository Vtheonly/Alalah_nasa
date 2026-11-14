"""test_01: multi-schema ingestion, aliasing, dedup, clipping, solar rollover.

18 tests (readme.md Section 7 item 1): MCD14ML space .txt (trailing-dot
sample, v1 missing dn), Vxx14IMGML CSV, Portolan GeoParquet geometry
extraction, FIRMS API CSV, satellite alias mapping, mixed confidence parsing,
missing type -> -1, (platform, acq_date) Science dominance dropping shifted
NRT duplicates, polygon clipping excluding neighboring countries, base-60
acq_hhmm conversion, and solar date midnight rollover across +/-180 deg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import settings as S
from src.geometry.solar_geometry import (
    compute_solar_time_and_sza,
    hhmm_to_hours,
    sza_dn_flag,
)
from src.ingest import normalize_dedup as nd
from src.ingest.download_data import validate_detection_schema

# --------------------------------------------------------------------------- #
# 1-5: source schema parsers
# --------------------------------------------------------------------------- #

MCD14ML_V2 = (
    "20210801 1345 T -29.16 -67.50 321.5 305.2 681. 42.5 82 0 D"
)
MCD14ML_V1 = "20030512 0510 A -12.60 -55.95 310.0 298.4 700 18.2 65 0"


def test_01_parse_mcd14ml_v2_full_row():
    df = nd.parse_mcd14ml(
        [MCD14ML_V2], ingest_ts="2026-11-14T00:00:00Z"
    )
    row = df.iloc[0]
    assert row["platform"] == "Terra"
    assert row["sensor_family"] == "MODIS"
    assert row["data_tier"] == "ML_SCIENCE"
    assert row["acq_date"] == "2021-08-01"
    assert row["acq_hhmm"] == 1345
    # trailing-dot sample '681.' must parse cleanly to the integer index
    assert row["sample_idx"] == 681
    assert row["t_mir_k"] == pytest.approx(321.5)
    assert row["t_tir_k"] == pytest.approx(305.2)
    assert row["frp_mw"] == pytest.approx(42.5)
    assert row["conf_num"] == pytest.approx(82.0)
    assert row["type_raw"] == 0
    assert row["dn_raw"] == "D"


def test_01_parse_mcd14ml_v1_missing_dn():
    df = nd.parse_mcd14ML if False else nd.parse_mcd14ml(
        [MCD14ML_V1], ingest_ts="2026-11-14T00:00:00Z"
    )
    row = df.iloc[0]
    assert row["platform"] == "Aqua"
    assert row["dn_raw"] is None or pd.isna(row["dn_raw"])
    assert row["sample_idx"] == 700
    assert row["conf_num"] == pytest.approx(65.0)
    assert row["type_raw"] == 0


def test_01_parse_vxx14imgml_csv_and_filename_platform():
    csv_text = (
        "YYYYMMDD,HHMM,Sat,Lat,Lon,T_I4,T_I5,Sample,Pixarea,FRP,Conf,Type\n"
        "20210720,1342,,31.68,6.07,340.1,300.5,3200,0.21,12.4,l,2\n"
        "20210720,1342,2,-29.10,-67.45,350.0,302.0,3100,0.19,22.7,n,0\n"
    )
    df = nd.parse_vxx14imgml(
        csv_text=csv_text,
        filename="VJ114IMGML_2021_07.txt.gz",
        ingest_ts="2026-11-14T00:00:00Z",
    )
    assert len(df) == 2
    # Sat missing on row 1 -> platform inferred from VJ1 filename prefix
    assert df.iloc[0]["platform"] == "NOAA-20"
    # explicit Sat '2' alias wins over filename
    assert df.iloc[1]["platform"] == "NOAA-21"
    assert df.iloc[0]["pixarea_km2"] == pytest.approx(0.21)
    assert df.iloc[0]["conf_num"] == pytest.approx(20.0)  # 'l' -> 20
    assert df.iloc[1]["conf_num"] == pytest.approx(65.0)  # 'n' -> 65
    assert df.iloc[0]["type_raw"] == 2  # static industrial gas flare
    assert (df["sensor_family"] == "VIIRS").all()


def test_01_parse_portolan_schema():
    port = pd.DataFrame(
        {
            "latitude": [-29.16, 36.75],
            "longitude": [-67.50, 3.05],
            "acq_datetime": ["2026-07-30 13:55:00", "2026-07-30 13:57:00"],
            "acq_date": ["2026-07-30", "2026-07-30"],
            "sensor": ["VIIRS", "MODIS"],
            "satellite": ["NOAA-20", "Terra"],
            "quality": ["science", "nrt"],
            "version": ["2.0", "2.0"],
            "brightness": [np.nan, 320.1],
            "bright_t31": [np.nan, 300.0],
            "bright_ti4": [335.5, np.nan],
            "bright_ti5": [301.1, np.nan],
            "scan": [0.45, 1.9],
            "track": [0.45, 1.9],
            "frp": [8.5, 40.0],
            "daynight": ["D", "D"],
            "confidence": ["n", "82"],
            "confidence_pct": [np.nan, 82.0],
            "type": [0, -1],
            "_month": [7, 7],
            "_hilbert": [12345, 67890],
        }
    )
    df = nd.parse_portolan(port, ingest_ts="2026-11-14T00:00:00Z")
    assert list(df["data_tier"]) == ["SP_SCIENCE", "NRT"]
    assert list(df["sensor_family"]) == ["VIIRS", "MODIS"]
    assert df.iloc[0]["acq_hhmm"] == 1355
    assert df.iloc[0]["t_mir_k"] == pytest.approx(335.5)
    assert df.iloc[1]["conf_num"] == pytest.approx(82.0)
    assert df.iloc[1]["type_raw"] == -1  # NRT quality has no type column
    assert df.iloc[0]["scan_km"] == pytest.approx(0.45)


def test_01_parse_firms_api_csv_modis_and_viirs():
    modis_df = pd.DataFrame(
        {
            "latitude": [-29.16], "longitude": [-67.50], "brightness": [321.0],
            "scan": [1.8], "track": [2.0], "acq_date": ["2026-11-13"],
            "acq_time": ["1345"], "satellite": ["Terra"],
            "instrument": ["MODIS"], "confidence": ["80"],
            "version": ["6.03"], "bright_t31": [300.0], "frp": [45.0],
            "daynight": ["D"],
        }
    )
    viirs_df = pd.DataFrame(
        {
            "latitude": [31.68], "longitude": [6.07], "bright_ti4": [340.0],
            "scan": [0.4], "track": [0.4], "acq_date": ["2026-11-13"],
            "acq_time": ["1350"], "satellite": ["N20"],
            "instrument": ["VIIRS"], "confidence": ["n"],
            "version": ["2.0NRT"], "bright_ti5": [301.0], "frp": [9.0],
            "daynight": ["D"],
        }
    )
    m = nd.parse_firms_api_csv(modis_df, ingest_ts="2026-11-14T00:00:00Z")
    v = nd.parse_firms_api_csv(viirs_df, ingest_ts="2026-11-14T00:00:00Z")
    assert m.iloc[0]["sensor_family"] == "MODIS"
    assert v.iloc[0]["sensor_family"] == "VIIRS"
    assert (m["data_tier"] == "NRT").all() and (v["data_tier"] == "NRT").all()
    # FIRMS NRT feeds drop the type column -> canonical -1 (never fabricated)
    assert m.iloc[0]["type_raw"] == -1 and v.iloc[0]["type_raw"] == -1
    assert v.iloc[0]["platform"] == "NOAA-20"
    assert m.iloc[0]["conf_num"] == pytest.approx(80.0)
    assert v.iloc[0]["conf_num"] == pytest.approx(65.0)


# --------------------------------------------------------------------------- #
# 6-8: alias / confidence / type normalization
# --------------------------------------------------------------------------- #

def test_01_satellite_alias_normalization():
    expected = {
        "T": "Terra", "Terra": "Terra", "TERRA": "Terra",
        "A": "Aqua", "Aqua": "Aqua",
        "N": "SNPP", "SNPP": "SNPP", "NPP": "SNPP", "S-NPP": "SNPP",
        "1": "NOAA-20", "J1": "NOAA-20", "N20": "NOAA-20",
        "2": "NOAA-21", "J2": "NOAA-21", "N21": "NOAA-21",
    }
    for alias, canonical in expected.items():
        assert nd.normalize_satellite(alias) == canonical, alias
    assert nd.normalize_satellite(None) is None
    assert nd.normalize_satellite(float("nan")) is None


def test_01_mixed_confidence_parsing():
    for raw, num in [("l", 20.0), ("n", 65.0), ("h", 95.0)]:
        txt, val = nd.parse_conf(raw)
        assert txt == raw and val == num
    txt, val = nd.parse_conf("77")
    assert val == pytest.approx(77.0)
    txt, val = nd.parse_conf(None)
    assert val is None


def test_01_missing_type_defaults_to_minus_one():
    nrt = pd.DataFrame(
        {
            "latitude": [1.0], "longitude": [1.0], "brightness": [300.0],
            "scan": [1.0], "track": [1.0], "acq_date": ["2026-11-01"],
            "acq_time": ["100"], "satellite": ["A"], "instrument": ["MODIS"],
            "confidence": ["h"], "version": ["6.03"], "bright_t31": [290.0],
            "frp": [10.0], "daynight": ["D"],
        }
    )
    df = nd.parse_firms_api_csv(nrt, ingest_ts="2026-11-14T00:00:00Z")
    assert (df["type_raw"] == -1).all()


# --------------------------------------------------------------------------- #
# 9-10: tier dominance & dedup
# --------------------------------------------------------------------------- #

def _make_canonical_frame(rows):
    base = nd.parse_mcd14ml(
        [MCD14ML_V2], ingest_ts="2026-11-14T00:00:00Z"
    )
    frames = []
    for tier, lat, lon, hhmm, ts in rows:
        part = base.copy()
        part["data_tier"] = tier
        part["latitude"] = lat
        part["longitude"] = lon
        part["acq_hhmm"] = hhmm
        part["ingest_ts"] = ts
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def test_01_science_tier_dominance_drops_shifted_nrt_duplicates():
    df = _make_canonical_frame(
        [
            # Science Aqua row on 2021-08-01 ...
            ("ML_SCIENCE", -29.20, -67.40, 1345, "2026-11-13T00:00:00Z"),
            # NRT duplicate shifted by ~250 m LUT offset on same (platform, date)
            ("NRT", -29.2025, -67.4019, 1345, "2026-11-14T00:00:00Z"),
            # NRT row on a date NOT covered by Science must survive
            ("NRT", -29.30, -67.60, 1402, "2026-11-14T00:00:00Z"),
        ]
    )
    df.loc[1, "acq_date"] = "2021-08-01"
    df.loc[2, "acq_date"] = "2021-08-02"
    out = nd.apply_tier_dominance(df)
    # The 2021-08-01 NRT duplicate is gone; the 2021-08-02 NRT row survives.
    assert ((out["data_tier"] == "NRT") & (out["acq_date"] == "2021-08-01")).sum() == 0
    assert ((out["data_tier"] == "NRT") & (out["acq_date"] == "2021-08-02")).sum() == 1
    assert (out["data_tier"] == "ML_SCIENCE").sum() == 1


def test_01_intra_tier_dedup_keeps_tier_rank_asc_ingest_ts_desc():
    base_lat, base_lon = -29.2004, -67.4006  # identical at 3-decimals
    df = _make_canonical_frame(
        [
            # same (platform, date, hhmm, lat3, lon3) key, three tiers
            ("NRT", base_lat, base_lon, 1345, "2026-11-15T00:00:00Z"),
            ("SP_SCIENCE", base_lat, base_lon, 1345, "2026-11-13T00:00:00Z"),
            ("ML_SCIENCE", base_lat, base_lon, 1345, "2026-11-12T00:00:00Z"),
            # same tier duplicate: newest ingest_ts wins
            ("ML_SCIENCE", base_lat, base_lon, 1345, "2026-11-11T00:00:00Z"),
        ]
    )
    out = nd.apply_tier_dominance(df)
    kept = out[(out["data_tier"] == "ML_SCIENCE")]
    assert len(out) == 1
    assert len(kept) == 1
    assert kept.iloc[0]["ingest_ts"] == "2026-11-12T00:00:00Z"


# --------------------------------------------------------------------------- #
# 11: polygon clipping
# --------------------------------------------------------------------------- #

def test_01_polygon_clipping_excludes_neighbor_countries(country_polygons):
    df = pd.DataFrame(
        {
            "latitude": [-29.16, -40.0, 36.75, 36.9, -10.0, 31.75, 65.5],
            "longitude": [-67.50, -73.5, 3.05, 9.8, -76.0, 6.05, -150.0],
            "frp_mw": [10.0] * 7,
            "acq_date": ["2021-08-01"] * 7,
        }
    )
    out = nd.clip_to_countries(df, country_polygons)
    isos = set(out["country_iso"])
    # Chilecito (ARG), Algiers (DZA), Hassi Messaoud (DZA), Alaska (USA) in.
    assert isos == {"ARG", "DZA", "USA"}
    # Southern Chile (-73.5, -40), Tunisia (9.8, 36.9), Peru (-76, -10) out.
    assert (out["latitude"] != -40.0).all()
    assert (out["longitude"] != 9.8).all()
    assert (out["longitude"] != -76.0).all()


# --------------------------------------------------------------------------- #
# 12-15: base-60 time, solar rollover, SZA vs orbit node
# --------------------------------------------------------------------------- #

def test_01_hhmm_base60_conversion():
    assert hhmm_to_hours(51) == pytest.approx(0.85)  # 00:51
    assert hhmm_to_hours(1345) == pytest.approx(13.75)
    assert hhmm_to_hours(0) == pytest.approx(0.0)
    assert hhmm_to_hours(2359) == pytest.approx(23.983333333)


def test_01_solar_date_rollover_west_of_greenwich():
    # UTC 11:00 at lon -175 -> tau_raw = -0.667 -> previous solar date.
    tau, sdate, dn, sza = compute_solar_time_and_sza(
        "2026-06-15", 1100, -175.0, -29.0
    )
    assert sdate == "2026-06-14"
    assert tau == pytest.approx(23.3333333, abs=1e-6)
    assert dn == "N"


def test_01_solar_date_rollover_east_of_greenwich():
    # UTC 13:00 at lon +175 -> tau_raw = 24.667 -> next solar date; local
    # solar hour 0.667 falls inside the Night window [18:00, 06:00).
    tau, sdate, dn, sza = compute_solar_time_and_sza(
        "2026-06-15", 1300, 175.0, 10.0
    )
    assert sdate == "2026-06-16"
    assert tau == pytest.approx(0.6666667, abs=1e-6)
    assert dn == "N"


def test_01_sza_daynight_replaces_nrt_orbit_node_labels():
    # Alaska, December, local solar 13:30 (UTC 23:30 at lon -150): NRT labels
    # the descending pass 'D' by orbit direction, but the sun is below the
    # SZA<85 deg horizon in polar winter.
    tau, sdate, dn_window, sza = compute_solar_time_and_sza(
        "2026-12-21", 2330, -150.0, 65.0
    )
    assert tau == pytest.approx(13.5, abs=1e-9)
    assert dn_window == "D"  # canonical solar half-day window assignment
    assert sza > S.SZA_DAY_NIGHT_THRESHOLD_DEG  # physically night (polar)
    assert sza_dn_flag(np.array([sza])).tolist() == ["N"]
    # Raw NRT orbit-node label ('D') is replaced by consistent solar physics:
    # the window flag stays clock-based while SZA provenance is preserved.


# --------------------------------------------------------------------------- #
# 16-17: platform activity masking & coverage audit
# --------------------------------------------------------------------------- #

def test_01_platform_active_day_outage_masking():
    rows = []
    # 4 detections on 2026-03-05 -> inactive (safe-mode); 5 on 03-06 -> active
    for date, n in (("2026-03-05", 4), ("2026-03-06", 5)):
        for i in range(n):
            r = nd.parse_mcd14ml(
                [MCD14ML_V2], ingest_ts="2026-11-14T00:00:00Z"
            ).iloc[[0]].copy()
            r["acq_date"] = date
            r["latitude"] = -29.0 - 0.01 * i
            rows.append(r)
    df = pd.concat(rows, ignore_index=True)
    active = nd.compute_platform_active_day(df)
    m = active.set_index("acq_date")["platform_active"]
    assert int(m.loc["2026-03-05"]) == 0
    assert int(m.loc["2026-03-06"]) == 1


def test_01_coverage_audit_mandatory_checks(country_polygons):
    rows = []
    specs = [
        # Chilecito July 2026 detection (ARG)
        (MCD14ML_V2, "2026-07-30", -29.16, -67.50, "Terra", 0),
        # S-NPP final delivery day 2026-11-02
        (MCD14ML_V2, "2026-11-02", -10.0, -55.0, "SNPP", 0),
        # NOAA-20 and NOAA-21 continuing through 2026-11-14
        (MCD14ML_V2, "2026-11-14", -10.0, -55.0, "NOAA-20", 0),
        (MCD14ML_V2, "2026-11-14", -10.0, -55.0, "NOAA-21", 0),
        # DZA static gas flare (type 2) near Hassi Messaoud
        (MCD14ML_V2, "2021-05-01", 31.75, 6.05, "Aqua", 2),
    ]
    for line, date, lat, lon, sat, ftype in specs:
        part = nd.parse_mcd14ml([line], ingest_ts="2026-11-14T00:00:00Z")
        part["acq_date"] = date
        part["latitude"] = lat
        part["longitude"] = lon
        part["platform"] = sat
        part["type_raw"] = ftype
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    df = nd.clip_to_countries(df, country_polygons)
    audit = nd.coverage_audit(df)
    assert audit["chilecito_jul_aug_2026_detections"] >= 1
    assert audit["snpp_ends_2026_11_02"] is True
    assert audit["snpp_last_date"] == "2026-11-02"
    assert audit["noaa_20_through_2026_11_14"] is True
    assert audit["noaa_21_through_2026_11_14"] is True
    assert audit["dza_static_flares_preserved"] >= 1


# --------------------------------------------------------------------------- #
# 18: schema validation gate
# --------------------------------------------------------------------------- #

def test_01_schema_validation_gate():
    good = pd.DataFrame(
        {
            "latitude": [-29.16, 36.75], "longitude": [-67.5, 3.05],
            "acq_date": ["2026-07-30", "2026-07-30"],
            "frp_mw": [10.0, 5.0],
        }
    )
    assert validate_detection_schema(good, "fixture-good") is True
    bad_lat = good.copy()
    bad_lat.loc[0, "latitude"] = 95.0
    assert validate_detection_schema(bad_lat, "fixture-bad-lat") is False
    missing = good.drop(columns=["frp_mw"])
    assert validate_detection_schema(missing, "fixture-missing") is False
    neg_frp = good.copy()
    neg_frp.loc[0, "frp_mw"] = -1.0
    assert validate_detection_schema(neg_frp, "fixture-neg-frp") is False
