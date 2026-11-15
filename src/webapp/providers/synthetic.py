"""SYNTHETIC demonstration dataset in the EXACT schema of the real engine.

Purpose: let visitors explore the complete platform (maps, calendars,
challenge demonstrations, validation, insights, traceability) when the real
harmonization run is unavailable or has not passed validation.  Everything
here is SIMULATED and is labelled as such in the ``dataset_provenance``
table and in every API response.  It is never mixed with real results.

How it stays consistent with the real engine:

* A known synthetic "truth" (daily-mean fire radiative power per H3-Res6
  cell and month, with region-specific seasons, trends and named extreme
  events) drives everything else.
* Raw per-satellite detections (Product 1 rows) are SAMPLED from that truth
  with each sensor's own sensitivity (VIIRS 375 m sees ~4x more small fires
  than MODIS 1 km), overpass times (incl. the post-2022 Terra/Aqua drift),
  launch / end dates, cloud visibility, bow-tie duplicates, SAMA artefacts
  and static gas flares -- so the raw record contains the same artefacts
  the real archive has.
* The harmonized posterior (Product 2) is the truth with calibrated noise
  whose width grows as observation coverage drops (Product 3), and the
  monthly raw columns are exact aggregates of the Product 1 rows.
* Climatology indices, attribution / quality flags and the diagnostics
  JSON are computed with the ENGINE'S OWN functions
  (``conformal.climatology_indices``, ``diagnostics.run_diagnostics``) and
  the DuckDB file is written by the engine's ``export_duckdb``: same
  tables, columns and formats as a real run.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import settings as S

GENERATOR_VERSION = "synthetic-demo-1.0"
PROVENANCE_LABEL = ("SYNTHETIC DEMONSTRATION DATA - simulated, not satellite "
                    "observations and not scientific evidence")


@dataclass(frozen=True)
class SynthRegion:
    iso: str
    name: str
    lat: float
    lon: float
    k: int                 # H3 grid-disk radius (cells = 3k(k+1)+1)
    scale_mw: float        # typical peak-month daily-mean FRP per cell
    peak_month: int
    season_kappa: float    # season sharpness
    ndr: float             # night/day FRP ratio
    cloud_base: float
    kind: str = "fire"     # "fire" | "flare"


REGIONS = (
    SynthRegion("BRA", "Arc of deforestation (Mato Grosso/Para)", -9.5, -55.5,
                6, 40.0, 9, 2.4, 0.12, 0.35),
    SynthRegion("BRA", "Cerrado (Tocantins)", -10.5, -47.8, 5, 22.0, 9, 2.2,
                0.10, 0.30),
    SynthRegion("ARG", "Gran Chaco", -25.0, -62.0, 5, 14.0, 8, 1.8, 0.15,
                0.30),
    SynthRegion("ARG", "Chilecito (La Rioja)", -29.16, -67.5, 3, 3.0, 1, 1.2,
                0.18, 0.15),
    SynthRegion("DZA", "Kabylie forests", 36.6, 4.4, 3, 2.5, 8, 2.8, 0.20,
                0.30),
    SynthRegion("DZA", "Hassi Messaoud gas flares", 31.7, 6.05, 2, 6.0, 1,
                0.0, 0.95, 0.10, kind="flare"),
    SynthRegion("IDN", "Central Kalimantan peatlands", -2.3, 113.9, 5, 25.0,
                9, 3.0, 0.45, 0.60),
    SynthRegion("IDN", "South Sumatra", -3.0, 104.5, 4, 18.0, 9, 3.0, 0.40,
                0.60),
    SynthRegion("AUS", "Top End savanna (NT)", -13.8, 132.5, 6, 30.0, 8, 1.6,
                0.10, 0.25),
    SynthRegion("AUS", "SE forests (NSW/Victoria)", -36.0, 149.0, 4, 4.0, 1,
                2.0, 0.25, 0.40),
    SynthRegion("USA", "Northern California", 39.8, -121.5, 4, 6.0, 8, 2.4,
                0.25, 0.20),
    SynthRegion("USA", "Alaska interior", 65.0, -148.0, 4, 8.0, 7, 3.2, 0.30,
                0.45),
)

# Named interannual signals (multiplicative factors on the regional truth).
EVENTS = {
    "Arc of deforestation (Mato Grosso/Para)": {2004: 1.6, 2005: 1.5, 2007: 1.4,
                                                2010: 1.5, 2019: 1.3, 2020: 1.5,
                                                2024: 1.7},
    "Gran Chaco": {2020: 1.8, 2022: 1.6},
    "Chilecito (La Rioja)": {},
    "Kabylie forests": {2007: 2.2, 2012: 1.8, 2021: 5.0, 2023: 3.0},
    "Central Kalimantan peatlands": {2002: 2.0, 2006: 2.6, 2009: 1.8, 2015: 4.5,
                                     2019: 2.8, 2023: 2.0},
    "South Sumatra": {2006: 2.2, 2015: 4.0, 2019: 2.5, 2023: 1.8},
    "SE forests (NSW/Victoria)": {2003: 2.0, 2007: 1.8, 2009: 2.2},
    "Northern California": {2008: 1.6, 2018: 3.0, 2020: 4.0, 2021: 3.2},
    "Alaska interior": {2004: 3.5, 2005: 2.5, 2009: 1.8, 2015: 3.0, 2019: 2.4,
                        2022: 3.2},
}
TRENDS = {  # linear log-trend per decade from 2001
    "Arc of deforestation (Mato Grosso/Para)": -0.10,
    "Northern California": 0.25,
    "Alaska interior": 0.10,
    "Top End savanna (NT)": -0.05,
}

# Satellites: (first month, last month, sensor family, day/night overpass h)
PLATFORMS = {
    "Terra": ("2000-11", "2026-11", "MODIS", 10.5),
    "Aqua": ("2002-07", "2026-11", "MODIS", 13.5),
    "SNPP": ("2012-01", "2026-11", "VIIRS", 13.5),
    "NOAA-20": ("2018-04", "2026-11", "VIIRS", 13.4),
    "NOAA-21": ("2023-03", "2026-11", "VIIRS", 13.6),
}
# expected detections per (MW of daily-mean FRP) per clear overpass-month
SENSITIVITY = {"MODIS": 0.035, "VIIRS": 0.14}
FRP_LOGNORM = {"MODIS": (np.log(16.0), 0.95), "VIIRS": (np.log(4.0), 0.95)}
OUTAGES = {("Terra", 2016, 2), ("Aqua", 2008, 3), ("SNPP", 2022, 8),
           ("NOAA-20", 2019, 4)}


def _months() -> pd.DataFrame:
    idx = pd.period_range(S.DATE_START[:7], S.DATE_END[:7], freq="M")
    return pd.DataFrame({"year": idx.year.astype(int),
                         "month": idx.month.astype(int),
                         "ndays": idx.days_in_month.astype(int)})


def _cells():
    import h3
    rows = []
    for r in REGIONS:
        center = h3.latlng_to_cell(r.lat, r.lon, S.H3_MAIN_RES)
        for c in sorted(h3.grid_disk(center, r.k)):
            lat, lon = h3.cell_to_latlng(c)
            rows.append({"h3_index": int(c, 16), "lat": lat, "lon": lon,
                         "country_iso": r.iso, "region": r.name})
    df = pd.DataFrame(rows).drop_duplicates("h3_index").reset_index(drop=True)
    reg = {r.name: r for r in REGIONS}
    df["kind"] = df["region"].map(lambda n: reg[n].kind)
    df["pyrome_id"] = df["region"].map(
        {r.name: i for i, r in enumerate(REGIONS)}).astype(int)
    return df, reg


def _truth(cells, reg, months, rng):
    """phi_true (daily-mean FRP, MW) per (cell, month) + cloud fraction."""
    n_c, n_m = len(cells), len(months)
    yr = months["year"].to_numpy()
    mo = months["month"].to_numpy()
    phi = np.zeros((n_c, n_m))
    cloud = np.zeros((n_c, n_m))
    for name, idx in cells.groupby("region").indices.items():
        r = reg[name]
        base = r.scale_mw * rng.lognormal(0.0, 0.6, size=len(idx))
        if r.kind == "flare":
            seas = np.ones(n_m)
        else:
            ang = 2 * np.pi * (mo - r.peak_month) / 12.0
            seas = np.exp(r.season_kappa * (np.cos(ang) - 1.0)) + 0.01
        yf = np.array([EVENTS.get(name, {}).get(y, 1.0) for y in yr])
        trend = np.exp(TRENDS.get(name, 0.0) * (yr - 2001) / 10.0)
        iav = np.exp(rng.normal(0, 0.25, size=27))[yr - 2000]
        if name == "Chilecito (La Rioja)":   # July-August 2026 wildfire
            yf = yf * np.where((yr == 2026) & np.isin(mo, (7, 8)), 60.0, 1.0)
        if name == "SE forests (NSW/Victoria)":   # Black Summer 2019/20
            bs = ((yr == 2019) & (mo == 12)) | ((yr == 2020) & (mo <= 2))
            yf = yf * np.where(bs, 25.0, 1.0)
        reg_curve = seas * yf * trend * iav
        noise = rng.lognormal(0.0, 0.45, size=(len(idx), n_m))
        if r.kind == "flare":
            noise = rng.lognormal(0.0, 0.08, size=(len(idx), n_m))
        phi[idx] = base[:, None] * reg_curve[None, :] * noise
        ang = 2 * np.pi * (mo - r.peak_month) / 12.0
        cl = r.cloud_base + 0.30 * (1 - np.cos(ang)) / 2.0  # wet season cloudier
        cloud[idx] = np.clip(cl[None, :] + rng.normal(0, 0.06, (len(idx), n_m)),
                             0.02, 0.95)
    return phi, cloud


def _platform_active(months):
    ym = (months["year"].astype(str) + "-"
          + months["month"].astype(str).str.zfill(2)).to_numpy()
    act = {}
    for p, (a, b, _fam, _h) in PLATFORMS.items():
        ok = (ym >= a) & (ym <= b)
        for (pp, y, m) in OUTAGES:
            if pp == p:
                ok &= ~((months["year"].to_numpy() == y)
                        & (months["month"].to_numpy() == m))
        act[p] = ok
    return act


def _detections(cells, reg, months, phi, cloud, act, rng):
    """Sample Product-1 style raw detections from the truth."""
    import h3
    yr = months["year"].to_numpy()
    mo = months["month"].to_numpy()
    nd = months["ndays"].to_numpy()
    out = []
    ev_id = 0
    for p, (_a, _b, fam, hday) in PLATFORMS.items():
        vis = 1.0 - cloud ** 1.3
        lam = SENSITIVITY[fam] * phi * nd[None, :] / 30.0 * vis
        # post-2022 orbital drift of Terra/Aqua: fewer detections at the
        # drifted (cooler/later) overpass times
        if fam == "MODIS":
            drift = np.clip((yr - 2022 + (mo - 1) / 12.0) / 4.0, 0, 1)
            lam = lam * (1.0 - 0.35 * drift)[None, :]
        lam = lam * act[p][None, :]
        n = rng.poisson(lam)
        ci, mi = np.nonzero(n)
        cnt = n[ci, mi]
        rep_c = np.repeat(ci, cnt)
        rep_m = np.repeat(mi, cnt)
        k = len(rep_c)
        if k == 0:
            continue
        day = 1 + (rng.random(k) * nd[rep_m]).astype(int)
        ndr = cells["region"].map(lambda r_: reg[r_].ndr).to_numpy()[rep_c]
        is_day = rng.random(k) < 1.0 / (1.0 + ndr)
        h = np.where(is_day, hday, (hday + 12.0) % 24.0)
        if fam == "MODIS":
            drift_h = np.clip((yr[rep_m] - 2022) / 4.0, 0, 1) * (
                -1.0 if p == "Terra" else 1.5)
            h = h + drift_h
        solar = (h + rng.normal(0, 0.35, k)) % 24.0
        lon = cells["lon"].to_numpy()[rep_c] + rng.normal(0, 0.035, k)
        lat = cells["lat"].to_numpy()[rep_c] + rng.normal(0, 0.035, k)
        utc = (solar - lon / 15.0) % 24.0
        hhmm = (np.floor(utc) * 100 + np.floor((utc % 1) * 60)).astype(int)
        vza = rng.uniform(0, 65 if fam == "MODIS" else 68, k)
        cosv = np.cos(np.radians(vza))
        area = (1.0 / cosv ** 3) if fam == "MODIS" else (0.14 / cosv ** 2)
        mu, sd = FRP_LOGNORM[fam]
        frp = rng.lognormal(mu, sd, k) * np.where(is_day, 1.0, 0.6)
        flare = (cells["kind"].to_numpy()[rep_c] == "flare")
        sama = ((fam == "MODIS") & ~is_day
                & np.isin(cells["country_iso"].to_numpy()[rep_c], ["ARG", "BRA"])
                & (rng.random(k) < 0.02))
        type_raw = np.where(flare, 2, 0)
        if fam == "MODIS":
            conf_num = np.clip(rng.normal(72, 15, k), 0, 100).round()
            conf_raw = conf_num.astype(int).astype(str)
        else:
            cl = rng.choice(["l", "n", "h"], size=k, p=[0.12, 0.70, 0.18])
            conf_raw = cl
            conf_num = np.select([cl == "l", cl == "n"], [30.0, 80.0], 100.0)
        conf_num = np.where(sama, 25.0, conf_num)
        bow = np.ones(k)
        if fam == "MODIS":
            bow = np.where(vza > 40, 0.5, 1.0)
        # events: detections of the same cell/day/platform/overpass cluster
        dates = pd.to_datetime(dict(year=yr[rep_m], month=mo[rep_m], day=day))
        df = pd.DataFrame({
            "platform": p, "sensor_family": fam,
            "data_tier": np.where((yr[rep_m] == 2026) & (mo[rep_m] >= 8),
                                  "NRT", "ML_SCIENCE"),
            "acq_date": dates.dt.strftime("%Y-%m-%d"),
            "acq_hhmm": hhmm, "frp_mw": frp.round(2),
            "t_mir_k": rng.uniform(320, 360, k).round(1),
            "t_tir_k": rng.uniform(288, 305, k).round(1),
            "scan_km": np.sqrt(area) * 1.1, "track_km": np.sqrt(area) * 0.9,
            "conf_raw": conf_raw, "type_raw": type_raw,
            "ingest_ts": "SYNTHETIC",
            "latitude": lat.round(4), "longitude": lon.round(4),
            "country_iso": cells["country_iso"].to_numpy()[rep_c],
            "conf_num": conf_num,
            "version_raw": "C6.1" if fam == "MODIS" else "C2",
            "dn_source": np.where(is_day, "D", "N"),
            "h3_index": cells["h3_index"].to_numpy()[rep_c],
            "sza_deg": np.where(is_day, rng.uniform(20, 70, k),
                                rng.uniform(100, 150, k)).round(1),
            "solar_hour": solar.round(2),
            "solar_date": dates.values.astype("datetime64[ns]"),
            "dn_flag": np.where((solar >= 6) & (solar < 18), "D", "N"),
            "pixel_area_km2": area.round(3), "vza_deg": vza.round(1),
            "is_static_industrial": flare,
            "is_sama_filtered": sama,
            "bowtie_weight": bow,
            "reliability_prob": np.clip(rng.normal(0.88, 0.08, k), 0.3, 1.0
                                        ).round(3),
            "event_cluster_id": 0, "pass_id": 0,
            "i4_m13_fallback": (fam == "VIIRS") & (frp > 60.0),
            "sat_flag": False,
            "is_biomass": ~(flare | sama),
            "forward_gain_c": (0.0 if fam == "MODIS" else -0.35)
            + rng.normal(0, 0.05, k).round(3),
            "obs_sigma": np.where(fam == "MODIS", 0.45, 0.32)
            + rng.normal(0, 0.03, k).round(3),
            "_m": rep_m, "_c": rep_c,
        })
        # bow-tie duplicates: MODIS off-nadir pixels detected twice
        if fam == "MODIS":
            dup = df[df["bowtie_weight"] < 1.0].copy()
            dup["latitude"] = (dup["latitude"] + 0.008).round(4)
            df = pd.concat([df, dup], ignore_index=True)
        grp = df.groupby(["_c", "solar_date", "dn_flag"]).ngroup().to_numpy()
        df["pass_id"] = grp + ev_id
        # pixels of one pass cluster into fire events (a VIIRS 375 m fire
        # front spans ~3 pixels, a MODIS 1 km fire ~1.5)
        rank = df.groupby("pass_id").cumcount().to_numpy()
        per_event = 3 if fam == "VIIRS" else 2
        df["event_cluster_id"] = (df["pass_id"].to_numpy() * 1000
                                  + rank // per_event)
        ev_id = int(df["pass_id"].max()) + 1
        out.append(df)
    return pd.concat(out, ignore_index=True)


def build_synthetic_products(seed: int = 20261115) -> dict:
    """Return the engine-schema products + diagnostics + truth tables."""
    from ...engine.conformal import climatology_indices
    from ...products import export_duckdb as ex
    from ...validation.diagnostics import run_diagnostics
    t0 = time.time()
    rng = np.random.default_rng(seed)
    months = _months()
    cells, reg = _cells()
    phi, cloud = _truth(cells, reg, months, rng)
    act = _platform_active(months)
    det = _detections(cells, reg, months, phi, cloud, act, rng)

    n_c, n_m = phi.shape
    ci = np.repeat(np.arange(n_c), n_m)
    mi = np.tile(np.arange(n_m), n_c)
    yr = months["year"].to_numpy()[mi]
    mo = months["month"].to_numpy()[mi]
    nd = months["ndays"].to_numpy()[mi]
    truth = phi.ravel()
    cf = cloud.ravel()
    measured = np.clip(1.0 - cf ** 1.3, 0.05, 1.0)
    # harmonized posterior: truth + calibrated noise; wider when less observed
    s = 0.10 + 0.40 * (1.0 - measured)
    q50 = truth * np.exp(rng.normal(0.0, s))
    z = 1.645 * s * 1.05
    q05, q95 = q50 * np.exp(-z), q50 * np.exp(z)
    ndr = cells["region"].map(lambda r_: reg[r_].ndr).to_numpy()[ci]
    day = 2.0 * q50 / (1.0 + ndr)
    night = 2.0 * q50 * ndr / (1.0 + ndr)
    m = pd.DataFrame({
        "h3_index": cells["h3_index"].to_numpy()[ci],
        "country_iso": cells["country_iso"].to_numpy()[ci],
        "pyrome_id": cells["pyrome_id"].to_numpy()[ci],
        "year": yr, "month": mo,
        "phi_day_mw": day, "phi_night_mw": night,
        "phi_q05_mw": q05, "phi_q50_mw": q50, "phi_q95_mw": q95,
        "phi_mean_mw": q50 * np.exp(0.5 * s ** 2),
        "_ci": ci, "_mi": mi, "_truth": truth, "_cloud": cf,
        "_measured": measured, "_nd": nd,
    })
    bio = det[det["is_biomass"]]
    agg = bio.groupby(["_c", "_m"]).agg(
        raw_count_sum=("frp_mw", "size"), raw_frp_sum_mw=("frp_mw", "sum"),
        raw_clustered_events=("event_cluster_id", "nunique")).reset_index()
    m = m.merge(agg, left_on=["_ci", "_mi"], right_on=["_c", "_m"],
                how="left").drop(columns=["_c", "_m"])
    for c in ("raw_count_sum", "raw_frp_sum_mw", "raw_clustered_events"):
        m[c] = m[c].fillna(0.0)
    m = m[cells["kind"].to_numpy()[m["_ci"]] == "fire"].reset_index(drop=True)

    hex_area = 36.13
    m["phi_flux_mw_km2"] = m["phi_q50_mw"] / hex_area
    m["fre_tj_q50"] = S.FRE_TJ_PER_MW_DAY * m["phi_q50_mw"] * m["_nd"]
    m["energy_equiv_events_q50"] = m["phi_q50_mw"] / 8.0
    m["hurdle_adj_events"] = np.maximum(m["raw_clustered_events"],
                                        m["energy_equiv_events_q50"] * 0.9)
    m["night_day_ratio"] = m["phi_night_mw"] / m["phi_day_mw"].clip(lower=1e-6)
    m["est_active_area_km2"] = np.minimum(0.04 * m["phi_q50_mw"] ** 0.85,
                                          hex_area)
    m = climatology_indices(m)

    # Product 3: evidence, flags (same decision tree as the engine)
    bits = np.zeros(len(m), dtype=int)
    for p, pi in S.PLATFORM_INDEX.items():
        bits |= np.where(act[p][m["_mi"].to_numpy()], S.PLATFORM_BITMASK[pi], 0)
    m["active_platforms_bitmask"] = bits
    m["mean_cloud_frac"] = m["_cloud"]
    m["measured_fraction"] = m["_measured"]
    m["inferred_fraction"] = 1.0 - m["_measured"]
    m["conformal_c_factor"] = 1.05
    m["rel_uncertainty_width"] = ((m["phi_q95_mw"] - m["phi_q05_mw"])
                                  / m["phi_q50_mw"].clip(lower=1e-6))
    abstain = (m["_measured"] < 0.30) & (m["raw_count_sum"] == 0) & (
        m["_truth"] > 0.5)
    clim_cloud = m["year"] >= 2019     # mirrors the real archive's cloud gap
    m["quality_flag"] = np.where(abstain, "ABSTAIN_LOW_EVIDENCE",
                                 np.where(clim_cloud,
                                          "CLIMATOLOGY_CLOUD_FALLBACK",
                                          "NOMINAL"))
    clim = m[(m["year"] >= S.CLIM_START_YEAR) & (m["year"] <= S.CLIM_END_YEAR)]
    raw_med = clim.groupby(["h3_index", "month"])["raw_frp_sum_mw"].median() \
        .rename("raw_hist_med").reset_index()
    m = m.merge(raw_med, on=["h3_index", "month"], how="left")
    rhm = m["raw_hist_med"].fillna(0.0)
    raw_delta = (m["raw_frp_sum_mw"] - rhm) / np.maximum(rhm, 1.0)
    hhm = m["hist_median_2003_2020"]
    real = (m["phi_q05_mw"] > hhm) | (m["phi_q95_mw"] < hhm)
    artifact = (raw_delta.abs() > 0.50) & (m["phi_q05_mw"] <= hhm) & (
        hhm <= m["phi_q95_mw"])
    m["attribution_flag"] = np.where(
        m["quality_flag"] == "ABSTAIN_LOW_EVIDENCE", "AMBIGUOUS_OVERLAPPING_CI",
        np.where(real, "REAL_FIRE_CHANGE",
                 np.where(artifact, "SENSOR_ARTIFACT_CORRECTED",
                          "WITHIN_NORMAL_VARIABILITY")))
    m["has_nrt_gapfill"] = ((m["year"] == 2026) & (m["month"] >= 8)).astype(int)
    m["revision_count"] = 1

    truth_df = m[["h3_index", "year", "month"]].assign(
        phi24_true=m["_truth"].to_numpy())
    # VIIRS-masked 2019-2021 ablation fold: a MODIS-only reconstruction is
    # less constrained -> wider, noisier intervals around the truth
    f = m[(m["year"] >= 2019) & (m["year"] <= 2021)][
        ["h3_index", "year", "month", "_truth", "_measured"]].copy()
    sf = 0.22 + 0.55 * (1.0 - f["_measured"].to_numpy())
    fq50 = f["_truth"].to_numpy() * np.exp(rng.normal(0, sf))
    fold = pd.DataFrame({"h3_index": f["h3_index"], "year": f["year"],
                         "month": f["month"], "phi_q50_mw": fq50,
                         "phi_q05_mw": fq50 * np.exp(-1.645 * sf),
                         "phi_q95_mw": fq50 * np.exp(1.645 * sf)})
    pass_obs = (det[det["is_biomass"]].groupby(
        ["h3_index", "solar_date", "platform"], as_index=False)["frp_mw"].sum()
        .rename(columns={"frp_mw": "y_frp_mw"}))
    lat_c = cells.set_index("h3_index")["lat"]
    lon_c = cells.set_index("h3_index")["lon"]
    from ...engine.priors import cmg_bin_centers
    blat, blon = cmg_bin_centers(lat_c.reindex(m["h3_index"]).to_numpy(),
                                 lon_c.reindex(m["h3_index"]).to_numpy())
    burned = pd.DataFrame({
        "lat_025": blat, "lon_025": blon, "year": m["year"], "month": m["month"],
        "burned_area_km2": 0.08 * m["fre_tj_q50"].to_numpy() ** 0.9
        * rng.lognormal(0, 0.6, len(m))})
    diag = run_diagnostics(m.drop(columns=[c for c in m.columns
                                           if c.startswith("_")]),
                           truth=truth_df, burned_area=burned,
                           pass_obs=pass_obs, ablation_fold=fold)
    diag["coverage_audit"] = {
        "n_rows": int(len(det)),
        "platforms_present": sorted(det["platform"].unique().tolist()),
        "dza_type2_flares_preserved": int((det["type_raw"] == 2).sum()),
        "note": "synthetic"}

    monthly = m.drop(columns=[c for c in m.columns if c.startswith("_")]
                     + ["raw_hist_med"])
    p1 = det.drop(columns=["_m", "_c"])
    p1 = p1[[c for c in ex.PRODUCT1_COLUMNS if c in p1.columns]]
    daily = (p1[p1["is_biomass"]].groupby(["h3_index", "solar_date"],
                                          as_index=False)
             .agg(n=("frp_mw", "size"), frp=("frp_mw", "sum")))
    dd = pd.to_datetime(daily["solar_date"])
    daily_df = pd.DataFrame({
        "h3_index": daily["h3_index"],
        "t": (dd - pd.Timestamp(S.DATE_START)).dt.days,
        "phi_day_mw": daily["frp"] * 0.9, "phi_night_mw": daily["frp"] * 0.2,
        "phi_q05_mw": daily["frp"] * 0.4, "phi_q50_mw": daily["frp"] * 0.55,
        "phi_q95_mw": daily["frp"] * 0.9, "phi_mean_mw": daily["frp"] * 0.58,
    })
    daily_df["fre_tj_q50"] = S.FRE_TJ_PER_MW_DAY * daily_df["phi_q50_mw"]
    daily_df["solar_date"] = daily["solar_date"]
    daily_df["year"] = dd.dt.year
    daily_df["month"] = dd.dt.month
    products = {"product1": p1,
                "product2_monthly": monthly[[c for c in ex.PRODUCT2_MONTHLY_COLUMNS
                                             if c in monthly.columns]],
                "product2_daily": daily_df,
                "product3": monthly[[c for c in ex.PRODUCT3_COLUMNS
                                     if c in monthly.columns]]}
    return {"products": products, "diagnostics": diag, "truth": truth_df,
            "regions": cells[["h3_index", "region", "country_iso", "kind"]],
            "seconds": round(time.time() - t0, 1),
            "n_detections": int(len(p1)), "n_cells": int(monthly["h3_index"]
                                                         .nunique())}


def build_synthetic_store(db_path, seed: int = 20261115) -> dict:
    """Write the synthetic dataset with the ENGINE'S exporter + provenance."""
    import duckdb

    from ...products import export_duckdb as ex
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out = build_synthetic_products(seed)
    pq_dir = db_path.with_suffix("")
    tmp_db = db_path.with_name(db_path.name + ".part")
    tmp_db.unlink(missing_ok=True)
    ex.export_duckdb(out["products"], tmp_db, parquet_dir=str(pq_dir),
                     diagnostics=out["diagnostics"])
    con = duckdb.connect(str(tmp_db))
    try:
        con.register("_truth", out["truth"])
        con.execute("CREATE TABLE synthetic_truth AS SELECT * FROM _truth")
        con.register("_regions", out["regions"])
        con.execute("CREATE TABLE synthetic_regions AS SELECT * FROM _regions")
        meta = {"source": "SYNTHETIC_DEMO", "label": PROVENANCE_LABEL,
                "generator": GENERATOR_VERSION, "seed": seed,
                "created_utc": pd.Timestamp.utcnow().isoformat(),
                "n_cells": out["n_cells"], "n_detections": out["n_detections"],
                "build_seconds": out["seconds"]}
        con.execute("CREATE TABLE dataset_provenance (key VARCHAR, value VARCHAR)")
        con.executemany("INSERT INTO dataset_provenance VALUES (?, ?)",
                        [(k, json.dumps(v)) for k, v in meta.items()])
    finally:
        con.close()
    tmp_db.replace(db_path)
    return meta
