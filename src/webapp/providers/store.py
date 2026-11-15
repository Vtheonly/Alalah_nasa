"""Query layer over the harmonization engine's export schema.

``HarmonizedStore`` reads ``fire_harmonized.duckdb`` as written by
``src.products.export_duckdb`` -- tables ``state_reconstructed_monthly``
(Product 2), ``uncertainty_diagnostics`` (Product 3),
``obs_harmonized_provenance`` (Product 1, one row per satellite detection),
``state_reconstructed_daily``, ``hex_centers`` and ``diagnostics``.  The SAME
class serves the real engine output and the synthetic demonstration store
(which is written by the same exporter), so every endpoint, chart and number
is computed identically for both; only the provenance differs.
"""
from __future__ import annotations

import functools
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ...config import settings as S
from ...validation.diagnostics import chow_test

REQUIRED_TABLES = ("state_reconstructed_monthly", "uncertainty_diagnostics",
                   "hex_centers")
FULL_YEARS = (2001, 2025)      # complete calendar years for annual statistics
COUNTRY_NAMES = {c.iso3: c.name for c in S.TARGET_COUNTRIES}
PLATFORM_ORDER = ["Terra", "Aqua", "SNPP", "NOAA-20", "NOAA-21"]
S_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
            "Oct", "Nov", "Dec"]


def _f(v, nd=4):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _iso_clause(iso, alias=""):
    col = f"{alias}country_iso" if alias else "country_iso"
    return (f" AND {col} = ?", [iso]) if iso else ("", [])


class HarmonizedStore:
    def __init__(self, path, source: str, label: str):
        import duckdb
        self.path = Path(path)
        self.source = source            # "REAL_ENGINE" | "SYNTHETIC_DEMO"
        self.label = label
        self._con = duckdb.connect(str(self.path), read_only=True)
        self._lock = threading.Lock()
        self.tables = {r[0] for r in self._rows("SHOW TABLES")}
        missing = [t for t in REQUIRED_TABLES if t not in self.tables]
        if missing:
            raise ValueError(f"{self.path.name} lacks tables {missing}")
        self.has_product1 = "obs_harmonized_provenance" in self.tables
        self.is_synthetic = "synthetic_truth" in self.tables

    # ------------------------------------------------------------ plumbing
    def _rows(self, sql, params=()):
        cur = self._con.cursor()
        try:
            return cur.execute(sql, list(params)).fetchall()
        finally:
            cur.close()

    def _df(self, sql, params=()) -> pd.DataFrame:
        cur = self._con.cursor()
        try:
            return cur.execute(sql, list(params)).df()
        finally:
            cur.close()

    def close(self):
        self._con.close()

    @functools.lru_cache(maxsize=1)
    def dataset_provenance(self) -> dict:
        if "dataset_provenance" not in self.tables:
            return {}
        return {k: json.loads(v) for k, v in
                self._rows("SELECT key, value FROM dataset_provenance")}

    @functools.lru_cache(maxsize=1)
    def diagnostics(self) -> dict:
        if "diagnostics" not in self.tables:
            return {}
        row = self._rows("SELECT diagnostics_json FROM diagnostics LIMIT 1")
        return json.loads(row[0][0]) if row and row[0][0] else {}

    # ------------------------------------------------------------ overview
    @functools.lru_cache(maxsize=1)
    def overview(self) -> dict:
        n_cells, y0, y1 = self._rows(
            "SELECT COUNT(DISTINCT h3_index), MIN(year), MAX(year) "
            "FROM state_reconstructed_monthly")[0]
        countries = self._df(f"""
            SELECT country_iso, COUNT(DISTINCT h3_index) AS n_cells,
                   SUM(fre_tj_q50) / {FULL_YEARS[1] - FULL_YEARS[0] + 1}
                       AS fre_tj_per_year
            FROM state_reconstructed_monthly
            WHERE year BETWEEN {FULL_YEARS[0]} AND {FULL_YEARS[1]}
            GROUP BY 1 ORDER BY 1""")
        out = {"n_cells": int(n_cells or 0), "first_year": y0, "last_year": y1,
               "countries": [{"iso": r.country_iso,
                              "name": COUNTRY_NAMES.get(r.country_iso,
                                                        r.country_iso),
                              "n_cells": int(r.n_cells),
                              "fre_tj_per_year": _f(r.fre_tj_per_year, 1)}
                             for r in countries.itertuples()],
               "n_detections": None, "platforms": []}
        if self.has_product1:
            p = self._df("""
                SELECT platform, sensor_family, COUNT(*) AS n,
                       MIN(acq_date) AS first, MAX(acq_date) AS last
                FROM obs_harmonized_provenance GROUP BY 1, 2""")
            p["o"] = p["platform"].map({k: i for i, k in
                                        enumerate(PLATFORM_ORDER)}).fillna(9)
            out["platforms"] = [{"platform": r.platform, "sensor": r.sensor_family,
                                 "n_detections": int(r.n),
                                 "first": str(r.first)[:10],
                                 "last": str(r.last)[:10]}
                                for r in p.sort_values("o").itertuples()]
            out["n_detections"] = int(p["n"].sum())
        return out

    def regions(self) -> list[dict]:
        cent = self._df("""SELECT country_iso, AVG(latitude) lat,
                                  AVG(longitude) lon, COUNT(*) n
                           FROM hex_centers GROUP BY 1""").set_index(
            "country_iso")
        out = []
        for c in S.TARGET_COUNTRIES:
            boxes = np.array(c.prefilter_boxes)
            bbox = [float(boxes[:, 0].min()), float(boxes[:, 1].min()),
                    float(boxes[:, 2].max()), float(boxes[:, 3].max())]
            if c.iso3 in cent.index:
                center = [float(cent.loc[c.iso3, "lon"]),
                          float(cent.loc[c.iso3, "lat"])]
                n = int(cent.loc[c.iso3, "n"])
            else:
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                n = 0
            out.append({"iso": c.iso3, "name": c.name, "bbox": bbox,
                        "center": center, "n_cells": n,
                        "spotlights": [{"key": s.key, "label": s.label,
                                        "bbox": list(s.bbox)}
                                       for s in S.SPOTLIGHT_PRESETS
                                       if s.country_iso == c.iso3]})
        return out

    # ------------------------------------------------- calendar / series / map
    def calendar(self, min_lon, min_lat, max_lon, max_lat, y0, y1, metric,
                 iso):
        cell_sql = ("SELECT h3_index FROM hex_centers WHERE longitude BETWEEN "
                    "? AND ? AND latitude BETWEEN ? AND ?")
        params = [min_lon, max_lon, min_lat, max_lat]
        if iso:
            cell_sql += " AND country_iso = ?"
            params.append(iso)
        q = f"""
            SELECT m.year, m.month, COUNT(*) AS n_cells,
                   AVG(m.{metric}) AS value, AVG(m.phi_q05_mw) AS q05,
                   AVG(m.phi_q95_mw) AS q95,
                   AVG(m.hist_median_2003_2020) AS hist_median,
                   AVG(m.anomaly_ratio) AS anomaly_ratio,
                   SUM(m.raw_count_sum) AS raw_count,
                   SUM(m.raw_frp_sum_mw) AS raw_frp,
                   AVG(u.measured_fraction) AS measured_fraction,
                   CASE WHEN BOOL_OR(u.quality_flag = 'ABSTAIN_LOW_EVIDENCE')
                        THEN 'ABSTAIN_LOW_EVIDENCE'
                        WHEN BOOL_OR(u.quality_flag = 'CLIMATOLOGY_CLOUD_FALLBACK')
                        THEN 'CLIMATOLOGY_CLOUD_FALLBACK' ELSE 'NOMINAL' END
                        AS quality_flag,
                   MODE(u.attribution_flag) AS attribution_flag,
                   SUM(m.fre_tj_q50) AS fre_tj
            FROM state_reconstructed_monthly m
            JOIN uncertainty_diagnostics u
              ON m.h3_index = u.h3_index AND m.year = u.year
             AND m.month = u.month
            WHERE m.h3_index IN ({cell_sql}) AND m.year BETWEEN ? AND ?
            GROUP BY m.year, m.month ORDER BY m.year, m.month"""
        return self._rows(q, params + [y0, y1])

    def map_hex(self, min_lon, min_lat, max_lon, max_lat, year, month,
                color_by, iso, limit):
        cell_sql = ("SELECT h3_index FROM hex_centers WHERE longitude BETWEEN "
                    "? AND ? AND latitude BETWEEN ? AND ?")
        params = [min_lon, max_lon, min_lat, max_lat]
        if iso:
            cell_sql += " AND country_iso = ?"
            params.append(iso)
        q = f"""
            SELECT m.h3_index, m.phi_q50_mw, m.{color_by} AS color_value,
                   m.phi_q95_mw - m.phi_q05_mw AS width, u.quality_flag,
                   u.measured_fraction, m.raw_count_sum, u.attribution_flag,
                   m.anomaly_ratio, m.fre_tj_q50
            FROM state_reconstructed_monthly m
            JOIN uncertainty_diagnostics u
              ON m.h3_index = u.h3_index AND m.year = u.year
             AND m.month = u.month
            WHERE m.h3_index IN ({cell_sql}) AND m.year = ? AND m.month = ?
            ORDER BY m.phi_q50_mw DESC LIMIT {int(limit)}"""
        return self._rows(q, params + [year, month])

    # ------------------------------------------------------------ helpers
    @functools.lru_cache(maxsize=32)
    def yearly(self, iso=None) -> pd.DataFrame:
        cl, p = _iso_clause(iso)
        return self._df(f"""
            SELECT year, SUM(raw_count_sum) AS raw_count,
                   SUM(raw_frp_sum_mw) AS raw_frp, SUM(fre_tj_q50) AS fre,
                   SUM(fre_tj_q50 * phi_q05_mw / GREATEST(phi_q50_mw, 1e-9))
                       AS fre_q05,
                   SUM(fre_tj_q50 * phi_q95_mw / GREATEST(phi_q50_mw, 1e-9))
                       AS fre_q95
            FROM state_reconstructed_monthly
            WHERE year BETWEEN {FULL_YEARS[0]} AND {FULL_YEARS[1]} {cl}
            GROUP BY 1 ORDER BY 1""", p)

    @functools.lru_cache(maxsize=32)
    def monthly_total(self, iso=None) -> pd.DataFrame:
        cl, p = _iso_clause(iso)
        return self._df(f"""
            SELECT year, month, SUM(fre_tj_q50) AS fre,
                   SUM(raw_count_sum) AS raw_count,
                   SUM(phi_day_mw) AS phi_day, SUM(phi_night_mw) AS phi_night
            FROM state_reconstructed_monthly WHERE 1=1 {cl}
            GROUP BY 1, 2 ORDER BY 1, 2""", p)

    def _p1_where(self, iso, extra=""):
        cl, p = _iso_clause(iso)
        return f"WHERE 1=1 {cl} {extra}", p

    @staticmethod
    def step_change(yearly: pd.DataFrame, col: str) -> float | None:
        a = yearly[yearly["year"].between(2009, 2011)][col].mean()
        b = yearly[yearly["year"].between(2012, 2014)][col].mean()
        return _f(b / a - 1.0) if a and np.isfinite(a) and a > 0 else None

    @staticmethod
    def chow(yearly: pd.DataFrame, col: str) -> dict:
        y = np.log1p(yearly[col].to_numpy(dtype=float))
        idx = int(np.searchsorted(yearly["year"].to_numpy(), 2012))
        if len(y) < 6 or idx < 3 or idx > len(y) - 3:
            return {"F": None, "p": None}
        r = chow_test(y, idx)
        return {"F": _f(r.get("F")), "p": _f(r.get("p"), 6)}

    # ------------------------------------------------------------- challenges
    def challenge_sensor_transition(self, iso):
        y = self.yearly(iso)
        base = y[y["year"].between(2003, 2011)]
        idx = lambda c: (100.0 * y[c] / max(base[c].mean(), 1e-9)).round(1)  # noqa: E731
        raw_step, harm_step = (self.step_change(y, "raw_count"),
                               self.step_change(y, "fre"))
        chow_raw, chow_harm = self.chow(y, "raw_count"), self.chow(y, "fre")
        return {
            "kpis": [
                {"label": "Raw detections, 2009-11 -> 2012-14", "value": raw_step,
                 "format": "pct", "tone": "bad"},
                {"label": "Harmonized energy, same period", "value": harm_step,
                 "format": "pct", "tone": "good"},
                {"label": "Chow test p (raw) at 2012", "value": chow_raw["p"],
                 "format": "p"},
                {"label": "Chow test p (harmonized)", "value": chow_harm["p"],
                 "format": "p"}],
            "charts": [{
                "kind": "line", "title": "Annual signal indexed to 2003-2011 = 100",
                "x": y["year"].tolist(), "yLabel": "index",
                "series": [
                    {"name": "Raw detection count", "role": "raw",
                     "data": idx("raw_count").tolist()},
                    {"name": "Raw FRP sum", "role": "raw2",
                     "data": idx("raw_frp").tolist()},
                    {"name": "Harmonized FRE", "role": "harmonized",
                     "data": idx("fre").tolist()}],
                "markers": [{"x": 2012, "label": "VIIRS (S-NPP) enters"}]}]}

    def challenge_resolution(self, iso):
        if not self.has_product1:
            return {"kpis": [], "charts": []}
        w, p = self._p1_where(iso, "AND is_biomass")
        h = self._df(f"""
            SELECT sensor_family, FLOOR(LOG10(GREATEST(frp_mw, 0.1)) * 5) / 5 AS b,
                   COUNT(*) AS n
            FROM obs_harmonized_provenance {w} GROUP BY 1, 2 ORDER BY 2""", p)
        s = self._df(f"""
            SELECT sensor_family, COUNT(*) AS n, MEDIAN(frp_mw) AS med_frp,
                   AVG(pixel_area_km2) AS area,
                   AVG(CASE WHEN frp_mw < 10 THEN 1.0 ELSE 0.0 END) AS small,
                   AVG(forward_gain_c) AS gain
            FROM obs_harmonized_provenance {w} GROUP BY 1""", p
                     ).set_index("sensor_family")
        bins = sorted(h["b"].unique())
        series = []
        for fam, role in (("MODIS", "raw"), ("VIIRS", "raw2")):
            sub = h[h["sensor_family"] == fam].set_index("b")["n"]
            tot = max(sub.sum(), 1)
            series.append({"name": f"{fam} detections", "role": role,
                           "data": [round(100 * float(sub.get(b, 0)) / tot, 2)
                                    for b in bins]})
        get = lambda fam, col: _f(s.loc[fam, col]) if fam in s.index else None  # noqa: E731
        return {
            "kpis": [
                {"label": "MODIS median FRP / detection", "value": get("MODIS", "med_frp"), "format": "mw"},
                {"label": "VIIRS median FRP / detection", "value": get("VIIRS", "med_frp"), "format": "mw"},
                {"label": "MODIS mean pixel area", "value": get("MODIS", "area"), "format": "km2"},
                {"label": "VIIRS mean pixel area", "value": get("VIIRS", "area"), "format": "km2"},
                {"label": "Small fires (<10 MW), MODIS", "value": get("MODIS", "small"), "format": "pct0"},
                {"label": "Small fires (<10 MW), VIIRS", "value": get("VIIRS", "small"), "format": "pct0"}],
            "charts": [{
                "kind": "bar", "title": "Share of detections by FRP (log scale)",
                "x": [f"{10 ** b:.3g}" for b in bins], "xLabel": "FRP per detection (MW)",
                "yLabel": "% of sensor's detections", "series": series}]}

    def challenge_overpass(self, iso):
        out = {"kpis": [], "charts": []}
        if self.has_product1:
            w, p = self._p1_where(iso)
            h = self._df(f"""
                SELECT platform, FLOOR(solar_hour) AS hr, COUNT(*) AS n
                FROM obs_harmonized_provenance {w} GROUP BY 1, 2""", p)
            hours = list(range(24))
            series = []
            for pl in PLATFORM_ORDER:
                sub = h[h["platform"] == pl].set_index("hr")["n"]
                if len(sub):
                    series.append({"name": pl, "role": f"platform:{pl}",
                                   "data": [int(sub.get(float(x), 0)) for x in hours],
                                   "stack": "obs"})
            out["charts"].append({
                "kind": "bar", "title": "When each satellite observes (local solar hour)",
                "x": [f"{x:02d}h" for x in hours], "yLabel": "detections",
                "series": series})
            d = self._df(f"""
                SELECT platform, YEAR(CAST(acq_date AS DATE)) AS year,
                       AVG(solar_hour) AS h
                FROM obs_harmonized_provenance {w}
                  AND dn_flag = 'D' AND sensor_family = 'MODIS'
                  AND YEAR(CAST(acq_date AS DATE)) >= 2012
                GROUP BY 1, 2 ORDER BY 2""", p)
            yrs = sorted(d["year"].unique().tolist())
            out["charts"].append({
                "kind": "line", "title": "MODIS daytime overpass drift (mean local hour)",
                "x": yrs, "yLabel": "local solar hour",
                "series": [{"name": pl, "role": f"platform:{pl}",
                            "data": [_f(v, 2) for v in d[d["platform"] == pl]
                                     .set_index("year")["h"].reindex(yrs)]}
                           for pl in ("Terra", "Aqua")],
                "markers": [{"x": 2022, "label": "orbit drift begins"}]})
            t = d[(d["platform"] == "Terra")]
            if len(t):
                out["kpis"].append({"label": "Terra daytime overpass, latest year",
                                    "value": _f(t.iloc[-1]["h"], 2), "format": "hour"})
            a = d[(d["platform"] == "Aqua")]
            if len(a):
                out["kpis"].append({"label": "Aqua daytime overpass, latest year",
                                    "value": _f(a.iloc[-1]["h"], 2), "format": "hour"})
        mt = self.monthly_total(iso)
        ndr = float(mt["phi_night"].sum() / max(mt["phi_day"].sum(), 1e-9))
        out["kpis"].append({"label": "Night / day FRP ratio (harmonized)",
                            "value": _f(ndr, 3), "format": "ratio"})
        return out

    def challenge_missing(self, iso):
        cl, p = _iso_clause(iso, "m.")
        b = self._df(f"""
            SELECT LEAST(FLOOR(u.measured_fraction * 10), 9) / 10 AS b,
                   MEDIAN(u.rel_uncertainty_width) AS w, COUNT(*) AS n
            FROM uncertainty_diagnostics u JOIN state_reconstructed_monthly m
              ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
            WHERE m.phi_q50_mw > 0.5 {cl} GROUP BY 1 ORDER BY 1""", p)
        f = self._df(f"""
            SELECT m.year, u.quality_flag, COUNT(*) AS n
            FROM uncertainty_diagnostics u JOIN state_reconstructed_monthly m
              ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
            WHERE m.year BETWEEN {FULL_YEARS[0]} AND 2026 {cl} GROUP BY 1, 2""", p)
        tot = self._rows(f"""
            SELECT AVG(u.measured_fraction), AVG(u.mean_cloud_frac),
                   AVG(CASE WHEN u.quality_flag='ABSTAIN_LOW_EVIDENCE' THEN 1.0 ELSE 0 END),
                   AVG(CASE WHEN u.quality_flag='CLIMATOLOGY_CLOUD_FALLBACK' THEN 1.0 ELSE 0 END)
            FROM uncertainty_diagnostics u JOIN state_reconstructed_monthly m
              ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
            WHERE 1=1 {cl}""", p)[0]
        yrs = sorted(f["year"].unique().tolist())
        piv = f.pivot_table(index="year", columns="quality_flag", values="n",
                            aggfunc="sum").reindex(yrs).fillna(0)
        share = piv.div(piv.sum(axis=1).clip(lower=1), axis=0) * 100
        flag_series = [{"name": fl.replace("_", " ").title(), "role": f"flag:{fl}",
                        "data": share[fl].round(2).tolist() if fl in share else
                        [0] * len(yrs), "stack": "q"}
                       for fl in ("NOMINAL", "CLIMATOLOGY_CLOUD_FALLBACK",
                                  "ABSTAIN_LOW_EVIDENCE")]
        return {
            "kpis": [
                {"label": "Mean share of days actually observed", "value": _f(tot[0]), "format": "pct0"},
                {"label": "Mean cloud fraction", "value": _f(tot[1]), "format": "pct0"},
                {"label": "Cell-months abstained (too little evidence)", "value": _f(tot[2]), "format": "pct1"},
                {"label": "Cell-months on climatological cloud", "value": _f(tot[3]), "format": "pct1"}],
            "charts": [
                {"kind": "line", "title": "Less observed -> wider (honest) uncertainty",
                 "x": [f"{v:.0%}-{v + 0.1:.0%}" for v in b["b"]],
                 "xLabel": "share of days with a clear observation",
                 "yLabel": "relative 90% interval width",
                 "series": [{"name": "Median interval width", "role": "harmonized",
                             "data": [_f(v, 3) for v in b["w"]]}]},
                {"kind": "bar", "title": "Evidence quality of every cell-month",
                 "x": yrs, "yLabel": "% of cell-months", "series": flag_series}]}

    def challenge_constellation(self, iso):
        out = {"kpis": [], "charts": []}
        mt = self.monthly_total(iso)
        x = [f"{y}-{m:02d}" for y, m in zip(mt["year"], mt["month"])]
        markers = [{"x": mo, "label": lbl} for mo, lbl in S.CONSTELLATION_EVENTS]
        if self.has_product1:
            w, p = self._p1_where(iso, "AND is_biomass")
            d = self._df(f"""
                SELECT platform, strftime(CAST(acq_date AS DATE), '%Y-%m') AS ym,
                       COUNT(*) AS n
                FROM obs_harmonized_provenance {w} GROUP BY 1, 2""", p)
            series = []
            for pl in PLATFORM_ORDER:
                sub = d[d["platform"] == pl].set_index("ym")["n"]
                if len(sub):
                    series.append({"name": pl, "role": f"platform:{pl}",
                                   "data": [int(sub.get(k, 0)) for k in x],
                                   "stack": "det"})
            out["charts"].append({"kind": "area", "title": "Raw detections by satellite",
                                  "x": x, "yLabel": "detections / month",
                                  "series": series, "markers": markers})
        out["charts"].append({"kind": "line", "title": "Harmonized fire energy (one continuous record)",
                              "x": x, "yLabel": "FRE (TJ / month)",
                              "series": [{"name": "Harmonized FRE", "role": "harmonized",
                                          "data": [_f(v, 2) for v in mt["fre"]]}],
                              "markers": markers})
        n_plat = self._rows("""SELECT MIN(bit_count(active_platforms_bitmask)),
                                      MAX(bit_count(active_platforms_bitmask))
                               FROM uncertainty_diagnostics""")[0]
        out["kpis"] += [{"label": "Satellites active (min over record)", "value": n_plat[0], "format": "int"},
                        {"label": "Satellites active (max over record)", "value": n_plat[1], "format": "int"},
                        {"label": "Constellation changes handled", "value": len(markers), "format": "int"}]
        return out

    def challenge_flares(self, iso):
        if not self.has_product1:
            return {"kpis": [], "charts": []}
        w, p = self._p1_where(iso)
        c = self._df(f"""
            SELECT country_iso, COUNT(*) AS n,
                   SUM(CASE WHEN is_static_industrial THEN 1 ELSE 0 END) AS flare,
                   SUM(CASE WHEN is_sama_filtered THEN 1 ELSE 0 END) AS sama,
                   SUM(CASE WHEN type_raw IN (1, 3) THEN 1 ELSE 0 END) AS other
            FROM obs_harmonized_provenance {w} GROUP BY 1 ORDER BY 1""", p)
        tot = c[["n", "flare", "sama", "other"]].sum()
        return {
            "kpis": [
                {"label": "Detections screened", "value": int(tot["n"]), "format": "int"},
                {"label": "Static industrial (gas flares) removed", "value": int(tot["flare"]), "format": "int"},
                {"label": "South Atlantic Anomaly artefacts removed", "value": int(tot["sama"]), "format": "int"},
                {"label": "Share kept as vegetation fire", "value": _f(1 - (tot["flare"] + tot["sama"] + tot["other"]) / max(tot["n"], 1)), "format": "pct1"}],
            "charts": [{
                "kind": "bar", "title": "Non-fire heat sources removed before fusion",
                "x": c["country_iso"].tolist(), "yLabel": "detections",
                "series": [{"name": "Gas flares / industrial", "role": "flag:flare",
                            "data": c["flare"].astype(int).tolist(), "stack": "f"},
                           {"name": "SAMA artefacts", "role": "flag:sama",
                            "data": c["sama"].astype(int).tolist(), "stack": "f"},
                           {"name": "Volcano / offshore", "role": "flag:other",
                            "data": c["other"].astype(int).tolist(), "stack": "f"}]}]}

    def challenge_duplicates(self, iso):
        out = {"kpis": [], "charts": []}
        if self.has_product1:
            w, p = self._p1_where(iso)
            b = self._df(f"""
                SELECT sensor_family, LEAST(FLOOR(vza_deg / 10) * 10, 60) AS b,
                       AVG(bowtie_weight) AS wgt, COUNT(*) AS n
                FROM obs_harmonized_provenance {w} GROUP BY 1, 2 ORDER BY 2""", p)
            bins = sorted(b["b"].unique().tolist())
            out["charts"].append({
                "kind": "line", "title": "Bow-tie de-duplication weight vs view angle",
                "x": [f"{int(v)}-{int(v) + 10}°" for v in bins],
                "xLabel": "view zenith angle", "yLabel": "mean weight per pixel",
                "series": [{"name": f"{fam}", "role": "raw" if fam == "MODIS" else "raw2",
                            "data": [_f(v, 3) for v in b[b["sensor_family"] == fam]
                                     .set_index("b")["wgt"].reindex(bins)]}
                           for fam in ("MODIS", "VIIRS")]})
            dup = self._rows(f"""
                SELECT AVG(CASE WHEN bowtie_weight < 1 THEN 1.0 ELSE 0 END)
                FROM obs_harmonized_provenance {w} AND sensor_family = 'MODIS'""", p)[0][0]
            out["kpis"].append({"label": "MODIS pixels in bow-tie overlap", "value": _f(dup), "format": "pct1"})
        cl, p2 = _iso_clause(iso)
        e = self._df(f"""
            SELECT year, SUM(raw_count_sum) AS det, SUM(raw_clustered_events) AS ev
            FROM state_reconstructed_monthly
            WHERE year BETWEEN {FULL_YEARS[0]} AND {FULL_YEARS[1]} {cl}
            GROUP BY 1 ORDER BY 1""", p2)
        out["charts"].append({
            "kind": "bar", "title": "Pixels vs independent fire events",
            "x": e["year"].tolist(), "yLabel": "count",
            "series": [{"name": "Fire pixels (raw)", "role": "raw",
                        "data": e["det"].astype(int).tolist()},
                       {"name": "Clustered fire events", "role": "harmonized",
                        "data": e["ev"].astype(int).tolist()}]})
        out["kpis"].append({"label": "Pixels per fire event", "value": _f(e["det"].sum() / max(e["ev"].sum(), 1), 2), "format": "ratio"})
        return out

    def challenge_uncertainty(self, iso):
        cl, p = _iso_clause(iso)
        top = self._rows(f"""
            SELECT h3_index FROM state_reconstructed_monthly WHERE 1=1 {cl}
            GROUP BY 1 ORDER BY SUM(fre_tj_q50) DESC LIMIT 1""", p)
        out = {"kpis": [], "charts": []}
        diag = self.diagnostics()
        ab = diag.get("ablation_2019_2021") or {}
        if ab:
            out["kpis"].append({"label": "90% interval coverage (VIIRS-masked test)",
                                "value": _f(ab.get("coverage_90")), "format": "pct1"})
        w = self._rows(f"""SELECT MEDIAN(u.rel_uncertainty_width), AVG(u.conformal_c_factor)
                          FROM uncertainty_diagnostics u JOIN state_reconstructed_monthly m
                          ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
                          WHERE m.phi_q50_mw > 0.5 {_iso_clause(iso, 'm.')[0]}""",
                       _iso_clause(iso, "m.")[1])[0]
        out["kpis"] += [{"label": "Median relative 90% interval width", "value": _f(w[0], 3), "format": "ratio"},
                        {"label": "Conformal calibration factor", "value": _f(w[1], 3), "format": "ratio"}]
        if top:
            h3 = int(top[0][0])
            s = self._df("""SELECT year, month, phi_q05_mw, phi_q50_mw, phi_q95_mw,
                                   raw_count_sum
                            FROM state_reconstructed_monthly
                            WHERE h3_index = ? AND year >= 2018 ORDER BY 1, 2""", [h3])
            out["charts"].append({
                "kind": "band", "title": "Most active hexagon: estimate with 90% interval",
                "x": [f"{y}-{m:02d}" for y, m in zip(s["year"], s["month"])],
                "yLabel": "daily-mean FRP (MW)", "h3_index": str(h3),
                "series": [{"name": "5th percentile", "role": "band_low",
                            "data": [_f(v, 2) for v in s["phi_q05_mw"]]},
                           {"name": "90% interval", "role": "band_high",
                            "data": [_f(v, 2) for v in s["phi_q95_mw"]]},
                           {"name": "Median estimate", "role": "harmonized",
                            "data": [_f(v, 2) for v in s["phi_q50_mw"]]},
                           {"name": "Raw detections", "role": "raw", "axis": 1,
                            "data": s["raw_count_sum"].astype(int).tolist()}]})
        return out

    # ------------------------------------------------------------- validation
    @functools.lru_cache(maxsize=1)
    def validation(self) -> dict:
        diag = self.diagnostics()
        y = self.yearly(None)
        checks = []

        def add(cid, title, question, value, threshold, passed, critical,
                evidence=None, fmt="num", detail=None):
            checks.append({"id": cid, "title": title, "question": question,
                           "value": value, "threshold": threshold,
                           "status": ("unavailable" if passed is None else
                                      "pass" if passed else "fail"),
                           "critical": critical, "format": fmt,
                           "detail": detail, "evidence": evidence})

        raw_step = self.step_change(y, "raw_count")
        harm_step = self.step_change(y, "fre")
        sb = diag.get("structural_breaks") or {}
        ch_raw = (sb.get("chow_raw_2012") or {}).get("p")
        ch_h = (sb.get("chow_harm_2012") or {}).get("p")
        base = y[y["year"].between(2003, 2011)]
        add("break_2012", "The 2012 sensor jump is removed",
            "Does the record still jump when VIIRS (375 m) joins MODIS (1 km)?",
            harm_step, "|change 2009-11 -> 2012-14| < 15%",
            None if harm_step is None else abs(harm_step) < 0.15, True,
            {"kind": "line", "title": "Raw vs harmonized, indexed to 2003-2011",
             "x": y["year"].tolist(), "yLabel": "index",
             "series": [{"name": "Raw detections", "role": "raw",
                         "data": (100 * y["raw_count"] / max(base["raw_count"].mean(), 1e-9)).round(1).tolist()},
                        {"name": "Harmonized FRE", "role": "harmonized",
                         "data": (100 * y["fre"] / max(base["fre"].mean(), 1e-9)).round(1).tolist()}],
             "markers": [{"x": 2012, "label": "S-NPP"}]}, "pct",
            {"raw_step": raw_step, "chow_p_raw": ch_raw, "chow_p_harmonized": ch_h})
        add("chow_2012", "No structural break at 2012 (Chow test)",
            "Is a break at 2012 statistically detectable in the harmonized series?",
            ch_h, "p > 0.05 (raw for comparison)",
            None if ch_h is None else ch_h > 0.05, False, None, "p",
            {"chow_p_raw": ch_raw})
        ab = diag.get("ablation_2019_2021") or {}
        rb = ab.get("raw_baseline") or {}
        cov = ab.get("coverage_90")
        add("ablation_coverage", "Honest uncertainty when VIIRS is withheld",
            "With all VIIRS satellites hidden for 2019-2021, do the 90% intervals "
            "contain the reference ~90% of the time?", _f(cov),
            "0.80 <= coverage <= 0.98", None if cov is None else 0.80 <= cov <= 0.98,
            True, {"kind": "bar", "title": "90% interval coverage",
                   "x": ["Harmonized (MODIS-only re-solve)", "Nominal"],
                   "yLabel": "coverage", "series": [{"name": "coverage", "role": "harmonized",
                                                     "data": [_f(cov), 0.90]}]}, "pct1")
        lr, lrr = ab.get("log_rmse"), rb.get("log_rmse")
        add("ablation_skill", "Beats raw MODIS when VIIRS is withheld",
            "Is the harmonized MODIS-only reconstruction closer to the reference "
            "than scaled raw MODIS?", _f(lr), "log-RMSE below raw baseline",
            None if lr is None or lrr is None else lr < lrr, True,
            {"kind": "bar", "title": "Error vs reference (log-RMSE, lower is better)",
             "x": ["Harmonized", "Raw MODIS (scaled)"], "yLabel": "log-RMSE",
             "series": [{"name": "log-RMSE", "role": "harmonized",
                         "data": [_f(lr), _f(lrr)]}]}, "num",
            {"spearman_harmonized": _f(ab.get("spearman_rho")),
             "spearman_raw": _f(rb.get("spearman_rho")), "n": ab.get("n"),
             "reference": diag.get("reference")})
        mc = diag.get("mcd64cmq_corr") or {}
        hf, rc = mc.get("harmonized_fre"), mc.get("raw_counts")
        add("burned_area", "Agrees with independent burned area (MCD64)",
            "Does harmonized fire energy track an independent satellite product "
            "better than raw counts do?", _f(hf), "r >= raw-count r and r > 0.3",
            None if hf is None or rc is None else (hf >= rc and hf > 0.3), False,
            {"kind": "bar", "title": "Correlation with MCD64 burned area (log)",
             "x": ["Harmonized FRE", "Raw counts"], "yLabel": "Pearson r",
             "series": [{"name": "r", "role": "harmonized", "data": [_f(hf), _f(rc)]}]},
            "num", {"n": mc.get("n")})
        rs = diag.get("residual_slopes") or {}
        sv = rs.get("slope_per_vza")
        add("view_angle_bias", "No residual view-angle bias",
            "After harmonization, do residuals still depend on the viewing angle?",
            _f(sv, 5), "|slope| x 60 deg < 0.25 log units",
            None if sv is None else abs(sv) * 60 < 0.25, False, None, "num",
            {"slope_per_solar_hour": _f(rs.get("slope_per_solar_hour"), 5), "n": rs.get("n")})
        if self.is_synthetic:
            t = self._df("""
                SELECT t.phi24_true AS t, m.phi_q05_mw AS lo, m.phi_q50_mw AS md,
                       m.phi_q95_mw AS hi
                FROM synthetic_truth t JOIN state_reconstructed_monthly m
                  USING (h3_index, year, month)""")
            r = float(np.corrcoef(np.log1p(t["t"]), np.log1p(t["md"]))[0, 1])
            cov_t = float(((t["lo"] <= t["t"]) & (t["t"] <= t["hi"])).mean())
            smp = t.sample(min(1500, len(t)), random_state=0)
            add("truth_recovery", "Recovers the known synthetic truth",
                "Synthetic only: how close is the estimate to the simulated truth?",
                _f(r), "log correlation > 0.9 and coverage 0.8-0.98",
                r > 0.9 and 0.8 <= cov_t <= 0.98, False,
                {"kind": "scatter", "title": "Estimate vs simulated truth (log1p MW)",
                 "xLabel": "truth", "yLabel": "estimate",
                 "series": [{"name": "cell-months", "role": "harmonized",
                             "data": [[_f(np.log1p(a), 3), _f(np.log1p(b), 3)]
                                      for a, b in zip(smp["t"], smp["md"])]}]},
                "num", {"coverage_vs_truth": _f(cov_t),
                        "top3_peak_month_match": _f(diag.get("top3_peak_month_match"))})
        crit = [c for c in checks if c["critical"]]
        gate = {"passed": bool(crit) and all(c["status"] == "pass" for c in crit),
                "critical": [c["id"] for c in crit],
                "failed": [c["id"] for c in crit if c["status"] != "pass"]}
        return {"checks": checks, "gate": gate,
                "reference": diag.get("reference"),
                "diagnostics_available": bool(diag)}

    # -------------------------------------------------------------- insights
    def insights(self, iso=None) -> dict:
        y = self.yearly(iso)
        years = y["year"].to_numpy()
        v = y["fre"].to_numpy(dtype=float)
        slope = stats.theilslopes(np.log1p(v), years) if len(v) > 3 else None
        tau = stats.kendalltau(years, v) if len(v) > 3 else None
        mt = self.monthly_total(iso)
        clim = mt[mt["year"].between(2003, 2020)]
        g = clim.groupby("month")["fre"]
        seas = pd.DataFrame({"mean": g.mean(), "p10": g.quantile(0.1),
                             "p90": g.quantile(0.9)}).reindex(range(1, 13))
        peak = int(seas["mean"].idxmax()) if seas["mean"].notna().any() else None
        season_len = int((seas["mean"] >= 0.3 * seas["mean"].max()).sum())
        med = clim.groupby("month")["fre"].median()
        a = mt.copy()
        a["ratio"] = a["fre"] / a["month"].map(med).clip(lower=1e-9)
        a = a[a["year"] >= 2003]
        top = a.sort_values("ratio", ascending=False).head(8)
        cl, p = _iso_clause(iso, "m.")
        att = self._df(f"""
            SELECT m.year, m.month, u.attribution_flag, COUNT(*) AS n
            FROM uncertainty_diagnostics u JOIN state_reconstructed_monthly m
              ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
            WHERE 1=1 {cl} GROUP BY 1, 2, 3""", p)
        anomalies = []
        for r in top.itertuples():
            sub = att[(att["year"] == r.year) & (att["month"] == r.month)]
            tot = max(sub["n"].sum(), 1)
            share = {k: round(float(n) / tot, 3) for k, n in
                     zip(sub["attribution_flag"], sub["n"])}
            anomalies.append({"year": int(r.year), "month": int(r.month),
                              "fre_tj": _f(r.fre, 1), "ratio_to_median": _f(r.ratio, 2),
                              "attribution": share})
        by_country = []
        if not iso:
            for c in S.TARGET_COUNTRY_ISOS:
                yc = self.yearly(c)
                if len(yc) > 3 and yc["fre"].sum() > 0:
                    sl = stats.theilslopes(np.log1p(yc["fre"]), yc["year"])
                    kt = stats.kendalltau(yc["year"], yc["fre"])
                    by_country.append({"iso": c, "name": COUNTRY_NAMES[c],
                                       "fre_tj_per_year": _f(yc["fre"].mean(), 1),
                                       "trend_pct_per_decade": _f(100 * (np.exp(10 * sl.slope) - 1), 1),
                                       "trend_p": _f(kt.pvalue, 4)})
        return {
            "trend": {"pct_per_decade": _f(100 * (np.exp(10 * slope.slope) - 1), 1)
                      if slope else None,
                      "mann_kendall_p": _f(tau.pvalue, 4) if tau else None,
                      "significant": bool(tau and tau.pvalue < 0.05)},
            "seasonality": {"peak_month": peak, "season_length_months": season_len},
            "charts": {
                "annual": {"kind": "band", "title": "Annual fire radiative energy",
                           "x": y["year"].tolist(), "yLabel": "FRE (TJ / year)",
                           "series": [{"name": "5th pct", "role": "band_low",
                                       "data": [_f(x, 1) for x in y["fre_q05"]]},
                                      {"name": "90% interval", "role": "band_high",
                                       "data": [_f(x, 1) for x in y["fre_q95"]]},
                                      {"name": "Harmonized FRE", "role": "harmonized",
                                       "data": [_f(x, 1) for x in y["fre"]]},
                                      {"name": "Trend (Theil-Sen)", "role": "trend",
                                       "data": [_f(np.expm1(slope.intercept + slope.slope * yy), 1)
                                                for yy in years] if slope else []}]},
                "seasonal": {"kind": "band", "title": "Typical fire season (2003-2020)",
                             "x": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                   "Aug", "Sep", "Oct", "Nov", "Dec"],
                             "yLabel": "FRE (TJ / month)",
                             "series": [{"name": "10th pct", "role": "band_low",
                                         "data": [_f(x, 1) for x in seas["p10"]]},
                                        {"name": "10th-90th pct of years", "role": "band_high",
                                         "data": [_f(x, 1) for x in seas["p90"]]},
                                        {"name": "Mean", "role": "harmonized",
                                         "data": [_f(x, 1) for x in seas["mean"]]}]}},
            "anomalies": anomalies, "by_country": by_country}

    # ----------------------------------------------------------------- replay
    def replay(self, iso: str | None, year: int, month: int, step: str = "day",
               max_cells: int = 2500, max_detections: int = 25000) -> dict:
        """Day-by-day (harmonized daily state + detections) or overpass-by-
        overpass ("hour": raw detections per UTC hour, daily state behind)
        replay of one month for the burning-calendar animation.

        Compact columnar payload: hexagon polygons once, then
        ``daily = [frame, cell, phi_q50, phi_q95]`` and
        ``detections = [frame, lon, lat, frp, platform_idx]``.
        Largest cells / strongest detections first when capped."""
        from ...geometry.h3_utils import h3_boundary_geojson
        cl, p = _iso_clause(iso, "m.")
        cells = self._df(f"""
            SELECT m.h3_index FROM state_reconstructed_monthly m
            WHERE m.year = ? AND m.month = ? AND m.phi_q50_mw > 0 {cl}
            ORDER BY m.fre_tj_q50 DESC LIMIT {int(max_cells)}""",
                         [year, month] + p)["h3_index"].astype("int64")
        cell_ids = cells.tolist()
        idx = {c: i for i, c in enumerate(cell_ids)}
        t0 = pd.Timestamp(S.DATE_START)
        first = pd.Timestamp(year=year, month=month, day=1)
        ndays = first.days_in_month
        t_first = int((first - t0).days)
        out = {"year": year, "month": month, "step": step, "frames": [],
               "cells": [str(c) for c in cell_ids],
               "polygons": [h3_boundary_geojson(c) for c in cell_ids],
               "daily": [], "detections": [], "platforms": PLATFORM_ORDER,
               "truncated": {"cells": len(cell_ids) >= max_cells,
                             "detections": False}}
        if not cell_ids:
            return out
        cid_list = ",".join(str(c) for c in cell_ids)
        d = self._df(f"""
            SELECT h3_index, t, phi_q50_mw, phi_q95_mw
            FROM state_reconstructed_daily
            WHERE t BETWEEN ? AND ? AND h3_index IN ({cid_list})
            ORDER BY t, h3_index""",
                     [t_first, t_first + ndays - 1])
        det = pd.DataFrame(columns=["h3_index", "day", "hh", "longitude",
                                    "latitude", "frp_mw", "platform"])
        if self.has_product1:
            det = self._df(f"""
                SELECT h3_index, DAY(CAST(solar_date AS DATE)) AS day,
                       CAST(acq_hhmm / 100 AS INTEGER) AS hh,
                       longitude, latitude, frp_mw, platform
                FROM obs_harmonized_provenance
                WHERE h3_index IN ({cid_list}) AND is_biomass
                  AND YEAR(CAST(solar_date AS DATE)) = ?
                  AND MONTH(CAST(solar_date AS DATE)) = ?
                ORDER BY frp_mw DESC, solar_date, acq_hhmm, latitude, longitude
                LIMIT {int(max_detections) + 1}""",
                           [year, month])
            if len(det) > max_detections:
                det = det.iloc[:max_detections]
                out["truncated"]["detections"] = True
        pidx = {pl: i for i, pl in enumerate(PLATFORM_ORDER)}
        # harmonized state is DAILY: one row per (day index, cell); every
        # frame carries the day it belongs to
        out["daily"] = [[int(r.t) - t_first, idx[int(r.h3_index)],
                         _f(r.phi_q50_mw, 3), _f(r.phi_q95_mw, 3)]
                        for r in d.itertuples(index=False)]
        if step == "hour":
            keys = sorted(set(zip(det["day"].astype(int), det["hh"].astype(int))))
            fidx = {k: i for i, k in enumerate(keys)}
            out["frames"] = [{"key": f"{year}-{month:02d}-{dd:02d}T{hh:02d}",
                              "label": f"{dd} {S_MONTHS[month - 1]} · {hh:02d}:00 UTC",
                              "day": dd} for dd, hh in keys]
            det_frame = [fidx[(int(a), int(b))] for a, b in zip(det["day"], det["hh"])]
        else:
            out["frames"] = [{"key": f"{year}-{month:02d}-{dd:02d}",
                              "label": f"{dd} {S_MONTHS[month - 1]} {year}",
                              "day": dd} for dd in range(1, ndays + 1)]
            det_frame = (det["day"].astype(int) - 1).tolist()
        out["detections"] = [[f, round(float(lo), 4), round(float(la), 4),
                              round(float(fr), 2), pidx.get(pl, 0)]
                             for f, lo, la, fr, pl in zip(
                                 det_frame, det["longitude"], det["latitude"],
                                 det["frp_mw"], det["platform"])]
        out["n_detections_month"] = int(len(det))
        return out

    # ----------------------------------------------------------- traceability
    def cell(self, h3: int, year: int | None, month: int | None) -> dict:
        info = self._rows("""SELECT latitude, longitude, country_iso
                             FROM hex_centers WHERE h3_index = ?""", [h3])
        if not info:
            return {}
        s = self._df("""
            SELECT m.year, m.month, m.phi_q05_mw, m.phi_q50_mw, m.phi_q95_mw,
                   m.raw_count_sum, m.raw_frp_sum_mw, m.anomaly_ratio,
                   m.pyrome_id, u.measured_fraction, u.quality_flag,
                   u.attribution_flag, u.active_platforms_bitmask,
                   u.mean_cloud_frac
            FROM state_reconstructed_monthly m JOIN uncertainty_diagnostics u
              ON m.h3_index = u.h3_index AND m.year = u.year AND m.month = u.month
            WHERE m.h3_index = ? ORDER BY 1, 2""", [h3])
        if year is None or month is None:
            best = s.loc[s["phi_q50_mw"].idxmax()]
            year, month = int(best["year"]), int(best["month"])
        sel = s[(s["year"] == year) & (s["month"] == month)]
        region = None
        if "synthetic_regions" in self.tables:
            r = self._rows("SELECT region FROM synthetic_regions WHERE h3_index = ?", [h3])
            region = r[0][0] if r else None
        out = {"h3_index": str(h3), "lat": info[0][0], "lon": info[0][1],
               "country_iso": info[0][2], "region": region,
               "pyrome_id": int(s["pyrome_id"].iloc[0]) if len(s) else None,
               "year": year, "month": month,
               "series": {"x": [f"{a}-{b:02d}" for a, b in zip(s["year"], s["month"])],
                          "q05": [_f(v, 2) for v in s["phi_q05_mw"]],
                          "q50": [_f(v, 2) for v in s["phi_q50_mw"]],
                          "q95": [_f(v, 2) for v in s["phi_q95_mw"]],
                          "raw_count": s["raw_count_sum"].astype(int).tolist()},
               "month_record": None, "detections": [], "by_platform": [],
               "reconciliation": None}
        if len(sel):
            r = sel.iloc[0]
            out["month_record"] = {
                "phi_q05_mw": _f(r["phi_q05_mw"], 3), "phi_q50_mw": _f(r["phi_q50_mw"], 3),
                "phi_q95_mw": _f(r["phi_q95_mw"], 3), "raw_count": int(r["raw_count_sum"]),
                "raw_frp_mw": _f(r["raw_frp_sum_mw"], 2), "anomaly_ratio": _f(r["anomaly_ratio"], 3),
                "measured_fraction": _f(r["measured_fraction"], 3),
                "mean_cloud_frac": _f(r["mean_cloud_frac"], 3),
                "quality_flag": r["quality_flag"], "attribution_flag": r["attribution_flag"],
                "active_platforms": [p for p, i in S.PLATFORM_INDEX.items()
                                     if int(r["active_platforms_bitmask"]) & S.PLATFORM_BITMASK[i]]}
        if self.has_product1:
            ym = f"{year}-{month:02d}"
            d = self._df("""
                SELECT platform, sensor_family, data_tier, acq_date, acq_hhmm,
                       solar_hour, dn_flag, latitude, longitude, frp_mw,
                       pixel_area_km2, vza_deg, conf_raw, is_biomass,
                       is_static_industrial, is_sama_filtered, bowtie_weight,
                       reliability_prob, forward_gain_c, obs_sigma,
                       event_cluster_id
                FROM obs_harmonized_provenance
                WHERE h3_index = ?
                  AND strftime(CAST(solar_date AS DATE), '%Y-%m') = ?
                ORDER BY acq_date, acq_hhmm""", [h3, ym])
            out["detections"] = [
                {k: (_f(v, 4) if isinstance(v, (float, np.floating)) else
                     (bool(v) if isinstance(v, (bool, np.bool_)) else
                      (int(v) if isinstance(v, (int, np.integer)) else str(v))))
                 for k, v in row.items()} for row in d.head(400).to_dict("records")]
            if len(d):
                bp = d.groupby("platform").agg(n=("frp_mw", "size"),
                                               frp=("frp_mw", "sum"),
                                               bio=("is_biomass", "sum"))
                out["by_platform"] = [{"platform": k, "n": int(v.n),
                                       "frp_mw": _f(v.frp, 2), "biomass": int(v.bio)}
                                      for k, v in bp.iterrows()]
            n_bio = int(d["is_biomass"].sum()) if len(d) else 0
            if out["month_record"] is not None:
                out["reconciliation"] = {
                    "biomass_detections_in_product1": n_bio,
                    "raw_count_in_product2": out["month_record"]["raw_count"],
                    "match": n_bio == out["month_record"]["raw_count"],
                    "n_detections_total": int(len(d))}
        return out
