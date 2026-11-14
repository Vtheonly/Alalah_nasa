"""Coordinate-free behavioral pyromes (K=7 GMM) -- INV-10.

Contract (readme Sections 4.7, 5.2.1): for every H3-Res6 hexagon with >= 20
active pass-days in 2012-2021, compute strictly non-coordinate behavioral
features ``[ln(median_cluster_frp + 1), night_frp_share,
seasonal_circular_concentration, interannual_cv, mean_cluster_pixels]``, fit
``GaussianMixture(n_components=7, covariance_type='full', reg_covar=1e-4,
random_state=42)`` over StandardScaler features, and assign every active
hexagon ``pyrome_id in 0..6`` (< 20-day cells are assigned to the nearest
GMM centroid in standardized space; cells with < 3 active days fall back to
the country modal pyrome).  Latitude/longitude are NEVER used (INV-10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..config import settings as S

FEATURES = ["ln_median_cluster_frp", "night_frp_share",
            "seasonal_circular_concentration", "interannual_cv",
            "mean_cluster_pixels"]


def extract_cluster_table(prov_bio: pd.DataFrame) -> pd.DataFrame:
    """Cluster-level records from biomass provenance rows (or a pre-built
    cluster table -- passed through unchanged).

    Besides the behavioural fields, the cluster keeps the pass geometry and
    sensor needed by the truncated Log-Normal fit (``sensor_family``,
    ``platform``, mean ``vza_deg``) and its centroid (``latitude`` /
    ``longitude``, used ONLY to join the 0.25 deg cloud grid -- never as a
    pyrome feature, INV-10) plus ``country_iso`` (country modal pyrome).
    """
    if "event_cluster_id" not in prov_bio.columns:
        out = prov_bio.copy()
        if "solar_date" in out.columns:
            out["solar_date"] = pd.to_datetime(out["solar_date"])
        return out
    aggs = dict(h3_index=("h3_index", "first"),
                solar_date=("solar_date", "first"),
                dn_flag=("dn_flag", "first"),
                cluster_frp_mw=("frp_mw", "sum"),
                n_pixels=("frp_mw", "size"))
    for col, how in (("sensor_family", "first"), ("platform", "first"),
                     ("vza_deg", "mean"), ("latitude", "mean"),
                     ("longitude", "mean"), ("country_iso", "first")):
        if col in prov_bio.columns:
            aggs[col] = (col, how)
    g = prov_bio.groupby("event_cluster_id", as_index=False).agg(**aggs)
    g["solar_date"] = pd.to_datetime(g["solar_date"])
    return g


def behavioral_features(clusters: pd.DataFrame) -> pd.DataFrame:
    """Per-hexagon coordinate-free behavioural features (vectorized).

    ``[ln(median_cluster_frp + 1), night_frp_share,
    seasonal_circular_concentration, interannual_cv, mean_cluster_pixels]``
    plus ``n_active_days`` (distinct active solar days).
    """
    c = clusters[["h3_index", "solar_date", "dn_flag", "cluster_frp_mw",
                  "n_pixels"]].copy()
    c["solar_date"] = pd.to_datetime(c["solar_date"])
    c["year"] = c["solar_date"].dt.year
    theta = 2.0 * np.pi * c["solar_date"].dt.dayofyear.to_numpy() / 365.0
    c["cos_t"] = np.cos(theta)
    c["sin_t"] = np.sin(theta)
    c["frp_night"] = c["cluster_frp_mw"] * (c["dn_flag"] == "N")
    c["day"] = c["solar_date"].dt.normalize()
    g = c.groupby("h3_index")
    out = pd.DataFrame({
        "n": g.size(),
        "med": g["cluster_frp_mw"].median(),
        "frp_sum": g["cluster_frp_mw"].sum(),
        "frp_night": g["frp_night"].sum(),
        "cos_sum": g["cos_t"].sum(),
        "sin_sum": g["sin_t"].sum(),
        "mean_cluster_pixels": g["n_pixels"].mean(),
        "n_active_days": g["day"].nunique(),
    })
    yearly = c.groupby(["h3_index", "year"])["cluster_frp_mw"].sum()
    yg = yearly.groupby(level="h3_index")
    y_mean = yg.mean()
    y_std = yg.std(ddof=1)
    y_n = yg.size()
    cv = np.where((y_mean > 0) & (y_n > 1), y_std / y_mean.where(y_mean > 0),
                  0.0)
    out["interannual_cv"] = pd.Series(cv, index=y_mean.index).reindex(
        out.index).fillna(0.0)
    out["ln_median_cluster_frp"] = np.log(out["med"] + 1.0)
    out["night_frp_share"] = out["frp_night"] / np.maximum(out["frp_sum"],
                                                           1e-9)
    out["seasonal_circular_concentration"] = np.hypot(
        out["cos_sum"], out["sin_sum"]) / out["n"]
    out["n_active_days"] = out["n_active_days"].astype(int)
    out = out.reset_index()
    return out[["h3_index"] + FEATURES + ["n_active_days"]]


def fit_pyromes(prov_bio: pd.DataFrame, calib_start: str = S.CALIB_START,
                calib_end: str = S.CALIB_END
                ) -> tuple[pd.DataFrame, GaussianMixture, StandardScaler]:
    """Fit the K=7 behavioural GMM and assign EVERY active hexagon.

    * Features are computed strictly on the 2012-01-20 .. 2021-12-31 window.
    * The GMM (+ StandardScaler) is fit ONLY on hexagons with
      ``>= PYROME_MIN_ACTIVE_DAYS`` (20) active days in that window (if
      fewer than K such hexagons exist, the fit set is widened to all
      window hexagons with >= 3 active days so the K-component GMM stays
      identifiable).
    * 3..19-day hexagons go to the NEAREST GMM centroid in standardized
      feature space; hexagons with < 3 window days -- including hexagons
      active only outside 2012-2021 -- take their country's modal pyrome
      (global modal pyrome when the country is unknown / has no fitted cell).

    Returns (hex_pyromes [h3_index, pyrome_id, n_active_days,
    assignment], fitted GMM, fitted StandardScaler).
    """
    clusters = extract_cluster_table(prov_bio)
    cols = ["h3_index", "pyrome_id", "n_active_days", "assignment"]
    if clusters.empty:
        return pd.DataFrame(columns=cols), None, None
    win = clusters[(clusters["solar_date"] >= calib_start)
                   & (clusters["solar_date"] <= calib_end)]
    all_cells = pd.Index(np.unique(clusters["h3_index"].astype(np.int64)))
    country = None
    if "country_iso" in clusters.columns:
        country = (clusters.groupby("h3_index")["country_iso"].first()
                   .reindex(all_cells))
    feats = behavioral_features(win) if len(win) else pd.DataFrame(
        columns=["h3_index"] + FEATURES + ["n_active_days"])
    n_days = feats.set_index("h3_index")["n_active_days"].reindex(
        all_cells).fillna(0).astype(int)

    fit_mask = feats["n_active_days"].to_numpy() >= S.PYROME_MIN_ACTIVE_DAYS
    if fit_mask.sum() < S.PYROME_K:
        fit_mask = feats["n_active_days"].to_numpy() >= 3
    if fit_mask.sum() < S.PYROME_K:
        fit_mask = np.ones(len(feats), dtype=bool)
    gmm = scaler = None
    labels = pd.Series(-1, index=all_cells, dtype=int)
    how = pd.Series("COUNTRY_MODAL", index=all_cells, dtype=object)
    if len(feats) and fit_mask.sum() >= 1:
        X = np.nan_to_num(feats[FEATURES].to_numpy(dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(X[fit_mask])
        Z = scaler.transform(X)
        n_comp = int(min(S.PYROME_K, fit_mask.sum()))
        gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                              reg_covar=S.GMM_REG_COVAR,
                              random_state=S.GMM_RANDOM_STATE)
        gmm.fit(Z[fit_mask])
        lab = np.full(len(feats), -1, dtype=int)
        lab[fit_mask] = gmm.predict(Z[fit_mask])
        nearest = np.argmin(((Z[:, None, :] - gmm.means_[None, :, :]) ** 2)
                            .sum(axis=-1), axis=1)
        nd = feats["n_active_days"].to_numpy()
        mid = (~fit_mask) & (nd >= 3)
        lab[mid] = nearest[mid]
        cell_ix = feats["h3_index"].astype(np.int64).to_numpy()
        labels.loc[cell_ix[lab >= 0]] = lab[lab >= 0]
        how.loc[cell_ix[fit_mask]] = "GMM_FIT"
        how.loc[cell_ix[mid]] = "NEAREST_CENTROID"
    assigned = labels[labels >= 0]
    global_mode = int(assigned.mode().iloc[0]) if len(assigned) else 0
    todo = labels.index[labels < 0]
    if len(todo):
        if country is not None and len(assigned):
            cmode = (pd.DataFrame({"c": country.reindex(assigned.index),
                                   "l": assigned})
                     .dropna().groupby("c")["l"]
                     .agg(lambda v: int(v.mode().iloc[0])))
            fill = country.reindex(todo).map(cmode).fillna(global_mode)
        else:
            fill = pd.Series(global_mode, index=todo)
        labels.loc[todo] = fill.astype(int).to_numpy()
    out = pd.DataFrame({"h3_index": all_cells.to_numpy(dtype=np.int64),
                        "pyrome_id": labels.to_numpy().astype(np.int16),
                        "n_active_days": n_days.to_numpy(),
                        "assignment": how.to_numpy()})
    return out, gmm, scaler


def assign_pyromes(feats_new: pd.DataFrame, gmm, scaler) -> np.ndarray:
    """Assign new hexagons to the nearest GMM centroid (standardized space)."""
    z = scaler.transform(feats_new[FEATURES].to_numpy(dtype=np.float64))
    return np.argmin(((z[:, None, :] - gmm.means_[None, :, :]) ** 2)
                     .sum(axis=-1), axis=1)


def run_pyromes(prov_bio: pd.DataFrame, paths=None) -> pd.DataFrame:
    """Fit + persist ``hex_pyromes.parquet`` (+ ``pyrome_gmm.pkl``)."""
    hex_pyromes, gmm, scaler = fit_pyromes(prov_bio)
    if paths is not None:
        hex_pyromes.to_parquet(paths.pyromes_parquet, index=False)
        import pickle
        with open(str(paths.pyromes_parquet).replace(
                "hex_pyromes.parquet", "pyrome_gmm.pkl"), "wb") as fh:
            pickle.dump({"gmm": gmm, "scaler": scaler,
                         "features": FEATURES}, fh)
    return hex_pyromes
