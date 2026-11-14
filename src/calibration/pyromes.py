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
    cluster table -- passed through unchanged)."""
    if "event_cluster_id" not in prov_bio.columns:
        return prov_bio.copy()
    g = prov_bio.groupby("event_cluster_id", as_index=False).agg(
        h3_index=("h3_index", "first"),
        solar_date=("solar_date", "first"),
        dn_flag=("dn_flag", "first"),
        cluster_frp_mw=("frp_mw", "sum"),
        n_pixels=("frp_mw", "size"))
    g["solar_date"] = pd.to_datetime(g["solar_date"])
    return g


def behavioral_features(clusters: pd.DataFrame) -> pd.DataFrame:
    """Per-hexagon coordinate-free behavioral features (2012-2021 window)."""
    c = clusters.copy()
    c["year"] = c["solar_date"].dt.year
    c["doy"] = c["solar_date"].dt.dayofyear
    theta = 2.0 * np.pi * c["doy"].to_numpy() / 365.0
    c["cos_t"] = np.cos(theta)
    c["sin_t"] = np.sin(theta)
    c["is_night"] = (c["dn_flag"] == "N").astype(float)
    c["frp_night"] = c["cluster_frp_mw"] * c["is_night"]

    def _feat(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        years = np.sort(g["year"].unique())
        yearly = g.groupby("year")["cluster_frp_mw"].sum().reindex(years,
                                                                   fill_value=0.0)
        mean_y = float(yearly.mean())
        cv = float(yearly.std(ddof=1) / mean_y) if (mean_y > 0 and len(years) > 1) else 0.0
        R = float(np.hypot(g["cos_t"].sum(), g["sin_t"].sum()) / n)
        return pd.Series({
            "ln_median_cluster_frp": float(np.log(np.median(g["cluster_frp_mw"]) + 1.0)),
            "night_frp_share": float(g["frp_night"].sum() / max(g["cluster_frp_mw"].sum(), 1e-9)),
            "seasonal_circular_concentration": R,
            "interannual_cv": cv,
            "mean_cluster_pixels": float(g["n_pixels"].mean()),
            "n_active_days": int(g["solar_date"].dt.normalize().nunique()),
        })

    return c.groupby("h3_index").apply(_feat, include_groups=False).reset_index()


def fit_pyromes(prov_bio: pd.DataFrame, calib_start: str = S.CALIB_START,
                calib_end: str = S.CALIB_END) -> tuple[pd.DataFrame, GaussianMixture, StandardScaler]:
    """Fit the K=7 behavioral GMM and assign every active hexagon.

    Returns (hex_pyromes frame [h3_index, pyrome_id, n_active_days], fitted
    GMM, fitted StandardScaler).
    """
    clusters = extract_cluster_table(prov_bio)
    win = clusters[(clusters["solar_date"] >= calib_start)
                   & (clusters["solar_date"] <= calib_end)]
    feats = behavioral_features(win if len(win) else clusters)
    if feats.empty:
        empty = pd.DataFrame(columns=["h3_index", "pyrome_id", "n_active_days"])
        return empty, None, None

    core = feats[FEATURES].to_numpy(dtype=np.float64)
    core = np.nan_to_num(core, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(core)
    z = scaler.transform(core)
    gmm = GaussianMixture(n_components=S.PYROME_K, covariance_type="full",
                          reg_covar=S.GMM_REG_COVAR,
                          random_state=S.GMM_RANDOM_STATE)
    gmm.fit(z)
    labels = gmm.predict(z)

    # < 20 active-day cells -> nearest centroid (already implied by predict);
    # cells with < 3 active days -> country modal pyrome (fallback mode).
    country_mode = int(pd.Series(labels).mode().iloc[0]) if len(labels) else 0
    low_evidence = feats["n_active_days"].to_numpy() < 3
    labels = np.where(low_evidence, country_mode, labels)
    out = feats[["h3_index", "n_active_days"]].copy()
    out["pyrome_id"] = labels.astype(np.int16)
    return out, gmm, scaler


def assign_pyromes(feats_new: pd.DataFrame, gmm, scaler) -> np.ndarray:
    """Assign new hexagons to the nearest GMM centroid (standardized space)."""
    z = scaler.transform(feats_new[FEATURES].to_numpy(dtype=np.float64))
    return gmm.predict(z)


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
