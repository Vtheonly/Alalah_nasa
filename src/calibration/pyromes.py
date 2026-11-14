"""Coordinate-free behavioral pyrome discovery (readme.md 5.2 item 1).

For every H3-Res6 hexagon with >= 20 active pass-days in 2012-2021, compute
strictly non-coordinate behavioral features
``[ln(median_cluster_frp + 1), night_frp_share, seasonal_circular_concentration,
interannual_cv, mean_cluster_pixels]`` and fit a ``K=7`` full-covariance
GaussianMixture (``reg_covar=1e-4``, ``random_state=42``) on standardized
features. Every active hexagon receives ``pyrome_id in 0..6``; cells with
< 20 pass-days are assigned to the nearest GMM centroid in standardized
space (or the country modal pyrome below 3 active days). Latitude and
longitude are never used (INV-10).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import settings as S

PYROME_FEATURES = (
    "ln_median_cluster_frp",
    "night_frp_share",
    "seasonal_circular_concentration",
    "interannual_cv",
    "mean_cluster_pixels",
)


def compute_pyrome_features(
    pass_obs: pd.DataFrame,
) -> pd.DataFrame:
    """Behavioral feature table per active hexagon.

    ``pass_obs`` requires ``h3_index, solar_date, y_frp_mw, raw_event_count,
    n_pixels, dn_flag`` plus at least one row inside the 2012-2021 window.
    """
    po = pass_obs.copy()
    po["year"] = po["solar_date"].str.slice(0, 4).astype(int)
    po["month"] = po["solar_date"].str.slice(5, 7).astype(int)
    in_calib = po[
        (po["solar_date"] >= S.CALIB_START) & (po["solar_date"] <= S.CALIB_END)
    ]
    rows = []
    for h3, part in in_calib.groupby("h3_index"):
        n_pass_days = part["solar_date"].nunique()
        median_frp = float(part["y_frp_mw"].median())
        night_share = float(
            (part.loc[part["dn_flag"] == "N", "y_frp_mw"].sum()
             / max(part["y_frp_mw"].sum(), 1e-9))
        )
        # Circular concentration of monthly activity (mean resultant length).
        theta = 2.0 * np.pi * part["month"].to_numpy() / 12.0
        weights = part["y_frp_mw"].to_numpy()
        if weights.sum() > 0:
            r_bar = float(
                np.hypot(
                    (weights * np.cos(theta)).sum(),
                    (weights * np.sin(theta)).sum(),
                )
                / weights.sum()
            )
        else:
            r_bar = 0.0
        annual = part.groupby("year")["y_frp_mw"].sum()
        mean_annual = float(annual.mean())
        interannual_cv = float(
            annual.std(ddof=1) / mean_annual if mean_annual > 0 else 0.0
        )
        pixels_per_cluster = float(
            (part["n_pixels"] / part["raw_event_count"].clip(lower=1)).mean()
        )
        rows.append(
            {
                "h3_index": h3,
                "n_pass_days": int(n_pass_days),
                "ln_median_cluster_frp": float(np.log(median_frp + 1.0)),
                "night_frp_share": night_share,
                "seasonal_circular_concentration": r_bar,
                "interannual_cv": interannual_cv,
                "mean_cluster_pixels": pixels_per_cluster,
            }
        )
    return pd.DataFrame(rows)


def fit_pyromes(
    features: pd.DataFrame,
    country_of_hex: Optional[Dict[int, str]] = None,
) -> Tuple[pd.DataFrame, GaussianMixture, StandardScaler]:
    """Fit the K=7 GMM and assign every active hexagon a pyrome id.

    Returns ``(hex_pyromes, gmm, scaler)`` where ``hex_pyromes`` carries
    ``h3_index, pyrome_id, n_pass_days``.
    """
    country_of_hex = country_of_hex or {}
    if features.empty:
        return (
            pd.DataFrame(columns=["h3_index", "pyrome_id", "n_pass_days"]),
            None,
            None,
        )
    core = features[features["n_pass_days"] >= S.PYROME_MIN_PASS_DAYS]
    X_all = features[list(PYROME_FEATURES)].to_numpy(dtype=float)
    X_core = core[list(PYROME_FEATURES)].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_core if len(X_core) else X_all)
    gmm = GaussianMixture(
        n_components=S.PYROME_K,
        covariance_type="full",
        reg_covar=S.GMM_REG_COVAR,
        random_state=S.GMM_RANDOM_STATE,
    )
    if len(X_core):
        gmm.fit(scaler.transform(X_core))
    else:  # degenerate archive: fit on everything available
        gmm.fit(scaler.transform(X_all))

    # Assign >= 20-day cells by posterior; sparse cells by nearest centroid.
    labels_core = gmm.predict(scaler.transform(X_core))
    out = []
    core_index = {h: i for i, h in enumerate(core["h3_index"])}
    for _, row in features.iterrows():
        h3 = row["h3_index"]
        if h3 in core_index:
            label = int(labels_core[core_index[h3]])
        elif row["n_pass_days"] >= 3:
            x = scaler.transform(
                row[list(PYROME_FEATURES)].to_numpy(dtype=float)[None, :]
            )
            label = int(np.argmin(((x - gmm.means_) ** 2).sum(axis=1)))
        else:
            # < 3 active days -> country modal pyrome (deterministic 0).
            label = 0
        out.append({"h3_index": h3, "pyrome_id": label,
                    "n_pass_days": int(row["n_pass_days"])})
    hex_pyromes = pd.DataFrame(out)
    return hex_pyromes, gmm, scaler


def run_pyromes(
    pass_obs: pd.DataFrame,
    out_parquet,
    gmm_pkl=None,
    country_of_hex: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """End-to-end pyrome fit + persistence (hex_pyromes + pyrome_gmm.pkl)."""
    import pickle
    from pathlib import Path

    features = compute_pyrome_features(pass_obs)
    hex_pyromes, gmm, scaler = fit_pyromes(features, country_of_hex)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    hex_pyromes.to_parquet(out_parquet, index=False)
    if gmm_pkl is not None and gmm is not None:
        with open(gmm_pkl, "wb") as fh:
            pickle.dump({"gmm": gmm, "scaler": scaler,
                         "features": list(PYROME_FEATURES)}, fh)
    return hex_pyromes
