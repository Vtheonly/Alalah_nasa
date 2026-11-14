"""test_07 -- Synthetic digital-twin truth tests (18 tests).

Readme Section 7.7: complete 26-year synthetic forward simulator with known
ground truth X_true (Day/Night contrast, 2002 Aqua, 2012 SNPP, 2018 N20,
2022 drift, 2023 N21, Nov 2026 SNPP end, 50% clouds, bow-tie, gas flares)
asserting: the 2012 raw jump (> +150%) is eliminated (< +-10%); 3-satellite
inflation eliminated; night_day_ratio preserved (never collapsing to 1.0);
the Nov 2026 cutoff unbiased; post-2022 drift corrected; monthly correlation
r > 0.90; the 5x drought spike preserved.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.validation.synthetic_twin import TwinConfig
from src.validation.twin_pipeline import twin_assertion_metrics
from tests.conftest import _full_twin


@pytest.fixture(scope="module")
def met():
    return twin_assertion_metrics(_full_twin())


# ---------------------------------------------------------------- structure
def test_07_twin_generates_all_structural_effects(full_twin):
    prov = full_twin["prov"]
    plats = set(prov["platform"].unique())
    assert {"Terra", "Aqua", "SNPP", "NOAA-20", "NOAA-21"} <= plats
    # Aqua begins July 2002
    aqua_first = prov[prov["platform"] == "Aqua"]["acq_date"].min()
    assert str(aqua_first)[:4] == "2002"
    # SNPP data ends 2026-11-02
    snpp_last = prov[prov["platform"] == "SNPP"]["acq_date"].max()
    assert str(snpp_last) <= "2026-11-02"
    # NOAA-20/21 continue through 2026-11-14
    for p in ("NOAA-20", "NOAA-21"):
        assert str(prov[prov["platform"] == p]["acq_date"].max()) \
            >= "2026-11-10"
    # static gas flares preserved in provenance (type 2, never destroyed)
    assert (prov["type_raw"] == 2).any()
    assert prov[prov["type_raw"] == 2]["is_static_industrial"].all()
    # bow-tie duplicated edge pixels exist
    assert (prov["bowtie_weight"] < 1.0).any()
    # Tobit-saturated pixels exist
    assert (prov["sat_flag"] == 1).any()
    # SAMA nighttime low-confidence artifacts exist and are flagged
    assert prov["is_sama_filtered"].any()


def test_07_raw_2012_jump_exceeds_150pct(met):
    assert met["raw_jump_2012_pct"] > 150.0


def test_07_harmonized_2012_step_within_10pct(met):
    assert abs(met["harm_step_2012_pct"]) < 10.0


def test_07_harmonized_step_tracks_truth(met):
    """The harmonized step matches the true step within 10 percentage
    points (raw is off by > 150 pp)."""
    assert abs(met["harm_step_2012_pct"] - met["true_step_2012_pct"]) < 10.0
    assert abs(met["raw_jump_2012_pct"] - met["true_step_2012_pct"]) > 150.0


def test_07_three_satellite_inflation_raw(met):
    """Raw counts inflate > +15% from 2013-2017 to 2019-2023."""
    assert met["raw_3sat_pct"] > 15.0


def test_07_three_satellite_inflation_eliminated(met):
    """Harmonized flux tracks the TRUE change within 5 pp (raw off by
    > 25 pp)."""
    assert abs(met["harm_3sat_pct"] - met["true_3sat_pct"]) < 5.0
    assert abs(met["raw_3sat_pct"] - met["true_3sat_pct"]) > 25.0


def test_07_nov_2026_cutoff_unbiased(met):
    """Oct->Nov 2026 raw counts crash (SNPP ends Nov 2); harmonized flux
    matches the true change far better."""
    assert met["raw_nov2026_pct"] < -60.0
    assert abs(met["harm_nov2026_pct"] - met["true_nov2026_pct"]) \
        < abs(met["raw_nov2026_pct"] - met["true_nov2026_pct"]) / 2


def test_07_post_2022_drift_corrected(met):
    """2023-2026 (drift era) harmonized level within 15% of truth."""
    assert 0.85 < met["harm_drift_ratio"] < 1.15


def test_07_monthly_correlation_above_090(met):
    assert met["log_corr"] > 0.90


def test_07_median_level_within_20pct(met):
    assert 0.8 < met["median_ratio"] < 1.2


def test_07_pyrome_levels_within_bounds(met):
    for b, ratio in met["pyrome_median_ratio"].items():
        assert 0.6 < ratio < 1.5, f"pyrome {b} level {ratio}"


# ---------------------------------------------------------------- ndr
def test_07_night_day_ratio_preserved(full_twin, met):
    """night_day_ratio recovered within 45% per cell and NEVER collapses
    toward 1.0 (the Day/Night states stay distinct)."""
    assert met["ndr_never_collapsed"]
    assert met["ndr_max_abs_err_pct"] < 45.0


def test_07_night_day_ratio_estimates(full_twin):
    m = full_twin["monthly"].merge(full_twin["truth"],
                                   on=["h3_index", "year", "month"])
    ndr_hat = m.groupby("h3_index").apply(
        lambda g: g["phi_night_mw"].mean()
        / max(g["phi_day_mw"].mean(), 1e-9), include_groups=False)
    ndr_true = m.groupby("h3_index").apply(
        lambda g: g["phi_night_true"].mean()
        / max(g["phi_day_true"].mean(), 1e-9), include_groups=False)
    # true twin ndr: agri ~0.08, forest ~0.20 -> estimates stay day-dominated
    assert (ndr_hat < 0.6).all()
    assert (ndr_hat > 0.01).all()
    # the two behavioural regimes stay separable (no collapse to one value)
    assert ndr_hat.std() > 0.02


# ---------------------------------------------------------------- spike
def test_07_drought_spike_preserved(met):
    """The 2015 5x drought fire spike is preserved (not oversmoothed)."""
    assert met["spike_true_x"] > 1.5
    assert met["spike_harm_x"] > 1.3
    assert 0.75 < met["spike_recovery"] < 1.25


def test_07_chilecito_2026_spotlight(full_twin):
    """CONAE July-Aug 2026 Chilecito wildfire (x8 local spike) is
    preserved in the harmonized record."""
    m = full_twin["monthly"].merge(full_twin["truth"],
                                   on=["h3_index", "year", "month"])
    spot = m[m["role"] == "chilecito"]
    aug_2026 = spot[(spot["year"] == 2026)
                    & (spot["month"].isin([7, 8]))]["phi_q50_mw"].sum()
    aug_2025 = spot[(spot["year"] == 2025)
                    & (spot["month"].isin([7, 8]))]["phi_q50_mw"].sum()
    t26 = spot[(spot["year"] == 2026)
               & (spot["month"].isin([7, 8]))]["phi24_true"].sum()
    t25 = spot[(spot["year"] == 2025)
               & (spot["month"].isin([7, 8]))]["phi24_true"].sum()
    raw_ratio = (aug_2026 / max(aug_2025, 1e-9)) / max(t26 / t25, 1e-9)
    assert 0.4 < raw_ratio < 2.0


def test_07_sensor_transitions_all_present(full_twin):
    """Every constellation transition is represented in the twin."""
    prov = full_twin["prov"]
    d = prov["acq_date"]
    assert (d >= "2000-11-01").all() and (d <= "2026-11-14").all()
    assert prov[prov["platform"] == "Terra"]["acq_date"].min() <= "2000-11-30"


def test_07_no_platform_summing_in_products(full_twin):
    """pass_observations are keyed per platform (INV-02), never summed."""
    pc = full_twin["pass_cal"]
    key_counts = pc.groupby(["h3_index", "solar_date", "dn_flag"]).size()
    multi = key_counts[key_counts > 1]
    assert len(multi) > 0            # same-day multi-platform rows exist ...
    sub = pc.set_index(["h3_index", "solar_date", "dn_flag"]).loc[
        multi.index[:5]]
    assert sub["platform"].nunique() > 1   # ... as SEPARATE rows
