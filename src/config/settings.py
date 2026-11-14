"""Global frozen constants for the MODIS-VIIRS harmonization engine.

Contract (readme.md Section 4, `src/config/settings.py`):
  * Pure constants & frozen dataclasses: TARGET_COUNTRIES, PLATFORM_TIMELINE,
    DATE_START, DATE_END, T_DAYS, CALIB_START, CALIB_END, Z_MIN, Z_MAX, F_REF_MW,
    SAMA_BOX, H3_MAIN_RES, H3_FLARE_RES, MC_SAMPLES, SOLVER_SWEEPS, NEWTON_STEPS,
    SPOTLIGHT_PRESETS, ...
  * Zero I/O logic, zero path hardcoding, zero heavy ML/GPU imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Temporal coverage (reference date: 2026-11-14)
# --------------------------------------------------------------------------- #
DATE_START = "2000-11-01"
DATE_END = "2026-11-14"
T_DAYS = 9510  # days from 2000-11-01 (t=0) through 2026-11-14 (t=9509)

# Pre-drift stable-orbit calibration window (INV-03).
CALIB_START = "2012-01-20"
CALIB_END = "2021-12-31"

# Historical climatology baseline for calendar anomaly indices.
CLIM_START_YEAR = 2003
CLIM_END_YEAR = 2020

# Cloud climatology fallback window for trailing-2026 CMG lag.
CLOUD_CLIM_START_YEAR = 2012
CLOUD_CLIM_END_YEAR = 2025

# --------------------------------------------------------------------------- #
# Latent-state bounds and physics anchors
# --------------------------------------------------------------------------- #
Z_MIN = math.log(0.01)  # dormant background floor: ln(0.01 MW) = -4.605170...
Z_MAX = 14.0            # upper latent log-flux bound
Z_CLIP_LO = -8.0        # solver clip range for z (readme.md 5.3.4)
Z_CLIP_HI = 14.0
PHI_FLOOR_MW = 0.01     # e**Z_MIN in MW
F_REF_MW = 0.5          # reference cluster FRP for left-truncated Log-Normal fit
FRE_TJ_PER_MW_DAY = 0.0864  # FRE (TJ/day) = 0.0864 * Phi (MW) [86.4 s/day * 1e-3]

# SAMA bounding box: (lon_min, lat_min, lon_max, lat_max).
SAMA_BOX = (-110.0, -55.0, 11.0, 7.0)

# SZA threshold retained for provenance Day/Night physics (raw sensor products
# use orbit-node; Science ML uses SZA < 85 deg). The canonical latent half-day
# assignment is the local-solar-time window (readme.md 5.1.1 item 3).
SZA_DAY_NIGHT_THRESHOLD_DEG = 85.0

# --------------------------------------------------------------------------- #
# H3 spatial grid
# --------------------------------------------------------------------------- #
H3_MAIN_RES = 6   # harmonization support grid (~36.13 km^2 mean hexagon area)
H3_FLARE_RES = 8  # static industrial gas-flare micro-cells (~0.74 km^2)

# --------------------------------------------------------------------------- #
# Solver numerics
# --------------------------------------------------------------------------- #
MC_SAMPLES = 200        # exact joint Cholesky posterior trajectories (M)
SOLVER_SWEEPS = 6       # outer 3-color x Day/Night Gauss-Seidel sweeps
NEWTON_STEPS = 15       # max damped Newton iterations per block
BATCH_SIZE = 1000       # active cells per GPU chunk (VRAM < 6 GB target)
ARMIJO_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625)
ARMIJO_C1 = 1e-4
NEWTON_TOL = 1e-4          # early exit when ||alpha*delta_z||_inf < tol
DIAG_FLOOR = 1e-6          # tridiagonal main-diagonal floor
PIVOT_FLOOR = 1e-7         # Thomas pivot floor d_{t+1} = max(..., 1e-7)
ALPHA_SPATIAL = 0.10       # alpha_s: spatial 1-ring coupling strength
LAMBDA_CLIP = (1e-12, 80.0)  # clip range for Hurdle Lambda = k e^z
TOBIT_TAIL_W = -30.0       # w below this uses the exact asymptotic expansion
SPATIAL_AREA_MODEL = "1-ring"  # neighbor coupling graph used by the GMRF

# --------------------------------------------------------------------------- #
# Platforms
# --------------------------------------------------------------------------- #
PLATFORM_INDEX: Dict[str, int] = {
    "Terra": 0,
    "Aqua": 1,
    "SNPP": 2,
    "NOAA-20": 3,
    "NOAA-21": 4,
}
PLATFORM_BITMASK: Dict[int, int] = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}
PLATFORMS: Tuple[str, ...] = tuple(PLATFORM_INDEX.keys())
N_PLATFORMS = len(PLATFORMS)  # P = 5

# Constellation transition months surfaced by /api/timeseries.
CONSTELLATION_EVENTS: Tuple[str, ...] = (
    "2002-07",  # Aqua first data
    "2012-01",  # S-NPP first data (2012-01-20)
    "2018-04",  # NOAA-20 first data
    "2022-01",  # Terra/Aqua exit station-keeping (drift begins)
    "2023-03",  # NOAA-21 first data
    "2026-11",  # S-NPP FIRMS delivery ceases (2026-11-02)
)


@dataclass(frozen=True)
class PlatformWindow:
    """Operational calendar for one platform (INV-04)."""

    name: str
    first_date: str
    last_date: str  # inclusive; "None" means open-ended through DATE_END
    sensor_family: str  # "MODIS" or "VIIRS"
    nominal_day_solar_hour: float   # expected Day half-day solar hour (pre-drift)
    nominal_night_solar_hour: float  # expected Night half-day solar hour


PLATFORM_TIMELINE: Dict[str, PlatformWindow] = {
    "Terra": PlatformWindow("Terra", "2000-11-01", "2026-11-14", "MODIS", 10.5, 22.5),
    "Aqua": PlatformWindow("Aqua", "2002-07-04", "2026-11-14", "MODIS", 13.5, 1.5),
    "SNPP": PlatformWindow("SNPP", "2012-01-20", "2026-11-02", "VIIRS", 13.5, 1.5),
    "NOAA-20": PlatformWindow("NOAA-20", "2018-04-01", "2026-11-14", "VIIRS", 13.5, 1.5),
    "NOAA-21": PlatformWindow("NOAA-21", "2023-03-01", "2026-11-14", "VIIRS", 13.5, 1.5),
}

# Post-2022 orbital drift of the MODIS constellation (readme.md 5.1.4 item 4):
# expected solar hour drifts linearly between 2022-01-01 and 2026-11-14.
DRIFT_START_DATE = "2022-01-01"
TERRA_DRIFT_HOURS: Tuple[float, float] = (10.5, 8.75)   # 2022-01-01 -> 2026-11-14
AQUA_DRIFT_HOURS: Tuple[float, float] = (13.5, 15.60)   # 2022-01-01 -> 2026-11-14

# Expected unseen-pass geometry (readme.md 5.1.4 item 3).
MODIS_AVG_AREA_KM2 = 2.35
MODIS_AVG_VZA_DEG = 35.0
VIIRS_AVG_AREA_KM2 = 0.23
VIIRS_AVG_VZA_DEG = 28.0

# MODIS swath degradation factor applied to p_swath from 2022-01-01.
MODIS_DRIFT_SWATH_FACTOR = 0.96

# --------------------------------------------------------------------------- #
# Data-tier dominance (readme.md 6.3 item 4)
# --------------------------------------------------------------------------- #
DATA_TIERS: Tuple[str, ...] = ("ML_SCIENCE", "SP_SCIENCE", "NRT")
TIER_RANK: Dict[str, int] = {"ML_SCIENCE": 0, "SP_SCIENCE": 1, "NRT": 2}

# Confidence mapping for low/nominal/high textual labels (readme.md 6.3 item 2).
CONF_LABEL_VALUES: Dict[str, float] = {"l": 20.0, "n": 65.0, "h": 95.0}

# platform_active_day threshold: platform inactive on a UTC day when total
# detections across the 6 target countries < 5 (spacecraft safe-mode masking).
PLATFORM_ACTIVE_DAY_MIN_DETECTIONS = 5

# --------------------------------------------------------------------------- #
# Geometry / clustering thresholds (readme.md 5.1.2 / 5.1.3)
# --------------------------------------------------------------------------- #
EARTH_RADIUS_KM = 6371.0
MODIS_ALTITUDE_KM = 705.0
VIIRS_ALTITUDE_KM = 824.0
MODIS_HALF_SWATH_DEG = 55.0
MODIS_SAMPLES = 1354          # samples 0..1353
MODIS_CENTER_SAMPLE = 676.5
MODIS_IFOV_RAD = 1.0 / 705.0  # Delta alpha_0
VIIRS_SAMPLES = 6400          # samples 0..6399
VIIRS_HALF_SWATH_DEG = 56.06
VIIRS_AGG_ZONES_DEG = (0.0, 31.59, 44.68, 56.06)  # 3:1 / 2:1 / 1:1 boundaries
VIIRS_PIXAREA_CLIP = (0.12, 0.75)
SCAN_TRACK_AREA_CLIP = (0.12, 12.0)

# Monotone area -> VZA lookup knots (np.interp), readme.md 5.1.2:
MODIS_AREA_VZA_KNOTS: Tuple[Tuple[float, float], ...] = (
    (1.00, 0.0), (1.35, 20.0), (2.25, 35.0), (4.20, 50.0), (9.72, 65.0)
)
VIIRS_AREA_VZA_KNOTS: Tuple[Tuple[float, float], ...] = (
    (0.141, 0.0), (0.180, 20.0), (0.280, 35.0), (0.430, 50.0), (0.620, 65.0)
)

PASS_SPLIT_MINUTES = 5.0        # >5 min gap starts a new overpass
BOWTIE_AREA_THRESHOLD_KM2 = 1.5  # MODIS bow-tie weighting applies when A_i > 1.5
BOWTIE_TRACK_FACTOR = 0.7       # connect if d_ij < 0.7 * mean(track_i, track_j)
CLUSTER_DISTANCE_FACTOR = 1.4   # connect if d_ij <= 1.4 * max(sqrt(A_i), sqrt(A_j))

# Static industrial gas-flare rule (H3-Res8 micro-cells, readme.md 5.1.3).
FLARE_MIN_YEARS = 3
FLARE_MIN_ACTIVE_DAYS_PER_YEAR = 15.0
FLARE_MIN_DAY_NIGHT_RATIO = 0.30

# Saturation classification (readme.md 5.1.3 item 3).
VIIRS_I4_SATURATION_K = 366.0        # i4_m13_fallback lower edge
VIIRS_I4_TOBIT_K = 366.9             # Tobit right-censoring edge
VIIRS_TOBIT_MIN_FRP_MW = 300.0
MODIS_CH21_SATURATION_K = 498.0      # Channel 21 saturates near 500 K
I4_M13_SIGMA_INFLATION = 1.35        # multiplies observation noise sigma

# --------------------------------------------------------------------------- #
# Observation-operator calibration anchors (readme.md 5.2)
# --------------------------------------------------------------------------- #
# F_min(A, DN) = alpha + beta * A  per (sensor_family, dn_flag)
FMIN_PARAMS: Dict[str, Tuple[float, float]] = {
    ("MODIS", "N"): (1.5, 2.2),
    ("MODIS", "D"): (2.5, 3.9),
    ("VIIRS", "N"): (0.25, 1.8),
    ("VIIRS", "D"): (0.40, 3.4),
}
PSI_CNT_CLIP = (0.02, 1.0)
PSI_FRP_CLIP = (0.05, 1.0)
PSI_FRP_EFF_FLOOR = 0.65      # isolated single-cluster detection blend floor
W_MULTI_RATE = 0.7            # w_multi(E) = 1 - exp(-0.7 (E-1))
LOGNORM_FIT_VZA_MAX_DEG = 20.0
LOGNORM_FIT_CLOUD_MAX = 0.30
RELIABILITY_CLIP = (0.15, 1.0)
EQM_KNOTS = 20
LN_GAMMA_CLIP = (-1.5, 1.5)
GAMMA_VZA_REF_MAX_DEG = 25.0
GAMMA_PAIR_MAX_DT_MIN = 45.0
DIURNAL_RIDGE_ALPHA = 10.0
DIURNAL_INT_POINTS = 120
SIGMA2_CLIP = (0.025, 2.50)
HURDLE_LAMBDA0_RANGE = (0.1, 5.0)
AR1_RHO_RANGE = (0.25, 0.90)
SIGMA_PROC_RANGE = (0.25, 1.50)
ETA_DN_RANGE = (0.05, 0.30)
EB_SHRINKAGE_WEIGHT = 5.0
PYROME_K = 7
PYROME_MIN_PASS_DAYS = 20
GMM_RANDOM_STATE = 42
GMM_REG_COVAR = 1e-4
F_EVENT_MIN_MW = 1.0
CORROBORATION_WINDOW_DAYS = 1
CORROBORATION_RADIUS_KM = 2.0

# Conformal calibration (readme.md 5.4).
CONFORMAL_ALPHA = 0.10
CONFORMAL_C_RANGE = (0.85, 3.5)
CONFORMAL_MIN_SCORE = 0.05
CQR_MODIS_FOLD = ("2018-01-01", "2020-12-31")
CQR_MULTI_FOLD = ("2018-01-01", "2021-12-31")

# Diagnostic flags.
ABSTAIN_CONSECUTIVE_DAYS = 5
ABSTAIN_CLOUD_FRAC = 0.85

# Sparse daily export filter (readme.md 5.4 item 4).
DAILY_EXPORT_PHI_MIN_MW = 0.25
DAILY_EXPORT_DETECTION_WINDOW_DAYS = 3

# --------------------------------------------------------------------------- #
# Target countries (readme.md Section 2.3 / 4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CountrySpec:
    """One benchmark country: ISO_A3 code plus coarse prefilter boxes.

    Prefilter boxes are (min_lon, min_lat, max_lon, max_lat) tuples used ONLY
    as a cheap pre-stage row filter in Environment 1; authoritative clipping
    is always exact Natural Earth Admin-0 polygon containment.
    """

    iso3: str
    prefilter_boxes: Tuple[Tuple[float, float, float, float], ...]


TARGET_COUNTRIES: Dict[str, CountrySpec] = {
    "ARG": CountrySpec("ARG", ((-73.8, -55.5, -53.5, -21.5),)),
    "DZA": CountrySpec("DZA", ((-8.7, 18.9, 12.0, 37.2),)),
    "BRA": CountrySpec("BRA", ((-74.0, -33.8, -34.7, 5.4),)),
    "IDN": CountrySpec("IDN", ((95.2, -11.0, 141.1, 6.3),)),
    "AUS": CountrySpec("AUS", ((112.9, -43.7, 153.7, -10.6),)),
    "USA": CountrySpec(
        "USA",
        (
            (-124.9, 24.4, -66.9, 49.5),   # CONUS
            (-179.2, 51.1, -129.9, 71.6),  # Alaska
        ),
    ),
}
TARGET_COUNTRY_ISOS: Tuple[str, ...] = tuple(TARGET_COUNTRIES.keys())

# --------------------------------------------------------------------------- #
# Spotlight presets (exact coordinates from readme.md Section 5.5)
# --------------------------------------------------------------------------- #
SPOTLIGHT_PRESETS: Dict[str, Dict[str, object]] = {
    "Chilecito ARG 2026": {
        "bbox": (-68.0, -29.6, -67.1, -28.7),
        "country_iso": "ARG",
        "center": (-67.5, -29.16),
    },
    "Kabylie DZA": {
        "bbox": (3.8, 36.3, 5.5, 37.0),
        "country_iso": "DZA",
        "center": (4.6, 36.65),
    },
    "Hassi Messaoud Gas Flares DZA": {
        "bbox": (5.5, 31.3, 6.6, 32.2),
        "country_iso": "DZA",
        "center": (6.05, 31.75),
    },
    "Alaska Boreal USA": {
        "bbox": (-156.0, 63.0, -142.0, 67.5),
        "country_iso": "USA",
        "center": (-149.0, 65.25),
    },
    "Black Summer AUS": {
        "bbox": (147.0, -37.8, 151.0, -34.5),
        "country_iso": "AUS",
        "center": (149.0, -36.15),
    },
    "Kalimantan Peat IDN": {
        "bbox": (112.5, -3.3, 115.2, -1.5),
        "country_iso": "IDN",
        "center": (113.85, -2.4),
    },
}

# --------------------------------------------------------------------------- #
# Web app constraints (readme.md 5.5)
# --------------------------------------------------------------------------- #
MAX_BBOX_SPAN_DEG = 25.0       # bbox span limit unless country_iso supplied
MAX_MAP_HEX_FEATURES = 3000    # /api/map_hex feature cap
API_LATENCY_BUDGET_MS = 300.0

# Ingestion source endpoints (URL constants only; no I/O executed here).
UMD_FUOCO_HOST = "fuoco.geog.umd.edu"
UMD_FUOCO_USER = "fire"
UMD_FUOCO_PASSWORD_ENV = "UMD_FUOCO_PASSWORD"  # password "burnt" read from env
PORTOLAN_STAC_URL = (
    "https://data.source.coop/portolan-mirrors/firms-catalog/catalog.json"
)
PORTOLAN_DETECTION_URL = (
    "https://data.source.coop/portolan-mirrors/firms-catalog/"
    "detections/year={year}/detections.parquet"
)
PORTOLAN_S3_URL = (
    "s3://us-west-2.opendata.source.coop/portolan-mirrors/firms-catalog/"
    "detections/year=*/detections.parquet"
)
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov"
FIRMS_MAP_KEY_ENV = "FIRMS_MAP_KEY"
FIRMS_MAX_DAY_RANGE = 5
FIRMS_RATE_LIMIT_SLEEP_S = 0.15
FIRMS_MAX_BBOX_SPAN_DEG = 25.0
NATURAL_EARTH_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
GEOBOUNDARIES_API_URL = (
    "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM0/"
)
NASA_CMR_GRANULE_URL = (
    "https://cmr.earthdata.nasa.gov/search/granules.json?short_name={short_name}"
    "&page_size=2000"
)

# HDF4 CMG grid geometry (readme.md Section 6.1): 720x1440 grid, 0.25 deg,
# upper-left cell center (-179.875, +89.875).
CMG_GRID_ROWS = 720
CMG_GRID_COLS = 1440
CMG_CELL_DEG = 0.25
CMG_LON0 = -179.875
CMG_LAT0 = 89.875

# Science-tier lag beyond which cloud grids fall back to climatology.
CLOUD_CLIMATOLOGY_FLAG = "CLIMATOLOGY_FALLBACK"

# Vxx14IMGML file-prefix -> platform inference (readme.md 6.1 item 1).
VIIRS_FILE_PREFIX_PLATFORM: Dict[str, str] = {
    "VNP": "SNPP",
    "VJ1": "NOAA-20",
    "VJ2": "NOAA-21",
}

# Satellite alias normalization map (readme.md 6.3 item 1).
SATELLITE_ALIASES: Dict[str, str] = {
    "T": "Terra",
    "TERRA": "Terra",
    "TERRA/AQUA": "Terra",
    "A": "Aqua",
    "AQUA": "Aqua",
    "N": "SNPP",
    "SNPP": "SNPP",
    "S-NPP": "SNPP",
    "SUOMI-NPP": "SNPP",
    "NPP": "SNPP",
    "1": "NOAA-20",
    "J1": "NOAA-20",
    "N20": "NOAA-20",
    "NOAA-20": "NOAA-20",
    "NOAA20": "NOAA-20",
    "2": "NOAA-21",
    "J2": "NOAA-21",
    "N21": "NOAA-21",
    "NOAA-21": "NOAA-21",
    "NOAA21": "NOAA-21",
}
