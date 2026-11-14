"""Frozen global constants for the Fire Harmonizer engine.

Contract (readme.md Section 4, item 2):
* Contains ONLY frozen dataclasses / plain constants.
* Zero I/O logic, zero path hardcoding, zero heavy ML/GPU imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dates and temporal grid
# ---------------------------------------------------------------------------
DATE_START = "2000-11-01"          # t = 0  (first Terra MODIS fire record month)
DATE_END = "2026-11-14"            # t = 9509 (reference date)
T_DAYS = 9510                      # number of daily latent states per cell

CALIB_START = "2012-01-20"         # S-NPP first data day (pre-drift overlap window)
CALIB_END = "2021-12-31"           # end of stable-orbit calibration window

CLIM_START_YEAR = 2003             # historical climatology window
CLIM_END_YEAR = 2020

# ---------------------------------------------------------------------------
# Latent state bounds (log-flux of cloud-free mean radiative power, MW)
# ---------------------------------------------------------------------------
Z_MIN = math.log(0.01)             # dormant baseline floor = -4.605170...
Z_MAX = 14.0                       # hard upper bound (Phi ~ 1.2e6 MW)
F_REF_MW = 0.5                     # VIIRS cluster reference FRP for psi integrals

# ---------------------------------------------------------------------------
# Spatial grids
# ---------------------------------------------------------------------------
H3_MAIN_RES = 6                    # main harmonization grid (~36.13 km^2 mean)
H3_FLARE_RES = 8                   # static flare micro-cells (~0.74 km^2)

# South Atlantic Magnetic Anomaly box: (min_lon, min_lat, max_lon, max_lat)
SAMA_BOX = (-110.0, -55.0, 11.0, 7.0)

# ---------------------------------------------------------------------------
# Solver numerics
# ---------------------------------------------------------------------------
MC_SAMPLES = 200                   # joint posterior trajectories
SOLVER_SWEEPS = 6                  # 3-color x Day/Night Gauss-Seidel sweeps (minimum)
# Convergence guard: after the 6 spec sweeps, keep sweeping until the max
# per-sweep state change < SWEEP_TOL (block Gauss-Seidel over the Day/Night
# + spatial couplings converges linearly, ~0.8/sweep on the twin).
MAX_SOLVER_SWEEPS = 40
SWEEP_TOL = 1e-4
NEWTON_STEPS = 15                  # damped Newton-Raphson iterations per block
BATCH_SIZE = 1000                  # active cells per GPU chunk (< 6 GB VRAM)
ARMIJO_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)
ARMIJO_C1 = 1e-4
NEWTON_TOL = 1e-4
DIAG_FLOOR = 1e-7                  # Thomas LDL^T d-floor
PRIOR_DIAG_FLOOR = 1e-6
SPATIAL_ALPHA_S = 0.10             # spatial 1-ring coupling strength
FIVE_MINUTES = 5.0                 # pass segmentation gap (minutes)

# 24h mean flux factors
FRE_TJ_PER_MW_DAY = 0.0864         # 1 MW sustained 24h = 0.0864 TJ

# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------
PLATFORM_INDEX = {"Terra": 0, "Aqua": 1, "SNPP": 2, "NOAA-20": 3, "NOAA-21": 4}
PLATFORM_BITMASK = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}
N_PLATFORMS = 5

MODIS_PLATFORMS = ("Terra", "Aqua")
VIIRS_PLATFORMS = ("SNPP", "NOAA-20", "NOAA-21")
SENSOR_FAMILY = {
    "Terra": "MODIS",
    "Aqua": "MODIS",
    "SNPP": "VIIRS",
    "NOAA-20": "VIIRS",
    "NOAA-21": "VIIRS",
}

# (start_date, end_date) operational delivery windows (INV-04).
PLATFORM_TIMELINE = {
    "Terra": ("2000-11-01", "2026-11-14"),
    "Aqua": ("2002-07-04", "2026-11-14"),
    "SNPP": ("2012-01-20", "2026-11-02"),   # FIRMS ceased S-NPP delivery Nov 2, 2026
    "NOAA-20": ("2018-04-01", "2026-11-14"),
    "NOAA-21": ("2023-03-01", "2026-11-14"),
}

# Satellite alias normalization table (readme Section 6.3.1)
SATELLITE_ALIASES = {
    "T": "Terra", "TERRA": "Terra", "TERRA/MODIS": "Terra",
    "A": "Aqua", "AQUA": "Aqua", "AQUA/MODIS": "Aqua",
    "N": "SNPP", "SNPP": "SNPP", "NPP": "SNPP", "SUOMI-NPP": "SNPP",
    "SUOMI NPP": "SNPP", "S-NPP": "SNPP",
    "1": "NOAA-20", "J1": "NOAA-20", "N20": "NOAA-20", "NOAA-20": "NOAA-20",
    "NOAA20": "NOAA-20",
    "2": "NOAA-21", "J2": "NOAA-21", "N21": "NOAA-21", "NOAA-21": "NOAA-21",
    "NOAA21": "NOAA-21",
}

# Nominal / drifting expected overpass solar hours per (platform, dn)
# Terra drifts 10.5 -> 8.75 over 2022-01-01..2026-11-14; Aqua 13.5 -> 15.60.
PLATFORM_SOLAR_HOUR = {
    "Terra": 10.5,
    "Aqua": 13.5,
    "SNPP": 13.5,
    "NOAA-20": 13.5,
    "NOAA-21": 13.5,
}
DRIFT_START = "2022-01-01"
TERRA_DRIFT_END_HOUR = 8.75
AQUA_DRIFT_END_HOUR = 15.60

# Data tier dominance (lower rank wins): ML Science > SP Science > NRT
TIER_RANK = {"ML_SCIENCE": 0, "SP_SCIENCE": 1, "NRT": 2}

# ---------------------------------------------------------------------------
# F_min detection-limit physical anchors: F_min(A, DN) = alpha + beta * A  [MW]
# ---------------------------------------------------------------------------
FMIN_PARAMS = {
    ("MODIS", "N"): (1.5, 2.2),
    ("MODIS", "D"): (2.5, 3.9),
    ("VIIRS", "N"): (0.25, 1.8),
    ("VIIRS", "D"): (0.40, 3.4),
}

# Expected (mean area km^2, mean VZA deg) for unseen passes
PASS_GEOMETRY_AVG = {
    "MODIS": (2.35, 35.0),
    "VIIRS": (0.23, 28.0),
}

# Swath visibility model
PSWATH_VIIRS = 0.99
PSWATH_MODIS_BASE = 0.85
PSWATH_MODIS_SLOPE = 0.15          # * |lat| / 30, clipped to [0.85, 1.00]
PSWATH_MODIS_POST2022 = 0.96       # multiplier for t >= 2022-01-01

# VIIRS I4 / M13 fallback saturation thresholds (K)
VIIRS_I4_SAT_K = 366.0
VIIRS_I4_M13FB_LO = 366.0          # i4_m13_fallback: t_mir >= 366.0 & frp > 0
VIIRS_I4_M13FB_HI = 366.9          # Tobit sat: t_mir >= 366.9 & frp >= 300
VIIRS_TOBIT_FRP_MIN = 300.0
VIIRS_M13_SIGMA_INFLATION = 1.35

# MODIS Ch21/22 saturation (Tobit right-censoring)
MODIS_SAT_K = 498.0
MODIS_SAT_T21_NOMINAL = 500.0      # Ch21 saturates near 500 K

# Bow-tie / clustering thresholds
BOWTIE_AREA_KM2 = 1.5              # MODIS pixels above this get bow-tie weighting
BOWTIE_TRACK_FACTOR = 0.7          # d_ij < 0.7 * mean(track_i, track_j)
CLUSTER_D_FACTOR = 1.4             # d_ij <= 1.4 * max(sqrt(A_i), sqrt(A_j))

# Static industrial flare persistence thresholds (H3-Res8 micro-cells)
FLARE_MIN_YEARS = 3
FLARE_MIN_DAYS_PER_YEAR = 15.0
FLARE_MIN_DN_SYMMETRY = 0.30

# Cloud visibility clip bounds
VIS_CLIP = (0.10, 0.98)

# Observation-noise and survival clip bounds
SIGMA2_CLIP = (0.025, 2.50)
PSI_CNT_CLIP = (0.02, 1.0)
PSI_FRP_CLIP = (0.05, 1.0)
LN_GAMMA_CLIP = (-1.5, 1.5)
RELIABILITY_CLIP = (0.15, 1.0)
W_MULTI_RATE = 0.7                 # w_multi(E) = 1 - exp(-0.7 (E-1))
PSI_EFF_ISOLATED_FLOOR = 0.65

# Prior parameter bounds
RHO_CLIP = (0.25, 0.90)
SIGMA_PROC_CLIP = (0.25, 1.50)
LAMBDA0_CLIP = (0.1, 5.0)
ETA_DN_CLIP = (0.05, 0.30)
SHRINKAGE_WEIGHT = 5.0             # Empirical Bayes cell-month prior weight
F_EVENT_FLOOR_MW = 1.0

# Conformal calibration
CONFORMAL_C_CLIP = (0.85, 3.5)
CONFORMAL_S_FLOOR = 0.05
ABSTAIN_CONSEC_DAYS = 5
ABSTAIN_CLOUD_FRAC = 0.85

# SZA day/night physical flag threshold (stored for provenance)
SZA_DAY_THRESHOLD_DEG = 85.0
# Canonical latent half-day windows by local solar time
DAY_WINDOW = (6.0, 18.0)           # Day  = [06:00, 18:00)
NIGHT_WINDOW = (18.0, 30.0)        # Night = [18:00, 06:00)

# Diurnal curve normalization grid (120-point trapezoid per 12h window)
DIURNAL_GRID_POINTS = 120

# Spherical Earth pixel geometry constants
R_EARTH_KM = 6371.0
MODIS_ALT_KM = 705.0
VIIRS_ALT_KM = 824.0
MODIS_N_SAMPLES = 1354             # s in [0, 1353]
MODIS_HALF_SWATH_DEG = 55.0
MODIS_IFOV_RAD = 1.0 / 705.0       # Delta alpha_0
VIIRS_N_SAMPLES = 6400             # s in [0, 6399]
VIIRS_HALF_SWATH_DEG = 56.06
VIIRS_IFOV_TRACK_KM = 0.375        # along-track nadir IFOV (km)
# VIIRS onboard aggregation zones: (max |alpha| deg, aggregated step multiplier)
VIIRS_AGG_ZONES = ((31.59, 3), (44.68, 2), (56.06, 1))

# Monotone VZA-from-area interpolation knots: area km^2 -> VZA deg
MODIS_AREA_VZA_KNOTS = ([1.00, 1.35, 2.25, 4.20, 9.72], [0.0, 20.0, 35.0, 50.0, 65.0])
VIIRS_AREA_VZA_KNOTS = ([0.141, 0.180, 0.280, 0.430, 0.620], [0.0, 20.0, 35.0, 50.0, 65.0])
AREA_CLIP_KM2 = (0.12, 12.0)
VIIRS_PIXAREA_CLIP = (0.12, 0.75)

# Pyrome GMM
PYROME_K = 7
GMM_REG_COVAR = 1e-4
GMM_RANDOM_STATE = 42
PYROME_MIN_ACTIVE_DAYS = 20

# Hurdle-adj events caps
MEASURED_FRACTION_FLOOR = 0.25
HURDLE_EVENTS_CAP_FACTOR = 5.0

# ---------------------------------------------------------------------------
# Target countries (readme Section 3.3 / 4.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Country:
    iso3: str
    name: str
    # Prefilter bounding boxes (min_lon, min_lat, max_lon, max_lat); exact
    # clipping is done against Natural Earth Admin-0 polygons downstream.
    prefilter_boxes: tuple = ()


TARGET_COUNTRIES = (
    Country("ARG", "Argentina", ((-73.8, -55.5, -53.5, -21.5),)),
    Country("DZA", "Algeria", ((-9.0, 18.0, 12.5, 38.0),)),
    Country("BRA", "Brazil", ((-74.5, -34.0, -34.0, 5.5),)),
    Country("IDN", "Indonesia", ((94.5, -11.5, 141.5, 6.5),)),
    Country("AUS", "Australia", ((112.0, -44.5, 154.5, -9.5),)),
    Country("USA", "United States (CONUS + Alaska)",
            ((-125.5, 24.0, -66.5, 50.0), (-170.0, 51.0, -129.0, 72.0))),
)
TARGET_COUNTRY_ISOS = tuple(c.iso3 for c in TARGET_COUNTRIES)

# ---------------------------------------------------------------------------
# Spotlight presets (exact coordinates from readme Section 5.5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Spotlight:
    key: str
    label: str
    bbox: tuple  # (min_lon, min_lat, max_lon, max_lat)
    country_iso: str


SPOTLIGHT_PRESETS = (
    Spotlight("chilecito_arg_2026", "Chilecito ARG 2026",
              (-68.0, -29.6, -67.1, -28.7), "ARG"),
    Spotlight("kabylie_dza", "Kabylie DZA",
              (3.8, 36.3, 5.5, 37.0), "DZA"),
    Spotlight("hassi_messaoud_flares", "Hassi Messaoud Gas Flares DZA",
              (5.5, 31.3, 6.6, 32.2), "DZA"),
    Spotlight("alaska_boreal", "Alaska Boreal USA",
              (-156.0, 63.0, -142.0, 67.5), "USA"),
    Spotlight("black_summer_aus", "Black Summer AUS",
              (147.0, -37.8, 151.0, -34.5), "AUS"),
    Spotlight("kalimantan_peat", "Kalimantan Peat IDN",
              (112.5, -3.3, 115.2, -1.5), "IDN"),
)

# CONAE Chilecito wildfire activation (July-August 2026 spotlight)
CHILECITO_LAT = -29.16
CHILECITO_LON = -67.50

# Constellation transition events for the web app timeseries view
CONSTELLATION_EVENTS = (("2002-07", "Aqua launch"),
                        ("2012-01", "S-NPP launch"),
                        ("2018-04", "NOAA-20 launch"),
                        ("2022-01", "Terra/Aqua drift start"),
                        ("2023-03", "NOAA-21 launch"),
                        ("2026-11", "S-NPP delivery end"))

# Web app limits
MAX_BBOX_SPAN_DEG = 25.0
MAX_HEX_FEATURES = 3000
API_LATENCY_BUDGET_MS = 300.0

# Sparse daily export filter (readme Section 5.4.4)
DAILY_SPARSE_PHI_MIN = 0.25
DAILY_SPARSE_DETECTION_PAD_DAYS = 3

# Gridded ancillary products (0.25 deg CMG)
CMG_GRID_SHAPE = (720, 1440)
CMG_UPPER_LEFT = (-179.875, 89.875)

# Mirror / API endpoints (Env 1 only; used by src/ingest/download_data.py)
UMD_FTP_HOST = "fuoco.geog.umd.edu"
UMD_FTP_USER = "fire"
UMD_FTP_PASS = "burnt"
PORTOLAN_BASE = "https://data.source.coop/portolan-mirrors/firms-catalog"
PORTOLAN_CATALOG_URL = PORTOLAN_BASE + "/catalog.json"
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
GEOBOUNDARIES_URL = "https://www.geoboundaries.org/api/current/gbOpen/{ISO3}/ADM0/"
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api"
NASA_CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
