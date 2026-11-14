# PART I — Complete Decision Record, Technical Evolution, and Brutally Honest Audit

---

## 1. Main Problem Definition, Underlying Challenges, Constraints, and Objectives

### 1.1 The Scientific and Practical Problem
The project addresses the **2026 NASA Space Apps Challenge: *"Harmonization of MODIS and VIIRS Hot Spots"*** (authored by Marina Compagnucci, CONAE). The objective is twofold:
1. **Scientific Data Reconstruction:** Synthesize over 26 years (November 2000 – November 14, 2026) of active-fire thermal anomaly records across MODIS (`Terra`, `Aqua`) and VIIRS (`Suomi-NPP`, `NOAA-20`, `NOAA-21`) into a single, physically consistent, uncertainty-quantified dataset where sensor transitions no longer induce artificial jumps.
2. **Operational Decision Support:** Expose the harmonized dataset inside an interactive **Burning Activity Calendar Web Application** allowing emergency managers, scientists, and land managers to select an area of interest (map bounding box or coordinates), examine a Year × Month calendar heatmap, compare current burning activity against historical medians, visualize trends and anomalies, inspect uncertainty, and download CSV analytics.

### 1.2 Why Existing Data Cannot Be Merged Directly
Concatenating or rescaling MODIS and VIIRS hotspot tables fails for physical, geometric, orbital, and algorithmic reasons:
* **Spatial Resolution & Detection Threshold Discontinuity:** MODIS detects at nominal `1 km` nadir (`~1.0 km²`, growing to `~9.6–9.7 km²` at `65°` view zenith angle [VZA]), with minimum detectable Fire Radiative Power ($F_{\min}$) ranging from `~3.7 MW` (nadir night) up to `~40 MW` (off-nadir day). VIIRS 375 m (`I-band`) detects at `~0.14 km²` nadir (`~0.62 km²` off-nadir), reaching `~0.5–2.5 MW`. Switching from MODIS to VIIRS in 2012 artificially multiplies raw hotspot counts by `3×–10×` even when physical burning is flat.
* **Spectral Bandpass & Saturation Asymmetry:** MODIS uses `3.96 μm` (Channels 21/22, saturating near `500 K` / `331 K`). VIIRS uses `3.74 μm` (`I4`, saturating at `367 K`), relying on co-located `750 m` `M13` (`4.05 μm`) pixels for FRP retrieval, which causes shared fractional FRP across adjacent 375 m pixels or null/zero FRP when M13/background retrieval fails. Raw brightness temperatures (`brightness`/`bright_t31` vs `bright_ti4`/`bright_ti5`) are physically non-comparable.
* **Scan-Edge Artifacts:** MODIS exhibits severe **bow-tie overlap** at high scan angles (adjacent scans overlap up to ~50%, double-counting the same fire pixels), whereas VIIRS applies **onboard pixel aggregation (`3:1`, `2:1`, `1:1`)** plus bow-tie pixel deletion.
* **Constellation Multiplicity:** From 2000–2002 only `Terra` operated; `Aqua` joined in July 2002; `Suomi-NPP` joined January 2012; `NOAA-20` joined 2018; `NOAA-21` joined 2023/2024; and `Suomi-NPP` ceased FIRMS delivery on **November 2, 2026**. `SNPP`, `NOAA-20`, and `NOAA-21` share the ~13:30 / 01:30 orbital plane (~50 minutes apart), meaning summing raw detections across afternoon platforms counts the same burning fire multiple times.
* **Post-2022 Orbital Clock Drift:** Starting in 2022, `Terra` (10:30 AM nominal) and `Aqua` (1:30 PM nominal) exited station-keeping, drifting progressively toward ~8:45 AM and ~3:30–3:50 PM respectively, confounding diurnal fire cycles with long-term trends.
* **Clouds, Quality Tiers, and Non-Biomass Heat Sources:** Equatorial cloud cover drops single-pass detection probability below 30–40%; `NRT` streams drop the `type` column and assign `daynight` purely by ascending/descending orbit node rather than Solar Zenith Angle (`SZA < 85°`) used in Science `ML` products; South Atlantic Magnetic Anomaly (SAMA) nighttime artifacts and persistent Saharan gas flares contaminate raw counts.
* **Fundamental Identifiability Limit:** Orbital thermal radiance (FRP) alone cannot uniquely invert sub-pixel flame temperature ($T_f$), flame area ($A_f$), fuel load, or combustion completeness without external fuel models. Any honest engine must restrict outputs to identifiable quantities (radiative flux $\Phi$, Fire Radiative Energy $\text{FRE}$, burning footprint area, and event proxies) and quantify ambiguity explicitly.

### 1.3 Hard Operational & Resource Constraints
1. **36-Hour Total Deadline** from start to finished dataset, validation suite, and web application.
2. **12 GPU-Hour Budget** originally planned on Kaggle `4x NVIDIA L4` (~96 GB aggregate VRAM), subject to strict **offline execution (no internet)** and a **~20 GB `/kaggle/working` output persistence cap** (plus a `2×` quota burn warning on multi-GPU instances).
3. **Bandwidth Asymmetry:** Operating from Algeria makes downloading 40–55 GB of global raw NASA CSVs locally and re-uploading to Kaggle prohibitive inside 36 hours.
4. **Kaggle Runtime & Policy Constraints:** Kaggle GPU images (`v173`, late 2026) run **Python 3.12**; `pyhdf` lacks Python 3.12 `manylinux` wheels and requires `libhdf4-dev` compilation; `L4x4` accelerators are restricted to authorized Kaggle Code Competitions, whereas general free Kaggle notebooks provide **`T4 x2`** (2 × 15 GB VRAM).

---

## 2. Core System Principles (Finalized Invariants)

We locked 10 non-negotiable system invariants (`INV-01` through `INV-10`):
1. **INV-01 (No Imitation Learning):** Never train `VIIRS -> MODIS` (destroying detail) or `MODIS -> VIIRS` (hallucinating unobserved 2000–2011 small fires). Both sensor families inform a shared physical latent state via forward observation operators.
2. **INV-02 (Never Sum Same-Orbit Platforms):** Treat `SNPP`, `NOAA-20`, and `NOAA-21` observations in the same half-day window as conditionally independent observations of the same latent physical state, never additive fires.
3. **INV-03 (Pre-Drift Calibration Freeze):** Calibrate all cross-sensor MODIS–VIIRS operators strictly on the stable-orbit overlap window (`2012-01-20` to `2021-12-31`).
4. **INV-04 (Explicit Operational Calendar & Outage Masking):** Only platforms physically operating and healthy on day $d$ enter the likelihood (`SNPP` closed after `2026-11-02`).
5. **INV-05 (Separate Physical State from Observation Model):** Explicitly separate $Y_{s,t} = h_s(X_t, \theta_s, G_{s,t}, A_{s,t}) + \epsilon_{s,t}$ and $D_{s,t} \sim \text{Bernoulli}(p_s)$ from inference $P(X_t \mid Y_{1:t}, D_{1:t}, Q_{1:t}, G_{1:t})$.
6. **INV-06 (No Raw Brightness Temperature Merging):** Never average `T21`/`T31` with `T_I4`/`T_I5`.
7. **INV-07 (Honest Zero-Detection Censoring):** Model MODIS sub-threshold blindness ($F_{\min}$) via a Hurdle likelihood so 2000–2011 uncertainty widens honestly instead of clamping small fires to zero.
8. **INV-08 (Static Thermal Anomaly Isolation):** Isolate volcanoes, offshore flares, SAMA nighttime artifacts, and multi-year industrial gas flares before biomass state reconstruction.
9. **INV-09 (Zero Data Destruction):** Emit 3 complementary products preserving untouched raw provenance alongside latent states and uncertainty diagnostics.
10. **INV-10 (No Coordinate Leakage):** Unsupervised pyrome discovery uses strictly behavioral fire features, never latitude or longitude.

---

## 3. Exhaustive Decision-by-Decision Record

### Decision D1: Three-Environment Execution Architecture `[FINALIZED]`
1. **Final Decision:** Decouple execution into **Environment 1** (Internet-ON CPU stager: cloud-to-cloud acquisition, HDF4 extraction, polygon filtering, Zstd Parquet conversion, Python 3.12 wheel packaging), **Environment 2** (Internet-OFF GPU compute: offline wheel install, geometry, calibration, GPU state-space solver, conformal calibration, validation, DuckDB/Parquet export), and **Environment 3** (Web server hosting FastAPI + DuckDB + frontend).
2. **Reasoning:** Eliminates residential bandwidth bottlenecks, satisfies offline GPU notebook constraints, isolates C-dependency compilation (`libhdf4-dev` for `pyhdf`) inside Env 1 so Env 2 installs pure `manylinux` wheels in seconds, and prevents wasting scarce GPU hours on CSV parsing or idling web servers.
3. **Alternatives Discussed & Rejected:** Local download/upload to Kaggle (fails 36h timeline); compiling CUDA/HDF4 extensions offline in Env 2 (high ABI breakage risk); tunneling a live UI out of Kaggle GPU notebooks (blocked by disabled internet and burns GPU quota).
4. **Layer Design Role:** Establishes clean data-artifact boundaries (`staged_data/` + `offline_wheels/` $\rightarrow$ `fire_harmonized.duckdb` + 3 Parquet tables $\rightarrow$ REST API).
5. **Consequences:** Requires manual dataset publication between Env 1 $\rightarrow$ Env 2 and artifact download between Env 2 $\rightarrow$ Env 3, while reducing Env 2 GPU waste to zero.
6. **Evidence & Confidence:** **High confidence (Engineering judgment backed by Kaggle Python 3.12 `v173` and offline execution rules).**

### Decision D2: Geographic Scope (`6 Multi-Biome Benchmark Countries`) `[FINALIZED — POLYGON PATCH REQUIRED]`
1. **Final Decision:** Run full pixel-to-hexagon harmonization across 6 scientifically diverse countries: `ARG` (CONAE July–Aug 2026 Chilecito spotlight, Pampas, SAMA), `DZA` (Kabylie crown fires vs. Saharan gas flares), `BRA` (Amazon/Cerrado 2012–2021 calibration density, clouds, SAMA), `IDN` (extreme clouds, nighttime peat smoldering, abstention), `AUS` (low-FRP northern savannas + Black Summer saturation), and `USA` (split into `CONUS` + `Alaska` for boreal cluster testing).
2. **Reasoning:** Our mathematical equations are 100% coordinate-agnostic; limiting data volume to 6 climate-diverse regimes keeps raw text to ~2–3 GB (~1.2–2.5 GB Zstd Parquet) instead of 35–55 GB globally, guaranteeing completion within 36 hours and `< 20 GB` disk while stress-testing every failure mode in EO literature.
3. **Alternatives Discussed & Rejected:** Full global 2000–2026 pixel-level ingestion (`150M+` rows, breaks disk/time budget); conditional dual-tier `H3-Res4` global branch (rejected in our single-system lock to eliminate dead/competing code paths); single-country prototype (fails to validate cross-biome generalization).
4. **Layer Design Role:** Implemented strictly as geographic masks in Env 1 and `src/config/settings.py`.
5. **Consequences & Honest Audit Critique:** **Flaw caught in audit:** Draft `01_download_and_stage.py` filtered solely by rectangular bounding boxes, which accidentally pull in large regions of neighboring countries (Bolivia, Peru, Paraguay, Uruguay, Chile, Tunisia, Morocco, Mali), silently inflating active cell count and GPU work by ~30–50%. **Mandatory fix:** Clip strictly against Natural Earth `50m` Admin-0 country polygons (`shapely.contains_xy`) inside Env 1 (note GeoPandas 1.0 removed bundled `naturalearth_lowres`, so Env 1 must fetch the raw GeoJSON directly).
6. **Evidence & Confidence:** **High confidence in the 6-country design; High confidence that polygon clipping must replace bbox-only filtering.**

### Decision D3: Spatial Reference Grid (`Uber H3 Hexagons`) `[FINALIZED — RESOLUTION & ACTIVE-CELL BUDGET AUDITED]`
1. **Final Decision (Prior Spec):** Equal-area Uber H3 hexagons at **Resolution 6** ($\bar{A} \approx 36.13\,\text{km}^2$) as the harmonization support grid, with fine micro-cells for static flare detection.
2. **Reasoning:** Point-to-point matching between `1 km` MODIS and `375 m` VIIRS fails under off-nadir footprint growth; lat/lon degree grids distort pixel density toward the poles by $\cos(\text{lat})$; H3 provides equal-area cells globally and uniform 6-neighbor topology.
3. **Alternatives Discussed & Rejected:** Raw point matching; $0.25^\circ$ lat/lon grid; dual `H3-Res6` + `H3-Res4` dual-resolution architecture.
4. **Layer Design Role:** `src/geometry/h3_utils.py` wraps `h3 v3/v4` integer conversion (`int64` storage, explicit `h3.int_to_str`/`str_to_int` conversion for `h3 v4`), builds sparse active-cell graphs, and computes a **3-coloring** of the planar hex graph.
5. **Consequences & Brutally Honest Audit Critique:**
   * **Bug #1 (`h3 v4` Crash):** Draft `01_download_and_stage.py` converted `latlng_to_cell` to `int(..., 16)`, which causes `TypeError` / `H3CellInvalidError` when passed directly to `h3 v4` functions like `grid_disk` unless `h3_utils.py` converts consistently.
   * **Bug #2 (Grossly Underestimated Cell Count):** Our earlier spec estimated `35,000–65,000` active H3-Res6 cells across the 6 countries. In reality, those 6 countries cover `~33.1 million km² ≈ 916,000` H3-Res6 hexagons; across 26 years (including 14 years of 375 m VIIRS agricultural/savanna burning), **250,000–450,000+ hexagons** will contain at least one detection—**5×–8× larger** than budgeted!
   * **Bug #3 (`H3-Res10` Flare Jitter):** `H3-Res10` cells (`~66 m` radius) are smaller than off-nadir/drifting MODIS geolocation jitter (`50–300+ m`), causing single Saharan gas flares to split across multiple Res-10 cells and fail persistence thresholds. **Mandatory fix:** Use **H3 Resolution 8** (`~0.74 km²`, ~460 m edge) for static flare persistence masking.
6. **Evidence & Confidence:** **High confidence in H3; Low confidence in our earlier 35k–65k Res-6 estimate—requires Polygon Clipping + GPU solver speedup (D10) to keep Res-6 viable within budget.**

### Decision D4: Implementation Stack & Offline Packaging `[FINALIZED]`
1. **Final Decision:** `DuckDB` + `Polars` + `PyArrow` for out-of-core CPU ETL; `scipy.spatial.cKDTree` + `csgraph.connected_components` + `shapely` for geometry/clustering; `scikit-learn` (`GaussianMixture`, `IsotonicRegression`) + `lightgbm` for calibration; pure `PyTorch` (`float64` kernels, `torch.multiprocessing`) for state-space solving; `FastAPI` + `DuckDB` + vanilla ES6 `MapLibre GL JS` + `Apache ECharts` for the web app. Offline wheel bundle strictly limited to `duckdb`, `polars`, and `h3` (`manylinux2014_x86_64`, Python 3.12).
2. **Reasoning:** Avoids recompiling CUDA/C++ extensions (`RAPIDS`, `JAX`, `pyhdf`) offline inside Env 2; DuckDB + Polars process 30M+ rows in seconds on CPU without spending GPU hours; zero-build frontend eliminates Node/npm failure modes in Hours 24–32.
3. **Alternatives Discussed & Rejected:** RAPIDS `cuDF`; `JAX`/`NumPyro`; `GeoPandas`/`PostGIS` for heavy ETL; dual React + Streamlit fallback UIs.
4. **Consequences:** Offline wheel package stays `< 80 MB`, installs in ~10 seconds, and uses Kaggle's preinstalled PyTorch/LightGBM/SciPy/Shapely stack natively.
5. **Evidence & Confidence:** **High confidence (95%).**

### Decision D5: Data Acquisition, Ancillary Grids, and Deduplication `[SUPERSEDED DRAFT -> REQUIRES PATCHES]`
1. **Final Decision (Intent):** Primary Science archives (`MCD14ML`, `VNP14IMGML`, `VJ114IMGML`, `VJ214IMGML`, `Mx14CMQ`/`Mx14C8Q` clouds, and burned area validation) from `fuoco.geog.umd.edu`; trailing 2026 NRT gap-fill from NASA FIRMS REST API; HDF4 extraction exclusively in Env 1; Science priority (`ML > SP > NRT`).
2. **Brutally Honest Audit of Draft Bugs & Necessary Fixes:**
   * **Broken Draft `01_download_and_stage.py`:** Still contained disabled `/api/country/csv` calls and brittle HTML scraping, completely mismatched UMD's real file formats (`MCD14ML` is space-delimited `.txt.gz` with `YYYYMMDD HHMM sat lat lon T21 T31 sample FRP conf type dn`, NOT `.csv.gz` with `scan/track`; `Vxx14IMGML` has `YYYYMMDD, HHMM, Sat, Lat, Lon, T_I4, T_I5, Sample, Pixarea, FRP, Conf, Type`, NOT `bright_ti4`/`scan`/`track`), and ignored FTP fallback (`fuoco.geog.umd.edu`).
   * **Broken SFTP Fallback Plan (`53,000` `/api/area` calls):** If port 22 is firewalled on Kaggle CPU runners, calling `/api/area/csv/.../5/...` over 26 years × 7 boxes × 4 sensors hits `~53,000+` calls and crashes NASA's `5,000 tx / 10 min` limit. **Verified Superior Fallback:** Use **Source Cooperative's NASA FIRMS Portolan GeoParquet 2.0 mirror** (`https://data.source.coop/portolan-mirrors/firms-catalog/detections/year=<YYYY>/detections.parquet` or `s3://us-west-2.opendata.source.coop/...`), which already mirrors all `631M+` global MODIS+VIIRS FIRMS detections from Nov 2000 to today (updated hourly, DuckDB-queryable via `httpfs` with Hilbert spatial pruning), reserving `/api/area/csv/.../5/...` strictly for trailing NRT gap-filling if needed!
   * **NRT vs. ML Deduplication Leak:** Draft `02_normalize_dedup_verify.py` keyed on `ROUND(lat, 4)` (~11 m) and exact `acq_hhmm`. Geolocation LUT shifts between NRT and Science ML move pixels by 40–300+ m, leaking duplicate rows. **Fix:** Drop `NRT` completely at the `(platform, acq_date)` level wherever `ML_SCIENCE` or `SP_SCIENCE` exists for that platform.
   * **Burned Area Bloat (`MCD64A1/SHP` vs `MCD64CMQ`):** Downloading/intersecting gigabytes of 500 m `MCD64A1/SHP` files in Env 1 wastes hours; replacing it with UMD's **`MCD64CMQ`** (`0.25°` monthly gridded burned area, 2001–2025) runs in seconds across the full 25-year span.
   * **Cloud Granularity Upgrade (Tentative -> Finalized Fix):** Upgrade from monthly `Mx14CMQ` to **8-day `MOD14C8Q` / `MYD14C8Q`** (falling back to monthly `Mx14CMQ` only if a month lacks 8-day files) so short cloudy storms are not smeared over 30 days.
   * **`acq_hhmm` & `Day/Night` SZA Bug:** `acq_time` (`HHMM`, e.g., `51` = `00:51`) must parse via `(hhmm // 100) + (hhmm % 100) / 60.0`. Moreover, NASA FIRMS FAQ (Nov 10, 2026) documents that `ML` assigns `Day/Night` via **Solar Zenith Angle (`SZA < 85°`)**, whereas `NRT` assigns it solely by **ascending/descending orbit**—corrupting high-latitude Alaska transitions unless `SZA` is computed uniformly for every row!
3. **Evidence & Confidence:** **High confidence in the corrected multi-source architecture; Medium confidence on Kaggle CPU port 22 availability until tested at Minute 15 (mitigated completely by the verified Source Cooperative Portolan GeoParquet mirror).**

### Decision D6: Geometry Correction, Hygiene, and Single-Pass Clustering `[FINALIZED WITH FIXES]`
1. **Final Decision:** Map `(lat, lon)` to `H3-Res6`; compute exact Solar Zenith Angle (`SZA < 85°`) and local solar time $\tau_i$ for consistent Day/Night assignment; derive `scan_km, track_km, A_i, vza_deg` from `Sample` via exact spherical Earth scan geometry when `scan/track` is absent (and preserve `Pixarea` when given); filter `type IN (1, 3)`, nighttime low-conf SAMA points, and `H3-Res8` persistent industrial flares (`type == 2` OR `Y >= 3 yrs, D_bar >= 15 days/yr, R_DN > 0.30`); within single-platform overpasses ($\Delta t \le 5\,\text{min}$), apply MODIS bow-tie overlap weighting ($\omega^{\text{bt}}_i = 1/(1 + \text{deg}(i))$ for $A_i > 1.5\,\text{km}^2$), cluster pixels via ECEF `cKDTree` + connected components ($d_{ij} \le 1.4\max(\sqrt{A_i},\sqrt{A_j})$), classify `i4_m13_fallback` (`sigma` inflation `1.35×`) vs true clipped Tobit saturation `sat_flag = 1`, and aggregate strictly per `(h3_index, solar_date, platform, dn_flag)`.
2. **Reasoning:** Eliminates bow-tie FRP inflation, VIIRS pixel-splitting count inflation, NRT industrial flare leakage, and `SNPP + N20 + N21` multi-counting.
3. **Alternatives Discussed & Rejected:** Multi-Hypothesis Tracking (MHT) across days (intractable across 26 years in 36h); dropping low-confidence fires (creates omission bias).
4. **Consequences & Honest Caveat:** Static persistence thresholds (`3 yrs, 15 days/yr, 0.30`) are heuristic engineering judgments that could occasionally flag stationary multi-year peat smoldering in Indonesia unless visually verified against Algeria/Indonesia diagnostics. Also, missing non-detection passes (`Y == 0`) require explicit analytical swath probability $p_{\text{swath}}$, swath-averaged area/VZA, and orbital-drift solar time $\bar{\tau}(d)$, PLUS an empirical **`platform_active_day[p, d]`** check (`>= 5` total detections across target regions) so spacecraft safe-mode outages are never treated as zero-fire evidence.
5. **Evidence & Confidence:** **High confidence (90%).**

### Decision D7: Pre-Drift Observation Operator Calibration (`2012–2021`) `[FINALIZED WITH 3 CRITICAL MATH FIXES]`
1. **Final Decision:** Calibrate strictly on stable-orbit `2012-01-20 .. 2021-12-31`: `K=7` coordinate-free GMM behavioral pyromes (`pyromes.py`), isotonic confidence reliability $w_i \in [0.15, 1.0]$, linear pixel-area detection limits $F_{\min, p}(A, \text{DN})$ + truncated Log-Normal MLE yielding analytical survival integrals $\psi^{\text{CNT}}, \psi^{\text{FRP}}$, Empirical Quantile Mapping (EQM) + LightGBM residual bias $\gamma_{\text{MODIS}, b}$ against near-nadir VIIRS reference (`VZA_VIIRS < 25°`), 2-harmonic Fourier diurnal & drift curve $D_{b,s}(\tau)$ (also fitting post-2022 Aqua vs NOAA-20 drift *strictly* on the diurnal coefficients), and heteroscedastic observation variance $\sigma^2_{p,g,d,\text{DN}} \in [0.025, 2.50]$.
2. **Reasoning:** Models physics-driven biome, intensity-tail, VZA, and diurnal differences without collapsing MODIS vs. VIIRS into a single broken global constant.
3. **Brutally Honest Math Bugs Caught & Corrected:**
   * **Math Bug #1 (Fatal Cloud Double-Counting):** Earlier specs placed $\ln v_{p,g,d}$ (cloud visibility) inside $\mu^{\text{gain}}$ for **detected** observations (`Case A/B`). That assumes clouds halve the radiance of fires seen through clear-sky gaps, artificially multiplying detected FRP by $2\times\text{–}10\times$ in cloudy Indonesia/Amazonia! **Correction:** Remove $\ln v_{p,g,d}$ completely from Case A/B; cloud visibility belongs **strictly inside Case C** ($P(D=0 \mid z) = (1 - v) + v e^{-\Lambda(z)}$).
   * **Math Bug #2 (24h vs. 12h Diurnal Collapse):** Earlier specs normalized $D(\tau)$ over 24 hours, divided both $Y_{\text{Day}}$ and $Y_{\text{Night}}$ by $D(\tau)$, and then weighted $\Phi_{\text{Day}}, \Phi_{\text{Night}}$ by $w_{\text{day}}, w_{\text{night}}$. That forced both $\Phi_{\text{Day}}$ and $\Phi_{\text{Night}}$ to estimate the 24h mean, collapsing `night_day_ratio` toward `~1.0` and causing extreme blowups on nocturnal smoldering fires! **Correction:** Normalize $D_{\text{day}}(\tau)$ to unit mean strictly within Daytime (`06:00–18:00`) and $D_{\text{night}}(\tau)$ to unit mean strictly within Nighttime (`18:00–06:00`), then integrate 24h mean flux as $0.5(\Phi_{\text{Day}} + \Phi_{\text{Night}})$.
   * **Math Bug #3 (Unconditional $\psi^{\text{FRP}}$ Blowup & Wrong Monotonicity Sign):** Dividing isolated single-fire detections (`raw_event_count == 1`) unconditionally by small $\psi^{\text{FRP}}$ assumes hidden companion fires and inflates isolated fires; forcing `monotone_constraints=[-1]` on `VZA` inside LightGBM fights real data geometry when bow-tie/background bias pushes the opposite direction; and pairing Aqua with off-nadir S-NPP mixes both sensors' angle errors. **Correction:** Restrict calibration pairs to near-nadir `VZA_SNPP < 25°`, enforce monotonicity strictly on **quantile ordering** inside EQM (not on VZA), and condition $\psi^{\text{FRP}}$ smoothly via Hurdle detection expectation $\frac{\psi^{\text{FRP}}}{1 - e^{-\Lambda}}$ (clamped cleanly) so isolated large fires aren't blown up.
4. **Remaining Limitation:** Polar-only satellites (~10:30, ~13:30, ~15:50) do not directly observe late-afternoon peaks (`16:00–19:00`), so absolute `FRE (TJ)` relies on 2-harmonic smoothness even while relative seasonal/annual anomalies remain accurate.
5. **Evidence & Confidence:** **High confidence (90%) once the three math fixes are applied.**

### Decision D8 & D9: Latent State-Space Model & Convexity-Guarded Likelihoods `[FINALIZED WITH CONSISTENCY FIX]`
1. **Final Decision:** Daily latent Day & Night log-fluxes $z_{g,d,\text{Day}}, z_{g,d,\text{Night}}$ reconstructed via a Space × Time × Day-Night Gaussian Markov Random Field (GMRF) MAP-Laplace model with Empirical Bayes shrinkage priors $(\mu_{g,m,\text{DN}}, \rho_b, \sigma_{\text{proc},b}, \lambda_{0,b}, \beta_{\text{dn},b}, \eta_{\text{dn},b})$ estimated deterministically from 2012–2021 VIIRS. Combine (A) Gaussian log-FRP likelihood, (B) Right-Censored Tobit saturation likelihood (via `log_ndtr` + asymptotic tail), and (C) Hurdle Zero-Detection Poisson-Cloud mixture likelihood using exact analytical gradients and a **Gauss-Newton convexity guard** ($h_{\text{miss}} = \max(0.0, \ell''_{\text{miss}})$) guaranteeing Global Symmetric Positive Definiteness ($H \succ 0$).
2. **Alternatives Discussed & Rejected:** Deep learning / ConvLSTMs (hallucination risk); Low-rank tensor completion (oversmooths high-rank extreme wildfire spikes); MCMC/HMC (computationally infeasible on 100k+ cells × 9,510 days in 36h); unguarded Newton (crashes Cholesky when $v < 1$ causes minor negative curvature in the 2-component Hurdle log-likelihood).
3. **Audit Fix Applied:** Earlier drafts mixed $\Phi = e^z$ in likelihoods with $\Phi = \max(0, e^z - 0.05)$ in post-processing (biasing small `0.5 MW` fires by 10% and leaving positive Poisson rate at $z_{\min}$). Fixed uniformly to $\Phi = e^z$ across all likelihoods and post-processing with dormant baseline floor $z_{\min} = \ln(0.01\,\text{MW})$. Also replaced misleading linear `std_events = Phi / F_event` with explicit labeling (`energy_equivalent_events`) alongside Hurdle-adjusted cluster counts.
4. **Evidence & Confidence:** **High confidence (95%).**

### Decision D10: Multi-GPU `float64` Solver (`3-Color` Gauss-Seidel + Thomas $LDL^\top$ + Takahashi + Joint Cholesky) `[FINALIZED — KERNEL OPTIMIZATION & HARDWARE FLEXIBILITY APPLIED]`
1. **Final Decision:** Partition active H3 cells across available GPUs (`torch.cuda.device_count()`, running on `T4 x2`, `1x L4`, or `4x L4`) in chunks sized safely under 6 GB VRAM (`batch_size = 1000`); run `6` outer sweeps over **H3 3-Coloring** (`color in {0, 1, 2}`, mathematically required for planar hexagons rather than 2-coloring) × alternating `Day/Night`; solve tridiagonal systems in **`torch.float64`** via Thomas $LDL^\top$ + Armijo backtracking; compute exact variances via Takahashi Recursion and sample $M=200$ joint correlated trajectories via $L^\top u = D^{-1/2}\varepsilon$ (soft-clamped to `z in [-8, 14]` before `exp` to prevent `4-sigma` outlier explosions).
2. **Brutally Honest Hardware & Runtime Audit:**
   * **Kaggle `L4x4` Policy Trap:** Kaggle Code Competitions restrict `L4x4` strictly to authorized competition notebooks; general free Kaggle users receive `T4 x2` (15 GB VRAM/GPU). Using dynamic `device_count()` and `< 6 GB` chunk memory makes the engine portable across both without OOM.
   * **Sequential `9,510`-Step CUDA Kernel Launch Trap:** Running `for t in range(9510)` inside `@torch.jit.script` launches 9,510 tiny CUDA slice kernels per pass (~0.15–0.25s launch latency per solve × 540 solves = ~1.5–2 min/chunk $\implies$ 4–8+ hours across 300k Res-6 cells!).
   * **Fix:** Implement a **single-launch CUDA C++ thread-per-cell Thomas/Takahashi/Cholesky kernel** via `torch.utils.cpp_extension.load_inline` (with the identical pure-PyTorch `float64` loop retained as the CPU verification reference tested against `load_inline` in `test_06`), eliminating 9,510× kernel launch overhead (~50× speedup!), PLUS require an Hour-14 single-chunk timing gate before full execution.
3. **Evidence & Confidence:** **High confidence in `float64` Thomas/Takahashi math (`< 1e-10` vs dense inverse); High confidence that CUDA kernel fusion (`load_inline`) + polygon clipping solves the Res-6 runtime bottleneck.**

### Decision D11, D12, D13: Conformal Uncertainty, Streaming Adaptation, and 3-Product Storage `[FINALIZED WITH SPARSE DAILY FIX]`
1. **Final Decisions:**
   * **Conformal Calibration (`conformal.py`):** Multiplicative log-space Split Conformal Quantile Rescaling (`CQR`) computed separately for `MODIS-only` (via held-out `2018–2020` VIIRS-masked folds) and `Multi-sensor` (`2018–2021` leave-one-VIIRS-platform-out folds), guaranteeing non-negative calibrated intervals `0 <= q05_cal <= q50 <= q95_cal`; explicit `ABSTAIN_LOW_EVIDENCE` flag when $\ge 5$ consecutive days have `cloud_frac > 0.85` with zero detections in cell + 1-ring neighbors; explicit `attribution_flag`.
   * **Streaming Adaptation (`run_fixed_lag_update(window_days=14)`):** Re-solves trailing 14-day slices anchored at `t_start - 1` to assimilate late NRT/Science revisions while preserving `obs_time`, `ingest_ts`, and `revision_count`.
   * **3 Output Products:** Stored in `fire_harmonized.duckdb` + Zstd Parquet (`obs_harmonized_provenance`, `state_reconstructed_monthly` & `daily`, `uncertainty_diagnostics`).
2. **Audit Catch & Mandatory Fix:** Exporting **dense** daily rows (`~300k cells × 9,510 days ≈ 2.85B rows`) exceeds Kaggle's **20 GB** `/kaggle/working` disk cap! **Fix:** Emit `state_reconstructed_monthly` densely (`~20–40M` rows) and **`state_reconstructed_daily` sparsely** (emit daily rows strictly where `phi_q50_mw > 0.25`, or within $\pm 3$ days of a detection, or inside spotlight validation regions), keeping total output under `~3–4 GB`.
3. **Unresolved Item Noted Honestly:** `run_fixed_lag_update` is an offline-callable API function, **not** an automated 24/7 polling cron daemon in Environment 3.
4. **Evidence & Confidence:** **High confidence in CQR and 3-product separation; 100% certainty that sparse daily export is mandatory.**

### Decision D14, D15, D16: Web App, Validation Protocol, and 166+ Test Suite `[FINALIZED]`
1. **Final Decisions:**
   * **Web App:** Single unified `FastAPI + DuckDB` backend (`/api/calendar`, `/api/timeseries`, `/api/map_hex`, `/api/diagnostics`, `/api/export.csv` sanitized against CSV formula injection) + vanilla HTML5/ES6 `MapLibre GL JS` + `Apache ECharts` frontend with accessible `viridis`/`cividis` ramps, uncertainty hatching/opacity, ROI bounding-box drawer, and 6 spotlight preset buttons (`Chilecito ARG 2026`, `Kabylie DZA`, `Hassi Messaoud Gas Flares`, `Alaska Boreal`, `Black Summer AUS`, `Kalimantan Peat IDN`).
   * **Validation Protocol:** Chow & Pettitt structural-break tests (`2002`, `2012`, `2018`, `2023`, `2026-11`), 2019–2021 VIIRS-masked ablation (`log_RMSE`, empirical `CRPS`, 90% coverage, `Spearman rho`, Top-3 Peak Month Match), `MCD64CMQ` burned-area cross-check, residual VZA/solar-time slope checks, and spotlight case reports.
   * **Testing:** 10 `pytest` suites (`166+` concrete unit, finite-difference, linear-algebra, synthetic digital-twin, conformal, structural-break, API security, and memory tests).
2. **Unresolved Question:** Hosting location for Environment 3 (local Docker presentation vs. public Hugging Face Space / VPS with 3–4 GB DuckDB storage) has not yet been selected.
3. **Evidence & Confidence:** **High confidence (95%).**

---

## 4. Consolidated Decision Status Ledger (Explicit Boundaries)

| Category / Decision | Status | Exact Position / Unresolved Item |
| :--- | :--- | :--- |
| 10 Scientific Invariants (`INV-01..10`) | **Finalized** | Enforced across every module and test. |
| 3-Environment Split (`Env1 -> Env2 -> Env3`) | **Finalized** | `pyhdf` isolated in Env 1; offline `manylinux` wheels in Env 2. |
| Space × Time × Day-Night GMRF + Hurdle + Tobit + CQR | **Finalized (Patched in Part II)** | Audit fixes applied: removed `ln(v)` from Case A/B, 12h $D(\tau)$ normalization, uniform $\Phi=e^z$, outage mask, `H3-Res8` flare mask, near-nadir VIIRS ref. |
| H3-Res6 Grid + Polygon Clipping + Sparse Daily + Fused Kernel | **Finalized in Part II** | Replaces buggy bbox-only + unoptimized sequential `9,510` loop + dense daily export. |
| Dual `H3-Res4` Global Tier & Dual Streamlit Fallback UI | **Rejected** | Eliminated permanently to preserve one coherent codebase. |
| `fuoco.geog.umd.edu` Port 22 & Source Coop Portolan Connectivity | **Unresolved (Verify at Hour 0.5)** | Port 22 vs HTTPS Portolan mirror must be checked at Minute 15 of Env 1. |
| Kaggle Free `T4 x2` vs Authorized `L4x4` Runtime | **Unresolved Hardware Reality** | Code dynamically scales across `1..4` GPUs within `< 6 GB` VRAM/GPU; user must confirm eligible competition hardware. |
| Automated Live Cron Polling for `run_fixed_lag_update` | **Deferred Past 36h** | Implemented and tested as a callable batch update API; 24/7 daemon deferred. |
| External Hosting Target for Environment 3 Web Server | **Unresolved** | Local Docker vs Hugging Face Spaces / cloud VPS decision pending. |

---
---

# PART II — Complete, Self-Contained Agent Implementation Specification (`AGENT_MASTER_SPEC.md`)

*(Everything below this line is a completely standalone, self-contained technical handoff specification written from scratch for an AI coding agent with zero prior context. All corrections from Part I are integrated seamlessly into one unified, bug-free architecture.)*

---

# MASTER IMPLEMENTATION SPECIFICATION: MODIS–VIIRS PHYSICS-INFORMED BAYESIAN HARMONIZATION ENGINE & BURNING ACTIVITY CALENDAR

## 0. Agent Directives and Scope Lock
You are responsible for implementing a single, cohesive, mathematically rigorous Earth Observation harmonization engine and interactive web application in Python, PyTorch, DuckDB, and FastAPI/MapLibre/ECharts.

Execute **strictly the single architecture specified in this document**:
* Implement every file, mathematical formula, parameter estimator, physical invariant, data check, and automated test defined below.
* Do **not** invent alternative modeling architectures, create duplicate scripts, leave stub functions, or modify mathematical definitions without maintaining exact equivalence verified by numerical unit tests.
* **Reference Date:** November 14, 2026.

---

## 1. Detailed Problem Statement

### 1.1 The Scientific and Operational Challenge
Satellite thermal anomaly ("active fire hotspot") archives span more than 26 years (November 2000 – November 2026), providing critical evidence for emergency response, early warning, climate research, and fire regime monitoring. However, this record is physically fractured across two incompatible satellite sensor families operating across five spacecraft:
1. **MODIS** (`1 km` nominal nadir spatial resolution) aboard **Terra** (launched Dec 1999; data Nov 2000–present) and **Aqua** (launched May 2002; data July 2002–present).
2. **VIIRS** (`375 m` I-band nominal nadir spatial resolution) aboard **Suomi-NPP / S-NPP** (Jan 20, 2012 – Nov 2, 2026), **NOAA-20** (April 2018 – present), and **NOAA-21** (2023/2024 – present).

The core challenge (authored by CONAE — the Argentinian Space Agency) is to build a harmonized active-fire record from 2000 to November 2026 in which **the sensor of origin no longer dictates trends or anomalies**, and expose it through an interactive **Burning Activity Calendar Web Application** enabling users to select an Area of Interest (AOI), inspect a Year × Month calendar heatmap, compare recent burning against multi-year historical medians, inspect uncertainty, and download sanitized CSV analytics.

### 1.2 Root Physical and Engineering Causes of Sensor Incompatibility
Raw MODIS and VIIRS records cannot be concatenated, averaged, or scaled by a single constant ratio for seven fundamental reasons:
1. **Spatial Resolution and Sub-Threshold Omission:** At nadir, one MODIS pixel covers $\sim 1.0\,\text{km}^2$ whereas one VIIRS I-band pixel covers $\sim 0.14\,\text{km}^2$ (`~7×` smaller area). Consequently, MODIS misses low-intensity agricultural, prescribed, and understory fires below $F_{\min} \approx 3.7\text{–}40\,\text{MW}$, whereas VIIRS detects fires down to $F_{\min} \approx 0.5\text{–}2.5\,\text{MW}$. Switching directly from MODIS to VIIRS in 2012 causes raw hotspot counts to jump by `3×–10×` purely from instrument sensitivity.
2. **Off-Nadir Pixel Stretching and Bow-Tie Artifacts:** Across its $\pm 55^\circ$ scan angle, MODIS pixels stretch to `~4.84 × 2.01 km ≈ 9.7 km²` and adjacent 10-line scans overlap by up to ~50% ("bow-tie effect"), double-counting off-nadir fires. VIIRS instead performs onboard pixel aggregation (`3:1` from $0^\circ\text{–}31.59^\circ$, `2:1` from $31.59^\circ\text{–}44.68^\circ$, `1:1` from $44.68^\circ\text{–}56.06^\circ$) and deletes redundant bow-tie fill pixels, growing only to `~0.62 km²` at swath edge.
3. **Spectral Bandpass and Saturation Differences:** MODIS uses `3.96 μm` (Channels 21/22) and `11 μm` (`Ch 31`). VIIRS uses `3.74 μm` (`I4`, saturating at `367 K`) and `11.45 μm` (`I5`), relying on co-located `750 m` `M13` channels for unsaturated Fire Radiative Power (FRP)—which splits one 750 m FRP retrieval across coincident 375 m sub-pixels or occasionally yields `0`/null FRP under extreme background failure. Raw brightness temperatures cannot be averaged across sensors.
4. **Constellation Multiplicity (Same-Orbit Overcounting):** `S-NPP`, `NOAA-20`, and `NOAA-21` share the ~13:30 / 01:30 solar plane separated by ~50 minutes. Adding their counts or FRPs together in 2018–2026 triples the apparent fire count relative to 2012–2017 (`S-NPP` alone). Conversely, `S-NPP` data delivery ceased on **November 2, 2026**, which causes a sudden drop if unmodeled.
5. **Post-2022 Orbital Drift:** Beginning in 2022, `Terra` (10:30 AM nominal) and `Aqua` (1:30 PM nominal) left controlled sun-synchronous orbits, drifting toward ~8:45 AM and ~3:30–3:50 PM respectively by late 2026. Because fire intensity peaks in early-to-mid afternoon, drifting sampling times confound sensor clock shifts with climate trends.
6. **Cloud Obscuration, SAMA False Alarms, Static Industrial Flares, and Processing Tiers:** Clouds block overpasses asymmetrically across tropical wet seasons (`< 30–40%` detection rate). Nighttime low-confidence VIIRS detections inside the South Atlantic Magnetic Anomaly (SAMA) produce spurious radiation spikes. Industrial gas flares (especially in Saharan oil fields) and volcanoes contaminate biomass wildfire calendars—complicated further because Near Real-Time (`NRT`) feeds omit the `type` column and assign `daynight` purely from ascending/descending orbit direction rather than Solar Zenith Angle (`SZA < 85°`) used in Science (`ML`) products.
7. **Physical Identifiability Bounds:** Orbital FRP alone cannot uniquely resolve sub-pixel flame temperature ($T_f$), exact flame area ($A_f$), fuel load, or combustion efficiency. The engine must output identifiable physical quantities (radiative flux $\Phi$, Fire Radiative Energy $\text{FRE}$, burning footprint area, and energy-equivalent/cluster event indices) and explicitly represent uncertainty and low-evidence abstention.

### 1.3 Success Criteria
The final implementation is successful if and only if:
1. **Structural Continuity:** Step jumps across sensor transitions (`2002` Aqua, `2012` S-NPP, `2018` N20, `2023` N21, `Nov 2026` S-NPP end) and `2022–2026` orbital drift are mathematically removed in stationary control regimes (verified by Chow/Pettitt structural break tests and synthetic twin falsification).
2. **Real Anomaly Preservation:** Genuine wildfire spikes—especially CONAE's spotlight **July–August 2026 Chilecito, La Rioja, Argentina** wildfire (`lat ~ -29.16, lon ~ -67.50`), **Kabylie, Algeria (2021/2023)**, and **Australian Black Summer (2019–2020)**—remain accurately preserved without oversmoothing.
3. **Static Anomaly Isolation:** Saharan industrial gas flares (e.g., Hassi Messaoud, Algeria) and SAMA artifacts are cleanly masked from wildfire calendars while remaining auditable in raw provenance outputs.
4. **Calibrated Uncertainty:** Pre-2012 MODIS-only uncertainty intervals are strictly wider than 2012–2026 multi-sensor intervals, cloudy gaps widen intervals and trigger explicit `ABSTAIN_LOW_EVIDENCE` flags under prolonged blindness, and Split Conformal Calibration achieves $\ge 90\%$ empirical coverage on held-out 2018–2020 ablation tests.
5. **Independent Physical Validation:** Harmonized monthly FRE correlates more strongly with independent `MCD64CMQ` burned area than unharmonized FIRMS hotspot counts.
6. **100% Automated Verification:** All 10 `pytest` suites (`166+` concrete mathematical, physical, numerical, and API tests) pass completely, and the interactive Burning Activity Calendar web app responds in `< 300 ms`.

---

## 2. Project Context, Hard Operational Constraints, and Execution Environments

### 2.1 Hard Operational Constraints
1. **Execution Window:** **36 hours** total end-to-end delivery.
2. **Compute & Offline Constraints:**
   * GPU compute executes in an **internet-disabled** environment (Kaggle notebook running Python 3.12 with preinstalled `torch`, `numpy`, `scipy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `shapely`, `rasterio`, `pytest`), constrained to **12 GPU-hours** quota and a **~20 GB `/kaggle/working`** disk cap.
   * **Hardware Portability Rule:** Code must inspect `torch.cuda.device_count()` dynamically and bound per-GPU memory `< 6 GB` per chunk (`batch_size = 1000`), executing seamlessly on `T4 x2` (15 GB/GPU), `1x L4` (24 GB), or `4x L4` (24 GB/GPU).
   * **Zero Wasted GPU Quota:** All data acquisition, HDF4 extraction, schema normalization, geometry preprocessing, calibration, and CPU unit testing execute on CPU (`0` GPU-hours). GPU runtime is reserved strictly for batched state-space solving, Monte Carlo sampling, conformal ablation runs, and validation checks.

### 2.2 Three-Environment Separation
1. **Environment 1 (Data & Wheel Stager — Internet ON, CPU only):**
   * Streams historical NASA/UMD active fires, `Mx14C8Q`/`Mx14CMQ` HDF4 cloud grids (using `libhdf4-dev` + `pyhdf` compiled strictly in Env 1), `MCD64CMQ` burned-area validation grids, and Natural Earth `50m` admin-0 country polygons cloud-to-cloud.
   * Clips strictly to the 6 target country polygons, normalizes schemas, deduplicates `ML > SP > NRT`, and outputs compressed Zstd Parquet datasets (`<= 2.5 GB`).
   * Downloads Python 3.12 `manylinux2014_x86_64` prebuilt wheels strictly for `duckdb`, `polars`, and `h3` (`<= 80 MB`) for offline installation in Environment 2.
2. **Environment 2 (Offline Harmonization Engine — Internet OFF, GPU batch):**
   * Installs `duckdb`, `polars`, `h3` via `--no-index`.
   * Executes CPU geometry/clustering and 2012–2021 operator calibration, runs multi-GPU `float64` Bayesian reconstruction + conformal calibration + validation suite, and exports `fire_harmonized.duckdb` + 3 Parquet products (`<= 4 GB` total).
3. **Environment 3 (Web Application Server):**
   * Runs `FastAPI + DuckDB` serving the static vanilla ES6 `MapLibre GL JS + Apache ECharts` frontend directly against `fire_harmonized.duckdb`.

---

## 3. Selected Technical Approach & Ten Non-Negotiable Invariants

### 3.1 Core Methodology: Physics-Informed Forward-Operator Bayesian State-Space Fusion
Rather than training regression models to map one sensor onto another, the engine separates the **Physical World** from the **Measurement World**:
* **Latent Physical State:** Every active equal-area hexagon $g$ (`Uber H3 Resolution 6`, $\bar{A} \approx 36.13\,\text{km}^2$) on local solar date $d \in \{0, \dots, T-1\}$ possesses two continuous latent states representing true cloud-free mean Daytime (`06:00–18:00`) and Nighttime (`18:00–06:00`) radiative flux in log-space:
  $$z_{g,d,\text{Day}} = \ln \Phi_{g,d,\text{Day}}, \qquad z_{g,d,\text{Night}} = \ln \Phi_{g,d,\text{Night}} \qquad (\text{MW})$$
* **Forward Sensor Operators:** Each operational platform $p \in \{\text{Terra}, \text{Aqua}, \text{SNPP}, \text{NOAA-20}, \text{NOAA-21}\}$ observes $z_{g,d,\text{DN}}$ through a sensor-specific physical forward operator accounting for sub-threshold detection survival $\psi$, viewing-angle/quantile calibration $\gamma$, intra-window diurnal solar time $D(\tau)$, heteroscedastic observation precision $\sigma^2$, detector saturation censoring (Tobit likelihood), and cloud-gated detection probability $P(D=1 \mid z, v, \psi^{\text{CNT}})$ (Hurdle likelihood).
* **Joint Inference & Uncertainty:** A Space × Time × Day-Night Gaussian Markov Random Field (GMRF) MAP-Laplace solver fuses all operational witnesses in `torch.float64` via H3 3-Coloring Gauss-Seidel sweeps, a fused `float64` Thomas $LDL^\top$ solver, Takahashi variance recursion, exact joint Cholesky posterior sampling ($M = 200$ trajectories), and Split Conformal Quantile Rescaling (`CQR`).

### 3.2 Ten Non-Negotiable System Invariants
1. **INV-01:** Never train `VIIRS -> MODIS` or `MODIS -> VIIRS` imitation mappings; fuse both into the shared physical latent state.
2. **INV-02:** Never sum detections or FRP across same-orbit platforms (`SNPP`, `NOAA-20`, `NOAA-21`) in the same half-day window; treat them as independent conditional likelihood terms.
3. **INV-03:** Freeze cross-sensor calibration strictly to the pre-drift overlap window **`2012-01-20` through `2021-12-31`**. Handle post-2022 Terra/Aqua clock drift strictly through the diurnal solar-time operator $D(\tau)$.
4. **INV-04:** Encode explicit platform operational calendars (`is_operational[p, d]`), closing `SNPP` after `2026-11-02` and masking multi-day spacecraft safe-mode outages empirically.
5. **INV-05:** Strictly separate forward observation models from state estimation; never bake cloud multipliers into detected FRP values.
6. **INV-06:** Never average raw MODIS and VIIRS brightness temperatures (`T21`/`T31` vs. `T_I4`/`T_I5`).
7. **INV-07:** Model MODIS zero-detection days via the area-dependent Hurdle survival probability so pre-2012 uncertainty widens honestly.
8. **INV-08:** Isolate volcanoes (`type=1`), offshore (`type=3`), SAMA nighttime artifacts, and `H3-Res8` persistent industrial flares prior to biomass fusion while preserving raw records in Product 1.
9. **INV-09:** Emit 3 distinct output products preserving untouched provenance, reconstructed physical states, and uncertainty diagnostics.
10. **INV-10:** Exclude `latitude` and `longitude` from unsupervised behavioral pyrome clustering.

### 3.3 Selected Target Scope (Six Multi-Biome Countries)
The mathematical engine is 100% coordinate-agnostic and operates uniformly at **H3 Resolution 6** across six polygon-clipped benchmark countries capturing all global fire regimes:
1. **Argentina (`ARG`):** CONAE July–Aug 2026 Chilecito spotlight (`lat ~ -29.16, lon ~ -67.50`), Pampas crop burns, SAMA zone.
2. **Algeria (`DZA`):** Kabylie Mediterranean crown fires (`2021`, `2023`) vs. Saharan Hassi Messaoud static gas flares.
3. **Brazil (`BRA`):** High-density 2012–2021 Amazon deforestation + Cerrado savanna calibration backbone, wet-season clouds, SAMA.
4. **Indonesia (`IDN`):** Severe equatorial cloud persistence, nighttime peat smoldering, low-evidence abstention regime.
5. **Australia (`AUS`):** Low-FRP northern savanna fires + 2019–2020 Black Summer VIIRS I4 saturation megafires.
6. **United States (`USA` — `CONUS` + `Alaska`):** Western crown megafires, SE prescribed small burns, and Alaska boreal cluster discrepancy benchmark.

---

## 4. Complete Codebase Architecture & Module Contracts

Implement strictly the repository layout below. Every file has a clear single responsibility, explicit inputs/outputs, and zero duplicate pipelines.

```text
fire-harmonizer/
├── Makefile
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── paths.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── build_wheels.sh
│   │   ├── download_data.py
│   │   └── normalize_dedup.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── h3_utils.py
│   │   ├── solar_geometry.py
│   │   └── preprocess_geometry.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── pyromes.py
│   │   └── operators.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── likelihoods.py
│   │   ├── priors.py
│   │   ├── thomas_takahashi.py
│   │   ├── cuda_kernels.py
│   │   ├── solver.py
│   │   └── conformal.py
│   ├── products/
│   │   ├── __init__.py
│   │   └── export_duckdb.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── synthetic_twin.py
│   │   └── diagnostics.py
│   └── webapp/
│       ├── __init__.py
│       ├── server.py
│       └── static/
│           ├── index.html
│           ├── app.js
│           └── styles.css
├── scripts/
│   ├── run_env1_stage.py
│   ├── run_env2_cpu_prep.py
│   ├── run_env2_gpu_solve.py
│   └── run_env2_validate.py
└── tests/
    ├── conftest.py
    ├── test_01_ingestion_schema.py
    ├── test_02_geometry_masking.py
    ├── test_03_event_clustering.py
    ├── test_04_observation_operators.py
    ├── test_05_likelihood_derivatives.py
    ├── test_06_gpu_solver_math.py
    ├── test_07_synthetic_twin_truth.py
    ├── test_08_conformal_uncertainty.py
    ├── test_09_validation_diagnostics.py
    └── test_10_webapp_api_scale.py
```

### File-by-File Module Contracts

* **`Makefile`:** Exposes `stage`, `cpu-prep`, `gpu-solve`, `validate`, `test`, `serve`.
* **`src/config/settings.py`:** Pure constants & dataclasses (`TARGET_COUNTRIES`, `PLATFORM_TIMELINE`, `DATE_START = "2000-11-01"`, `DATE_END = "2026-11-14"`, `T_DAYS = 9510`, `CALIB_START = "2012-01-20"`, `CALIB_END = "2021-12-31"`, `Z_MIN = ln(0.01)`, `Z_MAX = 14.0`, `F_REF_MW = 0.5`, `SAMA_BOX`, `H3_MAIN_RES = 6`, `H3_FLARE_RES = 8`, `MC_SAMPLES = 200`, `SOLVER_SWEEPS = 6`, `NEWTON_STEPS = 15`). Must contain zero I/O logic.
* **`src/config/paths.py`:** `DataPaths` dataclass auto-resolving `/kaggle/input/...` and `/kaggle/working/...` when running in Kaggle vs. `./data/...` locally via environment variables or directory detection. Single source of truth for every file path.
* **`src/ingest/build_wheels.sh`:** Runs in Env 1 to verify Python 3.12 (`sys.version_info`) and run `pip download --only-binary=:all: --platform manylinux2014_x86_64` strictly for `duckdb`, `polars`, `h3`. Must not bundle PyTorch, CUDA, or `pyhdf`.
* **`src/ingest/download_data.py`:** Env 1 downloader. Supports: (1) UMD `fuoco.geog.umd.edu` (SFTP + FTP fallback) for `MCD14ML` (`.txt.gz` space-delimited), `VNP14IMGML`/`VJ114IMGML`/`VJ214IMGML` (`.csv.gz`), `MOD14C8Q`/`MYD14C8Q`/`Mx14CMQ` HDF4 cloud grids (parsed via `pyhdf`), and `MCD64CMQ` burned area HDF4 grids; (2) Source Cooperative Portolan NASA FIRMS GeoParquet 2.0 mirror via DuckDB `httpfs` if UMD is unreachable; (3) NASA FIRMS REST API (`/api/data_availability/csv/{MAP_KEY}/ALL` + `/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/5/{DATE}`) strictly for trailing 2026 NRT backfill; and (4) Natural Earth `50m` Admin-0 GeoJSON download (`ne_50m_admin_0_countries.geojson`). Emits `staged_raw/`, `cloud_visibility_025deg.parquet`, `burned_area_val_025deg.parquet`, and `country_polygons.parquet`.
* **`src/ingest/normalize_dedup.py`:** Parses heterogeneous schemas (`MCD14ML` space format, `Vxx14IMGML` CSV format, FIRMS API/Portolan format), clips points strictly inside the 6 Natural Earth country polygons (`shapely.contains_xy`), enforces `(platform, acq_date)` Science-tier dominance (`ML_SCIENCE > SP_SCIENCE > NRT`, dropping NRT on any `(platform, acq_date)` covered by Science data), computes empirical `platform_active_day[P, T]`, writes Zstd Parquet partitioned by `(country_iso, year)` into `active_fires/`, and prints the coverage audit.
* **`src/geometry/h3_utils.py`:** Version-safe H3 wrapper (`h3 v3` & `v4`) storing 64-bit `int64` H3 IDs internally while converting cleanly to hex strings for `h3 v4` API calls (`latlng_to_cell`, `grid_disk`, `cell_to_boundary`, `cell_to_latlng`, `average_hexagon_area`). Implements deterministic greedy **3-coloring** of the active H3 hexagon graph (`color in {0, 1, 2}`) verified to have zero adjacent same-color neighbors.
* **`src/geometry/solar_geometry.py`:** Vectorized astronomical calculation of Solar Declination, Equation of Time, Solar Zenith Angle (`SZA` in degrees), local solar time $\tau_i \in [0, 24)$, local `solar_date`, and physical `dn_flag` (`'D'` if `SZA < 85.0` else `'N'`). Also implements exact spherical Earth pixel geometry (`Sample -> scan_km, track_km, vza_deg`) for MODIS (`1354` samples, $H=705\,\text{km}$) and VIIRS I-band (`6400` samples, $H=824\,\text{km}$, `3:1/2:1/1:1` aggregation zones), plus `scan*track -> vza_deg` lookup splines when `Sample` is absent.
* **`src/geometry/preprocess_geometry.py`:** Applies `type_raw IN (1, 3)` drop, SAMA nighttime low-confidence filter, `H3-Res8` persistent industrial/gas-flare mask (`is_static_industrial`), single-platform overpass segmentation ($\Delta t \le 5\,\text{min}$), ECEF `cKDTree` + `csgraph.connected_components` intra-pass bow-tie weighting ($\omega^{\text{bt}}_i$) and physical fire-event clustering (`event_cluster_id`), `i4_m13_fallback` vs Tobit `sat_flag` classification, and outputs **Product 1 base (`obs_provenance_base.parquet`)** and **`pass_observations.parquet`** keyed strictly by `(h3_index, solar_date, platform, dn_flag)` (never summing platforms).
* **`src/calibration/pyromes.py`:** Extracts strictly coordinate-free behavioral features per `H3-Res6` hexagon over `2012–2021`, fits `K=7` `GaussianMixture`, assigns all active hexagons `pyrome_id in 0..6`, and exports `hex_pyromes.parquet`.
* **`src/calibration/operators.py`:** Calibrates on `2012-01-20 .. 2021-12-31`: isotonic confidence reliability $w_i$, pixel-area detection limits $F_{\min,p}(A, \text{DN})$, truncated Log-Normal cluster MLE + exact closed-form survival integrals $\psi^{\text{CNT}}, \psi^{\text{FRP}}$, monotonic-quantile EQM + LightGBM residual calibration $\gamma_{\text{MODIS}, b}$ using near-nadir VIIRS reference (`VZA_VIIRS < 25°`), separate Daytime (`06–18`) and Nighttime (`18–06`) 2-harmonic Fourier diurnal curves $D_{b,s,\text{DN}}(\tau)$ (including post-2022 Aqua vs NOAA-20 drift *only* on diurnal harmonics), heteroscedastic observation variance $\sigma^2$, and active-area transfer model $g_{\text{area}, b}$. Joins `cloud_visibility_025deg.parquet` and emits `calibration_artifacts.pkl` + `pass_observations_calibrated.parquet`.
* **`src/engine/likelihoods.py`:** Pure PyTorch (`torch.float64`) vectorized negative log-likelihoods, exact 1st derivatives, and convexity-guarded Hessians (`Obs`, `Tobit Saturation`, `Hurdle Zero-Detection`).
* **`src/engine/priors.py`:** Deterministic Empirical Bayes estimators from `2012–2021` VIIRS data for $(\mu_{g,m,\text{DN}}, \rho_b, \sigma_{\text{proc}, b}, \lambda_{0, b}, \beta_{\text{dn}, b}, \eta_{\text{dn}, b}, \bar{F}_{\text{event}, b})$ and sparse 3-color neighbor graph tensor builder.
* **`src/engine/thomas_takahashi.py` & `src/engine/cuda_kernels.py`:** `thomas_takahashi.py` implements the exact `float64` reference Thomas $LDL^\top$, Takahashi variance/lag-1 recursion, and exact joint Cholesky sampler `L^T u = D^{-1/2} eps` in PyTorch. `cuda_kernels.py` compiles a clean single-launch CUDA C++ `float64` thread-per-cell Thomas/Takahashi/Cholesky kernel via `torch.utils.cpp_extension.load_inline` when CUDA & `nvcc` are present (automatically verified bit-for-bit against `thomas_takahashi.py` in `test_06`), eliminating the 9,510-kernel launch bottleneck.
* **`src/engine/solver.py`:** Multi-GPU worker orchestrator (`torch.multiprocessing` over `0..device_count()-1`, `batch_size = 1000`). Builds dense `(B, T=9510, P=5)` observation + non-detection tensors per chunk, executes `6` 3-color × Day/Night sweeps of `15` Damped Newton-Raphson iterations with Armijo backtracking, computes exact Takahashi variances, generates $M=200$ joint posterior trajectories, aggregates daily & monthly quantiles (`q05, q50, q95, mean`), and exposes `run_fixed_lag_update(window_days=14)`.
* **`src/engine/conformal.py`:** Computes Split Conformal Quantile Rescaling (`CQR`) multipliers from held-out `2018–2020` VIIRS-masked folds (`MODIS-only` multiplier) and `2018–2021` leave-one-VIIRS-platform-out folds (`multi-sensor` multiplier), rescales monthly/daily intervals, assigns `ABSTAIN_LOW_EVIDENCE` and `attribution_flag`, and computes 2003–2020 historical medians, anomaly ratios, z-scores, and percentiles.
* **`src/products/export_duckdb.py`:** Writes indexed Zstd Parquet for **Product 1 (`obs_harmonized_provenance`)**, **Product 2 (`state_reconstructed_monthly` dense + `state_reconstructed_daily` sparse episode filter)**, and **Product 3 (`uncertainty_diagnostics`)**, loading all views/tables into `fire_harmonized.duckdb`.
* **`src/validation/synthetic_twin.py` & `src/validation/diagnostics.py`:** Synthetic 26-year digital twin forward simulator with known ground truth $X_{\text{true}}$ (`synthetic_twin.py`), plus real-data Chow & Pettitt structural-break suite, 2019–2021 VIIRS-masked ablation, `MCD64CMQ` burned-area correlation comparison, residual VZA/solar-time slope tests, and spotlight reports (`Chilecito ARG 2026`, `Kabylie & Hassi Messaoud DZA`, `Nov 2, 2026 S-NPP end`).
* **`src/webapp/server.py` & `src/webapp/static/`:** FastAPI + read-only DuckDB backend with `/api/calendar`, `/api/timeseries`, `/api/map_hex`, `/api/diagnostics`, `/api/export.csv` (sanitized against formula injection), and MapLibre GL JS + Apache ECharts frontend.

---

## 5. Complete Mathematical & Computational Engine (Exact Derivations)

### 5.1 Coordinate Support, Solar Geometry, and Pixel Geometry

#### 5.1.1 Solar Zenith Angle & Local Solar Time
For UTC timestamp $t_{\text{UTC}}$ (Julian day $J$ and fractional UTC hours $h_{\text{UTC}} = \text{hh} + \text{mm}/60.0$) at $(\phi, \lambda)$:
1. Local solar time in $[0, 24)$:
   $$\tau_i = \left(h_{\text{UTC}, i} + \frac{\lambda_i}{15.0}\right) \bmod 24.0$$
   And local solar date shifts by $-1$ day if $h_{\text{UTC}, i} + \lambda_i / 15.0 < 0$ or $+1$ day if $\ge 24.0$.
2. Compute solar declination $\delta_\odot$ and equation of time $E_t$ using Spencer's standard astronomical Fourier approximation:
   Let day angle $\Gamma = \frac{2\pi(\text{doy} - 1)}{365}$:
   $$\delta_\odot = 0.006918 - 0.399912\cos\Gamma + 0.070257\sin\Gamma - 0.006758\cos 2\Gamma + 0.000907\sin 2\Gamma$$
   $$E_t = 229.18(0.000075 + 0.001868\cos\Gamma - 0.032077\sin\Gamma - 0.014615\cos 2\Gamma - 0.040849\sin 2\Gamma) \;\;(\text{min})$$
   True solar time $\tau_{\text{true}} = \tau_i + E_t / 60.0$, hour angle $\omega = 15^\circ(\tau_{\text{true}} - 12.0)$:
   $$\cos(\text{SZA}) = \sin\phi\sin\delta_\odot + \cos\phi\cos\delta_\odot\cos\omega$$
   Assign canonical regime `dn_flag = 'D'` if $\text{SZA} < 85.0^\circ$ else `'N'`. This guarantees identical physical Day/Night labeling across `ML`, `SP`, and `NRT` regardless of polar seasons or `NRT` ascending/descending node artifacts.

#### 5.1.2 Pixel Area, Scan/Track, and VZA Recovery
* **MODIS from `sample` ($s \in [0, 1353]$, $R_E = 6371.0\,\text{km}$, $H = 705.0\,\text{km}$, $\alpha_{\max} = 55.0^\circ$, $\Delta\alpha_0 = 1.0/705.0$):**
  $$\alpha(s) = (s - 676.5)\,\frac{2\alpha_{\max}}{1353}, \qquad \theta_{\text{VZA}}(s) = \arcsin\!\left(\frac{R_E + H}{R_E}\sin|\alpha(s)|\right)$$
  $$D(\alpha) = (R_E + H)\cos\alpha - \sqrt{R_E^2 - (R_E + H)^2\sin^2\alpha}$$
  $$\text{track}(s) = D(\alpha)\,\Delta\alpha_0, \qquad \text{scan}(s) = \frac{D(\alpha)\,\Delta\alpha_0}{\cos\theta_{\text{VZA}}(s)}, \qquad A_i = \text{scan}(s)\,\text{track}(s)$$
  *(Yields $1.00 \times 1.00 = 1.00\,\text{km}^2$ at nadir and $4.84 \times 2.01 = 9.72\,\text{km}^2$ at swath edge.)*
* **VIIRS I-band from `sample` ($s \in [0, 6399]$, $H = 824.0\,\text{km}$):**
  Map `sample` monotonically to scan angle $|\alpha| \in [0^\circ, 56.06^\circ]$ across the onboard `3:1` ($|\alpha| \le 31.59^\circ$), `2:1` ($31.59^\circ\text{–}44.68^\circ$), and `1:1` ($44.68^\circ\text{–}56.06^\circ$) aggregation zones, yielding $\theta_{\text{VZA}} = \arcsin\big(\frac{R_E+H}{R_E}\sin|\alpha|\big)$ and pixel area $A_i \in [0.141, 0.620]\,\text{km}^2$ (using `Pixarea` directly when supplied by `Vxx14IMGML`).
* **From `scan_km, track_km` (when `sample` is absent in Portolan/FIRMS API):**
  $A_i = \text{clip}(\text{scan}_i \times \text{track}_i,\; 0.12,\; 12.0)$ and recover $\theta_{\text{VZA}}$ via piecewise-linear monotone interpolation (`np.interp`): MODIS knots `[(1.00, 0.0), (1.35, 20.0), (2.25, 35.0), (4.20, 50.0), (9.65, 65.0)]`; VIIRS knots `[(0.141, 0.0), (0.180, 20.0), (0.280, 35.0), (0.430, 50.0), (0.620, 65.0)]`.

#### 5.1.3 Intra-Pass Bow-Tie Overlap & Connected-Component Clustering
Partition detections by `(platform, pass_id)` where sorted consecutive detections > `5 minutes` apart start a new pass. Convert `(lat, lon)` to 3D ECEF chords (`R_E = 6371.0 km`) and query neighbor pairs within radius via `scipy.spatial.cKDTree`:
1. **MODIS Bow-Tie Weight ($A_i > 1.5\,\text{km}^2$):** Connect pixels $i, j$ in the same pass if chord distance $d_{ij} < 0.7 \cdot \frac{\text{track}_i + \text{track}_j}{2}$. Set $\omega^{\text{bt}}_i = \frac{1.0}{1.0 + \text{deg}_{\text{bt}}(i)}$ (if two $40\,\text{MW}$ off-nadir edge pixels overlap, each gets weight $0.5$, conserving $40\,\text{MW}$). For VIIRS or MODIS $A_i \le 1.5\,\text{km}^2$, $\omega^{\text{bt}}_i = 1.0$.
2. **Physical Cluster Components:** Connect pixels $i, j$ in the same pass if $d_{ij} \le 1.4 \cdot \max(\sqrt{A_i}, \sqrt{A_j})$. Run `csgraph.connected_components` to assign `event_cluster_id`.
3. **SAMA & Static Gas-Flare Hygiene:**
   * Drop `type_raw IN (1, 3)` and VIIRS `dn_flag == 'N'` low-confidence (`conf_raw == 'l'` or `conf_num < 30`) inside SAMA (`-110 <= lon <= 11` and `-55 <= lat <= 7`).
   * Group detections into **`H3 Resolution 8`** micro-cells $m$ ($\sim 0.74\,\text{km}^2$, large enough to absorb MODIS off-nadir geolocation jitter). Compute distinct active years $Y_m$, mean active days/year $\bar{D}_m$, and day/night symmetry $R^{\text{DN}}_m = \frac{\min(n_{\text{D}}, n_{\text{N}})}{\max(n_{\text{D}}, n_{\text{N}}) + 1}$. Set `is_static_industrial = True` if `type_raw == 2` OR ($Y_m \ge 3$ AND $\bar{D}_m \ge 15.0$ AND $R^{\text{DN}}_m > 0.30$).
4. **Saturation Classification:**
   * `i4_m13_fallback = 1` when `sensor_family == 'VIIRS'` and `t_mir_k >= 366.0` and `frp_mw > 0.0` (M13 retrieved FRP; multiplies observation noise $\sigma$ by `1.35`).
   * `sat_flag = 1` (Tobit right-censored) when (`sensor_family == 'MODIS'` and `t_mir_k >= 498.0` and `frp_mw > 0.0`) OR (`sensor_family == 'VIIRS'` and `t_mir_k >= 366.9` and `frp_mw >= 300.0`).

#### 5.1.4 Non-Detection (`Y == 0`) Swath & Cloud Parameters
On any day $d$ where platform $p$ is operational (`is_operational[p, d] == 1` AND `platform_active_day[p, d] == 1`) with zero biomass detections in cell $g$, regime $\text{DN}$:
1. **Cloud & Swath Visibility $v_{p,g,d}$:**
   Join 8-day `Mx14C8Q` (or monthly `Mx14CMQ` fallback) cloud grid on `(0.25deg bin, year, period)`:
   $$v_{p,g,d} = p_{\text{swath}, p}(|\phi_g|, d) \cdot \text{clip}\!\Big(1.0 - f_{\text{cloud}} - f_{\text{missing}},\;\; 0.10,\;\; 0.98\Big)$$
   with $p_{\text{swath}, \text{VIIRS}} = 0.99$ and $p_{\text{swath}, \text{MODIS}} = \text{clip}(0.85 + 0.15\frac{|\phi_g|}{30.0}, 0.85, 1.00) \times (0.96 \text{ if } d \ge \text{2022-01-01})$.
2. **Expected Pass Geometry & Drift Solar Hour:**
   * MODIS: $\bar{A}_{\text{avg}} = 2.35\,\text{km}^2$, $\bar{\theta}_{\text{avg}} = 35.0^\circ$. VIIRS: $\bar{A}_{\text{avg}} = 0.23\,\text{km}^2$, $\bar{\theta}_{\text{avg}} = 28.0^\circ$.
   * Expected solar hour $\bar{\tau}_{p,d,\text{DN}}$: Terra drifts linearly from `10.5 -> 8.75` (`2022-01-01 .. 2026-11-14`), Aqua from `13.5 -> 15.60` (`+12.0 mod 24` for Night); VIIRS platforms fixed at `13.5` Day / `1.5` Night.

---

### 5.2 Pre-Drift Observation Operators (`2012-01-20` to `2021-12-31`)

1. **Coordinate-Free Behavioral Pyromes (`K=7` GMM):**
   For each hexagon with $\ge 20$ active pass-days in `2012–2021`, compute strictly non-coordinate behavioral features: `[log(median_cluster_frp + 1), night_frp_share, seasonal_circular_concentration, interannual_cv, mean_cluster_pixels]`. Fit `GaussianMixture(n_components=7, covariance_type='full', reg_covar=1e-4, random_state=42)` over standardized features and assign every active cell to `pyrome_id in 0..6`.
2. **Calibrated Confidence Reliability ($w_i \in [0.15, 1.0]$):**
   Bin 2012–2021 detections by `(sensor_family, dn_flag, conf_num)` and compute empirical corroboration rate (another overpass/day within $\pm 1$ day and 2 km). Fit `IsotonicRegression(y_min=0.15, y_max=1.0)` mapping raw confidence to calibrated reliability $w_i$.
3. **Physical Detection Limits & Analytical Survival Integrals ($\psi^{\text{CNT}}, \psi^{\text{FRP}}$):**
   $$F_{\min, \text{MODIS}}(A, \text{DN}) = \alpha_{M,\text{DN}} + \beta_{M,\text{DN}}\,A, \qquad F_{\min, \text{VIIRS}}(A, \text{DN}) = \alpha_{V,\text{DN}} + \beta_{V,\text{DN}}\,A$$
   using physical anchors: MODIS Night $(\alpha=1.5, \beta=2.2)$, MODIS Day $(\alpha=2.5, \beta=3.9)$, VIIRS Night $(\alpha=0.25, \beta=1.8)$, VIIRS Day $(\alpha=0.40, \beta=3.4)$.
   Fit left-truncated Log-Normal cluster FRP distribution $\ln F \sim \mathcal{N}(\mu_{F,b}, \sigma^2_{F,b})$ on clear-sky near-nadir (`VZA < 20°`, `cloud_frac < 0.30`) VIIRS clusters above $F_{\text{ref}} = 0.5\,\text{MW}$ per pyrome $b$.
   Compute closed-form survival integrals (`scipy.special.ndtr`):
   $$\psi^{\text{CNT}}_{p,b}(A, \text{DN}) = \text{clip}\!\left(1 - \Phi_{\mathcal{N}}\!\left(\frac{\ln F_{\min, p}(A,\text{DN}) - \mu_{F,b}}{\sigma_{F,b}}\right),\;\; 0.02,\;\; 1.0\right)$$
   $$\psi^{\text{FRP}}_{p,b}(A, \text{DN}) = \text{clip}\!\left(\frac{1 - \Phi_{\mathcal{N}}\!\left(\frac{\ln F_{\min, p}(A,\text{DN}) - \mu_{F,b} - \sigma^2_{F,b}}{\sigma_{F,b}}\right)}{1 - \Phi_{\mathcal{N}}\!\left(\frac{\ln F_{\text{ref}} - \mu_{F,b} - \sigma^2_{F,b}}{\sigma_{F,b}}\right)},\;\; 0.05,\;\; 1.0\right)$$
4. **VZA & Intensity Quantile Calibration ($\gamma_{p,b}$):**
   Pair synchronous 2012–2021 `Aqua` and near-nadir `SNPP` (`VZA_SNPP < 25°`, $\Delta t \le 45\,\text{min}$, same `h3_index`, same `dn_flag`, both $> 0$). Fit LightGBM regressor (`max_depth=4, n_estimators=100, min_child_samples=50`) on features `[pyrome_id, dn_is_day, mean_vza_modis, mean_area_modis, log_y_modis]` without forcing an artificial VZA sign constraint, followed by a 20-knot **strictly monotonic Empirical Quantile Mapping (EQM)** curve per `(pyrome, dn_flag)` so output quantile curves never cross. Set $\ln \gamma_{\text{VIIRS}} \equiv 0.0$, and clip $\ln \gamma_{\text{MODIS}} \in [-1.5, +1.5]$.
   Crucial Hurdle conditioning for detected observations ($Y > 0$):
   Because small isolated fires (`raw_event_count == 1`) already cleared $F_{\min}$ to be detected, model the detected completeness bias factor cleanly via conditional multi-fire adjustment:
   $$\mu^{\text{gain, det}}_{p,g,d,\text{DN}} = \ln D_{b,s,\text{DN}}(\bar{\tau}_{p,g,d}) + \ln \gamma_{p,b}(\bar{\theta}_p, q) + \ln \psi^{\text{FRP, eff}}_{p,b}(\bar{A}_p, \text{DN}, E_{\text{raw}})$$
   where $\psi^{\text{FRP, eff}} = \psi^{\text{FRP}}$ when multiple fire clusters coexist (`E_raw >= 2` or broad multi-pixel fronts) and smoothly blends toward $\max(\psi^{\text{FRP}}, 0.65)$ for isolated single-cluster detections—and **NEVER include $\ln v_{p,g,d}$ inside $\mu^{\text{gain, det}}$** (cloud visibility enters strictly inside the zero-detection Hurdle likelihood).
5. **Intra-Window Diurnal & Orbital-Drift Operator ($D_{b,s,\text{DN}}(\tau)$):**
   Fit a 2-harmonic Fourier curve per pyrome $b$ and season $s \in \{\text{DJF, MAM, JJA, SON}\}$:
   $$\ln D_{\text{raw}, b,s}(\tau) = a_1\cos\!\left(\frac{2\pi\tau}{24}\right) + b_1\sin\!\left(\frac{2\pi\tau}{24}\right) + a_2\cos\!\left(\frac{4\pi\tau}{24}\right) + b_2\sin\!\left(\frac{4\pi\tau}{24}\right)$$
   Fit $(a_1, b_1, a_2, b_2)$ via Ridge regression (`alpha=10.0`) on log-FRP differences between `Terra` vs `Aqua/SNPP` (`2003–2021`) plus `Aqua` post-2022 drifting hours vs stationary `NOAA-20` (updating *only* diurnal coefficients while keeping cross-sensor $\psi, \gamma$ frozen).
   Normalize **independently within each 12-hour half-day window** so $\Phi_{\text{Day}}$ and $\Phi_{\text{Night}}$ represent true mean Daytime (`06:00–18:00`) and Nighttime (`18:00–06:00`) MW flux:
   $$c_{0,\text{Day}} = -\ln\!\left(\frac{1}{12}\int_6^{18} D_{\text{raw}, b,s}(\tau)\,d\tau\right), \qquad c_{0,\text{Night}} = -\ln\!\left(\frac{1}{12}\int_{18}^{30} D_{\text{raw}, b,s}(\tau \bmod 24)\,d\tau\right)$$
   Then $D_{b,s,\text{DN}}(\tau) = \exp(\ln D_{\text{raw}, b,s}(\tau) + c_{0,\text{DN}})$ has exact unit mean over its 12-hour window, preserving physical `night_day_ratio = Phi_Night / Phi_Day` and making exact 24-hour mean flux $\Phi_{24\text{h}} = 0.5(\Phi_{\text{Day}} + \Phi_{\text{Night}})$.
6. **Heteroscedastic Observation Variance ($\sigma^2_{p,g,d,\text{DN}}$):**
   Fit OLS on `2012–2021` squared calibration pair residuals predicting $\ln(\hat{\epsilon}^2 + 10^{-3})$ from `[is_modis, vza_deg / 65.0, 1.0 - w_mean, 1.0 - v_cloud, i4_m13_fallback]`, clipped to $[0.025, 2.50]$.
7. **Active Burning Footprint Transfer ($g_{\text{area}, b}$):**
   Fit a quantile gradient-boosted/ridge curve on `2012–2021` VIIRS clusters mapping $(\ln \Phi, \text{night\_day\_ratio}, b) \mapsto \widehat{A}^{\text{burn}}_{g,d}$ (`km²`, bounded cleanly by `A_g`).

---

### 5.3 Space × Time × Day-Night State-Space Model & Exact Derivatives (`torch.float64`)

Let $z_t \in [-8.0, 14.0]$ denote $z_{g, t, \text{DN}} = \ln \Phi_{g, t, \text{DN}}$ (with $\Phi = e^z$ uniformly across all likelihoods and post-processing, and dormant baseline $z_{\min} = \ln(0.01) = -4.605170$).

#### 5.3.1 Prior Parameter Estimators (Deterministic from 2012–2021 VIIRS)
* $\bar{F}_{\text{event}, b}$: Median clear-sky nadir VIIRS cluster FRP in pyrome $b$ (`>= 1.0 MW`).
* $\mu_{g,m,\text{DN}}$: Empirical Bayes shrinkage of cell-month log-flux (treating clear non-detection days as $z_{\min}$) toward pyrome-month mean $\bar{\mu}_{b,m,\text{DN}}$ with shrinkage weight $5.0$:
  $$\mu_{g,m,\text{DN}} = \frac{n_{g,m}\,\bar{y}_{g,m,\text{DN}} + 5.0\,\bar{\mu}_{b,m,\text{DN}}}{n_{g,m} + 5.0}$$
* $\rho_b, \sigma_{\text{proc}, b}$: Lag-1 AR(1) OLS slope and residual std on consecutive active VIIRS cell-days (`rho_b in [0.25, 0.90]`, `sigma_proc_b in [0.25, 1.50]`).
* $\lambda_{0, b}$: Scalar intensity rate in $[0.1, 5.0]$ minimizing zero-detection rate discrepancy via `scipy.optimize.minimize_scalar`.
* $(\beta_{\text{dn},b}, \eta_{\text{dn},b})$: OLS slope and variance ratio between Day and Night anomalies ($\eta_{\text{dn},b} \in [0.05, 0.30]$).

#### 5.3.2 Exact Likelihood Terms, Analytical Gradients & Convexity-Guarded Hessians
For each operational platform pass $(p, g, t, \text{DN})$ with `is_operational[p, t] == 1` and `platform_active_day[p, t] == 1`:
Let $\mu = \mu^{\text{gain, det}}_{p,g,t,\text{DN}}$, $\tau_2 = 1/\sigma^2$, $\tau_1 = 1/\sigma$.

1. **Case A — Valid Unsaturated Detection ($Y > 0$, `sat_flag == 0`):**
   $$r = (z + \mu) - \ln Y, \qquad \ell_{\text{obs}} = \frac{1}{2}\tau_2 r^2, \qquad g_{\text{obs}} = \tau_2 r, \qquad h_{\text{obs}} = \tau_2$$
2. **Case B — Right-Censored Saturation Detection ($Y > 0$, `sat_flag == 1`, Tobit):**
   Let $w = \tau_1\big((z + \mu) - \ln Y\big)$. For $w \ge -30$:
   $$\ln\Phi(w) = \text{log\_ndtr}(w), \qquad \lambda(w) = \exp\!\left(-0.5 w^2 - 0.5\ln(2\pi) - \ln\Phi(w)\right)$$
   For $w < -30$ (exact asymptotic expansion avoiding $\log 0$):
   $$-\ln\Phi(w) \approx 0.5 w^2 + \ln(-w) + 0.5\ln(2\pi), \qquad \lambda(w) \approx -w - \frac{1}{w}$$
   $$\ell_{\text{sat}} = -\ln\Phi(w), \qquad g_{\text{sat}} = -\tau_1\,\lambda(w), \qquad h_{\text{sat}} = \tau_2\,\lambda(w)\big(w + \lambda(w)\big) > 0$$
3. **Case C — Non-Detection ($Y == 0$, Hurdle Poisson-Cloud Mixture):**
   Let $k = \frac{\lambda_{0,b}\,\psi^{\text{CNT}}_{p,b}(\bar{A}_{\text{avg}}, \text{DN})}{\bar{F}_{\text{event}, b}} > 0$, $\Lambda = \text{clip}(k\,e^z,\; 10^{-12},\; 80.0)$, $e^{-\Lambda} = \exp(-\Lambda)$, $q = (1 - v) + v e^{-\Lambda}$, and posterior fire-presence weight $r = \frac{v e^{-\Lambda}}{q} \in (0, 1]$:
   $$\ell_{\text{miss}} = -\ln q, \qquad g_{\text{miss}} = \frac{\partial \ell_{\text{miss}}}{\partial z} = r\,\Lambda$$
   Differentiating $r\,\Lambda$ once more with $\frac{\partial r}{\partial z} = -r(1 - r)\Lambda$:
   $$\ell''_{\text{miss}} = r\,\Lambda\,\big[1 - \Lambda(1 - r)\big]$$
   **Convexity Guard (SPD Guarantee):** Because $(1 - v) + v e^{-\Lambda}$ exhibits minor negative curvature when $v < 1$ and $\Lambda \in (1, 4)$, evaluate the exact gradient $g_{\text{miss}} = r\,\Lambda$ while applying the positive Gauss-Newton projection:
   $$h_{\text{miss}} = \max\!\Big(0.0,\; r\,\Lambda\,\big[1.0 - \Lambda(1.0 - r)\big]\Big)$$
   This preserves the exact stationary point $\nabla J(z^*) = 0$ while guaranteeing a Symmetric Positive Definite Hessian ($H \succ 0$) on every Newton iteration.

#### 5.3.3 Tridiagonal Temporal, Spatial, and Day-Night Coupling Prior
Let $\delta_t = z_t - \mu_t$, $\omega_p = 1/\sigma^2_{\text{proc}, b}$, $\alpha_s = 0.10$:
* **Temporal AR(1):**
  $$\ell_{\text{time}} = \frac{\omega_p}{2}\left[(1 - \rho^2)\delta_0^2 + \sum_{t=1}^{T-1}(\delta_t - \rho\,\delta_{t-1})^2\right]$$
  yielding tridiagonal main diagonal $a_0 = a_{T-1} = \omega_p$, $a_t = (1 + \rho^2)\omega_p$ ($1 \le t \le T-2$), and sub/super-diagonal $b_t = -\rho\,\omega_p$.
* **Spatial 1-Ring Coupling (Held Constant Per Color Sweep):**
  $$\Delta \ell_{\text{sp}} = \frac{\alpha_s\,\omega_p}{2}\sum_{g' \in \mathcal{N}(g)} (z_t - z_{g', t})^2 \implies \Delta g_t = \alpha_s\omega_p\big(|\mathcal{N}(g)|z_t - \sum z_{g', t}\big), \quad \Delta a_t = \alpha_s\omega_p|\mathcal{N}(g)|$$
* **Day–Night Coupling (Held Constant Per Half-Day Sweep):**
  $$\Delta \ell_{\text{dn}} = \frac{\eta_{\text{dn}}\omega_p}{2}\big(\delta_t - \beta_{\text{dn}}\delta_{\text{opp}, t}\big)^2 \implies \Delta g_t = \eta_{\text{dn}}\omega_p(\delta_t - \beta_{\text{dn}}\delta_{\text{opp}, t}), \quad \Delta a_t = \eta_{\text{dn}}\omega_p$$
Floor final main diagonal safely with $a_t = \max(a_t, 10^{-6})$.

#### 5.3.4 Fused `float64` Thomas $LDL^\top$, Armijo Line Search, Takahashi & Exact Joint Sampling
For each active chunk of size $B = 1000$ across $T = 9510$ days:
1. Run `6` outer sweeps over `color in (0, 1, 2)` and `dn in ('D', 'N')`.
2. Inside each `(color, dn)` block, run up to `15` Newton iterations:
   * Factorize $H = L D L^\top$: $d_0 = a_0$; for $t = 0\dots T-2$: $l_t = b_t / d_t$, $d_{t+1} = \max(a_{t+1} - l_t b_t,\; 10^{-7})$.
   * Solve $L D L^\top \Delta z = g$ by forward & backward substitution.
   * Try Armijo backtracking steps $\alpha \in \{1.0, 0.5, 0.25, 0.125, 0.0625\}$ updating $z_{\text{new}} = \text{clip}(z - \alpha\,\Delta z, -8.0, 14.0)$ until $J(z_{\text{new}}) \le J(z) - 10^{-4}\alpha\sum g\,\Delta z$. Exit early if $\|\alpha\,\Delta z\|_\infty < 10^{-4}$.
3. **Takahashi Exact Posterior Marginals:**
   $\Sigma_{T-1, T-1} = 1 / d_{T-1}$; for $t = T-2\dots 0$: $\Sigma_{t, t+1} = -l_t \Sigma_{t+1, t+1}$, $\Sigma_{t, t} = (1/d_t) - l_t \Sigma_{t, t+1}$.
4. **Exact Joint Cholesky Trajectory Sampling ($M = 200$):**
   Draw $\varepsilon \sim \mathcal{N}(0, 1)$ of shape `(B_sub, T, 200)`. Solve $L^\top u = D^{-1/2}\varepsilon$:
   $u_{T-1} = \varepsilon_{T-1} / \sqrt{d_{T-1}}$; for $t = T-2\dots 0$: $u_t = \varepsilon_t / \sqrt{d_t} - l_t u_{t+1}$.
   Compute $z^{(m)} = \text{clip}(\hat{z} + u^{(m)}, -8.0, 14.0)$, transform $\Phi^{(m)}_{g,t,\text{DN}} = \exp(z^{(m)})$, integrate 24h flux $\Phi^{(m)}_{g,t} = 0.5(\Phi^{(m)}_{g,t,\text{Day}} + \Phi^{(m)}_{g,t,\text{Night}})$, $\text{FRE}^{(m)}_{g,t} = 0.0864\,\Phi^{(m)}_{g,t}$ (`TJ/day`), and aggregate across month indices to extract `q05, q50, q95, mean` across the 200 trajectories.
5. **GPU Execution Kernel (`cuda_kernels.py`):** Implement the exact Thomas, Takahashi, and Cholesky loops inside a single-launch `torch.float64` CUDA C++ kernel (`load_inline`, one thread per hexagon) whenever CUDA is available, verified bit-for-bit (`< 1e-10`) against the pure PyTorch `thomas_takahashi.py` CPU reference.
6. **Fixed-Lag Rolling Assimilation (`run_fixed_lag_update(window_days=14)`):** Solves exclusively over `[t_start, T-1]` while anchoring `t_start - 1` via the AR(1) prior to the saved state.

### 5.4 Split Conformal Calibration (`CQR`), Attribution & Products

1. **Multiplicative Log-Space Split Conformal Calibration (`conformal.py`):**
   * Compute **MODIS-only** nonconformity scores on held-out `2018–2020` VIIRS-masked monthly tiles comparing `(q05_raw, q50_raw, q95_raw)` against full multi-sensor reference $\Phi_{\text{ref}}$:
     $$s_j = \max\!\left(\frac{\ln(q_{95,\text{raw}, j}+1) - \ln(q_{05,\text{raw}, j}+1)}{2},\; 0.05\right), \qquad r_j = \frac{|\ln(\Phi_{\text{ref}, j}+1) - \ln(q_{50,\text{raw}, j}+1)|}{s_j}$$
     Compute $\hat{c}^{\text{modis}}_{0.90, b} = \text{Quantile}_{\lceil(n_b+1)\cdot 0.90\rceil / n_b}(\{r_j\}) \in [0.85, 3.5]$.
   * Compute **Multi-sensor** conformal multipliers $\hat{c}^{\text{multi}}_{0.90, b}$ identically using leave-one-VIIRS-platform-out (`SNPP` vs `NOAA-20`) runs on `2018–2021`.
   * Rescale outputs:
     $$q_{05,\text{cal}} = \max\!\Big(0,\; \exp(\ln(q_{50}+1) - \hat{c}\,s) - 1\Big), \qquad q_{95,\text{cal}} = \exp(\ln(q_{50}+1) + \hat{c}\,s) - 1$$
2. **Diagnostic Flags:**
   * `quality_flag = 'ABSTAIN_LOW_EVIDENCE'` if $\ge 5$ consecutive days in the month have `mean_cloud_frac > 0.85` AND zero detections across the cell and its 1-ring neighbors.
   * `attribution_flag`: `'REAL_FIRE_CHANGE'` (raw & harmonized changes agree beyond conformal intervals), `'SENSOR_ARTIFACT_CORRECTED'` (raw changed artificially across sensor transition while harmonized CI spans zero step change), or `'AMBIGUOUS_OVERLAPPING_CI'`.
   * Monthly calendar indices relative to `2003–2020` climatology: `hist_median_2003_2020`, `anomaly_ratio = phi_q50 / max(hist_median_2003_2020, 0.5)`, `anomaly_zscore`, `climatology_pctile`.
3. **Three Final Complementary Products (`export_duckdb.py`):**
   * **Product 1 (`obs_harmonized_provenance.parquet`):** Untouched raw sensor fields (`platform`, `data_tier`, `acq_date`, `acq_hhmm`, `frp_mw`, `t_mir_k`, `t_tir_k`, `scan_km`, `track_km`, `conf_raw`, `type_raw`, `ingest_ts`) + derived diagnostics (`h3_index`, `sza_deg`, `solar_hour`, `dn_flag`, `pixel_area_km2`, `vza_deg`, `is_static_industrial`, `is_sama_filtered`, `bowtie_weight`, `reliability_prob`, `event_cluster_id`, `forward_gain_c`, `obs_sigma`).
   * **Product 2 (`state_reconstructed_monthly.parquet` dense + `state_reconstructed_daily.parquet` sparse where `phi_q50_mw > 0.25` OR within $\pm 3$ days of detection OR spotlight ROI):** `h3_index`, `country_iso`, `pyrome_id`, `year`, `month`, raw unharmonized baselines (`raw_count_sum`, `raw_frp_sum_mw`, `raw_clustered_events`), reconstructed states (`phi_day_mw`, `phi_night_mw`, `phi_q05_mw`, `phi_q50_mw`, `phi_q95_mw`, `phi_mean_mw`, `phi_flux_mw_km2`, `fre_tj_q50`, `energy_equiv_events_q50`, `hurdle_adj_events`, `est_active_area_km2`, `night_day_ratio`), and calendar indices.
   * **Product 3 (`uncertainty_diagnostics.parquet`, 1-to-1 with Product 2):** `conformal_c_factor`, `rel_uncertainty_width`, `measured_fraction`, `inferred_fraction`, `active_platforms_bitmask`, `mean_cloud_frac`, `has_nrt_gapfill`, `attribution_flag`, `quality_flag`, `revision_count`.

---

## 6. Complete Data Acquisition and Processing Pipeline (Environment 1 -> Environment 2)

1. **Primary Historical Active Fire Ingestion (`download_data.py`):**
   * Connect to `fuoco.geog.umd.edu` (`fire` / `burnt`, attempting SFTP first, FTP fallback) to stream monthly `MCD14ML` (`.txt.gz` space-delimited) and `VNP14IMGML`, `VJ114IMGML`, `VJ214IMGML` (`.csv.gz`) one month at a time into `/tmp`, deleting temp files immediately.
   * If `fuoco.geog.umd.edu` is unreachable, execute DuckDB `httpfs` spatial-pruned extraction directly from the **Source Cooperative NASA FIRMS Portolan GeoParquet 2.0 mirror** (`https://data.source.coop/portolan-mirrors/firms-catalog/detections/year=<YYYY>/detections.parquet`, partitioned 2000–present with Hilbert spatial row-group ordering).
   * Fetch trailing 2026 `*_NRT` detections (`MODIS_NRT`, `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT`) from the latest Science date through `2026-11-14` via NASA FIRMS `/api/data_availability/csv/{MAP_KEY}/ALL` + `/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/5/{DATE}` (`DAY_RANGE <= 5` strictly).
2. **Ancillary Cloud, Burned Area, and Country Boundary Ingestion:**
   * Fetch Natural Earth `50m` Admin-0 GeoJSON (`https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson`) and store geometries for `ARG`, `DZA`, `BRA`, `IDN`, `AUS`, `USA`.
   * Stream UMD `MOD14C8Q`/`MYD14C8Q` (8-day `0.25°`, falling back to monthly `Mx14CMQ` where absent) and `MCD64CMQ` (`0.25°` monthly burned area) HDF4 files inside Env 1, extract `CloudPix`, `MissingPix`, `LandPix`, `BurnedArea` via `pyhdf`, emit `cloud_visibility_025deg.parquet` (`< 25 MB`) and `burned_area_val_025deg.parquet` (`< 20 MB`), and delete raw HDF files.
3. **Normalization, Polygon Clipping & Verification (`normalize_dedup.py`):**
   * Parse heterogeneous source schemas cleanly into canonical types (`country_iso, sensor_family, platform, data_tier, latitude, longitude, acq_date, acq_hhmm, t_mir_k, t_tir_k, scan_km, track_km, sample_idx, pixarea_km2, frp_mw, conf_raw, conf_num, type_raw, version_raw, ingest_ts`).
   * Clip strictly with Natural Earth Admin-0 polygons (`shapely.contains_xy`).
   * **Tier Dominance Deduplication:** Drop `NRT` entirely for any `(platform, acq_date)` present in `ML_SCIENCE` or `SP_SCIENCE` (preventing LUT geolocation shift duplicates); resolve any intra-tier exact duplicates on `(platform, acq_date, acq_hhmm, ROUND(latitude, 3), ROUND(longitude, 3))` keeping `tier_rank ASC, ingest_ts DESC`.
   * Compute `platform_active_day[P, T]` (`>= 5` total detections/day across the 6 countries) and verify continuity: `ARG` Jul–Aug 2026 Chilecito detections present, `SNPP` ends `2026-11-02`, `NOAA-20/21` continue through `2026-11-14`, `DZA` preserves `type=2` static flares.

---

## 7. Complete Testing & Validation Framework (`pytest` — 166+ Required Tests)

All 10 test suites under `tests/` MUST pass 100% with concrete numerical/scientific assertions:

1. **`tests/test_01_ingestion_schema.py` (18 tests):** Multi-schema parsing (`MCD14ML` space `.txt`, `Vxx14IMGML` CSV, FIRMS API/Portolan), satellite alias normalization, mixed confidence parsing, missing `type -> -1`, `(platform, acq_date)` Science-tier dominance removing shifted NRT duplicates, polygon clipping excluding Bolivia/Tunisia/Chile neighbors, `(hhmm // 100) + (hhmm % 100)/60` base-60 time parsing, and SZA Day/Night assignment replacing orbit-node NRT errors at high latitudes.
2. **`tests/test_02_geometry_masking.py` (18 tests):** Exact spherical `Sample -> scan, track, VZA` geometry (checking nadir `1.00 km²` and edge `~9.7 km²` MODIS), monotone spline VZA inversion, MODIS bow-tie weight conservation (`2 × 40 MW` overlap -> `40 MW`), SAMA nighttime low-conf filter, `H3-Res8` gas-flare separation absorbing 200 m geolocation jitter while keeping seasonal wildfires, `h3 v3/v4` `int64 <-> str` conversion, and exact H3 **3-coloring validity** (`color[u] != color[v]` across every adjacent edge).
3. **`tests/test_03_event_clustering.py` (14 tests):** 3×3 contiguous VIIRS cluster -> 1 event, separated fires -> 2 events, same-afternoon `SNPP + N20 + N21` kept as 3 distinct pass rows (never summed), 5-minute pass boundary split, `i4_m13_fallback` vs Tobit `sat_flag` separation.
4. **`tests/test_04_observation_operators.py` (20 tests):** Detection threshold ordering (`MODIS_day > MODIS_night > VIIRS_night`), analytical survival integrals $\psi \in (0, 1]$ monotonicity, separate 12-hour `Day` (`06–18`) and `Night` (`18–06`) Fourier integral unit-mean checks (`1.0 ± 1e-4`), `ln(v)` strictly absent from detected gain $\mu^{\text{gain, det}}$, monotonic EQM quantile order, heteroscedastic $\sigma^2$ ordering, and coordinate-shift invariance of GMM pyromes.
5. **`tests/test_05_likelihood_derivatives.py` (22 tests):** Finite-difference gradient & Hessian parity (`< 1e-6`) across Obs, Tobit Saturation (over `w in [-35, +5]` testing asymptotic tail without `NaN`/`Inf`), and Hurdle Zero; Tobit asymmetry; convexity guard (`h_miss >= 0`) inside partial-cloud transition `(v=0.7, Lambda=2.5)`; uniform $\Phi = e^z$ consistency across Case A/B/C; and empirical `platform_active_day` masking out simulated spacecraft safe-mode outages.
6. **`tests/test_06_gpu_solver_math.py` (20 tests):** `float64` Thomas solver vs. `torch.linalg.solve` (`< 1e-10`); Takahashi diagonal & lag-1 vs. `torch.linalg.inv(H)` (`< 1e-10`); empirical covariance of $10{,}000$ joint Cholesky draws $L^\top u = D^{-1/2}\varepsilon$ matching $H^{-1}$; Armijo strict objective descent; 3-color spatial cloud-gap recovery; CUDA `load_inline` kernel bit-parity (`< 1e-10`) vs PyTorch CPU reference; extreme range `10^-4 .. 10^6 MW` stability.
7. **`tests/test_07_synthetic_twin_truth.py` (18 tests):** Full 26-year digital twin simulator with known $X_{\text{true}}$ (`Day/Night` contrast, 2002 Aqua, 2012 SNPP, 2018 N20, 2022 drift, 2023 N21, Nov 2026 SNPP end, 50% clouds, bow-tie, gas flares) asserting: 2012 raw jump `> +150%` eliminated (`< ±10%`); 3-satellite inflation eliminated; `night_day_ratio` accurately preserved (`error < 15%`, never collapsing to `1.0`); Nov 2026 cutoff unbiased; post-2022 drift corrected; monthly correlation `r > 0.90`; 5× drought fire spike preserved within `±15%`.
8. **`tests/test_08_conformal_uncertainty.py` (12 tests):** Split-conformal coverage inside `[1 - alpha - 0.03, 1 - alpha + 0.05]` across $\alpha \in \{0.05, 0.10, 0.20\}$; `q05 <= q50 <= q95` non-negativity; 2005 uncertainty strictly wider than 2024; cloudy uncertainty wider than clear months; `ABSTAIN_LOW_EVIDENCE` trigger; 14-day fixed-lag update matching full batch (`< 1e-3`) across an active fire transition.
9. **`tests/test_09_validation_diagnostics.py` (12 tests):** Chow & Pettitt break tests catching raw 2012 break (`p < 0.001`) and passing harmonized control (`p > 0.05`); 2019–2021 VIIRS-masked ablation outperforming raw MODIS on CRPS & Spearman $\rho$; `MCD64CMQ` burned-area correlation check; Product 1/2/3 schema & sparse daily export invariants.
10. **`tests/test_10_webapp_api_scale.py` (12 tests):** FastAPI endpoints return 200 & valid JSON; CSV injection sanitization (`=`, `+`, `-`, `@` prefixed with `'`); empty-bbox safety; DuckDB query latency `< 300 ms`; GPU chunk VRAM (`batch_size = 1000`, `T = 9510`) stays `< 6 GB` per GPU.

---

## 8. Sequential Implementation Plan & Verification Gates

1. **Phase 1 (Hours 0–4 — Env 1 Stager & Ingestion Tests):** Build repo skeleton, `settings.py`, `paths.py`, `build_wheels.sh`, `download_data.py`, `normalize_dedup.py`, and `test_01`. Verify connectivity to `fuoco.geog.umd.edu` / Source Coop Portolan + FIRMS API at Minute 15. Stage clean Parquet (`<= 2.5 GB`) and `.whl` bundle (`<= 80 MB`). **Gate 1: `test_01` passes 100% + coverage audit confirms Chilecito 2026, Nov 2 SNPP cutoff, DZA flares.**
2. **Phase 2 (Hours 4–10 — Geometry & Clustering):** Implement `h3_utils.py`, `solar_geometry.py`, `preprocess_geometry.py`, `test_02`, `test_03`. **Gate 2: `test_01..test_03` pass 100% on CPU.**
3. **Phase 3 (Hours 10–14 — Calibration & Derivatives):** Implement `pyromes.py`, `operators.py`, `likelihoods.py`, `priors.py`, `test_04`, `test_05`. **Gate 3: `test_04..test_05` pass 100% on CPU.**
4. **Phase 4 (Hours 14–17 — Solver, CUDA Kernel, Synthetic Twin & Timing Gate):** Implement `thomas_takahashi.py`, `cuda_kernels.py`, `solver.py`, `synthetic_twin.py`, `test_06`, `test_07`. **Gate 4: `test_01..test_07` pass 100%; single-chunk GPU timing benchmark verified `< 6 GB` VRAM and fast enough to complete all active cells inside `< 1.5` GPU-hours.**
5. **Phase 5 (Hours 17–21 — Full GPU Execution & Conformal Export):** Run `run_env2_cpu_prep.py` and `run_env2_gpu_solve.py` in Env 2; implement `conformal.py`, `export_duckdb.py`, `test_08`. **Gate 5: `fire_harmonized.duckdb` + 3 Parquet tables (`<= 4 GB`) exported cleanly and `test_08` passes.**
6. **Phase 6 (Hours 21–24 — Scientific Validation):** Implement `diagnostics.py`, run `run_env2_validate.py` (`test_09`). **Gate 6: All structural break, ablation, `MCD64CMQ`, and spotlight validations pass.**
7. **Phase 7 (Hours 24–34 — Web Application & Final Audit):** Implement `server.py` + `static/` (`MapLibre GL JS` + `Apache ECharts` interactive ROI map, calendar heatmap with `viridis`/`cividis` ramps and high-uncertainty hatching, historical median anomaly plots, provenance timeline, 6 spotlight presets, CSV export) and `test_10`. **Gate 7: Full `pytest` across all 166+ tests passes 100%, and UI spotlight views verified.**

---

## 9. Agent Behavioral Boundaries

### Must Do
* Maintain 100% adherence to the 10 Scientific Invariants and the exact equations in Section 5.
* Maintain strict Environment 1 / 2 / 3 separation (`pyhdf` strictly isolated in Env 1).
* Perform all tridiagonal/Takahashi/Cholesky algebra in `torch.float64`.
* Verify every mathematical module with analytical/finite-difference/dense-reference tests before proceeding to the next phase.
* Report exact empirical test metrics and wall-clock benchmarks honestly.

### Must Not Do
* Never introduce competing pipelines, alternative fallback UIs, or unrequested side branches.
* Never put cloud visibility $\ln v$ into detected observation likelihoods (`Case A/B`).
* Never normalize $D(\tau)$ over 24 hours across `z_Day` and `z_Night` (must normalize independently within 12-hour Day/Night windows).
* Never write dense `N_active × 9,510` daily rows to disk (always emit sparse episode rows).
* Never hardcode `/kaggle/...` paths outside `paths.py`, leave placeholder tests, or report unverified code as complete.

---

## 10. Final Deliverables & Acceptance Checklist

The completed repository and execution artifacts must provide:
1. Reproducible ingestion and wheel scripts (`src/ingest/`, `scripts/run_env1_stage.py`).
2. Complete offline harmonization engine (`src/geometry/`, `src/calibration/`, `src/engine/`, `scripts/run_env2_*.py`).
3. `fire_harmonized.duckdb` + 3 indexed Zstd Parquet products (`obs_harmonized_provenance`, `state_reconstructed_monthly` dense & `daily` sparse, `uncertainty_diagnostics`) within `<= 4 GB`.
4. Validation report (`diagnostics.json` + summary figures) confirming structural continuity, 2019–2021 VIIRS-masked ablation gains, `MCD64CMQ` burned-area correlation, and Chilecito 2026 / Algeria gas flare / Nov 2026 SNPP continuity.
5. Complete interactive FastAPI + MapLibre + ECharts Burning Activity Calendar Web App (`src/webapp/`).
6. **100% passing `pytest` run (`>= 166` tests across `test_01..test_10`)**.


## Brutally Honest Audit: Was the Previous Specification Truly Everything the AI Needed?

**No.** If you handed the previous specification straight to an autonomous coding agent, it would still hit **8 concrete failure points or missing contracts** where it would either crash on real NASA data or be forced to guess:

1. **Wrong HDF4 Layer Names in `MOD14CMQ` / `MOD14C8Q` (Guaranteed Env 1 Crash):**
   The previous spec told the agent to read `CloudPix`, `MissingPix`, and `LandPix` from `Mx14CMQ`. Verified directly against the official MODIS Collection 6 Active Fire User's Guide (Table 8), the actual CMG SDS layer names are **`MeanCloudFraction`**, **`CorrFirePix`**, **`CloudCorrFirePix`**, and **`RawFirePix`**. Similarly, `MCD64CMQ` contains **`BurnedArea`**, **`QA`**, **`UnmappedFraction`**, and optionally **`LandCoverDist`** (absent in recent trailing months due to the 1-year `MCD12Q1` lag) on a `720 × 1440` grid centered at `(-179.875°, +89.875°)`. Without these exact layer names and dynamic HDF4 `scale_factor` / `_FillValue` reading, `pyhdf` crashes immediately.
2. **Missing Trailing-2026 Cloud Data Fallback (`NaN` Propagation in Jul–Nov 2026):**
   `MOD14CMQ`/`MOD14C8Q` science grids lag real time by ~1–3 months, meaning **August–November 2026** (including the end of CONAE's Chilecito spotlight and the Nov 2, 2026 S-NPP cutoff) will not yet exist in CMG files. Without an explicit climatology fallback (`2012–2025` multi-year mean cloud fraction per `(0.25° cell, month)` tagged with `cloud_source = 'CLIMATOLOGY_FALLBACK'`), joining the cloud table drops or produces `NaN`s for late 2026.
3. **No Autonomous Link / Directory Discovery Protocol:**
   You specifically asked for instructions telling the agent how to find data elsewhere if any URL or folder path fails. The previous spec listed paths without instructing the agent how to recursively scan `fuoco.geog.umd.edu` via SFTP/FTP, read the Source Cooperative Portolan STAC `catalog.json`, query NASA Earthdata CMR (`cmr.earthdata.nasa.gov/search/granules.json`) or GeoBoundaries (`geoboundaries.org/api/current/gbOpen/{ISO3}/ADM0/`) as automated mirrors, or validate discovered schemas before ingesting.
4. **High-Latitude `SZA < 85°` vs. `[06:00, 18:00]` Normalization Mismatch (`Alaska` Bug):**
   Assigning `dn_flag` by `SZA < 85°` while normalizing `D_Day` over `[06:00, 18:00]` breaks in June/December Alaska (where `SZA < 85°` lasts ~20 hours in summer and ~4 hours in winter). Defining the two latent states strictly by **Local Solar Time Half-Day Windows** (`Day = [06:00, 18:00)`, `Night = [18:00, 06:00)`) while preserving `sza_deg` and raw `dn` in Product 1 keeps the 12-hour diurnal unit-mean integrals and `0.5 * (Phi_Day + Phi_Night)` mathematically exact at all latitudes.
5. **Five Equations Left in Prose Instead of Formulas:**
   The previous spec described (a) `psi^{FRP, eff}` blending, (b) `g_{area, b}` training target/features, (c) `hurdle_adj_events` aggregation across same-orbit sensors, (d) exact boolean rules for `attribution_flag`, and (e) `active_platforms_bitmask` bit positions in words instead of exact equations.
6. **Missing Exact Data-Source Schemas (`Portolan` GeoParquet & `Vxx14IMGML`):**
   `VNP14IMGML` already includes `Pixarea` (`km²`) directly alongside `YYYYMMDD, HHMM, Sat, Lat, Lon, T_I4, T_I5, Sample, FRP, Conf, Type`. Meanwhile, Source Cooperative's `portolan-mirrors/firms-catalog` stores GeoParquet 2.0 with `geometry` (`ST_Point`), `acq_datetime`, `sensor`, `satellite`, `quality`, `brightness`/`bright_t31`/`bright_ti4`/`bright_ti5`, `scan`, `track`, `frp`, `daynight`, `confidence`, `confidence_pct`, `type`, `_month`, `_hilbert`. Giving the agent all three exact schemas removes all guesswork.
7. **Missing Backend JSON Contracts, Spotlight Coordinates & UI Specification:**
   `server.py` and `app.js` require explicit JSON payload keys, error codes, bounding-box limits, and exact coordinates for the 6 spotlight presets so frontend and backend match on the first run.
8. **Missing `pyproject.toml` Dependency Split & `cuda_kernels.py` CPU/GPU Fallback Contract:**
   The agent needs explicit `[project.optional-dependencies]` separating Environment 1, Environment 2, and Environment 3, plus a clean `cuda_kernels.py` dispatch rule so CPU tests pass cleanly when `nvcc` is unavailable.

---

## Complete, Self-Contained AI Agent Prompt & Implementation Specification

Copy everything inside the horizontal rules below directly as your prompt / `AGENT_MASTER_SPEC.md` for your coding agent. It transfers 100% of the context, mathematical derivations, exact file schemas, autonomous data-discovery failover protocol, testing framework, and implementation rules cleanly from scratch.

---

# MASTER ENGINEERING HANDOFF SPECIFICATION
# MODIS–VIIRS Physics-Informed Bayesian Harmonization Engine & Burning Activity Calendar

## 0. Agent Execution Mandate
You are the lead scientific software engineer implementing a complete, end-to-end Earth Observation active-fire harmonization engine and interactive web application. You have **zero prior context**; every scientific principle, mathematical derivation, file schema, architectural boundary, data source, failover discovery protocol, and acceptance test you need is contained strictly inside this specification.

**Strict Rules:**
* Implement **one single unified system** obeying this exact specification. Do **not** invent alternative pipelines, fallback UIs, competing models, stub implementations, or unverified assumptions.
* **Reference Date:** November 14, 2026.

---

## 1. Complete Problem Statement

### 1.1 Origin, Goal, and Core Challenge
This system solves the **2026 NASA Space Apps Challenge: *"Harmonization of MODIS and VIIRS Hot Spots"*** (authored by Marina Compagnucci, CONAE — The Argentinian Space Agency). Satellite thermal anomaly ("active fire hotspot") observations span over 26 years (November 2000 – November 14, 2026), yet are fractured across sensors whose raw detections cannot be directly compared or combined.

Your mission has two tightly coupled deliverables:
1. **Scientific Harmonization Engine:** Reconstruct a sensor-independent, uncertainty-quantified, physically consistent 2000–2026 record of terrestrial burning activity combining MODIS and VIIRS observations so that instrument switches no longer induce artificial steps, trends, or duplicate fire counts.
2. **Interactive Burning Activity Calendar Web Application:** Expose the harmonized dataset in a fast, interactive web application allowing emergency managers, scientists, and land managers to select any Area of Interest (AOI) on a map, view a Year × Month burning activity calendar heatmap, compare current burning against historical medians (`2003–2020`), examine trends, anomalies, and uncertainty bands, inspect spotlight wildfire case studies (notably CONAE's **July–August 2026 wildfire activation in Chilecito, La Rioja, Argentina**, `lat ~ -29.16, lon ~ -67.50`), and download sanitized CSV analytics.

### 1.2 Why Existing Implementations and Naive Merging Fail
Raw hotspot records come from two fundamentally different instrument families aboard five polar-orbiting spacecraft:
* **MODIS (`1 km` nadir resolution, `3.96 μm` Ch 21/22 & `11 μm` Ch 31):** **Terra** (`2000-11-01` to present) and **Aqua** (`2002-07-04` to present).
* **VIIRS (`375 m` I-band nadir resolution, `3.74 μm` I4 & `11.45 μm` I5):** **Suomi-NPP / S-NPP** (`2012-01-20` to **`2026-11-02`**, when FIRMS ceased S-NPP delivery), **NOAA-20** (`2018-04-01` to present), and **NOAA-21** (`2023-03-01` / `2024-01-17` to present).

Naive concatenation, raw averaging, or global linear scaling fails for seven physical and engineering reasons:
1. **Resolution & Sensitivity Step-Jump:** At nadir, a MODIS pixel covers $\sim 1.0\,\text{km}^2$ with a minimum detectable Fire Radiative Power ($F_{\min}$) of `~3.7–6.4 MW`, whereas a VIIRS 375 m pixel covers $\sim 0.141\,\text{km}^2$ (`~7×` finer) and detects fires down to `~0.5–0.9 MW`. Switching from MODIS to VIIRS in 2012 multiplies raw hotspot counts by `3×–10×` even under identical burning.
2. **Off-Nadir Stretching & Bow-Tie Distortion:** MODIS pixels grow to `~4.84 × 2.01 km ≈ 9.72 km²` at `65°` View Zenith Angle (VZA), where $F_{\min}$ rises to `~22–40 MW` and adjacent 10-line scans overlap up to ~50% ("bow-tie effect"), double-counting edge fire pixels. VIIRS instead uses onboard `3:1 / 2:1 / 1:1` pixel aggregation and deletes redundant bow-tie fill pixels, reaching only `~0.62 km²` at swath edge.
3. **Spectral Bandpass & Saturation Asymmetry:** Raw brightness temperatures (`brightness`/`T21` vs `bright_ti4`/`T_I4`) measure different wavelengths over different footprint dilutions and cannot be averaged. VIIRS `I4` saturates at `367 K` and relies on co-located `750 m` `M13` (`4.05 μm`) pixels for FRP—splitting single `M13` retrievals across multiple `375 m` pixels or occasionally failing/clipping under extreme fire conditions.
4. **Same-Orbit Constellation Multiplicity:** `S-NPP`, `NOAA-20`, and `NOAA-21` share the ~13:30 / 01:30 orbital plane ~50 minutes apart. Summing their detections between 2018 and 2026 triples apparent fire counts/FRP compared to 2012–2017 (`S-NPP` alone), and `S-NPP`'s termination on `2026-11-02` induces a sharp drop unless modeled via conditional sensor likelihoods.
5. **Post-2022 Orbital Drift:** Beginning in 2022, `Terra` (~10:30 AM nominal) and `Aqua` (~1:30 PM nominal) exited controlled station-keeping, drifting progressively toward ~8:45 AM and ~3:30–3:50 PM respectively by late 2026. Diurnal fire-cycle variation across drifting overpass hours masquerades as climate trends unless modeled explicitly in continuous local solar time.
6. **Clouds, Artifacts, Gas Flares & Processing Tiers:** Tropical clouds obscure >60–70% of passes; nighttime low-confidence VIIRS detections inside the South Atlantic Magnetic Anomaly (SAMA) create false radiation spikes; static oil/gas flares (e.g., Saharan Algeria) and volcanoes contaminate biomass fire counts—complicated because Near Real-Time (`NRT`) data omits the `type` column and determines `daynight` purely by ascending/descending orbit node rather than consistent solar geometry; and NRT geolocation offsets (40–300 m relative to Science `ML` lookup tables) cause duplicate leaks under naive lat/lon deduplication.
7. **Physical Identifiability Limit:** Orbital mid/thermal infrared FRP alone cannot uniquely invert sub-pixel flame temperature ($T_f$), exact flame area ($A_f$), fuel load, or combustion efficiency. The engine must output strictly identifiable physical quantities ($\Phi$, $\text{FRE}$, estimated burning area, and calibrated event indices) and explicitly quantify uncertainty and low-evidence abstention.

### 1.3 Success and Verification Criteria
The solution is complete and verified if and only if:
1. All sensor step jumps (`2002`, `2012`, `2018`, `2023`, `2026-11`) and `2022–2026` drift biases are eliminated on stationary control benchmarks (`p > 0.05` on Chow/Pettitt structural-break tests after harmonization, vs `p < 0.001` raw).
2. True extreme wildfire episodes (**Chilecito, Argentina Jul–Aug 2026**; **Kabylie, Algeria 2021/2023**; **Black Summer Australia 2019–2020**) are accurately reconstructed while static Saharan flares (**Hassi Messaoud, Algeria**) and SAMA artifacts are isolated.
3. Uncertainty is empirically valid: 2000–2011 MODIS-only intervals are strictly wider than 2012–2026 multi-sensor intervals; heavy cloud blindness widens bounds and triggers `ABSTAIN_LOW_EVIDENCE`; Split Conformal Calibration achieves $\ge 90\%$ empirical coverage on held-out `2018–2020` VIIRS-masked folds and beats unharmonized MODIS on CRPS and Spearman rank correlation.
4. Harmonized monthly $\text{FRE}$ correlates more strongly with independent `MCD64CMQ` burned area (`2001–2025`) than unharmonized FIRMS hotspot counts.
5. All `166+` automated `pytest` tests across `test_01..test_10` pass 100%, output DuckDB + Parquet artifacts fit within `<= 4 GB`, and the FastAPI + DuckDB web app responds in `< 300 ms`.

---

## 2. Selected Technical Approach & Core Scientific Invariants

### 2.1 Theoretical Foundation: Forward-Operator State-Space Bayesian Data Assimilation
Instead of training one sensor to imitate another, the system separates the **Physical World** from the **Measurement World**:
* **Spatial & Temporal Support:** Space is partitioned globally into equal-area **Uber H3 Resolution 6** hexagons ($\bar{A} \approx 36.13\,\text{km}^2$). Time runs daily over $T = 9{,}510$ local solar days from `2000-11-01` (`t = 0`) to `2026-11-14` (`t = 9509`), split cleanly into two 12-hour Local Solar Time half-day regimes: **Day (`[06:00, 18:00)`)** and **Night (`[18:00, 06:00)`)**.
* **Latent Physical State:** Every active hexagon $g$ on solar day $t$ holds two continuous log-flux states representing true cloud-free mean radiative power (MW) during Day and Night:
  $$z_{g, t, \text{Day}} = \ln \Phi_{g, t, \text{Day}}, \qquad z_{g, t, \text{Night}} = \ln \Phi_{g, t, \text{Night}}$$
  with uniform forward mapping $\Phi = \exp(z)$ everywhere in likelihoods and trajectory sampling, and dormant background floor $z_{\min} = \ln(0.01) = -4.605170$.
* **Forward Sensor Observation Operators:** For every operational platform $p$, an actual detection $Y_{p,g,t,\text{DN}} > 0$ satisfies:
  $$\ln Y_{p,g,t,\text{DN}} = z_{g,t,\text{DN}} + \ln D_{b(g), s(t), \text{DN}}(\bar{\tau}) + \ln \gamma_{p, b}(\bar{\theta}, q) + \ln \psi^{\text{FRP, eff}}_{p,b}(\bar{A}, \text{DN}, E_{\text{raw}}) + \epsilon_{p,g,t,\text{DN}}$$
  while non-detection (`Y == 0`) on an operational, healthy satellite pass follows a **Cloud-Gated Poisson Hurdle Model**:
  $$P(D_{p,g,t,\text{DN}} = 0 \mid z_{g,t,\text{DN}}) = (1 - v_{p,g,t}) + v_{p,g,t}\exp\!\left(-\lambda_{0,b}\,\psi^{\text{CNT}}_{p,b}(\bar{A}_{\text{avg}}, \text{DN})\,\frac{\exp(z_{g,t,\text{DN}})}{\bar{F}_{\text{event}, b}}\right)$$
  Crucially, cloud visibility $v_{p,g,t}$ governs **only** detection probability (`D = 0` vs `D = 1`) and **never** scales detected FRP magnitude (`Y > 0`).
* **Joint Reconstruction & Uncertainty:** A **Space × Time × Day-Night Gaussian Markov Random Field (GMRF)** MAP-Laplace solver fuses all platforms jointly in `torch.float64` across H3 3-Coloring Gauss-Seidel sweeps using a fused Thomas $LDL^\top$ solver, Takahashi variance recursion, $M=200$ exact joint Cholesky posterior trajectories, and multiplicative log-space **Split Conformal Quantile Rescaling (`CQR`)**.

### 2.2 Ten Non-Negotiable Scientific Invariants
1. **INV-01 (No Imitation Learning):** Never train `VIIRS -> MODIS` or `MODIS -> VIIRS`.
2. **INV-02 (Never Sum Same-Orbit Satellites):** `SNPP`, `NOAA-20`, and `NOAA-21` enter as separate conditional likelihood terms within the same half-day window, never summed counts or FRP.
3. **INV-03 (Pre-Drift Calibration Freeze):** Calibrate all cross-sensor operators strictly on **`2012-01-20` to `2021-12-31`**. Handle 2022–2026 Terra/Aqua drift strictly through diurnal solar-time operator $D(\tau)$.
4. **INV-04 (Explicit Operational & Outage Masking):** Encode launch/retirement windows (closing `SNPP` on `2026-11-02`) AND automatically mask multi-day spacecraft safe-mode outages (`platform_active_day[p, t]`).
5. **INV-05 (Separate Physical State from Observation Model):** Never inject cloud multipliers into detected FRP magnitudes.
6. **INV-06 (No Raw Brightness Averaging):** Never average MODIS `T21`/`T31` with VIIRS `T_I4`/`T_I5`.
7. **INV-07 (Honest Censoring):** Account for MODIS pixel-area detection threshold $F_{\min}$ on zero-detection days via the Hurdle model so pre-2012 uncertainty widens naturally.
8. **INV-08 (Static Thermal Anomaly Isolation):** Exclude volcanoes (`type=1`), offshore (`type=3`), SAMA nighttime low-confidence artifacts, and `H3-Res8` persistent industrial flares from biomass fusion while retaining raw records in Product 1.
9. **INV-09 (Zero Data Destruction):** Export 3 distinct output products separating raw provenance, latent states, and diagnostic uncertainty.
10. **INV-10 (No Coordinate Leakage):** Pyrome clustering uses strictly behavioral fire features, never latitude or longitude.

### 2.3 Target Benchmark Scope
While all mathematical equations are 100% coordinate-agnostic, the system executes at **H3 Resolution 6** across **six polygon-clipped countries** covering every major fire regime on Earth while remaining under tight compute/disk limits:
* `ARG` (Argentina): CONAE Jul–Aug 2026 Chilecito spotlight (`lat ~ -29.16, lon ~ -67.50`), Pampas crop burns, SAMA.
* `DZA` (Algeria): Kabylie Mediterranean crown fires (`2021`, `2023`) vs. Saharan Hassi Messaoud static gas flares.
* `BRA` (Brazil): Amazon deforestation + Cerrado savanna 2012–2021 calibration backbone, heavy wet-season clouds, SAMA.
* `IDN` (Indonesia): Extreme equatorial cloud cover, nighttime peat smoldering, low-evidence abstention stress test.
* `AUS` (Australia): Northern low-FRP savannas + 2019–2020 Black Summer VIIRS `I4` saturation megafires.
* `USA` (United States — `CONUS` + `Alaska`): Western megafires, SE prescribed small burns, Alaska boreal clusters.

### 2.4 Limitations and Assumptions (Explicitly Documented)
* Polar-orbiting satellites sample ~10:30, ~13:30, and drifting ~08:45–15:50 local times; therefore, 24-hour diurnal curve curvature during late afternoon (`16:00–20:00`) relies on 2-harmonic smoothness.
* Applying `2012–2021` within-pyrome fire cluster intensity shapes backward to `2000–2011` assumes stationarity of the underlying sub-threshold fire-size distribution shape within each behavioral pyrome.
* Active-fire FRP cannot distinguish fuel consumption or combustion efficiency without auxiliary biomass/fuel models; outputs strictly avoid unidentifiable claims.
* `run_fixed_lag_update(window_days=14)` provides incremental sliding-window NRT/Science revision capability via callable Python API rather than an always-on 24/7 ingestion daemon.

---

## 3. Project Context, Execution Environments, and Deliverables

### 3.1 Three-Environment Architecture
1. **Environment 1 (Data & Wheel Stager — Internet ON, CPU, Python 3.12):**
   * Performs cloud-to-cloud ingestion, HDF4 extraction (`apt-get install -y libhdf4-dev && pip install pyhdf`), Natural Earth 50m Admin-0 polygon clipping, schema normalization, Science-over-NRT deduplication, and packages Python 3.12 `manylinux2014_x86_64` wheels strictly for `duckdb`, `polars`, and `h3`.
   * Emits staged Input Datasets (`staged_data/` <= `2.5 GB`, `offline_wheels/` <= `80 MB`).
2. **Environment 2 (Offline Harmonization Compute — Internet OFF, GPU, Python 3.12):**
   * Kaggle GPU Docker preinstalls `torch`, `numpy`, `scipy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `shapely`, `rasterio`, `pytest`. Never imports `pyhdf`.
   * Installs `duckdb`, `polars`, `h3` via `pip install --no-index --find-links=...`.
   * Detects GPUs dynamically (`torch.cuda.device_count()`), bounding per-GPU VRAM `< 6 GB` (`batch_size = 1000` active cells) so execution runs cleanly on `T4 x2` (15 GB/GPU), `1x L4` (24 GB), or `4x L4` (24 GB/GPU).
   * Exports `fire_harmonized.duckdb` + 3 Zstd Parquet products (`<= 4 GB` total under 20 GB working disk limit) + validation diagnostics.
3. **Environment 3 (Web Application Server):**
   * Runs `FastAPI + DuckDB` serving the static vanilla ES6 `MapLibre GL JS + Apache ECharts` application directly from `fire_harmonized.duckdb`.

---

## 4. Complete Codebase Architecture & Module Contracts

Implement **strictly** this directory layout and module contract:

```text
fire-harmonizer/
├── Makefile
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── paths.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── build_wheels.sh
│   │   ├── download_data.py
│   │   └── normalize_dedup.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── h3_utils.py
│   │   ├── solar_geometry.py
│   │   └── preprocess_geometry.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── pyromes.py
│   │   └── operators.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── likelihoods.py
│   │   ├── priors.py
│   │   ├── thomas_takahashi.py
│   │   ├── cuda_kernels.py
│   │   ├── solver.py
│   │   └── conformal.py
│   ├── products/
│   │   ├── __init__.py
│   │   └── export_duckdb.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── synthetic_twin.py
│   │   └── diagnostics.py
│   └── webapp/
│       ├── __init__.py
│       ├── server.py
│       └── static/
│           ├── index.html
│           ├── app.js
│           └── styles.css
├── scripts/
│   ├── run_env1_stage.py
│   ├── run_env2_cpu_prep.py
│   ├── run_env2_gpu_solve.py
│   └── run_env2_validate.py
└── tests/
    ├── conftest.py
    ├── test_01_ingestion_schema.py
    ├── test_02_geometry_masking.py
    ├── test_03_event_clustering.py
    ├── test_04_observation_operators.py
    ├── test_05_likelihood_derivatives.py
    ├── test_06_gpu_solver_math.py
    ├── test_07_synthetic_twin_truth.py
    ├── test_08_conformal_uncertainty.py
    ├── test_09_validation_diagnostics.py
    └── test_10_webapp_api_scale.py
```

### Exact Module Responsibilities & Interfaces

1. **`pyproject.toml` & `Makefile`:**
   * `pyproject.toml` defines package `fire_harmonizer` (`requires-python = ">=3.10,<3.13"`) with extras: `ingest = ["requests", "paramiko", "beautifulsoup4", "lxml", "pyhdf", "duckdb", "polars", "shapely", "pyarrow"]`, `engine = ["torch", "numpy", "scipy", "pandas", "pyarrow", "duckdb", "polars", "h3", "scikit-learn", "lightgbm", "shapely"]`, `webapp = ["fastapi", "uvicorn", "duckdb", "h3", "pydantic"]`, `dev = ["pytest", "httpx"]`.
   * `Makefile` exposes `stage`, `cpu-prep`, `gpu-solve`, `validate`, `test`, `serve`.

2. **`src/config/settings.py`:**
   * **Contains:** Frozen dataclasses & global constants: `TARGET_COUNTRIES` (`ARG`, `DZA`, `BRA`, `IDN`, `AUS`, `USA` with prefilter boxes `[(-73.8, -55.5, -53.5, -21.5)]`, etc., and ISO_A3 codes), `PLATFORM_INDEX = {"Terra": 0, "Aqua": 1, "SNPP": 2, "NOAA-20": 3, "NOAA-21": 4}`, `PLATFORM_BITMASK = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}`, `PLATFORM_TIMELINE`, `DATE_START = "2000-11-01"`, `DATE_END = "2026-11-14"`, `T_DAYS = 9510`, `CALIB_START = "2012-01-20"`, `CALIB_END = "2021-12-31"`, `CLIM_START_YEAR = 2003`, `CLIM_END_YEAR = 2020`, `Z_MIN = math.log(0.01)`, `Z_MAX = 14.0`, `F_REF_MW = 0.5`, `SAMA_BOX = (-110.0, -55.0, 11.0, 7.0)`, `H3_MAIN_RES = 6`, `H3_FLARE_RES = 8`, `MC_SAMPLES = 200`, `SOLVER_SWEEPS = 6`, `NEWTON_STEPS = 15`, `SPOTLIGHT_PRESETS` (exact coordinates for the 6 case studies).
   * **Excludes:** Zero I/O, zero path hardcoding, zero imports of heavy ML/GPU libraries.

3. **`src/config/paths.py`:**
   * **Responsibility:** Single path resolver class `DataPaths.from_env(base_dir=None)` auto-detecting `/kaggle/input` datasets (`staged-data`, `offline-wheels`) and `/kaggle/working/output` vs local `./data` and `./output`.
   * **Exposed Paths:** `raw_stage_dir`, `active_fires_dir`, `cloud_parquet`, `burned_area_parquet`, `country_polygons_parquet`, `provenance_base_parquet`, `pass_obs_parquet`, `pyromes_parquet`, `calibration_pkl`, `pass_calibrated_parquet`, `priors_pkl`, `state_checkpoint_pt`, `duckdb_path`, `product1_parquet`, `product2_monthly_parquet`, `product2_daily_parquet`, `product3_parquet`, `diagnostics_json`.

4. **`src/ingest/build_wheels.sh`:**
   * Detects active Python minor version (`3.12`) and executes `pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version ${PY_VER}` strictly for `duckdb`, `polars`, and `h3`.

5. **`src/ingest/download_data.py` & `src/ingest/normalize_dedup.py`:**
   * **Responsibility (`download_data.py`):** Env 1 cloud-to-cloud downloader implementing the **Autonomous Discovery & Failover Protocol** (Section 6). Downloads active fires, `MOD14C8Q`/`MYD14C8Q`/`Mx14CMQ` HDF4 cloud grids, `MCD64CMQ` burned area HDF4 grids, and Natural Earth `50m` Admin-0 polygons. Saves `cloud_visibility_025deg.parquet`, `burned_area_val_025deg.parquet`, `country_polygons.parquet`, and raw Parquet chunks under `staged_raw/`.
   * **Responsibility (`normalize_dedup.py`):** Parses all source formats into canonical schema, clips points strictly inside Natural Earth country polygons via `shapely.contains_xy`, enforces `(platform, acq_date)` Science-tier dominance (`ML_SCIENCE > SP_SCIENCE > NRT`), computes `platform_active_day[P, T]`, writes `(country_iso, year)` partitioned `active_fires/`, and emits a structured ingestion audit report.

6. **`src/geometry/h3_utils.py`, `solar_geometry.py`, `preprocess_geometry.py`:**
   * **`h3_utils.py`:** `latlon_to_h3_int(lats, lons, res) -> np.ndarray[int64]`, `h3_int_to_str(cell_int) -> str`, `h3_neighbors(cell_int) -> list[int]`, `h3_boundary_geojson(cell_int) -> list[list[float]]`, `h3_area_km2(cell_int) -> float`, `three_color_hex_graph(active_cells: np.ndarray) -> tuple[np.ndarray, dict]`. Works uniformly across `h3 v3` and `h3 v4`.
   * **`solar_geometry.py`:** Vectorized `compute_solar_time_and_sza(acq_date, acq_hhmm, lons, lats)` returning `(solar_hour, solar_date, dn_window_flag, sza_deg)`, plus `modis_sample_to_geometry(sample)` and `area_to_vza_deg(areas, sensor_family)`.
   * **`preprocess_geometry.py`:** Applies static/SAMA/`H3-Res8` gas-flare hygiene, single-pass ECEF `cKDTree` bow-tie weights ($\omega^{\text{bt}}_i$) and connected-component clustering (`event_cluster_id`), saturation flagging (`i4_m13_fallback` vs Tobit `sat_flag`), emits `obs_provenance_base.parquet` and `pass_observations.parquet` keyed strictly by `(h3_index, solar_date, platform, dn_flag)`.

7. **`src/calibration/pyromes.py` & `src/calibration/operators.py`:**
   * **`pyromes.py`:** Extracts coordinate-free behavioral features (`2012–2021`), fits `K=7` `GaussianMixture`, assigns all active hexagons `pyrome_id in 0..6`, saves `hex_pyromes.parquet` + `pyrome_gmm.pkl`.
   * **`operators.py`:** Fits all `2012-01-20 .. 2021-12-31` forward observation operators (isotonic confidence $w_i$, detection limits $F_{\min}$ & Log-Normal MLE survival $\psi^{\text{CNT}}, \psi^{\text{FRP}}, \psi^{\text{FRP, eff}}$, near-nadir VIIRS EQM + LightGBM residual $\gamma_{\text{MODIS}, b}$, 12-hour intra-window diurnal harmonics $D_{b,s,\text{DN}}(\tau)$ including post-2022 Aqua-vs-N20 drift strictly on diurnal coefficients, heteroscedastic $\sigma^2$, active-area model $g_{\text{area}, b}$, plus `2012–2025` cloud climatology table for trailing 2026 lag). Saves `calibration_artifacts.pkl` (exact dictionary schema in Section 5.2) and `pass_observations_calibrated.parquet`.

8. **`src/engine/likelihoods.py`, `priors.py`, `thomas_takahashi.py`, `cuda_kernels.py`, `solver.py`, `conformal.py`:**
   * **`likelihoods.py`:** Pure PyTorch `float64` functions `eval_pointwise_nll_grad_hess(z, obs_y, obs_sat, obs_mu_gain, obs_inv_sig2, obs_inv_sig, obs_k_miss, obs_vis, is_op_pass)` returning `(nll, grad, hess_spd)` with asymptotic Tobit tail and convexity-guarded Hurdle Hessian.
   * **`priors.py`:** Estimates `(\mu_{g,m,DN}, \rho_b, \sigma_{proc, b}, \lambda_{0,b}, \beta_{dn,b}, \eta_{dn,b}, \bar{F}_{event, b})` from `2012–2021` VIIRS data and packs sparse 3-color neighbor CSR indices.
   * **`thomas_takahashi.py`:** Pure PyTorch `float64` reference implementations of `thomas_factor_and_solve`, `takahashi_marginals`, and `joint_cholesky_sample`.
   * **`cuda_kernels.py`:** Lazy-compiled `torch.utils.cpp_extension.load_inline` single-launch CUDA C++ `float64` kernels (`thomas_solve_cuda`, `takahashi_cuda`, `cholesky_sample_cuda`), automatically falling back to `thomas_takahashi.py` on CPU.
   * **`solver.py`:** `solve_all_chunks(...)` distributing active cells across `0..device_count()-1` (`batch_size = 1000`), running `6` 3-color × Day/Night sweeps, saving MAP state checkpoint (`state_checkpoint_pt`) and raw monthly/daily trajectory statistics, plus `run_fixed_lag_update(window_days=14)`.
   * **`conformal.py`:** Runs held-out `2018–2020` VIIRS-masked calibration and `2018–2021` leave-one-VIIRS-out calibration to fit `c_modis[b]` and `c_multi[b]`, calibrates intervals, evaluates `ABSTAIN_LOW_EVIDENCE`, `attribution_flag`, `hurdle_adj_events`, `energy_equiv_events`, and `2003–2020` climatology anomalies.

9. **`src/products/export_duckdb.py`:**
   * Emits Zstd Parquet for **Product 1 (`obs_harmonized_provenance.parquet`)**, **Product 2 (`state_reconstructed_monthly.parquet` dense + `state_reconstructed_daily.parquet` sparse)**, **Product 3 (`uncertainty_diagnostics.parquet`)**, and creates indexed DuckDB tables + materialized summary views in `fire_harmonized.duckdb`.

10. **`src/validation/synthetic_twin.py` & `src/validation/diagnostics.py`:**
    * **`synthetic_twin.py`:** Deterministic 26-year forward generator producing ground truth `(Phi_day_true, Phi_night_true)` and simulated noisy, censored, bow-tie, cloud-blocked, drifting, S-NPP-terminating sensor observations across all 5 spacecraft.
    * **`diagnostics.py`:** Computes Chow & Pettitt structural breaks, 2019–2021 VIIRS ablation (CRPS, log-RMSE, 90% coverage, Spearman $\rho$, Top-3 Peak Month Match), `MCD64CMQ` burned-area correlations, residual VZA/solar-time slopes, and spotlight audit metrics (`Chilecito ARG 2026`, `Kabylie & Hassi Messaoud DZA`, `Nov 2, 2026 SNPP`). Saves `diagnostics.json`.

11. **`src/webapp/server.py` & `src/webapp/static/`:**
    * FastAPI read-only DuckDB API + static MapLibre GL JS + Apache ECharts web app matching exact JSON contracts in Section 5.5.

---

## 5. Complete Mathematical & Computational Engine

### 5.1 Spatial-Temporal Geometry, SZA, Bow-Tie & Clustering

#### 5.1.1 Astronomical Local Solar Time & Half-Day Window
Given UTC date `(YYYY-MM-DD)`, `acq_hhmm` (`HHMM` integer, e.g., `51` = `00:51`), and longitude $\lambda_i \in [-180, 180]$:
1. Convert `acq_hhmm` carefully in base-60:
   $$h_{\text{UTC}, i} = \lfloor \text{acq\_hhmm}_i / 100 \rfloor + \frac{\text{acq\_hhmm}_i \bmod 100}{60.0}$$
2. Local mean solar time and solar date shift:
   $$\tau_{\text{raw}, i} = h_{\text{UTC}, i} + \frac{\lambda_i}{15.0}, \qquad \tau_i = \tau_{\text{raw}, i} \bmod 24.0$$
   $$\text{solar\_date}_i = \text{acq\_date}_i + \lfloor \tau_{\text{raw}, i} / 24.0 \rfloor \text{ days}$$
3. Canonical latent half-day assignment (ensuring 1-to-1 mathematical alignment with 12-hour diurnal normalization even at Arctic latitudes):
   $$\text{dn\_flag}_i = \begin{cases} \text{'D'} & \text{if } 6.0 \le \tau_i < 18.0 \\ \text{'N'} & \text{otherwise} \end{cases}$$
4. Also compute physical **Solar Zenith Angle ($\text{SZA}$)** using Spencer's Fourier solar declination $\delta_\odot(\text{doy})$ and Equation of Time $E_t(\text{doy})$:
   $$\Gamma = \frac{2\pi(\text{doy} - 1)}{365}, \quad \omega = 15^\circ\left(\tau_i + \frac{E_t}{60.0} - 12.0\right)$$
   $$\text{SZA}_i = \arccos\!\big(\text{clip}(\sin\phi_i\sin\delta_\odot + \cos\phi_i\cos\delta_\odot\cos\omega,\; -1.0,\; 1.0)\big) \cdot \frac{180^\circ}{\pi}$$
   Store `sza_deg` and raw sensor `daynight` inside Product 1 for scientific provenance.

#### 5.1.2 Spherical Pixel Geometry & VZA Recovery
* **When MODIS `sample` ($s \in [0, 1353]$) is present (`MCD14ML`):**
  With $R_E = 6371.0\,\text{km}$, $H = 705.0\,\text{km}$, half-swath angle $\alpha_{\max} = 55.0^\circ \cdot \frac{\pi}{180}$, nominal nadir IFOV $\Delta\alpha_0 = 1.0 / 705.0$:
  $$\alpha(s) = (s - 676.5)\,\frac{2\alpha_{\max}}{1353}, \qquad \theta_{\text{VZA}}(s) = \arcsin\!\left(\frac{R_E + H}{R_E}\sin|\alpha(s)|\right) \cdot \frac{180^\circ}{\pi}$$
  $$D(\alpha) = (R_E + H)\cos\alpha - \sqrt{R_E^2 - (R_E + H)^2\sin^2\alpha}$$
  $$\text{track}(s) = D(\alpha)\,\Delta\alpha_0, \qquad \text{scan}(s) = \frac{D(\alpha)\,\Delta\alpha_0}{\cos\theta_{\text{VZA}}(s)}, \qquad A_i = \text{scan}(s)\cdot\text{track}(s)$$
  *(Check: $s = 676.5 \implies 1.00 \times 1.00 = 1.00\,\text{km}^2$, $\theta = 0^\circ$; $s \in \{0, 1353\} \implies 4.84 \times 2.01 = 9.72\,\text{km}^2$, $\theta = 65.48^\circ$.)*
* **When VIIRS `Pixarea` is present (`Vxx14IMGML`):** Use $A_i = \text{clip}(\text{Pixarea}_i,\; 0.12,\; 0.75)$ directly (`scan = track = sqrt(A_i)` proxy if absent) and recover $\theta_{\text{VZA}}$ from monotone spline below.
* **When `scan_km, track_km` are present (Portolan / FIRMS API):** Compute $A_i = \text{clip}(\text{scan}_i \times \text{track}_i,\; 0.12,\; 12.0)$ and interpolate $\theta_{\text{VZA}} \in [0^\circ, 65^\circ]$ via `np.interp`:
  * MODIS knots: `A = [1.00, 1.35, 2.25, 4.20, 9.72] -> VZA = [0.0, 20.0, 35.0, 50.0, 65.0]`
  * VIIRS knots: `A = [0.141, 0.180, 0.280, 0.430, 0.620] -> VZA = [0.0, 20.0, 35.0, 50.0, 65.0]`

#### 5.1.3 Hygiene, Bow-Tie Weighting, Event Clustering & Pass Aggregation
1. **Static & SAMA Filtering:**
   * Drop `type_raw IN (1, 3)` (volcanoes & offshore).
   * Flag `is_sama_filtered = True` (and exclude from biomass fusion) when `sensor_family == 'VIIRS'` AND `dn_flag == 'N'` AND (`conf_raw == 'l'` OR `conf_num < 30`) AND `-110.0 <= longitude <= 11.0` AND `-55.0 <= latitude <= 7.0`.
   * Group remaining detections into **`H3 Resolution 8`** micro-cells $m$ ($\sim 0.74\,\text{km}^2$, robust to 50–300 m MODIS geolocation jitter). Compute distinct active years $Y_m$, mean active days/year $\bar{D}_m = \text{active\_days}_m / \max(1, Y_m)$, and day/night symmetry $R^{\text{DN}}_m = \frac{\min(n_{\text{D}}, n_{\text{N}})}{\max(n_{\text{D}}, n_{\text{N}}) + 1}$. Set `is_static_industrial = True` if `type_raw == 2` OR ($Y_m \ge 3$ AND $\bar{D}_m \ge 15.0$ AND $R^{\text{DN}}_m > 0.30$). Keep flagged points in Product 1; exclude from biomass fusion.
2. **Pass Segmentation & Spatial Graphs (`cKDTree` in ECEF 3D):**
   * Sort by `(platform, utc_epoch_minutes)` and split `pass_id` wherever consecutive timestamps differ by `> 5 minutes`.
   * Map `(lat, lon)` to 3D ECEF chords ($R_E = 6371.0\,\text{km}$). Within each `(platform, pass_id)`:
     * **Bow-tie graph (MODIS pixels with $A_i > 1.5\,\text{km}^2$):** Connect $i, j$ if $d_{ij} < 0.7 \cdot \frac{\text{track}_i + \text{track}_j}{2}$. Weight $\omega^{\text{bt}}_i = \frac{1.0}{1.0 + \text{deg}_{\text{bt}}(i)}$ (conserving total FRP across overlapping scans). Otherwise $\omega^{\text{bt}}_i = 1.0$.
     * **Event clustering graph:** Connect $i, j$ if $d_{ij} \le 1.4 \cdot \max(\sqrt{A_i}, \sqrt{A_j})$. Run `scipy.sparse.csgraph.connected_components` to assign `event_cluster_id`.
3. **Saturation Flags:**
   * `i4_m13_fallback = 1` if `sensor_family == 'VIIRS'` and `t_mir_k >= 366.0` and `frp_mw > 0.0` (inflates observation std by `1.35×`).
   * `sat_flag = 1` (Tobit right-censored) if (`sensor_family == 'MODIS'` and `t_mir_k >= 498.0` and `frp_mw > 0.0`) OR (`sensor_family == 'VIIRS'` and `t_mir_k >= 366.9` and `frp_mw >= 300.0`).
4. **Overpass Aggregation (`pass_observations.parquet`):**
   Aggregate biomass rows strictly per `(h3_index, solar_date, platform, dn_flag)` (never summing platforms):
   * `y_frp_mw = sum(omega_bt * rel_weight * frp_mw)` (note `rel_weight` enters after calibration in `operators.py`, initialized `1.0` before calibration and weighted during calibration pass)
   * `raw_event_count = count_distinct(event_cluster_id)`
   * `det_footprint_km2 = sum(omega_bt * A_i)`
   * `mean_area_km2`, `mean_vza_deg`, `mean_solar_hour`, `mean_conf_num`, `has_i4_m13_fallback`, `sat_flag`.

#### 5.1.4 Non-Detection (`Y == 0`) Cloud & Swath Parameters
For every operational pass `(p, g, t, DN)` where `is_operational[p, t] == 1` AND `platform_active_day[p, t] == 1` (`>= 5` total detections across all 6 countries on UTC day `t`):
1. Join `cloud_visibility_025deg.parquet` on `(lat_025, lon_025, year, period_8d_or_month)`—falling back to `2012–2025` multi-year mean `MeanCloudFraction` for that `(0.25° cell, month)` whenever trailing 2026 CMG files have not yet been published (flagging `cloud_source = 'CLIMATOLOGY_FALLBACK'`).
2. Compute clear-sky observation probability:
   $$v_{p,g,t} = p_{\text{swath}, p}(|\phi_g|, t) \cdot \text{clip}\!\Big(1.0 - f_{\text{cloud}} - f_{\text{unmapped}},\;\; 0.10,\;\; 0.98\Big)$$
   with $p_{\text{swath}, \text{VIIRS}} = 0.99$ and $p_{\text{swath}, \text{MODIS}} = \text{clip}(0.85 + 0.15\frac{|\phi_g|}{30.0}, 0.85, 1.00) \times (0.96 \text{ if } t \ge \text{2022-01-01})$.
3. Expected unseen-pass averages: MODIS $\bar{A}_{\text{avg}} = 2.35\,\text{km}^2, \bar{\theta}_{\text{avg}} = 35.0^\circ$; VIIRS $\bar{A}_{\text{avg}} = 0.23\,\text{km}^2, \bar{\theta}_{\text{avg}} = 28.0^\circ$.
4. Expected solar hour $\bar{\tau}_{p,t,\text{DN}}$: Terra drifts linearly `10.5 -> 8.75` (`2022-01-01 .. 2026-11-14`), Aqua drifts `13.5 -> 15.60` (`+12.0 mod 24` for Night); VIIRS fixed at `13.5` Day / `1.5` Night.

---

### 5.2 Calibration of Observation Operators (`2012-01-20` to `2021-12-31`)

1. **Behavioral Pyromes ($K=7$ GMM, Coordinate-Free):**
   For every hexagon with $\ge 20$ active pass-days in `2012–2021`, compute strictly non-coordinate features: `[ln(median_cluster_frp + 1), night_frp_share, seasonal_circular_concentration, interannual_cv, mean_cluster_pixels]`. Fit `GaussianMixture(n_components=7, covariance_type='full', reg_covar=1e-4, random_state=42)` over `StandardScaler` features. Assign `< 20`-day cells to the nearest GMM centroid in standardized space (or country modal pyrome if `< 3` active days).
2. **Confidence Reliability Mapping ($w_i \in [0.15, 1.0]$):**
   On `2012–2021`, mark each detection as corroborated (`1`) if another detection (different pass or platform) occurs within $\pm 1$ day and $\le 2\,\text{km}$. Fit `IsotonicRegression(y_min=0.15, y_max=1.0, out_of_bounds='clip')` per `(sensor_family, dn_flag)` over `conf_num`. Recompute clean weighted pass FRP `y_frp_mw = sum(omega_bt * w_i * frp_mw)` (`w_i` defaults to `1.0` in high-confidence single-pixel checks where calibrated `w_i >= 0.90`).
3. **Detection Limits & Analytical Survival Fractions ($\psi^{\text{CNT}}, \psi^{\text{FRP}}, \psi^{\text{FRP, eff}}$):**
   $$F_{\min, \text{MODIS}}(A, \text{DN}) = \alpha_{M,\text{DN}} + \beta_{M,\text{DN}}\,A, \qquad F_{\min, \text{VIIRS}}(A, \text{DN}) = \alpha_{V,\text{DN}} + \beta_{V,\text{DN}}\,A$$
   with physical anchors: MODIS Night `(1.5, 2.2)`, MODIS Day `(2.5, 3.9)`, VIIRS Night `(0.25, 1.8)`, VIIRS Day `(0.40, 3.4)`.
   Fit per-pyrome left-truncated Log-Normal $(\mu_{F,b}, \sigma_{F,b})$ via L-BFGS-B on clear-sky near-nadir (`VZA < 20°`, `cloud_frac < 0.30`) VIIRS clusters above $F_{\text{ref}} = 0.5\,\text{MW}$:
   $$\psi^{\text{CNT}}_{p,b}(A, \text{DN}) = \text{clip}\!\left(1 - \Phi_{\mathcal{N}}\!\left(\frac{\ln F_{\min, p}(A,\text{DN}) - \mu_{F,b}}{\sigma_{F,b}}\right),\;\; 0.02,\;\; 1.0\right)$$
   $$\psi^{\text{FRP}}_{p,b}(A, \text{DN}) = \text{clip}\!\left(\frac{1 - \Phi_{\mathcal{N}}\!\left(\frac{\ln F_{\min, p}(A,\text{DN}) - \mu_{F,b} - \sigma^2_{F,b}}{\sigma_{F,b}}\right)}{1 - \Phi_{\mathcal{N}}\!\left(\frac{\ln F_{\text{ref}} - \mu_{F,b} - \sigma^2_{F,b}}{\sigma_{F,b}}\right)},\;\; 0.05,\;\; 1.0\right)$$
   Smooth conditional multi-fire adjustment for detected passes ($E_{\text{raw}} \ge 1$):
   $$w_{\text{multi}}(E_{\text{raw}}) = 1.0 - \exp\!\big(-0.7\,(E_{\text{raw}} - 1)\big)$$
   $$\psi^{\text{FRP, eff}}_{p,b}(A, \text{DN}, E_{\text{raw}}) = (1 - w_{\text{multi}})\,\max\!\big(\psi^{\text{FRP}}_{p,b}(A, \text{DN}),\; 0.65\big) + w_{\text{multi}}\,\psi^{\text{FRP}}_{p,b}(A, \text{DN})$$
   *(Prevents 4× artificial inflation of isolated single-cluster fires that already cleared $F_{\min}$, while applying full small-fire omission correction when multiple clusters coexist.)*
4. **VZA & Quantile Residual Transfer ($\gamma_{p,b}$):**
   Pair synchronous `2012–2021` `Aqua` and near-nadir `SNPP` (`VZA_SNPP < 25°`, $\Delta t \le 45\,\text{min}$, same `h3_index`, same `dn_flag`, both $> 0$). Compute target log-ratio:
   $$r_k = \ln Y_{\text{SNPP}, k} - \Big(\ln Y_{\text{Aqua}, k} - \ln \psi^{\text{FRP, eff}}_{\text{Aqua}, b, k} + \ln \psi^{\text{FRP, eff}}_{\text{SNPP}, b, k}\Big)$$
   Fit a LightGBM regressor (`max_depth=4, n_estimators=100, min_child_samples=50`, unconstrained VZA sign) predicting $r_k$ from `[pyrome_id, dn_is_day, mean_vza_modis, mean_area_modis, ln_y_modis]`, plus a 20-knot **strictly monotonic Empirical Quantile Mapping (EQM)** curve (`isotonic`/`PCHIP`) per `(pyrome, dn_flag)` mapping adjusted `ln Y_MODIS` quantiles onto `ln Y_SNPP` quantiles so quantile curves never cross. Clip $\ln \gamma_{\text{MODIS}} \in [-1.5, +1.5]$; set $\ln \gamma_{\text{VIIRS}} \equiv 0.0$.
5. **12-Hour Intra-Window Diurnal & Orbital-Drift Curve ($D_{b,s,\text{DN}}(\tau)$):**
   Per pyrome $b$ and season $s \in \{\text{DJF, MAM, JJA, SON}\}$:
   $$\ln D_{\text{raw}, b,s}(\tau) = a_1\cos\!\left(\frac{2\pi\tau}{24}\right) + b_1\sin\!\left(\frac{2\pi\tau}{24}\right) + a_2\cos\!\left(\frac{4\pi\tau}{24}\right) + b_2\sin\!\left(\frac{4\pi\tau}{24}\right)$$
   Fit $(a_1, b_1, a_2, b_2)$ by Ridge regression (`alpha=10.0`) on same-day paired log-FRP differences (`Terra` vs `Aqua/SNPP` `2003–2021`, plus `Aqua` `2022–2025` drifting solar hours vs stationary `NOAA-20` strictly updating diurnal parameters while keeping $\psi, \gamma$ frozen).
   Normalize **independently within each 12-hour half-day window** via 120-point trapezoidal integration:
   $$c_{0,\text{Day}} = -\ln\!\left(\frac{1}{12}\int_6^{18} D_{\text{raw}, b,s}(\tau)\,d\tau\right), \qquad c_{0,\text{Night}} = -\ln\!\left(\frac{1}{12}\int_{18}^{30} D_{\text{raw}, b,s}(\tau \bmod 24)\,d\tau\right)$$
   $$D_{b,s,\text{DN}}(\tau) = \exp\!\big(\ln D_{\text{raw}, b,s}(\tau) + c_{0,\text{DN}}\big)$$
   Because each half-day curve integrates to unit mean over its own 12 hours, $\Phi_{\text{Day}}$ and $\Phi_{\text{Night}}$ retain their exact physical meaning (mean Daytime MW vs mean Nighttime MW) and exact 24h flux is $\Phi_{24\text{h}} = 0.5(\Phi_{\text{Day}} + \Phi_{\text{Night}})$.
6. **Detected Forward Log-Gain (Strictly Cloud-Free):**
   $$\mu^{\text{gain, det}}_{p,g,t,\text{DN}} = \ln D_{b,s,\text{DN}}(\bar{\tau}_{p,g,t}) + \ln \gamma_{p,b}(\bar{\theta}_p, q) + \ln \psi^{\text{FRP, eff}}_{p,b}(\bar{A}_p, \text{DN}, E_{\text{raw}})$$
7. **Heteroscedastic Variance ($\sigma^2_{p,g,t,\text{DN}}$):**
   OLS on `2012–2021` squared calibration residuals (`SNPP` vs `NOAA-20` for VIIRS; `Aqua` vs `SNPP` minus VIIRS variance for MODIS) predicting $\ln(\hat{\epsilon}^2 + 10^{-3})$ from `[is_modis, vza_deg / 65.0, 1.0 - w_mean, 1.0 - v_cloud, i4_m13_fallback]`, clipped to $[0.025, 2.50]$.
8. **Active Burning Footprint Model ($g_{\text{area}, b}$):**
   Fit a Ridge regressor on `2012–2021` clear-sky near-nadir (`VZA < 25°`) VIIRS detections predicting $\ln(A^{\text{det}}_{\text{VIIRS}} + 10^{-3})$ from $[\ln(\Phi + 0.01),\, \ln(\text{night\_day\_ratio} + 0.05),\, \text{pyrome\_one\_hot}]$, outputting $\widehat{A}^{\text{burn}}_{g,t} = \text{clip}(\exp(\hat{y}) - 10^{-3},\; 0.0,\; A_g)$.
9. **`calibration_artifacts.pkl` Exact Dictionary Schema:**
   `{"reliability_models": dict, "fmin_params": dict, "lognorm_params": dict, "lgb_gamma": LightGBM, "eqm_knots": dict, "diurnal_coeffs": dict, "sigma_ols": LinearRegression, "area_model": Ridge, "cloud_climatology": DataFrame}`.

---

### 5.3 State-Space Priors, Exact Likelihoods, and `float64` GPU Solver

#### 5.3.1 Deterministic Empirical Bayes Priors (`priors.py`)
From `2012–2021` VIIRS data:
* $\bar{F}_{\text{event}, b} = \max(1.0,\; \text{median}(\text{clear near-nadir VIIRS cluster FRP in pyrome } b))$
* Cell-month baseline prior (treating clear zero-detection days as $z_{\min} = \ln(0.01)$):
  $$\mu_{g,m,\text{DN}} = \frac{n_{g,m}\,\bar{y}_{g,m,\text{DN}} + 5.0\,\bar{\mu}_{b,m,\text{DN}}}{n_{g,m} + 5.0}$$
* AR(1) temporal persistence $\rho_b \in [0.25, 0.90]$ and innovation std $\sigma_{\text{proc}, b} \in [0.25, 1.50]$ via lag-1 OLS on consecutive active VIIRS cell-days.
* Hurdle rate constant $\lambda_{0, b} \in [0.1, 5.0]$ via `scipy.optimize.minimize_scalar` matching empirical zero-detection probability on active pyrome months.
* Day–Night coupling $(\beta_{\text{dn},b}, \eta_{\text{dn},b})$ via OLS slope and variance ratio ($\eta_{\text{dn},b} \in [0.05, 0.30]$).

#### 5.3.2 Exact Likelihood Equations & Convexity-Guarded Hessians (`likelihoods.py`)
Let $z \in [-8.0, 14.0]$ (`float64`), $\mu = \mu^{\text{gain, det}}$, $\tau_2 = 1/\sigma^2$, $\tau_1 = 1/\sigma$:
1. **Case A (`Y > 0, sat_flag == 0`):**
   $$r = (z + \mu) - \ln Y, \qquad \ell = \frac{1}{2}\tau_2 r^2, \qquad g = \tau_2 r, \qquad h = \tau_2$$
2. **Case B (`Y > 0, sat_flag == 1`, Right-Censored Tobit):**
   Let $w = \tau_1\big((z + \mu) - \ln Y\big)$.
   If $w \ge -30$: $\ln\Phi(w) = \text{torch.special.log\_ndtr}(w)$, $\lambda(w) = \exp(-0.5 w^2 - 0.5\ln(2\pi) - \ln\Phi(w))$.
   If $w < -30$: $-\ln\Phi(w) = 0.5 w^2 + \ln(-w) + 0.5\ln(2\pi)$, $\lambda(w) = -w - 1/w$.
   $$\ell = -\ln\Phi(w), \qquad g = -\tau_1 \lambda(w), \qquad h = \tau_2\,\lambda(w)\big(w + \lambda(w)\big) > 0$$
3. **Case C (`Y == 0` on active operational pass):**
   Let $k = \frac{\lambda_{0,b}\,\psi^{\text{CNT}}_{p,b}(\bar{A}_{\text{avg}}, \text{DN})}{\bar{F}_{\text{event}, b}} > 0$, $\Lambda = \text{clip}(k\,e^z,\; 10^{-12},\; 80.0)$, $e^{-\Lambda} = \exp(-\Lambda)$, $q = (1 - v) + v e^{-\Lambda}$, $r = \frac{v e^{-\Lambda}}{q}$:
   $$\ell = -\ln q, \qquad g = r\,\Lambda, \qquad h = \max\!\Big(0.0,\; r\,\Lambda\,\big[1.0 - \Lambda(1.0 - r)\big]\Big)$$

#### 5.3.3 Space × Time × Day-Night Prior Tridiagonal System
Let $\delta_t = z_t - \mu_t$, $\omega_p = 1/\sigma^2_{\text{proc}, b}$, $\alpha_s = 0.10$:
* Main diagonal $a_0 = a_{T-1} = \omega_p$, $a_t = (1 + \rho^2)\omega_p$ ($1 \le t \le T-2$); off-diagonal $b_t = -\rho\,\omega_p$.
* Neighbor 1-ring term (fixed per H3 color): $\Delta a_t = \alpha_s\omega_p|\mathcal{N}(g)|$, $\Delta g_t = \alpha_s\omega_p(|\mathcal{N}(g)| z_t - \sum_{g'} z_{g', t})$.
* Opposite half-day term: $\Delta a_t = \eta_{\text{dn}}\omega_p$, $\Delta g_t = \eta_{\text{dn}}\omega_p(\delta_t - \beta_{\text{dn}}\delta_{\text{opp}, t})$.
* Floor diagonal: $a_t = \max(a_t, 10^{-6})$.

#### 5.3.4 Fused `float64` Thomas $LDL^\top$, Takahashi, and Joint Sampling (`thomas_takahashi.py` & `cuda_kernels.py`)
1. **Factorization ($H = L D L^\top$):** $d_0 = a_0$; for $t = 0\dots T-2$: $l_t = b_t / d_t$, $d_{t+1} = \max(a_{t+1} - l_t b_t, 10^{-7})$.
2. **Solve ($H\,\Delta z = g$):** $y_0 = g_0$, $y_t = g_t - l_{t-1} y_{t-1}$; $\Delta z_{T-1} = y_{T-1} / d_{T-1}$, $\Delta z_t = y_t / d_t - l_t \Delta z_{t+1}$.
3. **Armijo Backtracking:** Test $\alpha \in \{1.0, 0.5, 0.25, 0.125, 0.0625\}$; accept when $J(z_{\text{new}}) \le J(z) - 10^{-4}\alpha\sum g\Delta z$; stop Newton loop early when $\|\alpha\Delta z\|_\infty < 10^{-4}$.
4. **Takahashi Recursion:** $\Sigma_{T-1, T-1} = 1 / d_{T-1}$; for $t = T-2\dots 0$: $\Sigma_{t, t+1} = -l_t \Sigma_{t+1, t+1}$, $\Sigma_{t, t} = 1/d_t - l_t \Sigma_{t, t+1}$.
5. **Exact Joint Cholesky Sampling ($M = 200$):** Draw $\varepsilon \sim \mathcal{N}(0, 1)$, solve $L^\top u = D^{-1/2}\varepsilon$ backwards ($u_{T-1} = \varepsilon_{T-1}/\sqrt{d_{T-1}}$, $u_t = \varepsilon_t/\sqrt{d_t} - l_t u_{t+1}$). Compute $z^{(m)} = \text{clip}(\hat{z} + u^{(m)}, -8.0, 14.0)$, $\Phi^{(m)}_{g,t,\text{DN}} = \exp(z^{(m)})$, $\Phi^{(m)}_{g,t} = 0.5(\Phi^{(m)}_{\text{Day}} + \Phi^{(m)}_{\text{Night}})$, $\text{FRE}^{(m)}_{g,t} = 0.0864\,\Phi^{(m)}_{g,t}$, aggregate within `(g, year, month)`, and extract `q05, q50, q95, mean`.
6. **GPU Acceleration (`cuda_kernels.py`):** Implement `thomas_solve_cuda`, `takahashi_cuda`, and `cholesky_sample_cuda` as single-launch thread-per-cell `float64` CUDA C++ kernels via `torch.utils.cpp_extension.load_inline` (dispatching transparently to `thomas_takahashi.py` on CPU). Chunk size `B = 1000` (`~2.5 GB` peak VRAM per GPU across `T = 9510`).
7. **3-Coloring & Fixed-Lag Streaming (`solver.py`):** Run `6` outer sweeps over **3-coloring** (`color in {0, 1, 2}`) × alternating `('D', 'N')`. Implement `run_fixed_lag_update(window_days=14)` anchoring `t_start - 1` via AR(1) to the saved state.

### 5.4 Conformal Calibration, Diagnostics & Product Equations

1. **Split Conformal Quantile Rescaling (`conformal.py`):**
   Evaluate nonconformity scores $r_j = \frac{|\ln(\Phi_{\text{ref}, j}+1) - \ln(q_{50,\text{raw}, j}+1)|}{s_j}$ with $s_j = \max\big(\frac{\ln(q_{95,\text{raw}, j}+1) - \ln(q_{05,\text{raw}, j}+1)}{2},\; 0.05\big)$ on:
   * Held-out `2018–2020` VIIRS-masked runs $\implies \hat{c}^{\text{modis}}_{0.90, b} \in [0.85, 3.5]$.
   * Held-out `2018–2021` leave-one-VIIRS-platform-out runs $\implies \hat{c}^{\text{multi}}_{0.90, b} \in [0.85, 3.5]$.
   Rescale intervals: $q_{05,\text{cal}} = \max(0, \exp(\ln(q_{50}+1) - \hat{c} s) - 1)$, $q_{95,\text{cal}} = \exp(\ln(q_{50}+1) + \hat{c} s) - 1$.
2. **Hurdle-Adjusted vs Energy-Equivalent Event Counts (Respecting INV-02):**
   * `energy_equiv_events_q50 = phi_q50_mw / F_event_b`
   * `hurdle_adj_events`: Within each `(h3_index, solar_date, dn_flag)` half-day that had detections, take the **mean** across detecting platforms (never sum `SNPP + N20 + N21`!):
     $$E^{\text{halfday}}_{g,t,\text{DN}} = \frac{1}{|P_{\text{det}}|}\sum_{p \in P_{\text{det}}} \frac{E^{\text{raw}}_{p,g,t,\text{DN}}}{\psi^{\text{CNT}}_{p,b}(\bar{A}_p, \text{DN}) \cdot v_{p,g,t}}$$
     Sum $E^{\text{halfday}}$ over detected half-days in the month, and scale unobserved blind days smoothly by `1 / max(measured_fraction, 0.25)` (capped at `5.0 * raw_clustered_events` to prevent extreme noise blowups).
3. **Diagnostic Flags & Bitmasks:**
   * `active_platforms_bitmask`: Bit `0` (1) = Terra, Bit `1` (2) = Aqua, Bit `2` (4) = SNPP, Bit `3` (8) = NOAA-20, Bit `4` (16) = NOAA-21.
   * `quality_flag`: `'ABSTAIN_LOW_EVIDENCE'` if $\ge 5$ consecutive days in month have `mean_cloud_frac > 0.85` AND zero detections in cell + 1-ring neighbors; `'CLIMATOLOGY_CLOUD_FALLBACK'` if cloud fraction came from 2012–2025 climatology; else `'NOMINAL'`.
   * `attribution_flag`: Let $\Delta_{\text{raw}} = \frac{\text{raw\_frp\_sum} - \text{raw\_hist\_med}}{\max(\text{raw\_hist\_med}, 1.0)}$ and `harm_hist_med` = 2003–2020 cell-month median of `phi_q50_mw`:
     - If `quality_flag == 'ABSTAIN_LOW_EVIDENCE'`: `'AMBIGUOUS_OVERLAPPING_CI'`
     - Else if `q05_cal > harm_hist_med` or `q95_cal < harm_hist_med`: `'REAL_FIRE_CHANGE'`
     - Else if $|\Delta_{\text{raw}}| > 0.50$ and `q05_cal <= harm_hist_med <= q95_cal`: `'SENSOR_ARTIFACT_CORRECTED'`
     - Else: `'WITHIN_NORMAL_VARIABILITY'`.
4. **Sparse Daily Export Filter (`export_duckdb.py`):**
   Write `state_reconstructed_monthly.parquet` densely for all active cells × months, and write `state_reconstructed_daily.parquet` **sparsely** (include row iff `phi_q50_mw > 0.25` OR within $\pm 3$ days of a detection OR inside any `SPOTLIGHT_PRESETS` bounding box).

### 5.5 Exact FastAPI JSON & Web Contracts (`src/webapp/`)
* `GET /api/calendar?min_lon=...&min_lat=...&max_lon=...&max_lat=...&start_year=2000&end_year=2026&metric=phi_q50_mw` (validates `min_lon < max_lon, min_lat < max_lat`, max bbox span `<= 25° × 25°` unless `country_iso` supplied):
  Returns `{"meta": {...}, "cells": [{"year": int, "month": int, "value": float, "q05": float, "q95": float, "hist_median": float, "anomaly_ratio": float, "raw_count": int, "raw_frp": float, "measured_fraction": float, "quality_flag": str, "attribution_flag": str}]}`.
* `GET /api/timeseries?...`: Returns monthly array + `constellation_events` (`2002-07`, `2012-01`, `2018-04`, `2022-01`, `2023-03`, `2026-11`).
* `GET /api/map_hex?min_lon=...&year=2026&month=8&color_by=anomaly_ratio`: Returns GeoJSON `FeatureCollection` of H3 polygons (`<= 3000` features per query).
* `GET /api/diagnostics`: Returns JSON contents of `diagnostics.json`.
* `GET /api/export.csv?...`: Streams CSV sanitizing any string field starting with `=`, `+`, `-`, `@`, `\t`, `\r` by prefixing `'`.
* **Frontend (`app.js`, `index.html`, `styles.css`):** Vanilla ES6 + MapLibre GL JS + Apache ECharts. Includes bounding-box ROI drawing, lat/lon input form, 6 spotlight preset buttons (`Chilecito ARG 2026: [-68.0,-29.6,-67.1,-28.7]`, `Kabylie DZA: [3.8,36.3,5.5,37.0]`, `Hassi Messaoud Gas Flares DZA: [5.5,31.3,6.6,32.2]`, `Alaska Boreal USA: [-156.0,63.0,-142.0,67.5]`, `Black Summer AUS: [147.0,-37.8,151.0,-34.5]`, `Kalimantan Peat IDN: [112.5,-3.3,115.2,-1.5]`), colorblind-safe `viridis`/`cividis` heatmap with diagonal hatching overlay on `ABSTAIN_LOW_EVIDENCE` / low-`measured_fraction` months, comparative raw-vs-harmonized time-series chart with conformal CI ribbons, and CSV download button.

---

## 6. Complete Data Acquisition, Autonomous Mirror Discovery & Processing Pipeline

### 6.1 Primary Verified Data Sources & Exact File Schemas
1. **University of Maryland Fire Archive (`fuoco.geog.umd.edu`, login `fire` / password `burnt` via SFTP port 22 or FTP port 21):**
   * `MCD14ML` (`MODIS/C6/MCD14ML` or `C61`): Fixed-width **space-delimited** `.txt.gz` with columns: `YYYYMMDD HHMM sat lat lon T21 T31 sample FRP conf type dn` (`version 1` omits `dn`; parse robustly, strip trailing dot on `sample` e.g. `681.`).
   * `VNP14IMGML`, `VJ114IMGML`, `VJ214IMGML` (`VIIRS/VNP14IMGML`, `data/VIIRS/C2/...`): Comma-separated `.csv.gz` with columns: `YYYYMMDD, HHMM, Sat, Lat, Lon, T_I4, T_I5, Sample, Pixarea, FRP, Conf, Type`. Infer platform from filename (`VNP`->`SNPP`, `VJ1`->`NOAA-20`, `VJ2`->`NOAA-21`) if `Sat` is missing.
   * **Cloud Denominator (`MOD14C8Q` / `MYD14C8Q` 8-day `0.25°` HDF4, falling back to `MOD14CMQ` / `MYD14CMQ` monthly HDF4):** Verified Table 8 SDS layers: `MeanCloudFraction`, `CorrFirePix`, `CloudCorrFirePix`, `RawFirePix` (`720 × 1440` grid, `0.25°`, upper-left center `-179.875°, +89.875°`). Read `scale_factor`, `add_offset`, and `_FillValue` dynamically from `pyhdf` SDS attributes; mask negative/fill values; normalize `[0, 100]` values to `[0.0, 1.0]`.
   * **Burned Area Validation (`MCD64CMQ` monthly `0.25°` HDF4):** Verified SDS layers: `BurnedArea` (scaled by `scale_factor` `0.01` -> hectares $\times 0.01 = \text{km}^2$), `QA` (`2 = processed land`), `UnmappedFraction` (`0–100%`).
2. **Cloud-Native GeoParquet 2.0 Global Mirror (`portolan-mirrors/firms-catalog` on Source Cooperative):**
   * Verified root STAC catalog: `https://data.source.coop/portolan-mirrors/firms-catalog/catalog.json`.
   * Year-partitioned GeoParquet files: `https://data.source.coop/portolan-mirrors/firms-catalog/detections/year=<YYYY>/detections.parquet` or `s3://us-west-2.opendata.source.coop/portolan-mirrors/firms-catalog/detections/year=*/detections.parquet`.
   * Verified schema: `geometry` (`ST_Point`, extract via `ST_X(geometry), ST_Y(geometry)` using DuckDB `spatial` + `httpfs`), `acq_datetime`, `acq_date`, `sensor`, `satellite`, `quality`, `version`, `brightness`, `bright_t31`, `bright_ti4`, `bright_ti5`, `scan`, `track`, `frp`, `daynight`, `confidence`, `confidence_pct`, `type`, `_month`, `_hilbert`.
3. **NASA FIRMS REST API (Strictly for Trailing 2026 NRT Gap-Fill):**
   * Query `GET https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{MAP_KEY}/ALL`.
   * Fetch trailing windows via `GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{YYYY-MM-DD}` with **`DAY_RANGE <= 5`** strictly, respecting the 5,000 tx / 10 min limit (`time.sleep(0.15)` + `429` exponential backoff).
4. **Country Admin-0 Polygons:**
   * Fetch `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson`, extracting `ARG, DZA, BRA, IDN, AUS, USA`.

### 6.2 Autonomous Data Source Discovery & Failover Protocol (If Links/Paths Change)
If ANY primary URL, directory path, or file pattern returns `404`, `NoSuchFile`, or connection refusal, **do NOT crash, silently skip, or fabricate synthetic replacements**. Execute this exact deterministic discovery sequence and log the resolved source in `data_provenance_manifest.json`:
1. **Remote Directory Tree Traversal (UMD SFTP / FTP):** If a hardcoded subpath (e.g. `MODIS/C6/MCD14ML`) does not exist, try FTP (`ftplib.FTP`) if SFTP port 22 fails, and recursively walk top-level directories (`sftp.listdir` / `ftp.nlst`) matching regex patterns `MCD14ML`, `V(NP|J1|J2)14IMGML`, `M[OY]D14C(8|M)Q`, `MCD64CMQ`.
2. **STAC & Mirror Failover (Source Cooperative):** If `fuoco.geog.umd.edu` is unreachable or times out, fetch `https://data.source.coop/portolan-mirrors/firms-catalog/catalog.json` (and inspect its `child` links dynamically) or query `https://data.source.coop/portolan-mirrors/firms-catalog/detections/year=<YYYY>/detections.parquet` via DuckDB `httpfs` with bounding-box pruning (`WHERE ST_X(geometry) BETWEEN ... AND ST_Y(geometry) BETWEEN ...`)! Notice Portolan uses FIRMS `scan`/`track` schema directly from 2000 to today.
3. **NASA CMR / Earthdata & FIRMS Bulk Archive Discovery:** For missing `MOD14C8Q` / `MOD14CMQ` / `MCD64CMQ` granules or if Portolan lacks ancillary grids, query NASA's unauthenticated Common Metadata Repository search API (`https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MCD64CMQ&page_size=2000`) or FIRMS static download catalog (`https://firms.modaps.eosdis.nasa.gov/usfs/api/` / `https://firms.modaps.eosdis.nasa.gov/download/`) to locate alternate download endpoints. If no cloud CMG source can be reached for a historical period, verify whether `Mx14CMQ` exists; if all CMG fetches fail completely, log a critical warning in `data_provenance_manifest.json` and abort rather than silently fabricating zero-cloud assumptions.
4. **Polygon Fallback Discovery:** If GitHub raw `natural-earth-vector` fails, automatically query GeoBoundaries Open API (`https://www.geoboundaries.org/api/current/gbOpen/{ISO3}/ADM0/`) and follow its `gjDownloadURL`.
5. **Schema Validation Gate Before Acceptance:** Every dynamically discovered file must pass column-presence assertions (`lat/lon/frp/date/time` non-null rates `>= 99%`, `latitude in [-90, 90]`, `frp >= 0`, target date span populated) before entering `staged_raw/`.

### 6.3 Normalization & Verification (`normalize_dedup.py`)
1. Normalize satellite descriptors (`T/Terra/TERRA -> Terra`, `A/Aqua/AQUA -> Aqua`, `N/SNPP/NPP/Suomi-NPP -> SNPP`, `1/J1/N20/NOAA-20 -> NOAA-20`, `2/J2/N21/NOAA-21 -> NOAA-21`).
2. Preserve `conf_raw` natively (`'l'/'n'/'h'` or `'0'..'100'`), map `conf_num` (`low=20, nominal=65, high=95`, float parse for numeric). Set `type_raw = -1` when missing (`NRT`).
3. Clip rows strictly inside the 6 Natural Earth Admin-0 polygons via `shapely.contains_xy`.
4. **Science-Tier Dominance Deduplication:** Drop all `NRT` rows for any `(platform, acq_date)` covered by `ML_SCIENCE` or `SP_SCIENCE` for that platform (eliminating 40–300 m LUT geolocation-shift duplicates cleanly). Resolve intra-tier overlaps by `(platform, acq_date, acq_hhmm, ROUND(latitude, 3), ROUND(longitude, 3))` keeping `tier_rank ASC, ingest_ts DESC`.
5. Compute `platform_active_day[P, T]`: mark platform $p$ inactive on UTC date $t$ if total detections across all 6 countries `< 5` (masking spacecraft safe-mode outages).
6. Verify mandatory coverage checks: `ARG` Jul–Aug 2026 Chilecito detections present; `SNPP` ends on `2026-11-02`; `NOAA-20` & `NOAA-21` continue through `2026-11-14`; `DZA` preserves `type=2` static flares.

---

## 7. Complete Testing & Validation Framework (`pytest` — 166+ Tests)

Every test file must execute deterministically via `pytest` with concrete assertions (zero placeholder tests):

1. **`tests/test_01_ingestion_schema.py` (18 tests):** Space-delimited `MCD14ML` (`sample` with trailing dot, v1 missing `dn`), `Vxx14IMGML` CSV, Portolan GeoParquet `geometry` extraction, FIRMS API CSV, satellite alias mapping, mixed confidence parsing, missing `type -> -1`, `(platform, acq_date)` Science dominance dropping shifted NRT duplicates, polygon clipping excluding neighboring countries (`Bolivia/Tunisia/Chile`), base-60 `acq_hhmm` conversion (`51 -> 0.85h`), and solar date midnight rollover across $\pm 180^\circ$.
2. **`tests/test_02_geometry_masking.py` (18 tests):** Exact spherical `Sample -> scan, track, VZA` (MODIS nadir `1.00 km²`, edge `~9.72 km²`), monotone spline VZA inversion `[0, 65]`, MODIS bow-tie overlap conservation (`2 × 40 MW -> 40 MW`), SAMA nighttime low-conf filter, `H3-Res8` gas flare isolation absorbing 200 m geolocation jitter while preserving seasonal fires, `h3 v3/v4` `int64 <-> str` compatibility, and strict 3-coloring validity (`color[u] != color[v]` across all adjacent edges).
3. **`tests/test_03_event_clustering.py` (14 tests):** 3×3 contiguous VIIRS cluster -> 1 event, separated clusters -> 2 events, `SNPP + N20 + N21` within 50 min kept as 3 distinct pass rows (never summed), 5-minute pass gap split, `i4_m13_fallback` vs Tobit `sat_flag` classification.
4. **`tests/test_04_observation_operators.py` (20 tests):** Detection threshold ordering (`MODIS_day > MODIS_night > VIIRS_night`), analytical survival integrals $\psi \in (0, 1]$ monotonicity, conditional $\psi^{\text{FRP, eff}}$ smoothing (`E_raw=1` vs `E_raw=5`), separate 12-hour `Day` (`06–18`) and `Night` (`18–06`) Fourier integrals equaling `1.0 ± 1e-4`, `ln(v)` strictly absent from $\mu^{\text{gain, det}}$, monotonic EQM ordering, heteroscedastic $\sigma^2$ ordering, coordinate-shift invariance of GMM pyromes, and trailing-2026 cloud climatology fallback.
5. **`tests/test_05_likelihood_derivatives.py` (22 tests):** Finite-difference gradient & Hessian parity (`< 1e-6`) for Obs, Tobit Saturation (over `w in [-35, +5]` testing asymptotic tail without `NaN`/`Inf`), and Hurdle Zero; Tobit asymmetry; convexity guard (`h_miss >= 0`) inside `(v=0.7, Lambda=2.5)`; uniform $\Phi = e^z$ consistency; empirical `platform_active_day` outage masking.
6. **`tests/test_06_gpu_solver_math.py` (20 tests):** `float64` Thomas solver vs `torch.linalg.solve` (`< 1e-10`); Takahashi diagonal & lag-1 vs `torch.linalg.inv(H)` (`< 1e-10`); empirical covariance of $10{,}000$ joint Cholesky draws $L^\top u = D^{-1/2}\varepsilon$ matching $H^{-1}$; Armijo strict objective descent; 3-color spatial cloud recovery; `cuda_kernels.py` CUDA-vs-CPU parity (`< 1e-10`); dynamic range `10^-4 .. 10^6 MW` stability.
7. **`tests/test_07_synthetic_twin_truth.py` (18 tests):** Complete 26-year synthetic forward simulator with known ground truth $X_{\text{true}}$ (`Day/Night` contrast, 2002 Aqua, 2012 SNPP, 2018 N20, 2022 drift, 2023 N21, Nov 2026 SNPP end, 50% clouds, bow-tie, gas flares) asserting: 2012 raw jump `> +150%` eliminated (`< ±10%`); 3-satellite inflation eliminated; `night_day_ratio` preserved (`error < 15%`); Nov 2026 cutoff unbiased; post-2022 drift corrected; monthly correlation `r > 0.90`; 5× drought spike preserved within `±15%`.
8. **`tests/test_08_conformal_uncertainty.py` (12 tests):** Split-conformal coverage inside `[1 - alpha - 0.03, 1 - alpha + 0.05]` for $\alpha \in \{0.05, 0.10, 0.20\}$; `q05 <= q50 <= q95`; 2005 uncertainty strictly wider than 2024; cloudy uncertainty wider than clear; `ABSTAIN_LOW_EVIDENCE` trigger; 14-day fixed-lag update matching full batch (`< 1e-3`) across an active fire transition.
9. **`tests/test_09_validation_diagnostics.py` (12 tests):** Chow & Pettitt structural breaks catching raw 2012 jump (`p < 0.001`) and passing harmonized control (`p > 0.05`); 2019–2021 VIIRS ablation outperforming raw MODIS on CRPS & Spearman $\rho$; `MCD64CMQ` burned-area correlation improvement; Product 1/2/3 schema & sparse daily export invariants.
10. **`tests/test_10_webapp_api_scale.py` (12 tests):** All 5 FastAPI endpoints return 200 and expected JSON keys; CSV formula-injection neutralization (`=`, `+`, `-`, `@`); empty-bbox handling; DuckDB query latency `< 300 ms`; single-GPU chunk (`batch_size = 1000, T = 9510`) VRAM `< 6 GB`.

---

## 8. Sequential Implementation Plan & Verification Gates

1. **Phase 1 (Hours 0–4 — Env 1 Stager & Ingestion):** Implement repo skeleton, `pyproject.toml`, `Makefile`, `src/config/`, `src/ingest/`, `scripts/run_env1_stage.py`, and `test_01`. Run connectivity checks against UMD / Source Coop Portolan / FIRMS API; stage Parquet (`<= 2.5 GB`) and `.whl` bundle (`<= 80 MB`). **Gate 1:** `test_01` passes 100% and coverage audit verifies Chilecito 2026, Nov 2 SNPP end, and Algeria flares.
2. **Phase 2 (Hours 4–10 — Geometry & Clustering):** Implement `src/geometry/`, `test_02`, `test_03`. **Gate 2:** `test_01..test_03` pass 100% on CPU.
3. **Phase 3 (Hours 10–14 — Operators & Likelihoods):** Implement `src/calibration/`, `likelihoods.py`, `priors.py`, `test_04`, `test_05`. **Gate 3:** `test_04..test_05` pass 100% on CPU.
4. **Phase 4 (Hours 14–17 — Solver, CUDA Kernel & Synthetic Twin):** Implement `thomas_takahashi.py`, `cuda_kernels.py`, `solver.py`, `synthetic_twin.py`, `test_06`, `test_07`. **Gate 4:** `test_01..test_07` pass 100%; single-chunk (`B=1000`) GPU timing benchmark confirms `< 6 GB` VRAM and full run fits well under GPU quota.
5. **Phase 5 (Hours 17–21 — Full GPU Execution & Export):** Execute `run_env2_cpu_prep.py`, `run_env2_gpu_solve.py`, `conformal.py`, `export_duckdb.py`, `test_08`. **Gate 5:** `fire_harmonized.duckdb` + 3 Parquet products (`<= 4 GB`) exported cleanly and `test_08` passes.
6. **Phase 6 (Hours 21–24 — Scientific Validation):** Implement `diagnostics.py`, run `run_env2_validate.py` (`test_09`). **Gate 6:** Structural breaks, ablation, `MCD64CMQ` cross-check, and spotlight reports pass.
7. **Phase 7 (Hours 24–34 — Web App & Final Audit):** Implement `src/webapp/`, `test_10`, run complete `pytest` (`166+` tests), verify spotlight views visually, and freeze final documentation.

---

## 9. Agent Behavioral Boundaries

### Must Do
* Enforce all 10 Scientific Invariants (`INV-01..10`) and every formula in Section 5 verbatim.
* Execute the Autonomous Discovery Protocol (Section 6.2) if any data URL fails, logging provenance transparently.
* Perform all tridiagonal/Takahashi/Cholesky solves in `torch.float64`.
* Verify every module with its corresponding `pytest` suite before advancing phases.

### Must Not Do
* Never introduce competing pipelines, alternate UIs, or dead configuration branches.
* Never fabricate or substitute synthetic data when a real data download fails.
* Never put cloud visibility $\ln v$ into detected observation likelihoods (`Case A/B`).
* Never normalize $D(\tau)$ over 24 hours across `z_Day` and `z_Night`.
* Never export dense `N_active × 9,510` daily rows (always emit sparse episode rows).
* Never import `pyhdf` inside Environment 2 or hardcode paths outside `paths.py`.
* Never claim completion without citing concrete passing `pytest` counts and measured validation metrics.

---

## 10. Final Deliverables & Acceptance Checklist

1. Complete tested Python repository (`src/`, `scripts/`, `tests/`, `pyproject.toml`, `Makefile`).
2. Clean staged data & wheel artifacts (`<= 2.5 GB` + `<= 80 MB`).
3. `fire_harmonized.duckdb` + indexed Zstd Parquet **Product 1 (`obs_harmonized_provenance`)**, **Product 2 (`state_reconstructed_monthly` dense & `daily` sparse)**, and **Product 3 (`uncertainty_diagnostics`)** (`<= 4 GB` total).
4. Reproducible validation report (`diagnostics.json`) demonstrating structural continuity, VIIRS-masked ablation gains, `MCD64CMQ` correlation superiority, and Chilecito 2026 / Algeria flare / Nov 2026 SNPP verification.
5. Working `FastAPI + DuckDB + MapLibre + ECharts` Burning Activity Calendar web application (`< 300 ms` query latency, sanitized CSV download).
6. **100% passing `pytest` execution across all `166+` tests in `tests/test_01`..`tests/test_10`.**