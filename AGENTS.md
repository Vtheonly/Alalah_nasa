# AGENTS.md — Agent Handoff, Decisions, Tasks & Known Issues

This file is the dedicated handoff log required by the project's agent workflow protocol.
It records blockers, decisions, task state, and known issues so any future agent can
continue coherently without repeating work or inheriting mistakes.
It does NOT replace `readme.md` (the authoritative specification, which must never be
modified or deleted).

---

## Repository State Snapshot

- Repository: https://github.com/Vtheonly/Alalah_nasa.git (branch `main`)
- Authoritative specification: `readme.md` (159,451 bytes, 1,112 lines; read 100%
  sequentially in chunks — no skimming). Reference date: 2026-11-14.
- Implementation target: the exact `fire-harmonizer` tree from Part II /
  "MASTER ENGINEERING HANDOFF SPECIFICATION" of `readme.md`.

## Ground Rules Locked In (from readme.md Section 9 + user instructions)

- Execute the single architecture in `readme.md`; no competing pipelines, no fallback UIs,
  no stub functions, no placeholder tests, no synthetic substitution for failed downloads.
- Cloud visibility `ln v` NEVER enters detected-observation likelihoods (Case A/B).
- `D(tau)` normalized independently within each 12-hour Day/Night window, never over 24 h.
- Never write dense `N_active x 9,510` daily rows; daily product is sparse.
- `pyhdf` imports strictly inside Environment-1 ingestion code paths only.
- No `/kaggle/...` paths outside `src/config/paths.py`.
- All tridiagonal/Takahashi/Cholesky algebra in `torch.float64`.
- Commit + push after every meaningful edit cycle; never leave unpushed commits.

## Environment Notes (build sandbox)

- Python 3.12.16 (matches `requires-python >=3.10,<3.13`).
- CPU-only (torch 2.15.0+cpu, `torch.cuda.is_available() == False`). CUDA-parity tests in
  `test_06` are hardware-gated and skip on CPU-only machines (documented, not weakened).
- Installed: numpy, scipy, pandas, pyarrow, scikit-learn, lightgbm, shapely, h3 4.5.0,
  duckdb 2.0.0, polars, torch (CPU), fastapi, uvicorn, pydantic, pytest, httpx, requests,
  paramiko, bs4, lxml, pyhdf (built against Debian `libhdf4-alt-dev`, headers at
  `/usr/include/hdf4`).
- Constraints: 2 CPUs, 4 GB RAM, ~8 GB free disk. Tests are designed memory-frugal.
- Sandbox has internet; live UMD SFTP/FIRMS-API staging was NOT executed against NASA
  archives (no FIRMS MAP_KEY provided; UMD FTP reachability from sandbox unverified).
  Connectivity probing results are recorded under "Known Issues".

## Decision Log

- D-001: Repo root = the cloned repository itself (`Makefile`, `pyproject.toml`, `src/`,
  `scripts/`, `tests/` live at the repo root, matching the mandated tree exactly).
- D-002: Import namespace is `src.*` (the spec tree contains `src/__init__.py`);
  `pyproject.toml` distribution name is `fire_harmonizer` with explicit package list.
- D-003: Git push auth via `core.askpass` helper reading a token file OUTSIDE the repo
  (`/home/z/my-project/.secrets/`), never embedded in the remote URL or any tracked file.
- D-004: VIIRS forward pixel-geometry model (when `Pixarea` absent) uses the spec's own
  area->VZA knot table (0.141/0.180/0.280/0.430/0.620 km2 at 0/20/35/50/65 deg VZA) with a
  monotone zone-aware sample->scan-angle mapping across 3:1/2:1/1:1 aggregation-zone
  boundaries (0/31.59/44.68/56.06 deg), exactly per Section 5.1.2.
- D-005: `dn_flag` canonical assignment is the Local-Solar-Time half-day window
  (Day = [06:00,18:00)) per the audited final spec (line 859), with `sza_deg` and raw
  sensor `daynight` preserved in Product 1 for provenance (Section 5.1.1).
- D-006: CUDA kernels compile lazily via `torch.utils.cpp_extension.load_inline` only when
  CUDA + nvcc are present; otherwise dispatch transparently to the pure-PyTorch
  `thomas_takahashi.py` reference (Section 5.3.4 / audit item 8).
- D-007: Sequential Thomas/Takahashi/Cholesky recurrences are implemented in pure PyTorch
  and accelerated with `torch.jit.script` (still pure PyTorch float64, bit-identical
  algorithm); eager fallback retained.
- D-008: Tests requiring live credentials/network (UMD SFTP fetch, FIRMS MAP_KEY calls)
  are not executed; all 166 tests run on deterministic local fixtures as the spec's test
  descriptions prescribe (parsing/masking/math tests, zero network dependence).

## Known Issues / Blockers

- KI-001: No FIRMS MAP_KEY token available in this sandbox -> trailing-2026 NRT backfill
  code path implemented and unit-tested against fixture CSVs, but NOT executed live.
- KI-002: `fuoco.geog.umd.edu` reachability from the sandbox untested until Env-1 run;
  the autonomous failover protocol (Section 6.2) is implemented for that eventuality.
- KI-003: No GPU in sandbox -> `test_06` CUDA-parity test hardware-gated (skips on CPU);
  all other 165 tests must pass 100% on CPU.

## Task Ledger

- [x] Read 100% of readme.md (lines 1..1112), extract every contract.
- [x] .gitignore created before any code.
- [x] Phase 1: skeleton + config + ingest + test_01 — Gate 1 PASSED (18/18).
- [x] Phase 2: geometry + test_02/test_03 — Gate 2 PASSED (50/50 cumulative).
- [x] Phase 3: calibration + likelihoods/priors + test_04/test_05 — Gate 3 PASSED (92/92).
- [x] Phase 4: solver + cuda kernels + synthetic twin + test_06/test_07 — Gate 4 PASSED (140 pass + 1 CUDA-gated skip).
- [x] Phase 5: conformal + products + scripts + test_08 — Gate 5 PASSED.
- [x] Phase 6: diagnostics + test_09 — Gate 6 PASSED.
- [x] Phase 7: webapp + test_10 + full pytest — Gate 7 PASSED: 165 passed + 1 CUDA hardware-gated skip (test_06 CUDA bit-parity; CPU-only sandbox).

## Final Audit Notes (2026-11-14)

- Repository tree matches the mandated layout exactly (find src scripts tests | sort).
- No TODO/NotImplementedError/stub patterns in src/ (single rg hit is prose in a
  docstring); pyhdf confined to src/ingest; /kaggle only in src/config/paths.py.
- 52 tracked files; largest is readme.md (159 KB); every tracked file << 10 MB;
  no secrets/datasets staged (verified via git ls-files).
- All entrypoints verified: scripts/run_env1_stage.py, run_env2_cpu_prep.py,
  run_env2_gpu_solve.py, run_env2_validate.py --help; python -m
  src.ingest.download_data --help; src.webapp.server import; build_wheels.sh
  bash -n.
- Live connectivity probes from this sandbox: Portolan STAC catalog 200 OK
  (mirror path usable); Natural Earth GeoJSON 200 OK; fuoco.geog.umd.edu and
  firms.modaps.eosdis.nasa.gov unreachable from here (failover protocol covers
  UMD; FIRMS NRT backfill additionally requires a MAP_KEY that was not
  provided) — no live staging executed, all 166 tests run on deterministic
  local fixtures per the spec's test descriptions.
- Twin harmonization quality (5 H3 cells x 9,510 days, seed 42): monthly
  correlation r = 0.977; 2012-step residual < 7%; 3-satellite inflation
  residual < 9%; post-2022 drift residual < 4%; night/day ratio 0.301 (true
  0.300); Nov-2026 cutoff unbiased (0.998); drought spike within 8%; MODIS-only
  posterior variance ~2.9x wider than the 5-platform era.
