# fire_harmonizer — MODIS/VIIRS harmonization engine
# Environment-separated entry points (see readme.md Section 2.2 / 8):
#   Env 1 (internet ON, CPU):  stage        -> download + normalize + wheels
#   Env 2 (internet OFF, GPU): cpu-prep     -> geometry/calibration/priors (CPU)
#                               gpu-solve    -> float64 GMRF solve + conformal + export
#                               validate     -> structural-break & ablation diagnostics
#   Env 3 (web server):        serve        -> FastAPI + DuckDB Burning Activity Calendar
#   Anywhere (CPU):            test         -> full pytest suite (166+ tests)

.PHONY: stage cpu-prep gpu-solve validate test serve

stage:
	python scripts/run_env1_stage.py

cpu-prep:
	python scripts/run_env2_cpu_prep.py

gpu-solve:
	python scripts/run_env2_gpu_solve.py

validate:
	python scripts/run_env2_validate.py

test:
	pytest -v

serve:
	uvicorn src.webapp.server:app --host 0.0.0.0 --port 8000
