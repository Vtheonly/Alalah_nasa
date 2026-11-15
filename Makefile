# Fire Harmonizer - Environment-orchestrated build targets
# Env 1 (stager, internet ON) -> Env 2 (offline GPU engine) -> Env 3 (web server)

.PHONY: stage cpu-prep gpu-solve validate test serve web demo synthetic-demo

stage:  ## Env 1: download + normalize + polygon-clip + stage Parquet + offline wheels
	python3 scripts/run_env1_stage.py

cpu-prep:  ## Env 2 CPU: geometry, clustering, pyromes, operators, priors (0 GPU-hours)
	python3 scripts/run_env2_cpu_prep.py

gpu-solve:  ## Env 2 GPU: multi-GPU float64 state-space solve + conformal + export
	python3 scripts/run_env2_gpu_solve.py

validate:  ## Env 2: structural-break / ablation / MCD64CMQ / spotlight validation
	python3 scripts/run_env2_validate.py

test:  ## Full 166+ test suite (CPU-runnable; CUDA parity tests are hardware-gated)
	python3 -m pytest tests -v

serve:  ## Env 3: FastAPI + DuckDB Burning Activity Calendar web app
	python3 -m uvicorn src.webapp.server:app --host 0.0.0.0 --port 8000

web:  ## Build the React frontend (served by `make serve` at /)
	cd Frontend && npm install && npm run build

synthetic-demo:  ## Pre-build the clearly labelled synthetic demonstration store
	python3 scripts/build_synthetic_demo.py

demo:  ## Serve the web app on the SYNTHETIC demonstration store only
	FH_DATA_PROVIDER=synthetic python3 -m uvicorn src.webapp.server:app --host 0.0.0.0 --port 8000
