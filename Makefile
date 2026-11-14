# Fire Harmonizer - Environment-orchestrated build targets
# Env 1 (stager, internet ON) -> Env 2 (offline GPU engine) -> Env 3 (web server)

.PHONY: stage cpu-prep gpu-solve validate test serve

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
