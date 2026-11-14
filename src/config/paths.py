"""Single source of truth for every file path used by the engine.

Contract (readme.md Section 4, `src/config/paths.py`):
  * `DataPaths.from_env(base_dir=None)` auto-detects Kaggle input datasets
    (`/kaggle/input/staged-data`, `/kaggle/input/offline-wheels`) and the
    Kaggle working output (`/kaggle/working/output`) versus local `./data`
    and `./output` layouts, driven by environment variables or directory
    detection.
  * No other module may hardcode `/kaggle/...` paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_KAGGLE_INPUT = Path("/kaggle/input")
_KAGGLE_WORKING = Path("/kaggle/working")


def _detect_kaggle() -> bool:
    """Kaggle notebooks expose both the input mount and a working directory."""
    return _KAGGLE_INPUT.is_dir() and _KAGGLE_WORKING.is_dir()


@dataclass(frozen=True)
class DataPaths:
    """Resolved filesystem layout for one execution environment."""

    base_dir: Path
    raw_stage_dir: Path        # Env 1: heterogeneous raw parquet chunks
    active_fires_dir: Path     # normalized, country/year partitioned parquet
    cloud_parquet: Path        # cloud_visibility_025deg.parquet
    burned_area_parquet: Path  # burned_area_val_025deg.parquet
    country_polygons_parquet: Path
    provenance_base_parquet: Path  # Product 1 base (obs_provenance_base)
    pass_obs_parquet: Path         # pass_observations.parquet
    pyromes_parquet: Path          # hex_pyromes.parquet
    calibration_pkl: Path          # calibration_artifacts.pkl
    pass_calibrated_parquet: Path  # pass_observations_calibrated.parquet
    priors_pkl: Path               # priors.pkl
    state_checkpoint_pt: Path      # state_checkpoint.pt
    duckdb_path: Path              # fire_harmonized.duckdb
    product1_parquet: Path         # obs_harmonized_provenance.parquet
    product2_monthly_parquet: Path  # state_reconstructed_monthly.parquet
    product2_daily_parquet: Path    # state_reconstructed_daily.parquet (sparse)
    product3_parquet: Path          # uncertainty_diagnostics.parquet
    diagnostics_json: Path          # diagnostics.json
    offline_wheels_dir: Path        # Env 1 -> Env 2 wheel bundle
    provenance_manifest_json: Path  # data_provenance_manifest.json
    in_kaggle: bool

    @classmethod
    def from_env(cls, base_dir: Optional[Path] = None) -> "DataPaths":
        """Resolve the layout for the current environment.

        Detection order:
          1. `FIRE_HARMONIZER_BASE` environment variable (explicit override).
          2. Kaggle notebooks: inputs from `/kaggle/input/staged-data`,
             outputs into `/kaggle/working/output`.
          3. Local execution: `./data` (inputs) and `./output` (artifacts)
             under `base_dir` (default: repository root = parent of `src/`).
        """
        env_base = os.environ.get("FIRE_HARMONIZER_BASE")
        if base_dir is None:
            if env_base:
                base_dir = Path(env_base).resolve()
            else:
                base_dir = Path(__file__).resolve().parents[2]

        kaggle = _detect_kaggle()
        if kaggle:
            staged_root = _KAGGLE_INPUT / "staged-data"
            data_root = staged_root if staged_root.is_dir() else _KAGGLE_INPUT
            out_root = _KAGGLE_WORKING / "output"
            wheels = _KAGGLE_INPUT / "offline-wheels"
        else:
            data_root = Path(base_dir) / "data"
            out_root = Path(base_dir) / "output"
            wheels = Path(base_dir) / "offline_wheels"

        raw_stage = data_root / "staged_raw"
        active_fires = data_root / "active_fires"
        return cls(
            base_dir=Path(base_dir),
            raw_stage_dir=raw_stage,
            active_fires_dir=active_fires,
            cloud_parquet=data_root / "cloud_visibility_025deg.parquet",
            burned_area_parquet=data_root / "burned_area_val_025deg.parquet",
            country_polygons_parquet=data_root / "country_polygons.parquet",
            provenance_base_parquet=out_root / "obs_provenance_base.parquet",
            pass_obs_parquet=out_root / "pass_observations.parquet",
            pyromes_parquet=out_root / "hex_pyromes.parquet",
            calibration_pkl=out_root / "calibration_artifacts.pkl",
            pass_calibrated_parquet=out_root
            / "pass_observations_calibrated.parquet",
            priors_pkl=out_root / "priors.pkl",
            state_checkpoint_pt=out_root / "state_checkpoint.pt",
            duckdb_path=out_root / "fire_harmonized.duckdb",
            product1_parquet=out_root / "obs_harmonized_provenance.parquet",
            product2_monthly_parquet=out_root
            / "state_reconstructed_monthly.parquet",
            product2_daily_parquet=out_root / "state_reconstructed_daily.parquet",
            product3_parquet=out_root / "uncertainty_diagnostics.parquet",
            diagnostics_json=out_root / "diagnostics.json",
            offline_wheels_dir=wheels,
            provenance_manifest_json=raw_stage / "data_provenance_manifest.json",
            in_kaggle=kaggle,
        )

    def ensure_dirs(self) -> None:
        """Create every output directory (idempotent). No other side effects."""
        for p in (
            self.raw_stage_dir,
            self.active_fires_dir,
            self.provenance_base_parquet.parent,
            self.offline_wheels_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
