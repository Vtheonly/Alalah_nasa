"""Single source of truth for every file path in the pipeline.

Contract (readme Section 4, item 3): ``DataPaths.from_env`` auto-detects the
Kaggle staged datasets (``/kaggle/input/staged-data``, ``/kaggle/input/
offline-wheels``) and ``/kaggle/working`` output root when present, falling
back to local ``./data`` and ``./output``.  No other module may hardcode a
filesystem path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_kaggle_dataset(kaggle_input: Path, name: str,
                         marker: str | None = None) -> Path | None:
    """Locate an attached Kaggle dataset directory by slug.

    Kaggle mounts datasets at ``/kaggle/input/<slug>`` (classic) or
    ``/kaggle/input/datasets/<owner>/<slug>`` (current layout), and
    competitions under ``/kaggle/input/competitions/<slug>``; search up to
    four levels deep.  ``marker`` (e.g. ``active_fires``) also identifies the
    staged-data root when the dataset was published under another slug.
    """
    if not kaggle_input.is_dir():
        return None
    frontier = [kaggle_input]
    for _depth in range(4):
        nxt = []
        for d in frontier:
            try:
                children = sorted(c for c in d.iterdir() if c.is_dir())
            except OSError:
                continue
            for c in children:
                if c.name == name:
                    return c
                if marker is not None and (c / marker).is_dir():
                    return c
                nxt.append(c)
        frontier = nxt
    return None


def _detect_roots(base_dir: str | os.PathLike | None = None) -> tuple[Path, Path, Path]:
    """Return (staged_input_root, wheels_dir, output_root).

    Precedence: explicit ``base_dir`` > per-path environment overrides
    (``FIRE_HARMONIZER_STAGED_DIR`` / ``FIRE_HARMONIZER_WHEELS_DIR`` /
    ``FIRE_HARMONIZER_OUTPUT_DIR``) > Kaggle auto-detection > repository
    ``FIRE_HARMONIZER_BASE`` > local ``./data`` + ``./output``.
    """
    if base_dir is not None:
        root = Path(base_dir)
        return root / "data", root / "offline_wheels", root / "output"
    env_base = os.environ.get("FIRE_HARMONIZER_BASE")
    if env_base:
        root = Path(env_base)
        staged, wheels, output = (root / "data", root / "offline_wheels",
                                  root / "output")
    else:
        root = Path(__file__).resolve().parents[2]
        staged, wheels, output = (root / "data", root / "offline_wheels",
                                  root / "output")
        kaggle_input = Path("/kaggle/input")
        k_staged = _find_kaggle_dataset(kaggle_input, "staged-data",
                                        marker="active_fires")
        if k_staged is not None:
            staged = k_staged
            k_wheels = _find_kaggle_dataset(kaggle_input, "offline-wheels")
            wheels = k_wheels if k_wheels is not None else staged / "offline_wheels"
        if Path("/kaggle/working").is_dir():
            output = Path("/kaggle/working/output")
    staged = Path(os.environ.get("FIRE_HARMONIZER_STAGED_DIR", staged))
    wheels = Path(os.environ.get("FIRE_HARMONIZER_WHEELS_DIR", wheels))
    output = Path(os.environ.get("FIRE_HARMONIZER_OUTPUT_DIR", output))
    return staged, wheels, output


@dataclass
class DataPaths:
    """All pipeline artifact paths (Env1 -> Env2 -> Env3)."""

    root: Path = field(default_factory=Path.cwd)
    # --- Environment 1 staged inputs -------------------------------------
    raw_stage_dir: Path = None            # staged_raw/ (raw parquet chunks)
    active_fires_dir: Path = None         # active_fires/ (country,year)-partitioned
    cloud_parquet: Path = None            # cloud_visibility_025deg.parquet
    burned_area_parquet: Path = None      # burned_area_val_025deg.parquet
    country_polygons_parquet: Path = None  # country_polygons.parquet
    wheels_dir: Path = None               # offline_wheels/
    # --- Environment 2 intermediates --------------------------------------
    provenance_base_parquet: Path = None  # obs_provenance_base.parquet (Product 1 base)
    pass_obs_parquet: Path = None         # pass_observations.parquet
    pyromes_parquet: Path = None          # hex_pyromes.parquet
    calibration_pkl: Path = None          # calibration_artifacts.pkl
    pass_calibrated_parquet: Path = None  # pass_observations_calibrated.parquet
    priors_pkl: Path = None               # priors.pkl
    state_checkpoint_pt: Path = None      # MAP state checkpoint
    state_monthly_parquet: Path = None    # raw monthly trajectory statistics
    state_daily_parquet: Path = None      # raw daily trajectory statistics
    # --- Environment 2/3 products -----------------------------------------
    duckdb_path: Path = None              # fire_harmonized.duckdb
    product1_parquet: Path = None         # obs_harmonized_provenance.parquet
    product2_monthly_parquet: Path = None  # state_reconstructed_monthly.parquet
    product2_daily_parquet: Path = None   # state_reconstructed_daily.parquet (sparse)
    product3_parquet: Path = None         # uncertainty_diagnostics.parquet
    diagnostics_json: Path = None         # diagnostics.json
    provenance_manifest: Path = None      # data_provenance_manifest.json
    # --- Tests -------------------------------------------------------------
    fixtures_dir: Path = None             # tests/fixtures

    @classmethod
    def from_env(cls, base_dir: str | os.PathLike | None = None) -> "DataPaths":
        staged_root, wheels, output = _detect_roots(base_dir)
        staged_root = Path(staged_root)
        wheels = Path(wheels)
        output = Path(output)
        p = cls()
        p.root = output.parent
        # Env 1 outputs live in the staged root (they become Env 2 inputs).
        p.raw_stage_dir = staged_root / "staged_raw"
        p.active_fires_dir = staged_root / "active_fires"
        p.cloud_parquet = staged_root / "cloud_visibility_025deg.parquet"
        p.burned_area_parquet = staged_root / "burned_area_val_025deg.parquet"
        p.country_polygons_parquet = staged_root / "country_polygons.parquet"
        p.wheels_dir = wheels
        # Env 2 intermediates + products
        p.provenance_base_parquet = output / "obs_provenance_base.parquet"
        p.pass_obs_parquet = output / "pass_observations.parquet"
        p.pyromes_parquet = output / "hex_pyromes.parquet"
        p.calibration_pkl = output / "calibration_artifacts.pkl"
        p.pass_calibrated_parquet = output / "pass_observations_calibrated.parquet"
        p.priors_pkl = output / "priors.pkl"
        p.state_checkpoint_pt = output / "state_checkpoint.pt"
        p.state_monthly_parquet = output / "state_monthly_raw.parquet"
        p.state_daily_parquet = output / "state_daily_raw.parquet"
        p.duckdb_path = output / "fire_harmonized.duckdb"
        p.product1_parquet = output / "obs_harmonized_provenance.parquet"
        p.product2_monthly_parquet = output / "state_reconstructed_monthly.parquet"
        p.product2_daily_parquet = output / "state_reconstructed_daily.parquet"
        p.product3_parquet = output / "uncertainty_diagnostics.parquet"
        p.diagnostics_json = output / "diagnostics.json"
        p.provenance_manifest = staged_root / "data_provenance_manifest.json"
        p.fixtures_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
        return p

    def ensure_output_dirs(self) -> None:
        for attr in (
            self.raw_stage_dir, self.active_fires_dir, self.wheels_dir,
            self.duckdb_path.parent,
        ):
            if attr is None:
                continue
            try:
                Path(attr).mkdir(parents=True, exist_ok=True)
            except OSError:
                # read-only Kaggle input mounts (Env 2 never writes there)
                if attr == self.duckdb_path.parent:
                    raise
