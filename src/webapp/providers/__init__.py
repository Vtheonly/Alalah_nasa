"""Data provider selection (server side only).

Configuration (environment variables):

* ``FH_DATA_PROVIDER`` = ``auto`` (default) | ``real`` | ``synthetic``
* ``FH_SYNTHETIC_FALLBACK`` = ``true`` (default) | ``false``
* ``FIRE_HARMONIZED_DB`` -- real engine output (default
  ``output/fire_harmonized.duckdb``)
* ``FH_SYNTHETIC_DB`` -- synthetic demonstration store (default
  ``output/synthetic_demo.duckdb``; generated on first use, ~15 s)

Rules:

* ``synthetic``: always the synthetic demonstration store.
* ``real`` / ``auto``: the real engine output is used when it exists, has the
  export schema AND passes the critical validation checks (the gate in
  ``HarmonizedStore.validation``).  Otherwise:
    - fallback enabled  -> synthetic store, with the reason reported;
    - fallback disabled -> the real store is still served if it exists, but
      labelled ``validated: false`` (never presented as validated), else 503.

Every API response carries the active provenance; the frontend never
decides anything about providers.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .store import HarmonizedStore
from .synthetic import PROVENANCE_LABEL, build_synthetic_store

logger = logging.getLogger("fire_harmonizer.webapp.providers")

_LOCK = threading.Lock()
_ACTIVE: dict | None = None

REAL_LABEL = "Real MODIS/VIIRS harmonization engine output"


def _bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def config() -> dict:
    return {"mode": os.environ.get("FH_DATA_PROVIDER", "auto").strip().lower(),
            "synthetic_fallback": _bool("FH_SYNTHETIC_FALLBACK", True),
            "real_db": os.environ.get("FIRE_HARMONIZED_DB",
                                      "output/fire_harmonized.duckdb"),
            "synthetic_db": os.environ.get("FH_SYNTHETIC_DB",
                                           "output/synthetic_demo.duckdb")}


def _open_synthetic(cfg) -> HarmonizedStore:
    path = Path(cfg["synthetic_db"])
    if not path.exists():
        logger.info("building synthetic demonstration store at %s", path)
        build_synthetic_store(path)
    return HarmonizedStore(path, "SYNTHETIC_DEMO", PROVENANCE_LABEL)


def _evaluate_real(cfg):
    path = Path(cfg["real_db"])
    if not path.exists():
        return None, {"available": False,
                      "reason": f"real engine output not found ({path.name})"}
    try:
        store = HarmonizedStore(path, "REAL_ENGINE", REAL_LABEL)
    except Exception as exc:  # noqa: BLE001
        return None, {"available": False,
                      "reason": f"real engine output unreadable: {exc}"}
    gate = store.validation()["gate"]
    status = {"available": True, "validated": gate["passed"],
              "failed_checks": gate["failed"],
              "reason": ("passed the critical validation checks" if gate["passed"]
                         else "did not pass critical validation checks: "
                         + ", ".join(gate["failed"] or ["no diagnostics"]))}
    return store, status


def _select() -> dict:
    cfg = config()
    real_status = {"available": False, "reason": "not evaluated (synthetic mode)"}
    if cfg["mode"] == "synthetic":
        return {"store": _open_synthetic(cfg), "validated": False,
                "reason": "FH_DATA_PROVIDER=synthetic", "config": cfg,
                "real_engine": real_status}
    store, real_status = _evaluate_real(cfg)
    if store is not None and real_status.get("validated"):
        return {"store": store, "validated": True, "config": cfg,
                "reason": "real engine output passed validation",
                "real_engine": real_status}
    if cfg["synthetic_fallback"]:
        if store is not None:
            store.close()
        return {"store": _open_synthetic(cfg), "validated": False, "config": cfg,
                "reason": "synthetic fallback: " + real_status["reason"],
                "real_engine": real_status}
    if store is not None:
        return {"store": store, "validated": False, "config": cfg,
                "reason": "real engine output served UNVALIDATED (fallback "
                          "disabled): " + real_status["reason"],
                "real_engine": real_status}
    raise RuntimeError("no data provider available: " + real_status["reason"]
                       + " and FH_SYNTHETIC_FALLBACK is disabled")


def get_active() -> dict:
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is None:
            _ACTIVE = _select()
        return _ACTIVE


def reset() -> None:
    """Drop the cached provider (re-evaluated on the next request)."""
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            try:
                _ACTIVE["store"].close()
            except Exception:  # noqa: BLE001
                pass
        _ACTIVE = None


def provenance() -> dict:
    a = get_active()
    st = a["store"]
    return {"source": st.source, "is_synthetic": st.source == "SYNTHETIC_DEMO",
            "validated": a["validated"], "label": st.label,
            "reason": a["reason"]}
