"""Version-safe Uber H3 wrapper (h3 v3 and v4) and hex-graph 3-coloring.

Contract (readme Sections 4.6, 5.1): store 64-bit ``int64`` H3 IDs internally
while converting cleanly to hex strings for ``h3 v4`` API calls
(``latlng_to_cell``, ``grid_disk``, ``cell_to_boundary``, ``cell_to_latlng``,
``average_hexagon_area``).  Implements a deterministic greedy **3-coloring**
of the active H3 hexagon graph (``color in {0, 1, 2}``) verified to have
zero adjacent same-color neighbors.
"""
from __future__ import annotations

import numpy as np

from ..config import settings as S

try:  # h3 v4.x
    import h3 as _h3
    if hasattr(_h3, "latlng_to_cell"):
        _H3_V4 = True
    else:  # pragma: no cover - h3 v3 fallback
        _H3_V4 = False
except ImportError as _exc:  # pragma: no cover
    raise ImportError("h3 is required (offline wheel bundle includes it)") from _exc


# ---------------------------------------------------------------------------
# Version-safe primitives
# ---------------------------------------------------------------------------
def latlon_to_h3_int(lats, lons, res: int = S.H3_MAIN_RES) -> np.ndarray:
    """Vectorized (lat, lon) -> H3 int64 cell IDs at resolution ``res``."""
    lats = np.atleast_1d(np.asarray(lats, dtype=np.float64))
    lons = np.atleast_1d(np.asarray(lons, dtype=np.float64))
    out = np.empty(lats.shape, dtype=np.int64)
    flat_lat = lats.ravel()
    flat_lon = lons.ravel()
    flat_out = out.ravel()
    if _H3_V4:
        for k in range(flat_lat.size):
            cell = _h3.latlng_to_cell(float(flat_lat[k]), float(flat_lon[k]), res)
            flat_out[k] = _h3.str_to_int(cell)
    else:  # pragma: no cover - h3 v3
        for k in range(flat_lat.size):
            flat_out[k] = _h3.geo_to_h3(float(flat_lat[k]), float(flat_lon[k]), res)
    return out.reshape(lats.shape) if lats.ndim > 1 else out


def h3_int_to_str(cell_int) -> str:
    """int64 cell -> canonical H3 hex string."""
    if _H3_V4:
        return _h3.int_to_str(int(cell_int))
    return _h3.h3_to_string(int(cell_int))  # pragma: no cover


def h3_str_to_int(cell_str: str) -> int:
    if _H3_V4:
        return _h3.str_to_int(cell_str)
    return int(cell_str, 16)  # pragma: no cover


def h3_neighbors(cell_int) -> list[int]:
    """1-ring neighbor cell IDs (excluding the cell itself), int64."""
    if _H3_V4:
        cell = _h3.int_to_str(int(cell_int))
        disk = _h3.grid_disk(cell, 1)
        return sorted(_h3.str_to_int(c) for c in disk if c != cell)
    disk = _h3.k_ring(int(cell_int), 1)  # pragma: no cover
    return sorted(c for c in disk if c != int(cell_int))  # pragma: no cover


def h3_boundary_geojson(cell_int) -> list[list[float]]:
    """Cell boundary as GeoJSON [[lon, lat], ...] ring (closed implicitly)."""
    if _H3_V4:
        cell = _h3.int_to_str(int(cell_int))
        ring = _h3.cell_to_boundary(cell)     # iterable of (lat, lon)
        return [[float(lon), float(lat)] for lat, lon in ring]
    ring = _h3.h3_to_geo_boundary(int(cell_int))  # pragma: no cover
    return [[float(lon), float(lat)] for lat, lon in ring]  # pragma: no cover


def h3_cell_center(cell_int) -> tuple[float, float]:
    """Cell centroid (lat, lon)."""
    if _H3_V4:
        lat, lon = _h3.cell_to_latlng(_h3.int_to_str(int(cell_int)))
        return float(lat), float(lon)
    lat, lon = _h3.h3_to_geo(int(cell_int))  # pragma: no cover
    return float(lat), float(lon)  # pragma: no cover


def h3_area_km2(cell_int_or_res) -> float:
    """Average hexagon area (km^2) for the cell's resolution."""
    if isinstance(cell_int_or_res, int) and cell_int_or_res < 16:
        res = cell_int_or_res
    else:
        res = h3_resolution(cell_int_or_res)
    if _H3_V4:
        try:
            return float(_h3.average_hexagon_area(res, unit="km^2"))
        except (ValueError, TypeError):
            return float(_h3.average_hexagon_area(res, unit="km2"))
    return float(_h3.hex_area(res, unit="km^2"))  # pragma: no cover


def h3_resolution(cell_int) -> int:
    if _H3_V4:
        return _h3.get_resolution(_h3.int_to_str(int(cell_int)))
    return _h3.h3_get_resolution(int(cell_int))  # pragma: no cover


def h3_is_valid(cell_int) -> bool:
    if _H3_V4:
        return _h3.is_valid_cell(_h3.int_to_str(int(cell_int)))
    return _h3.h3_is_valid(int(cell_int))  # pragma: no cover


# ---------------------------------------------------------------------------
# Deterministic greedy 3-coloring of the active hex graph
# ---------------------------------------------------------------------------
def build_adjacency(active_cells: np.ndarray) -> tuple[list[list[int]], int]:
    """Adjacency lists (indices) + edge count over the active-cell set."""
    cells = np.array(sorted(set(int(c) for c in active_cells)), dtype=np.int64)
    pos = {int(c): i for i, c in enumerate(cells)}
    adj: list[list[int]] = [[] for _ in range(len(cells))]
    edges = 0
    for i, c in enumerate(cells):
        for nb in h3_neighbors(int(c)):
            j = pos.get(nb)
            if j is not None:
                adj[i].append(j)
                edges += 1
    return adj, edges // 2


def _greedy_three_color(adj: list[list[int]]) -> np.ndarray:
    colors = np.full(len(adj), -1, dtype=np.int8)
    for i in range(len(adj)):
        used = {int(colors[j]) for j in adj[i] if colors[j] >= 0}
        for col in (0, 1, 2):
            if col not in used:
                colors[i] = col
                break
        if colors[i] < 0:
            colors[i] = 0          # over-constrained locally; repaired below
    return colors


def _conflicts(adj, colors) -> list[int]:
    return [i for i in range(len(adj))
            if any(colors[i] == colors[j] for j in adj[i])]


def _backtrack_three_color(adj, colors_init=None, max_nodes: int = 2_000_000
                          ) -> np.ndarray:
    """Exact DFS 3-coloring (the H3 hex graph is always 3-colorable)."""
    n = len(adj)
    colors = np.full(n, -1, dtype=np.int8)
    order = sorted(range(n), key=lambda i: -len(adj[i]))
    # seed from greedy where consistent (speeds convergence)
    if colors_init is not None:
        for i in range(n):
            if colors_init[i] >= 0:
                colors[i] = colors_init[i]
        for i in _conflicts(adj, colors):
            colors[i] = -1
        # uncolor anything adjacent to uncolored conflicts is unnecessary;
        # verify partial consistency
        for i in range(n):
            if colors[i] >= 0 and any(colors[j] == colors[i] and j > i
                                      for j in adj[i]):
                colors[i] = -1
    nodes = 0

    def ok(i: int, col: int) -> bool:
        return all(colors[j] != col for j in adj[i])

    def dfs(k: int) -> bool:
        nonlocal nodes
        # next uncolored vertex with most constraints
        cand = [v for v in order if colors[v] < 0]
        if not cand:
            return True
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError("3-coloring backtracking budget exceeded")
        v = cand[0]
        for col in (0, 1, 2):
            if ok(v, col):
                colors[v] = col
                if dfs(k + 1):
                    return True
                colors[v] = -1
        return False

    if not dfs(0):
        raise RuntimeError("graph is not 3-colorable (not an H3 hex graph?)")
    return colors


def three_color_hex_graph(active_cells: np.ndarray) -> tuple[np.ndarray, dict]:
    """Deterministic valid 3-coloring (colors in {0,1,2}) of the active H3
    hexagon graph.  Returns (colors aligned to sorted unique cells, meta).

    ``meta`` maps sorted-cell index -> color and includes graph statistics.
    The returned coloring is verified: ``color[u] != color[v]`` across every
    adjacent edge (guaranteed by construction + hard verification).
    """
    cells = np.array(sorted(set(int(c) for c in np.asarray(active_cells).ravel())),
                     dtype=np.int64)
    if len(cells) == 0:
        return np.array([], dtype=np.int8), {"n_cells": 0, "n_edges": 0,
                                             "cells": [], "colors": []}
    adj, n_edges = build_adjacency(cells)
    colors = _greedy_three_color(adj)
    if _conflicts(adj, colors):
        colors = _backtrack_three_color(adj, colors_init=colors)
    bad = _conflicts(adj, colors)
    if bad:  # pragma: no cover - safety net
        raise RuntimeError(f"3-coloring verification failed on {len(bad)} vertices")
    meta = {"n_cells": int(len(cells)), "n_edges": int(n_edges),
            "cells": [int(c) for c in cells], "colors": colors.tolist()}
    return colors, meta


def sorted_active_cells(active_cells) -> np.ndarray:
    """Canonical ordering of the active cell set (sorted int64)."""
    return np.array(sorted(set(int(c) for c in np.asarray(active_cells).ravel())),
                    dtype=np.int64)
