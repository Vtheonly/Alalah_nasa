"""Version-safe Uber H3 utilities for the harmonization grid.

Contract (readme.md Section 4, `src/geometry/h3_utils.py`):
  * 64-bit ``int64`` H3 IDs stored internally, clean conversion to hex strings
    for ``h3 v4`` API calls, working uniformly across ``h3`` v3 and v4.
  * ``latlon_to_h3_int(lats, lons, res) -> np.ndarray[int64]``
  * ``h3_int_to_str(cell_int) -> str``
  * ``h3_neighbors(cell_int) -> list[int]`` (1-ring)
  * ``h3_boundary_geojson(cell_int) -> list[[lon, lat]]``
  * ``h3_area_km2(cell_int) -> float``
  * ``three_color_hex_graph(active_cells) -> (colors, graph)`` — deterministic
    greedy 3-coloring of the planar hexagon graph with zero adjacent
    same-color neighbors (verified by tests).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:  # h3 v4
    import h3 as _h3

    _H3_V4 = hasattr(_h3, "latlng_to_cell")
except ImportError:  # pragma: no cover - h3 is a hard engine dependency
    _h3 = None
    _H3_V4 = False


def _require_h3():
    if _h3 is None:
        raise ImportError("the `h3` package is required for the harmonizer")
    return _h3


# --------------------------------------------------------------------------- #
# int64 <-> string conversion (v3/v4 safe)
# --------------------------------------------------------------------------- #
def latlon_to_h3_int(
    lats, lons, res: int
) -> np.ndarray:
    """Vectorized (lat, lon) -> H3 cell id at ``res``, as int64."""
    h3 = _require_h3()
    lats_arr = np.atleast_1d(np.asarray(lats, dtype=float))
    lons_arr = np.atleast_1d(np.asarray(lons, dtype=float))
    out = np.empty(lats_arr.shape, dtype=np.int64)
    if _H3_V4:
        for i, (la, lo) in enumerate(zip(lats_arr, lons_arr)):
            out[i] = np.int64(
                int(h3.latlng_to_cell(float(la), float(lo), res), 16)
            )
    else:  # h3 v3
        for i, (la, lo) in enumerate(zip(lats_arr, lons_arr)):
            out[i] = np.int64(h3.geo_to_h3(float(la), float(lo), res))
    if lats_arr.size == 1 and np.ndim(lats) == 0:
        return out[:1]  # callers index element 0
    return out


def h3_int_to_str(cell_int) -> str:
    """int64 H3 id -> canonical hex string."""
    h3 = _require_h3()
    if isinstance(cell_int, (np.integer, int)):
        if _H3_V4:
            return h3.int_to_str(int(cell_int))
        # v3: int is already the canonical index form; format as hex.
        return _int_to_hex_v3(int(cell_int))
    raise TypeError(f"expected integer H3 id, got {type(cell_int)}")


def _int_to_hex_v3(value: int) -> str:
    return format(value & 0xFFFFFFFFFFFFFFFF, "016x")


def h3_str_to_int(cell_str: str) -> int:
    """Hex string -> int64 H3 id."""
    h3 = _require_h3()
    if _H3_V4:
        return int(h3.str_to_int(cell_str))
    return int(cell_str, 16)


def h3_neighbors(cell_int) -> List[int]:
    """1-ring neighbor cell ids (excluding the cell itself)."""
    h3 = _require_h3()
    if _H3_V4:
        cell_str = h3_int_to_str(cell_int)
        return sorted(int(c, 16) for c in h3.grid_disk(cell_str, 1)
                      if c != cell_str)
    return sorted(
        int(c) for c in h3.k_ring(int(cell_int), 1) if int(c) != int(cell_int)
    )


def h3_boundary_geojson(cell_int) -> List[List[float]]:
    """Cell boundary as a [[lon, lat], ...] ring (GeoJSON orientation)."""
    h3 = _require_h3()
    if _H3_V4:
        cell_str = h3_int_to_str(cell_int)
        return [[float(lon), float(lat)]
                for lon, lat in h3.cell_to_boundary(cell_str)]
    return [[float(lon), float(lat)]
            for lon, lat in h3.cell_to_boundary(int(cell_int))]


def h3_area_km2(cell_int) -> float:
    """Mean hexagon area in km^2."""
    h3 = _require_h3()
    if _H3_V4:
        res = h3.get_resolution(h3_int_to_str(cell_int))
        return float(h3.average_hexagon_area(res, unit="km^2"))
    res = h3.h3_get_resolution(int(cell_int))
    return float(h3.hex_area(res, unit="km^2"))


def h3_resolution(cell_int) -> int:
    h3 = _require_h3()
    if _H3_V4:
        return int(h3.get_resolution(h3_int_to_str(cell_int)))
    return int(h3.h3_get_resolution(int(cell_int)))


def h3_center_latlon(cell_int) -> Tuple[float, float]:
    """Cell center as (lat, lon)."""
    h3 = _require_h3()
    if _H3_V4:
        lat, lon = h3.cell_to_latlng(h3_int_to_str(cell_int))
        return float(lat), float(lon)
    lat, lon = h3.h3_to_geo(int(cell_int))
    return float(lat), float(lon)


# --------------------------------------------------------------------------- #
# Active-cell graph and deterministic greedy 3-coloring
# --------------------------------------------------------------------------- #
def build_active_cell_graph(
    active_cells: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Neighbor lists restricted to the active set.

    Returns ``(active_sorted, adjacency)`` where ``adjacency[i]`` lists the
    positions (row indices into ``active_sorted``) of active 1-ring neighbors
    of cell ``active_sorted[i]``.
    """
    active_sorted = np.array(sorted(set(int(c) for c in active_cells)),
                             dtype=np.int64)
    pos = {int(c): i for i, c in enumerate(active_sorted)}
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(active_sorted))}
    for i, cell in enumerate(active_sorted):
        for nb in h3_neighbors(int(cell)):
            j = pos.get(nb)
            if j is not None:
                adjacency[i].append(j)
    return active_sorted, adjacency


def three_color_hex_graph(
    active_cells: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Deterministic 3-coloring of the active H3 hexagon graph.

    A planar hexagonal grid is 3-colorable; the coloring uses a deterministic
    DSATUR greedy sweep (highest-saturation cell first, ties broken by larger
    degree then smaller cell index) followed by a verification pass and a
    bounded local recolor repair, guaranteeing ``color[u] != color[v]`` for
    every adjacency edge (which the Gauss-Seidel solver and the tests rely
    on). Raises ``RuntimeError`` if the graph admits no 3-coloring rather
    than silently emitting a corrupted coloring.

    Returns ``(colors, adjacency)`` with ``colors[i] in {0, 1, 2}`` aligned
    with the sorted active-cell array, plus the adjacency map used later by
    the sparse spatial prior.
    """
    import heapq

    active_sorted, adjacency = build_active_cell_graph(active_cells)
    n = len(active_sorted)
    colors = np.full(n, -1, dtype=np.int8)
    if n == 0:
        return colors, adjacency

    neighbor_colors: List[set] = [set() for _ in range(n)]
    degree = [len(adjacency[i]) for i in range(n)]
    heap = [(0, -degree[i], i) for i in range(n)]
    heapq.heapify(heap)
    conflicts: List[int] = []

    while heap:
        neg_sat, neg_deg, v = heapq.heappop(heap)
        if colors[v] != -1:
            continue  # stale heap entry
        sat = len(neighbor_colors[v])
        if -neg_sat != sat:
            # saturation grew since this entry was pushed; re-queue.
            heapq.heappush(heap, (-sat, neg_deg, v))
            continue
        pick = next((c for c in (0, 1, 2) if c not in neighbor_colors[v]), -1)
        if pick < 0:
            # Should not occur on hex-lattice graphs; repair below.
            pick = 0
            conflicts.append(v)
        colors[v] = pick
        for u in adjacency[v]:
            if colors[u] == -1:
                neighbor_colors[u].add(pick)
                heapq.heappush(
                    heap, (-len(neighbor_colors[u]), -degree[u], u)
                )

    if conflicts:
        _repair_conflicts(colors, adjacency, conflicts)

    # Hard verification: zero adjacent same-color neighbors.
    for i, neighbors in adjacency.items():
        ci = colors[i]
        for j in neighbors:
            if colors[j] == ci:
                raise RuntimeError(
                    "3-coloring failed: adjacent cells share a color; the "
                    "active hex graph is not 3-colorable"
                )
    return colors, adjacency


def _repair_conflicts(
    colors: np.ndarray,
    adjacency: Dict[int, List[int]],
    conflicts: List[int],
    max_depth: int = 3,
) -> None:
    """Depth-limited deterministic local recoloring for conflict vertices."""

    def try_recolor(v: int, depth: int) -> bool:
        used = {colors[u] for u in adjacency[v]}
        free = [c for c in (0, 1, 2) if c not in used]
        if free:
            colors[v] = free[0]
            return True
        if depth <= 0:
            return False
        for u in sorted(adjacency[v]):
            for c in (0, 1, 2):
                if c == colors[v]:
                    continue
                old = colors[u]
                colors[u] = c
                if try_recolor(u, depth - 1):
                    return True
                colors[u] = old
        return False

    for v in conflicts:
        if not try_recolor(v, max_depth):
            raise RuntimeError(
                "3-coloring repair failed: graph is not 3-colorable"
            )
