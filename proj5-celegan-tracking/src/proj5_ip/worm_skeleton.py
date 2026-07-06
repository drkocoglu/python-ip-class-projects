"""Skeleton geometry: centerline, head/tail, equidistant points, normals.

The worm mask is thinned to a 1-pixel skeleton, then reduced to an ordered
centerline by taking the longest path through the skeleton (a double
breadth-first search). This is robust to the small spurs that thinning can
leave behind, and its two ends are the head and tail of the worm. Equidistant
points are sampled along the centerline and a unit normal to the local tangent
is computed at each one.
"""

from collections import deque

import numpy as np
from skimage.morphology import skeletonize

from proj5_ip import tracking_config as cfg

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeletonize_worm(worm: np.ndarray) -> np.ndarray:
    """Thin the worm mask to a 1-pixel-wide boolean skeleton."""
    return skeletonize(worm)


def _bfs_farthest(
    start: tuple[int, int], pixels: set[tuple[int, int]]
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        farthest = node
        row, col = node
        for d_row, d_col in _NEIGHBORS:
            neighbor = (row + d_row, col + d_col)
            if neighbor in pixels and neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    return farthest, parent


def extract_centerline(skeleton: np.ndarray) -> np.ndarray:
    """Return the ordered centerline as an ``(N, 2)`` array of ``(row, col)``."""
    coords = np.argwhere(skeleton)
    pixels = {(int(r), int(c)) for r, c in coords}
    if not pixels:
        return np.empty((0, 2), dtype=int)

    source = next(iter(pixels))
    end_a, _ = _bfs_farthest(source, pixels)
    end_b, parent = _bfs_farthest(end_a, pixels)

    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = end_b
    while node is not None:
        path.append(node)
        node = parent[node]
    return np.array(path, dtype=int)


def equidistant_indices(length: int, count: int) -> np.ndarray:
    """Return ``count`` indices evenly spaced along a centerline of ``length``."""
    if length <= 2:
        return np.arange(length)
    margin = min(3, (length - 1) // 2)
    return np.unique(
        np.linspace(margin, length - 1 - margin, count).round().astype(int)
    )


def normal_at(centerline: np.ndarray, index: int, half_window: int) -> np.ndarray:
    """Return the unit normal (as ``[x, y]``) to the tangent at ``index``."""
    last = len(centerline) - 1
    lo = max(0, index - half_window)
    hi = min(last, index + half_window)
    d_row = float(centerline[hi, 0] - centerline[lo, 0])
    d_col = float(centerline[hi, 1] - centerline[lo, 1])
    normal = np.array([-d_row, d_col], dtype=float)  # rotate tangent by 90 deg
    magnitude = np.hypot(*normal)
    if magnitude == 0.0:
        return np.zeros(2, dtype=float)
    return normal / magnitude


def sample_geometry(
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centerline, sample_points_xy, normals_xy)`` for a skeleton.

    ``sample_points_xy`` and ``normals_xy`` are in ``(x, y)`` image order to
    match OpenCV drawing conventions.
    """
    centerline = extract_centerline(skeleton)
    if len(centerline) == 0:
        empty = np.empty((0, 2), dtype=float)
        return centerline, empty, empty

    indices = equidistant_indices(len(centerline), cfg.NUM_EQUIDISTANT_POINTS)
    points_xy = centerline[indices][:, ::-1].astype(float)
    normals_xy = np.array(
        [normal_at(centerline, int(i), cfg.TANGENT_HALF_WINDOW) for i in indices]
    )
    return centerline, points_xy, normals_xy
