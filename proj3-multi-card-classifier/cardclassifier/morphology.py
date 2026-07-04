"""Binary morphology implemented from scratch.

Provides:

* :func:`fill_holes`  -- MATLAB ``imfill(bw, 'holes')`` equivalent, via a
  flood-fill of the background that starts from the image border. Any
  background pixel not reachable from the border is a hole and gets filled.
* :func:`connected_components` -- MATLAB ``bwlabel`` equivalent, 8-connected,
  via an iterative breadth-first flood fill.
* :func:`largest_component` -- convenience helper used all over the pipeline.

Everything uses an explicit stack/queue so behaviour is deterministic and easy
to unit-test; no ``scipy.ndimage`` / OpenCV.
"""

from __future__ import annotations

import numpy as np

# 8-connectivity neighbour offsets.
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
# 4-connectivity offsets (used for background flood fill in fill_holes).
_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask (``imfill(bw, 'holes')``).

    Parameters
    ----------
    mask:
        2-D boolean (or 0/1) array. Foreground is ``True``.

    Returns
    -------
    np.ndarray
        Boolean array where every background region *not* connected to the
        border has been turned into foreground.
    """
    bw = np.asarray(mask).astype(bool)
    h, w = bw.shape

    # `reachable` marks background pixels connected to the image border.
    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    # Seed the flood fill with every border pixel that is background.
    for x in range(w):
        for y in (0, h - 1):
            if not bw[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not bw[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                stack.append((y, x))

    # 4-connected flood fill through the background.
    while stack:
        y, x = stack.pop()
        for dy, dx in _NEIGHBORS_4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not bw[ny, nx] and not reachable[ny, nx]:
                reachable[ny, nx] = True
                stack.append((ny, nx))

    # A hole is any background pixel the border flood fill never reached.
    holes = (~bw) & (~reachable)
    return bw | holes


def connected_components(mask: np.ndarray, connectivity: int = 8):
    """Label connected foreground components (``bwlabel``).

    Parameters
    ----------
    mask:
        2-D boolean array. Foreground is ``True``.
    connectivity:
        ``8`` (default) or ``4``.

    Returns
    -------
    labels : np.ndarray
        ``int32`` array where background is 0 and each component has a unique
        label ``1..n``.
    n : int
        Number of components found.
    """
    bw = np.asarray(mask).astype(bool)
    h, w = bw.shape
    labels = np.zeros((h, w), dtype=np.int32)
    neigh = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4

    current = 0
    for sy in range(h):
        for sx in range(w):
            if not bw[sy, sx] or labels[sy, sx] != 0:
                continue
            current += 1
            labels[sy, sx] = current
            stack = [(sy, sx)]
            while stack:
                y, x = stack.pop()
                for dy, dx in neigh:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and bw[ny, nx] and labels[ny, nx] == 0):
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


def component_masks(labels: np.ndarray, n: int) -> list[np.ndarray]:
    """Return a boolean mask for each label ``1..n``."""
    return [labels == i for i in range(1, n + 1)]


def largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Return a boolean mask of the single largest foreground component."""
    labels, n = connected_components(mask, connectivity)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0  # ignore background
    return labels == int(np.argmax(areas))
