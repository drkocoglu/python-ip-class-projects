"""Adaptive histogram equalization (CLAHE) -- from-scratch ``adapthisteq``.

Contrast-Limited Adaptive Histogram Equalization:

1. split the image into a grid of tiles,
2. build a clipped, redistributed CDF for each tile,
3. bilinearly interpolate the four surrounding tile mappings for every pixel.

Operates on small rank/suit crops in the pipeline, so speed is a non-issue.
"""

from __future__ import annotations

import numpy as np


def _tile_cdf(tile: np.ndarray, n_bins: int, clip_limit: float) -> np.ndarray:
    """Clipped, redistributed CDF mapping ``0..n_bins-1 -> 0..255``."""
    hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.float64)

    if clip_limit > 0:
        clip = max(1.0, clip_limit * tile.size / n_bins)
        excess = np.maximum(hist - clip, 0.0).sum()
        hist = np.minimum(hist, clip)
        hist += excess / n_bins  # redistribute clipped mass uniformly

    cdf = np.cumsum(hist)
    if cdf[-1] == 0:
        return np.arange(n_bins, dtype=np.float64) * 255.0 / (n_bins - 1)
    return (cdf - cdf[0]) / (cdf[-1] - cdf[0]) * 255.0


def adapthisteq(image: np.ndarray, n_tiles=(8, 8), clip_limit: float = 0.01,
                n_bins: int = 256) -> np.ndarray:
    """Apply CLAHE to a grayscale image.

    Parameters
    ----------
    image:
        2-D grayscale image (``uint8`` or float in ``[0, 255]``).
    n_tiles:
        ``(rows, cols)`` tile grid. MATLAB default is ``(8, 8)``
        the card
        pipeline uses ``(2, 2)`` on the tiny glyph crops.
    clip_limit:
        Contrast clip in ``[0, 1]`` (fraction of tile size per bin).
    n_bins:
        Number of histogram bins (grey levels).
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape
    ty, tx = n_tiles
    th = int(np.ceil(h / ty))
    tw = int(np.ceil(w / tx))

    # Mapping table for every tile: shape (ty, tx, n_bins).
    maps = np.zeros((ty, tx, n_bins), dtype=np.float64)
    for i in range(ty):
        for j in range(tx):
            y0, y1 = i * th, min((i + 1) * th, h)
            x0, x1 = j * tw, min((j + 1) * tw, w)
            tile = img[y0:y1, x0:x1]
            if tile.size == 0:
                maps[i, j] = np.arange(n_bins) * 255.0 / (n_bins - 1)
            else:
                maps[i, j] = _tile_cdf(tile, n_bins, clip_limit)

    # Tile centre coordinates.
    cy = (np.arange(ty) + 0.5) * th
    cx = (np.arange(tx) + 0.5) * tw

    out = np.zeros((h, w), dtype=np.float64)
    yy = np.arange(h)
    xx = np.arange(w)

    # Locate each pixel between tile centres and bilinearly blend mappings.
    iy = np.clip(np.searchsorted(cy, yy) - 1, 0, ty - 2) if ty > 1 else np.zeros(h, int)
    ix = np.clip(np.searchsorted(cx, xx) - 1, 0, tx - 2) if tx > 1 else np.zeros(w, int)

    for yi in range(h):
        i0 = iy[yi]
        i1 = min(i0 + 1, ty - 1)
        fy = 0.0 if ty == 1 else np.clip((yy[yi] - cy[i0]) / (cy[i1] - cy[i0] + 1e-9), 0, 1)
        row = img[yi]
        for k, xi in enumerate(range(w)):
            j0 = ix[xi]
            j1 = min(j0 + 1, tx - 1)
            fx = 0.0 if tx == 1 else np.clip((xx[xi] - cx[j0]) / (cx[j1] - cx[j0] + 1e-9), 0, 1)
            v = row[xi]
            m00 = maps[i0, j0, v]
            m01 = maps[i0, j1, v]
            m10 = maps[i1, j0, v]
            m11 = maps[i1, j1, v]
            top = m00 * (1 - fx) + m01 * fx
            bot = m10 * (1 - fx) + m11 * fx
            out[yi, xi] = top * (1 - fy) + bot * fy

    return np.clip(np.round(out), 0, 255).astype(np.uint8)
