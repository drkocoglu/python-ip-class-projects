"""Histogram of Oriented Gradients (HOG) -- from-scratch ``extractHOGFeatures``.

Standard Dalal-Triggs HOG:

1. gradients -> unsigned orientation (0..180 deg) + magnitude,
2. accumulate magnitude into orientation histograms per cell,
3. group cells into overlapping blocks and L2-normalise,
4. concatenate into one feature vector.

The card project used a small cell size to get a dense descriptor of the suit
pip; :func:`hog_features` exposes ``cell_size`` / ``block_size`` / ``n_bins`` so
the suit classifier can reproduce that.
"""

from __future__ import annotations

import numpy as np

from .edges import sobel_gradients


def hog_features(image: np.ndarray, cell_size=(6, 6), block_size=(2, 2),
                 n_bins: int = 9, eps: float = 1e-6) -> np.ndarray:
    """Compute a HOG descriptor for a grayscale image.

    Returns a 1-D float feature vector (block-normalised).
    """
    img = np.asarray(image, dtype=np.float64)
    _, _, mag, direction = sobel_gradients(img)

    # Unsigned orientation in [0, 180).
    ang = direction % 180.0

    h, w = img.shape
    cy, cx = cell_size
    n_cells_y = h // cy
    n_cells_x = w // cx
    if n_cells_y == 0 or n_cells_x == 0:
        return np.zeros(0, dtype=np.float64)

    bin_width = 180.0 / n_bins
    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float64)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            m = mag[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx].ravel()
            a = ang[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx].ravel()
            b = np.clip((a / bin_width).astype(int), 0, n_bins - 1)
            cell_hist[i, j] = np.bincount(b, weights=m, minlength=n_bins)

    by, bx = block_size
    feats: list[np.ndarray] = []
    for i in range(n_cells_y - by + 1):
        for j in range(n_cells_x - bx + 1):
            block = cell_hist[i:i + by, j:j + bx].ravel()
            norm = np.sqrt((block ** 2).sum() + eps ** 2)
            feats.append(block / norm)

    if not feats:
        return cell_hist.ravel()
    return np.concatenate(feats)


def cell_histograms(image: np.ndarray, cell_size=(4, 4), n_bins: int = 9) -> np.ndarray:
    """Raw (un-normalised) HOG cell orientation histograms.

    Returns an array of shape ``(n_cells_y, n_cells_x, n_bins)`` -- the first
    stage of HOG, useful for deriving interpretable orientation statistics
    (e.g. how much vertical-stroke energy sits in the bottom of a glyph).
    """
    img = np.asarray(image, dtype=np.float64)
    _, _, mag, direction = sobel_gradients(img)
    ang = direction % 180.0
    h, w = img.shape
    cy, cx = cell_size
    n_y, n_x = h // cy, w // cx
    bin_width = 180.0 / n_bins
    out = np.zeros((max(n_y, 1), max(n_x, 1), n_bins))
    for i in range(n_y):
        for j in range(n_x):
            m = mag[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx].ravel()
            a = ang[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx].ravel()
            b = np.clip((a / bin_width).astype(int), 0, n_bins - 1)
            out[i, j] = np.bincount(b, weights=m, minlength=n_bins)
    return out
