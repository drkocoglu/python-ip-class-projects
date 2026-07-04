"""Gradients and edge detection -- from-scratch ``imgradient`` / ``edge``.

* :func:`sobel_gradients` -- Gx, Gy, magnitude, direction (degrees).
* :func:`edge_sobel`      -- boolean edge map (MATLAB ``edge(I,'sobel')`` uses an
  automatic threshold on the gradient magnitude; we use the same rule of thumb:
  a multiple of the mean squared gradient).
"""

from __future__ import annotations

import numpy as np

from .filters import convolve2d

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel_gradients(image: np.ndarray):
    """Return ``(gx, gy, magnitude, direction_deg)``."""
    img = np.asarray(image, dtype=np.float64)
    gx = convolve2d(img, _SOBEL_X)
    gy = convolve2d(img, _SOBEL_Y)
    mag = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(-gy, gx))
    return gx, gy, mag, direction


def edge_sobel(image: np.ndarray, thresh: float | None = None) -> np.ndarray:
    """Boolean edge map via Sobel magnitude thresholding.

    When ``thresh`` is ``None`` an automatic threshold is chosen from the mean
    squared gradient magnitude, matching MATLAB's default behaviour closely.
    """
    _, _, mag, _ = sobel_gradients(image)
    mag2 = mag ** 2
    if thresh is None:
        thresh = np.sqrt(4.0 * mag2.mean()) if mag2.mean() > 0 else 0.0
    return mag > thresh
