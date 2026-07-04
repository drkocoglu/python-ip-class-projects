"""Image rotation from scratch (inverse mapping + bilinear interpolation).

Reproduces MATLAB ``imrotate(I, angle, 'bilinear', 'crop')``:

* positive ``angle`` rotates counter-clockwise,
* ``'crop'`` keeps the output the same size as the input, rotating about the
  image centre.

Vectorised with NumPy so it runs fast even on full 960x1280 frames.
"""

from __future__ import annotations

import numpy as np


def rotate(image: np.ndarray, angle_deg: float, fill: float = 0.0) -> np.ndarray:
    """Rotate ``image`` by ``angle_deg`` degrees (CCW) about its centre.

    Output has the same shape as the input (``'crop'`` behaviour). Areas that
    map outside the source are set to ``fill``.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # Destination pixel grid.
    yy, xx = np.mgrid[0:h, 0:w]
    xr = xx - cx
    yr = yy - cy

    # Inverse rotation: for each destination pixel find its source coordinate.
    # Forward rotation R(theta) rotates CCW in image coords (y-down), so the
    # inverse map uses R(-theta) with the y-axis sign handled consistently.
    src_x = cos_t * xr + sin_t * yr + cx
    src_y = -sin_t * xr + cos_t * yr + cy

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = src_x - x0
    wy = src_y - y0

    # A destination pixel is valid when its source *centre* lies within the
    # image; the interpolation neighbours are clamped below. This makes an
    # angle of 0 an exact identity (no spurious blank border row/column).
    valid = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy

    out = np.where(valid, out, fill)

    if np.issubdtype(np.asarray(image).dtype, np.integer):
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out
