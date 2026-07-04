"""Image binarization -- from-scratch replacement for MATLAB ``imbinarize``.

MATLAB's ``imbinarize`` uses Otsu's method by default (global threshold that
maximises between-class variance). We reimplement it here from the intensity
histogram.
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(image: np.ndarray) -> int:
    """Compute a global threshold with Otsu's method.

    Parameters
    ----------
    image:
        2-D ``uint8`` grayscale image.

    Returns
    -------
    int
        Threshold level in ``[0, 255]``. Pixels *strictly greater* than the
        level are considered foreground by :func:`binarize`.
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        # Scale arbitrary ranges into 0..255 before histogramming.
        lo, hi = float(img.min()), float(img.max())
        if hi <= lo:
            return 0
        img = ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)

    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    prob = hist / total

    # Cumulative zeroth (weight) and first (mean) moments.
    omega = np.cumsum(prob)                    # class 0 weight up to each level
    mu = np.cumsum(prob * np.arange(256))      # class 0 cumulative mean
    mu_total = mu[-1]

    # Between-class variance for every candidate threshold; guard divide-by-0.
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_total * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = 0.0

    return int(np.argmax(sigma_b))


def binarize(image: np.ndarray, level: float | None = None) -> np.ndarray:
    """Binarize an image to a boolean foreground mask.

    Parameters
    ----------
    image:
        2-D grayscale image.
    level:
        Optional threshold. If a value in ``(0, 1]`` is given it is treated as a
        normalised level (like ``imbinarize(I, 0.4)`` in MATLAB) and scaled to
        the image's ``uint8`` range. If ``None``, Otsu's threshold is used.

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` where ``pixel > threshold``.
    """
    img = np.asarray(image)
    if level is None:
        thr = otsu_threshold(img)
    elif 0.0 < level <= 1.0:
        thr = level * 255.0
    else:
        thr = float(level)
    return img.astype(np.float64) > thr
