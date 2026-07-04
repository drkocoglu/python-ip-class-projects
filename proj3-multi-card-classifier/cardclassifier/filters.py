"""Small from-scratch spatial filters (Gaussian blur, box convolution).

Used wherever the MATLAB original called ``imgaussfilt``. Implemented as a
separable convolution with reflected borders so there is no dependency on
``scipy.ndimage`` / OpenCV.
"""

from __future__ import annotations

import numpy as np


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Return a normalised 1-D Gaussian kernel."""
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _convolve1d(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve along one axis with reflected (edge-symmetric) padding."""
    r = len(kernel) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(image, pad, mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for i, kv in enumerate(kernel):
        if axis == 0:
            out += kv * padded[i:i + image.shape[0], :]
        else:
            out += kv * padded[:, i:i + image.shape[1]]
    return out


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a 2-D image with an isotropic Gaussian (separable)."""
    img = np.asarray(image, dtype=np.float64)
    if sigma <= 0:
        return img.copy()
    k = gaussian_kernel1d(sigma)
    tmp = _convolve1d(img, k, axis=0)
    return _convolve1d(tmp, k, axis=1)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Full 2-D convolution with reflected borders (small kernels only)."""
    img = np.asarray(image, dtype=np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def unsharp(image: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: sharpen by adding back the high-frequency residual.

    ``out = img + amount * (img - gaussian_blur(img))`` -- a from-scratch
    anti-blur step for defocused frames (e.g. a card caught while the camera
    was refocusing).
    """
    img = np.asarray(image, dtype=np.float64)
    blurred = gaussian_blur(img, sigma)
    out = img + amount * (img - blurred)
    return np.clip(out, 0, 255)
