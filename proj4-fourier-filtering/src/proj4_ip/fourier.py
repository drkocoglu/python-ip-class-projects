"""Shared Fourier building blocks: FFT, frequency grid, spectral threshold.

The frequency grid places DC (zero frequency) at index (0, 0), matching
:func:`numpy.fft.fft2`, exactly like the MATLAB code::

    u = 0:(M-1);  u(u > M/2) = u(u > M/2) - M;
    v = 0:(N-1);  v(v > N/2) = v(v > N/2) - N;
    [V, U] = meshgrid(v, u);  D = sqrt(U.^2 + V.^2);
"""

from __future__ import annotations

import numpy as np


def fft2(image: np.ndarray) -> np.ndarray:
    """Return the unshifted 2-D FFT of ``image`` (DC at index [0, 0])."""
    return np.fft.fft2(image)


def ifft2_real(spectrum: np.ndarray) -> np.ndarray:
    """Inverse 2-D FFT, keeping the real part (MATLAB: ``real(ifft2(...))``)."""
    return np.real(np.fft.ifft2(spectrum))


def log_magnitude(spectrum: np.ndarray, shift: bool = True) -> np.ndarray:
    """Return ``log(1 + |F|)`` for display, centred by default."""
    data = np.fft.fftshift(spectrum) if shift else spectrum
    return np.log1p(np.abs(data))


def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Euclidean distance (in cycles) from DC on a wrapped frequency grid."""
    rows, cols = shape
    u = np.arange(rows)
    v = np.arange(cols)
    u[u > rows / 2] -= rows
    v[v > cols / 2] -= cols
    grid_v, grid_u = np.meshgrid(v, u)  # MATLAB [V, U] = meshgrid(v, u)
    return np.sqrt(grid_u.astype(np.float64) ** 2 + grid_v.astype(np.float64) ** 2)


def spectral_threshold_mask(spectrum: np.ndarray, log_threshold: float) -> np.ndarray:
    """Boolean mask of components to *keep* (``log(1 + |F|) >= threshold``)."""
    return np.log1p(np.abs(spectrum)) >= log_threshold


def apply_threshold(
    spectrum: np.ndarray, log_threshold: float, attenuation: float = 0.01
) -> np.ndarray:
    """Keep strong spectral components; scale the weak ones by ``attenuation``.

    ``attenuation = 0.01`` matches the MATLAB script; use 0.0 to delete them.
    """
    keep = spectral_threshold_mask(spectrum, log_threshold)
    filtered = spectrum.copy()
    filtered[~keep] *= attenuation
    return filtered
