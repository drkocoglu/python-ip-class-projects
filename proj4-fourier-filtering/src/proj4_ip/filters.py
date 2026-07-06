"""Butterworth filters: ``low_pass``, ``high_pass`` and ``band_pass``.

low_pass
    Keeps the LOW spatial frequencies (near DC) and drops the high ones - the
    smooth brightness/shading, not the sharp detail/noise.
        H = 1 / (1 + (D / cutoff) ** (2 * order))

high_pass
    The opposite: BLOCKS the low frequencies and keeps the high ones. Because
    non-uniform illumination is a slow gradient near DC, a high-pass removes it
    while keeping texture.
        H = 1 / (1 + (cutoff / D) ** (2 * order))

band_pass
    Keeps only a RING of frequencies between an inner and outer cut-off; it is
    a high-pass times a low-pass:
        H = high_pass(inner_cutoff) * low_pass(outer_cutoff)
    Used to extract the pattern: the pattern's spikes sit in a mid-frequency
    ring, and the inner cut removes the illumination that a low-pass alone would
    leave behind as a bright central blob.
"""

from __future__ import annotations

import numpy as np

from .fourier import distance_grid


def low_pass(shape: tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth low-pass transfer function (DC at index [0, 0])."""
    dist = distance_grid(shape)
    return 1.0 / (1.0 + (dist / cutoff) ** (2 * order))


def high_pass(shape: tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth high-pass transfer function (DC at index [0, 0]).

    The DC distance is nudged off zero to avoid a divide warning, then DC gain
    is forced to exactly 0, as a high-pass intends.
    """
    dist = distance_grid(shape)
    safe = np.where(dist == 0.0, 1e-12, dist)
    transfer = 1.0 / (1.0 + (cutoff / safe) ** (2 * order))
    transfer[dist == 0.0] = 0.0
    return transfer


def band_pass(
    shape: tuple[int, int],
    inner_cutoff: float,
    outer_cutoff: float,
    order: int = 2,
) -> np.ndarray:
    """Butterworth band-pass = ``high_pass(inner) * low_pass(outer)``.

    Frequencies below ``inner_cutoff`` (illumination) and above ``outer_cutoff``
    (noise) are removed; only the ring between them survives.
    """
    return high_pass(shape, inner_cutoff, order) * low_pass(shape, outer_cutoff, order)
