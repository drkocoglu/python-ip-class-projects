"""Day/night classification for camera-trap images via HSV colour statistics.

The decision rule is ported directly from the original MATLAB assignment:

1. Convert the RGB frame to HSV and scale every channel to the ``0-255`` range.
2. Add the Hue and Saturation channels and take their mean over all pixels.
3. Label the frame ``NIGHT`` when that mean falls below a threshold, else ``DAY``.

Night frames come from a monochrome infra-red sensor, so they carry almost no
hue or saturation and their score sits near zero. Daylight frames are colourful
and score well above the threshold, which makes a single scalar enough to
separate the two regimes without any training data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Class labels.
DAY = "DAY"
NIGHT = "NIGHT"

# Decision threshold carried over from the MATLAB assignment.
DEFAULT_THRESHOLD = 5.0


def _to_matlab_uint8(array: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Cast a non-negative float array to ``uint8`` the way MATLAB does.

    MATLAB rounds halves away from zero and saturates to ``[0, 255]``, whereas
    NumPy's ``astype(np.uint8)`` truncates and wraps around. The inputs here are
    HSV channels scaled to ``[0, 255]`` (always non-negative), so adding 0.5 and
    flooring reproduces MATLAB's rounding before the values are clipped.
    """
    rounded = np.floor(np.asarray(array, dtype=np.float64) + 0.5)
    return np.clip(rounded, 0, 255).astype(np.uint8)


def _rgb_to_hsv(rgb: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert a float RGB array in ``[0, 1]`` to HSV in ``[0, 1]``.

    A small vectorised implementation of the standard HSV definition (the same
    formula used by ``colorsys`` and ``matplotlib.colors.rgb_to_hsv``). It is
    written in plain NumPy so the core classifier has no dependency on a plotting
    library: only the optional figure helpers in ``daynight_viz`` need matplotlib.

    Hue, Saturation, and Value are each returned in ``[0, 1]``; achromatic pixels
    (where the channels are equal) get hue and saturation of 0.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc

    value = maxc
    saturation = np.zeros_like(maxc)
    coloured = maxc > 0
    saturation[coloured] = delta[coloured] / maxc[coloured]

    # Avoid dividing by zero on achromatic pixels; their hue is fixed to 0 below.
    safe_delta = np.where(delta == 0, 1.0, delta)
    rc = (maxc - red) / safe_delta
    gc = (maxc - green) / safe_delta
    bc = (maxc - blue) / safe_delta
    hue = np.zeros_like(maxc)
    hue = np.where(maxc == red, bc - gc, hue)
    hue = np.where(maxc == green, 2.0 + rc - bc, hue)
    hue = np.where(maxc == blue, 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    hue[delta == 0] = 0.0

    return np.stack((hue, saturation, value), axis=-1)


def rgb_to_hsv_uint8(image: NDArray) -> NDArray[np.uint8]:
    """Convert an 8-bit RGB image to HSV scaled to ``[0, 255]``.

    Mirrors ``uint8(255 .* rgb2hsv(image))`` from MATLAB. Grayscale (2-D) inputs
    are promoted to three channels and any alpha channel is dropped.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack((array, array, array), axis=-1)
    rgb = array[..., :3].astype(np.float64) / 255.0
    hsv = _rgb_to_hsv(rgb)  # H, S, V each in [0, 1]
    return _to_matlab_uint8(255.0 * hsv)


def hue_saturation_score(image: NDArray) -> float:
    """Return the mean of (Hue + Saturation) for an RGB image.

    This scalar is the day/night decision statistic. Hue and Saturation are
    added with ``uint8`` saturation (capped at 255) to match MATLAB's integer
    arithmetic before the mean is taken.
    """
    hsv = rgb_to_hsv_uint8(image)
    hue = hsv[..., 0].astype(np.uint16)
    saturation = hsv[..., 1].astype(np.uint16)
    combined = np.minimum(hue + saturation, 255)
    return float(combined.mean())


@dataclass
class DayNightClassifier:
    """Label an image ``DAY`` or ``NIGHT`` from its HSV colour statistics.

    Parameters
    ----------
    threshold:
        Frames whose Hue+Saturation mean is below this value are ``NIGHT``;
        the rest are ``DAY``.

    Examples
    --------
    >>> import numpy as np
    >>> clf = DayNightClassifier()
    >>> clf.predict(np.zeros((4, 4, 3), dtype=np.uint8))  # an all-black frame
    'NIGHT'
    """

    threshold: float = DEFAULT_THRESHOLD

    def score(self, image: NDArray) -> float:
        """Mean Hue+Saturation statistic used to make the decision."""
        return hue_saturation_score(image)

    def label_for_score(self, score: float) -> str:
        """Map a precomputed score to a ``DAY``/``NIGHT`` label."""
        return NIGHT if score < self.threshold else DAY

    def predict(self, image: NDArray) -> str:
        """Label a single RGB image as ``DAY`` or ``NIGHT``."""
        return self.label_for_score(self.score(image))

    def predict_batch(self, images: Iterable[NDArray]) -> list[str]:
        """Label each image in an iterable, preserving order."""
        return [self.predict(image) for image in images]
