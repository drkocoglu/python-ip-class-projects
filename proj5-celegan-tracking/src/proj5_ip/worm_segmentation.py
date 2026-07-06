"""Segment the worm from a single frame.

Pipeline for one frame: inverse-binarize, subtract the static background,
keep the largest 8-connected component, then close and fill it into a solid
worm mask. Frames whose largest object is smaller than ``MIN_WORM_AREA`` are
reported as "no worm" so the caller can pass the raw frame through.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi

from proj5_ip import tracking_config as cfg
from proj5_ip.background_model import binarize_inverse

_CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


def _disk(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, num = ndi.label(mask, structure=_CONNECTIVITY_8)
    if num == 0:
        return np.zeros_like(mask, dtype=bool), 0
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background label
    best = int(counts.argmax())
    return labels == best, int(counts[best])


def segment_worm(
    gray: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray | None, int]:
    """Return ``(worm_mask, area)`` or ``(None, area)`` if no worm is present."""
    foreground = binarize_inverse(gray) & ~background
    worm, area = _largest_component(foreground)

    if area < cfg.MIN_WORM_AREA:
        return None, area

    if area >= cfg.LARGE_WORM_AREA:
        opened = cv2.morphologyEx(
            foreground.astype(np.uint8), cv2.MORPH_OPEN, _disk(cfg.OPEN_DISK_RADIUS)
        ).astype(bool)
        worm, area = _largest_component(opened)
        if area < cfg.MIN_WORM_AREA:
            return None, area

    closed = cv2.morphologyEx(
        worm.astype(np.uint8), cv2.MORPH_CLOSE, _disk(cfg.CLOSE_DISK_RADIUS)
    ).astype(bool)
    filled = ndi.binary_fill_holes(closed)
    return filled, int(filled.sum())


def bounding_box(worm: np.ndarray) -> tuple[int, int, int, int]:
    """Return the ``(x_min, y_min, x_max, y_max)`` extent of the worm mask."""
    ys, xs = np.where(worm)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
