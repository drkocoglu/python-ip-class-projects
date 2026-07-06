"""Tests for the per-frame worm segmentation stage."""

import numpy as np

from proj5_ip import tracking_config as cfg
from proj5_ip import worm_segmentation


def _worm_frame(size: int = 200) -> np.ndarray:
    """Light frame with one big dark worm-like bar and a tiny dark speck."""
    gray = np.full((size, size), 240, dtype=np.uint8)
    gray[60:130, 40:160] = 10  # large dark object (the "worm")
    gray[5:9, 5:9] = 10  # small noise speck
    return gray


def test_segment_returns_largest_object() -> None:
    gray = _worm_frame()
    background = np.zeros_like(gray, dtype=bool)
    worm, area = worm_segmentation.segment_worm(gray, background)
    assert worm is not None
    assert area >= cfg.MIN_WORM_AREA
    assert worm[95, 100]  # centre of the big object is filled
    assert not worm[7, 7]  # tiny speck was dropped


def test_segment_rejects_small_object() -> None:
    gray = np.full((200, 200), 240, dtype=np.uint8)
    gray[10:20, 10:20] = 10  # only 100 px -> below MIN_WORM_AREA
    background = np.zeros_like(gray, dtype=bool)
    worm, area = worm_segmentation.segment_worm(gray, background)
    assert worm is None
    assert area < cfg.MIN_WORM_AREA


def test_bounding_box_covers_object() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:70, 30:80] = True
    x_min, y_min, x_max, y_max = worm_segmentation.bounding_box(mask)
    assert (x_min, y_min) == (30, 20)
    assert (x_max, y_max) == (79, 69)
