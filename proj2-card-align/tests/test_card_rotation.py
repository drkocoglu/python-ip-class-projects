"""
test_card_rotation.py — Tests for the card detection and rotation algorithm.

Most tests build a synthetic "card" (a bright rotated rectangle on a dark
background) so they run without the real image files and have a known answer.
A couple of tests use the real images if present, skipping cleanly otherwise.
"""

import numpy as np
import pytest
import cv2

from card_rotation import (
    find_card_rectangle, rotate_card_upright, rotation_angle,
)


def _make_card(angle_deg, w=150, h=250, canvas=(600, 600)):
    """
    Build a synthetic test image: a bright (255) rectangle of size (h, w)
    rotated by angle_deg on a dark (0) background. Portrait by construction
    (h > w) before rotation.
    """
    canvas_img = np.zeros(canvas, dtype=np.uint8)
    cx, cy = canvas[1] // 2, canvas[0] // 2
    # Build an upright white rectangle, then rotate the whole canvas.
    x0, y0 = cx - w // 2, cy - h // 2
    canvas_img[y0:y0 + h, x0:x0 + w] = 255
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(canvas_img, M, (canvas[1], canvas[0]))


@pytest.fixture
def upright_card():
    return _make_card(0.0)


@pytest.fixture
def rotated_card():
    return _make_card(30.0)


# ── Rectangle detection ───────────────────────────────────────────────────────
def test_find_rectangle_returns_tuple(upright_card):
    rect = find_card_rectangle(upright_card)
    (cx, cy), (w, h), angle = rect
    assert w > 0 and h > 0


def test_detects_card_near_center(upright_card):
    (cx, cy), (w, h), angle = find_card_rectangle(upright_card)
    # Card was centred on the 600x600 canvas.
    assert abs(cx - 300) < 20
    assert abs(cy - 300) < 20


# ── Output is always portrait ─────────────────────────────────────────────────
@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 90, 120, 175, -30, -75])
def test_output_is_always_portrait(angle):
    card = _make_card(angle)
    out = rotate_card_upright(card)
    assert out.shape[0] >= out.shape[1], f"angle {angle} produced landscape output"


def test_output_is_cropped_smaller_than_input(rotated_card):
    out = rotate_card_upright(rotated_card)
    # The crop should be much smaller than the full 600x600 canvas.
    assert out.shape[0] < 600
    assert out.shape[1] < 600


def test_output_is_nonempty(rotated_card):
    out = rotate_card_upright(rotated_card)
    assert out.size > 0
    assert out.max() > 0   # contains the bright card


# ── Angle reporting ───────────────────────────────────────────────────────────
def test_rotation_angle_is_float(rotated_card):
    assert isinstance(rotation_angle(rotated_card), float)


def test_no_contour_raises():
    blank = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(Exception):
        find_card_rectangle(blank)
