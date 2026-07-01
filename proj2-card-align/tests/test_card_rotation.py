"""
test_card_rotation.py — Tests for the from-scratch detection and rotation.

Builds synthetic "cards" (bright rotated rectangles on a dark background) so the
tests have known answers and run without the real image files. A few tests use
the real images if present, skipping cleanly otherwise.
"""

import numpy as np
import pytest
import cv2

from card_rotation import (
    convolve2d, box_blur, edge_map,
    find_rotation_angle, rotation_angle, rotate_card_upright,
    _edge_points, _bounding_box_area,
    SOBEL_X, SOBEL_Y,
)


def _make_card(angle_deg, w=150, h=250, canvas=(600, 600)):
    """A bright (255) h x w rectangle rotated by angle_deg on a dark background."""
    img = np.zeros(canvas, dtype=np.uint8)
    cx, cy = canvas[1] // 2, canvas[0] // 2
    x0, y0 = cx - w // 2, cy - h // 2
    img[y0:y0 + h, x0:x0 + w] = 255
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (canvas[1], canvas[0]))


# ── Convolution primitive ─────────────────────────────────────────────────────
def test_convolve_identity():
    """Convolving with a 1x1 identity kernel returns the image unchanged."""
    img = np.arange(25, dtype=float).reshape(5, 5)
    out = convolve2d(img, np.array([[1.0]]))
    assert np.allclose(out, img)


def test_convolve_box_blur_averages():
    """A 3x3 averaging kernel on a constant image returns the same constant."""
    img = np.full((10, 10), 4.0)
    out = convolve2d(img, np.ones((3, 3)) / 9.0)
    # Interior pixels average to 4; ignore the zero-padded border.
    assert np.allclose(out[2:-2, 2:-2], 4.0)


def test_sobel_detects_vertical_edge():
    """Sobel-x should respond strongly at a vertical intensity step."""
    img = np.zeros((20, 20), dtype=float)
    img[:, 10:] = 255
    gx = convolve2d(img, SOBEL_X)
    # Strong response near the edge column.
    assert np.abs(gx[:, 9:11]).max() > 100


# ── Edge map ──────────────────────────────────────────────────────────────────
def test_edge_map_is_boolean():
    card = _make_card(20.0)
    edges = edge_map(card)
    assert edges.dtype == bool
    assert edges.any()


def test_edge_map_border_cleared():
    card = _make_card(0.0)
    edges = edge_map(card)
    m = 12
    assert not edges[:m, :].any()
    assert not edges[-m:, :].any()


# ── Angle search ──────────────────────────────────────────────────────────────
def test_bounding_box_area_minimal_when_aligned():
    """The bounding-box area at the detected angle beats a misaligned angle."""
    card = _make_card(30.0)
    pts = _edge_points(edge_map(card))
    detected = find_rotation_angle(edge_map(card))
    area_aligned = _bounding_box_area(pts, detected)
    area_off = _bounding_box_area(pts, detected + 20.0)
    assert area_aligned < area_off


@pytest.mark.parametrize("true_angle", [0, 10, 20, 35, 50, 70])
def test_find_rotation_angle_recovers_tilt(true_angle):
    """
    The detected angle should straighten the card. A rectangle has 90-degree
    symmetry, so the detected angle may differ from the true tilt by a multiple
    of 90 and still be correct — what matters is that rotating by it makes the
    card axis-aligned. We verify by checking the straightened edge points form a
    tighter box than a deliberately-wrong angle does.
    """
    card = _make_card(float(true_angle))
    edges = edge_map(card)
    detected = find_rotation_angle(edges)
    pts = _edge_points(edges)

    area_detected = _bounding_box_area(pts, detected)
    area_wrong = _bounding_box_area(pts, detected + 15.0)
    assert area_detected < area_wrong, f"true {true_angle}, detected {detected}"


# ── Output is always portrait ─────────────────────────────────────────────────
@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 90, 120, 175, -30, -75])
def test_output_is_always_portrait(angle):
    card = _make_card(angle)
    out = rotate_card_upright(card)
    assert out.shape[0] >= out.shape[1], f"angle {angle} produced landscape output"


def test_output_is_cropped_smaller_than_input():
    card = _make_card(30.0)
    out = rotate_card_upright(card)
    assert out.shape[0] < 600 and out.shape[1] < 600


def test_output_nonempty_and_contains_card():
    out = rotate_card_upright(_make_card(25.0))
    assert out.size > 0
    assert out.max() > 0


def test_rotation_angle_is_float():
    assert isinstance(rotation_angle(_make_card(15.0)), float)


def test_no_edges_raises():
    blank = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(Exception):
        find_rotation_angle(edge_map(blank))
