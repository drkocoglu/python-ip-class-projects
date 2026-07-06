"""Tests for skeleton centerline extraction and geometry."""

import numpy as np

from proj5_ip import worm_skeleton


def _horizontal_bar() -> np.ndarray:
    mask = np.zeros((60, 200), dtype=bool)
    mask[27:33, 20:180] = True  # thick horizontal bar
    return mask


def test_centerline_is_ordered_and_spans_bar() -> None:
    skeleton = worm_skeleton.skeletonize_worm(_horizontal_bar())
    centerline = worm_skeleton.extract_centerline(skeleton)
    assert len(centerline) > 100
    # ends should sit at opposite sides of the bar
    left_col, right_col = centerline[0][1], centerline[-1][1]
    assert abs(int(left_col) - int(right_col)) > 100
    # consecutive centerline pixels are 8-connected (no jumps)
    steps = np.abs(np.diff(centerline, axis=0)).max(axis=1)
    assert steps.max() <= 1


def test_equidistant_indices_count_and_bounds() -> None:
    indices = worm_skeleton.equidistant_indices(length=100, count=10)
    assert len(indices) == 10
    assert indices.min() >= 0
    assert indices.max() <= 99
    assert np.all(np.diff(indices) > 0)


def test_normal_is_unit_and_perpendicular() -> None:
    # Straight horizontal centerline -> tangent is horizontal, normal vertical.
    centerline = np.stack([np.full(50, 30), np.arange(50)], axis=1)
    normal = worm_skeleton.normal_at(centerline, index=25, half_window=3)
    assert np.isclose(np.hypot(*normal), 1.0)
    assert np.isclose(abs(normal[1]), 1.0, atol=1e-6)  # points in +/- y
    assert np.isclose(normal[0], 0.0, atol=1e-6)  # no x component


def test_sample_geometry_shapes_match() -> None:
    skeleton = worm_skeleton.skeletonize_worm(_horizontal_bar())
    centerline, points_xy, normals_xy = worm_skeleton.sample_geometry(skeleton)
    assert len(centerline) > 0
    assert points_xy.shape[0] == normals_xy.shape[0]
    assert points_xy.shape[1] == 2
    assert normals_xy.shape[1] == 2
