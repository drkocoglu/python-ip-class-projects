"""Unit tests for the day/night classifier core."""

from __future__ import annotations

import numpy as np

from daynight_classifier import (
    DAY,
    DEFAULT_THRESHOLD,
    NIGHT,
    DayNightClassifier,
    _rgb_to_hsv,
    _to_matlab_uint8,
    hue_saturation_score,
    rgb_to_hsv_uint8,
)


def _solid(color, size=8):
    """Return a *size* x *size* RGB image filled with *color*."""
    return np.tile(np.array(color, dtype=np.uint8), (size, size, 1))


class TestRgbToHsv:
    def test_primary_colours(self):
        rgb = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        hsv = _rgb_to_hsv(rgb)
        # Hue: red=0, green=1/3, blue=2/3; full saturation and value throughout.
        np.testing.assert_allclose(hsv[:, 0], [0.0, 1 / 3, 2 / 3])
        np.testing.assert_allclose(hsv[:, 1], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(hsv[:, 2], [1.0, 1.0, 1.0])

    def test_achromatic_pixels_have_zero_hue_and_saturation(self):
        rgb = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        hsv = _rgb_to_hsv(rgb)
        np.testing.assert_array_equal(hsv[:, 0], [0.0, 0.0, 0.0])  # hue
        np.testing.assert_array_equal(hsv[:, 1], [0.0, 0.0, 0.0])  # saturation
        np.testing.assert_allclose(hsv[:, 2], [0.0, 0.5, 1.0])  # value == max channel

    def test_preserves_image_shape(self):
        hsv = _rgb_to_hsv(np.zeros((5, 7, 3)))
        assert hsv.shape == (5, 7, 3)


class TestMatlabUint8:
    def test_rounds_half_away_from_zero(self):
        result = _to_matlab_uint8(np.array([0.5, 1.5, 2.4, 2.6]))
        assert result.tolist() == [1, 2, 2, 3]

    def test_saturates_high_and_low(self):
        result = _to_matlab_uint8(np.array([-10.0, 300.0, 255.0]))
        assert result.tolist() == [0, 255, 255]

    def test_dtype_is_uint8(self):
        assert _to_matlab_uint8(np.array([1.0])).dtype == np.uint8


class TestRgbToHsvUint8:
    def test_shape_and_dtype(self):
        hsv = rgb_to_hsv_uint8(_solid((10, 20, 30)))
        assert hsv.shape == (8, 8, 3)
        assert hsv.dtype == np.uint8

    def test_pure_red_is_fully_saturated(self):
        hsv = rgb_to_hsv_uint8(_solid((255, 0, 0)))
        # Hue of pure red is 0; saturation and value are maximal.
        assert hsv[..., 0].max() == 0
        assert hsv[..., 1].min() == 255
        assert hsv[..., 2].min() == 255

    def test_grayscale_input_is_promoted(self):
        gray = np.full((8, 8), 128, dtype=np.uint8)
        hsv = rgb_to_hsv_uint8(gray)
        assert hsv.shape == (8, 8, 3)
        assert hsv[..., 1].max() == 0  # achromatic pixels have no saturation

    def test_alpha_channel_is_ignored(self):
        rgba = np.dstack([_solid((255, 0, 0)), np.full((8, 8), 123, np.uint8)])
        hsv = rgb_to_hsv_uint8(rgba)
        assert hsv.shape == (8, 8, 3)


class TestHueSaturationScore:
    def test_black_scores_zero(self):
        assert hue_saturation_score(_solid((0, 0, 0))) == 0.0

    def test_gray_scores_zero(self):
        # Achromatic pixels carry no hue or saturation.
        assert hue_saturation_score(_solid((128, 128, 128))) == 0.0

    def test_saturated_colour_scores_high(self):
        assert hue_saturation_score(_solid((255, 0, 0))) > DEFAULT_THRESHOLD


class TestDayNightClassifier:
    def test_night_for_dark_frame(self):
        assert DayNightClassifier().predict(_solid((0, 0, 0))) == NIGHT

    def test_day_for_colourful_frame(self):
        assert DayNightClassifier().predict(_solid((200, 120, 40))) == DAY

    def test_threshold_separates_scores(self):
        clf = DayNightClassifier(threshold=10.0)
        assert clf.predict(_solid((0, 0, 0))) == NIGHT
        assert clf.predict(_solid((255, 0, 0))) == DAY

    def test_label_for_score_boundary(self):
        clf = DayNightClassifier(threshold=5.0)
        assert clf.label_for_score(4.999) == NIGHT
        assert clf.label_for_score(5.0) == DAY  # boundary is inclusive on the DAY side

    def test_predict_batch_preserves_order(self):
        clf = DayNightClassifier()
        images = [_solid((0, 0, 0)), _solid((200, 120, 40)), _solid((0, 0, 0))]
        assert clf.predict_batch(images) == [NIGHT, DAY, NIGHT]
