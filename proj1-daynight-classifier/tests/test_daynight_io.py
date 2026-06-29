"""Tests for the image discovery and loading helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from daynight_io import find_images, load_image


def _save(path, color=(120, 60, 30), size=8):
    Image.fromarray(np.tile(np.array(color, np.uint8), (size, size, 1))).save(path)


class TestFindImages:
    def test_finds_jpgs_sorted(self, tmp_path):
        for name in ("b.jpg", "a.jpg", "c.jpeg"):
            _save(tmp_path / name)
        paths = find_images(tmp_path)
        assert [p.name for p in paths] == ["a.jpg", "b.jpg", "c.jpeg"]

    def test_matching_is_case_insensitive(self, tmp_path):
        _save(tmp_path / "UPPER.JPG")
        assert [p.name for p in find_images(tmp_path)] == ["UPPER.JPG"]

    def test_ignores_non_image_files(self, tmp_path):
        _save(tmp_path / "frame.jpg")
        (tmp_path / "notes.txt").write_text("not an image")
        assert [p.name for p in find_images(tmp_path)] == ["frame.jpg"]

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_images(tmp_path / "missing")


class TestLoadImage:
    def test_returns_rgb_uint8_array(self, tmp_path):
        path = tmp_path / "frame.jpg"
        _save(path, size=12)
        image = load_image(path)
        assert image.shape == (12, 12, 3)
        assert image.dtype == np.uint8

    def test_grayscale_is_converted_to_rgb(self, tmp_path):
        path = tmp_path / "gray.png"
        Image.fromarray(np.full((6, 6), 90, np.uint8), mode="L").save(path)
        image = load_image(path)
        assert image.shape == (6, 6, 3)
