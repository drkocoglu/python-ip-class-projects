"""
test_card_data_loader.py — Tests for the image loader.

Uses the real images if present (skips cleanly if not), plus a synthetic-folder
test for the upward search.
"""

import os
import numpy as np
import pytest
import cv2

import card_data_loader as cdl


@pytest.fixture
def real_images():
    try:
        return cdl.load_all_grayscale()
    except FileNotFoundError:
        pytest.skip("Test images not found in data/proj2_data")


def test_finds_six_images(real_images):
    assert len(real_images) == 6


def test_images_are_2d_uint8(real_images):
    _, gray = real_images[0]
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_find_data_dir_walks_upward(tmp_path):
    # Build data/proj2_data two levels above a fake script dir.
    data = tmp_path / "data" / "proj2_data"
    data.mkdir(parents=True)
    cv2.imwrite(str(data / "Testimage1.tif"), np.zeros((10, 10), np.uint8))
    deep = tmp_path / "proj2" / "nested"
    deep.mkdir(parents=True)
    found = cdl.find_data_dir(start_dir=str(deep))
    assert os.path.isdir(found)


def test_missing_data_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cdl.find_data_dir(start_dir=str(tmp_path))
