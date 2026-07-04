import numpy as np

from cardclassifier import binarize


def test_otsu_separates_bimodal():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, :10] = 30
    img[:, 10:] = 220
    thr = binarize.otsu_threshold(img)
    # Any level in [30, 220) separates the two intensities.
    assert 30 <= thr < 220


def test_binarize_otsu_mask(solid_square):
    mask = binarize.binarize(solid_square)
    assert mask.dtype == bool
    # The bright square is foreground, the dark frame is not.
    assert mask[50, 50] and not mask[0, 0]
    assert mask.sum() == 40 * 40


def test_binarize_fixed_level():
    img = (np.arange(256, dtype=np.uint8)).reshape(16, 16)
    mask = binarize.binarize(img, level=0.5)  # ~127.5
    assert mask.sum() == np.count_nonzero(img.astype(float) > 127.5)
