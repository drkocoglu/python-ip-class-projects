import numpy as np

from cardclassifier import edges, enhance, filters, hog


def test_gaussian_blur_preserves_mean_and_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (30, 30)).astype(float)
    b = filters.gaussian_blur(img, 2.0)
    assert b.shape == img.shape
    # Blurring conserves overall brightness to within a small tolerance.
    assert abs(b.mean() - img.mean()) < 1.0


def test_gaussian_blur_zero_sigma_noop():
    img = np.ones((5, 5)) * 3.0
    assert np.allclose(filters.gaussian_blur(img, 0.0), img)


def test_sobel_detects_vertical_edge():
    img = np.zeros((20, 20), dtype=float)
    img[:, 10:] = 255.0
    e = edges.edge_sobel(img)
    # Edge pixels concentrate near the column-10 boundary.
    cols = np.where(e.any(axis=0))[0]
    assert cols.size > 0
    assert 8 <= cols.mean() <= 12


def test_edge_map_is_boolean():
    img = np.random.default_rng(1).integers(0, 255, (16, 16)).astype(np.uint8)
    e = edges.edge_sobel(img)
    assert e.dtype == bool


def test_adapthisteq_range_and_shape():
    rng = np.random.default_rng(2)
    img = rng.integers(50, 120, (32, 32)).astype(np.uint8)  # low-contrast
    out = enhance.adapthisteq(img, n_tiles=(2, 2))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    # Contrast should not collapse.
    assert out.std() >= img.std() - 1


def test_hog_feature_vector_nonempty_and_finite():
    img = np.zeros((24, 24), dtype=float)
    img[:, 12:] = 255.0
    feat = hog.hog_features(img, cell_size=(6, 6), block_size=(2, 2), n_bins=9)
    assert feat.ndim == 1 and feat.size > 0
    assert np.all(np.isfinite(feat))


def test_hog_too_small_returns_empty():
    feat = hog.hog_features(np.zeros((3, 3)), cell_size=(6, 6))
    assert feat.size == 0
