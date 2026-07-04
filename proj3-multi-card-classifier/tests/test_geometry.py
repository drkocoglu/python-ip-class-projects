import numpy as np

from cardclassifier import morphology, regionprops, rotation


def test_regionprops_area_and_bbox():
    m = np.zeros((30, 30), dtype=bool)
    m[5:15, 8:20] = True
    labels, n = morphology.connected_components(m)
    props = regionprops.regionprops(labels, n)
    assert len(props) == 1
    p = props[0]
    assert p.area == 10 * 12
    x, y, w, h = p.bounding_box
    assert (x, y, w, h) == (8, 5, 12, 10)


def test_orientation_horizontal_bar_is_zero():
    # A wide, short bar has its major axis along x -> orientation ~ 0 deg.
    m = np.zeros((40, 40), dtype=bool)
    m[19:21, 5:35] = True
    labels, n = morphology.connected_components(m)
    p = regionprops.largest_region(labels, n)
    assert abs(p.orientation) < 5


def test_orientation_vertical_bar_is_90():
    m = np.zeros((40, 40), dtype=bool)
    m[5:35, 19:21] = True
    labels, n = morphology.connected_components(m)
    p = regionprops.largest_region(labels, n)
    assert abs(abs(p.orientation) - 90) < 5


def test_rotate_shape_preserved_and_reversible():
    img = np.zeros((60, 60), dtype=np.uint8)
    img[20:40, 25:35] = 255
    r = rotation.rotate(img, 90.0)
    assert r.shape == img.shape
    # Rotating by +90 then -90 returns close to the original content mass.
    back = rotation.rotate(r, -90.0)
    assert abs(int(back.sum()) - int(img.sum())) < 0.1 * img.sum()


def test_rotate_zero_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (20, 20)).astype(np.uint8)
    r = rotation.rotate(img, 0.0)
    assert np.array_equal(r, img)
