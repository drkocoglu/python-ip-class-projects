import numpy as np

from cardclassifier import morphology


def test_fill_holes(square_with_hole):
    filled = morphology.fill_holes(square_with_hole)
    # The interior hole is now filled ...
    assert filled[25, 25]
    # ... and the solid square is a single 30x30 block.
    assert filled[10:40, 10:40].all()
    # Background outside the square is untouched.
    assert not filled[0, 0]


def test_fill_holes_no_hole_is_noop():
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True
    assert np.array_equal(morphology.fill_holes(m), m)


def test_connected_components_counts_two_blobs():
    m = np.zeros((20, 40), dtype=bool)
    m[5:15, 5:15] = True
    m[5:15, 25:35] = True
    labels, n = morphology.connected_components(m)
    assert n == 2
    assert labels.max() == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_connectivity_diagonal():
    m = np.zeros((5, 5), dtype=bool)
    m[1, 1] = True
    m[2, 2] = True  # touches only diagonally
    _, n8 = morphology.connected_components(m, connectivity=8)
    _, n4 = morphology.connected_components(m, connectivity=4)
    assert n8 == 1
    assert n4 == 2


def test_largest_component():
    m = np.zeros((20, 40), dtype=bool)
    m[2:5, 2:5] = True            # small
    m[5:18, 20:38] = True         # large
    big = morphology.largest_component(m)
    assert big.sum() == 13 * 18
