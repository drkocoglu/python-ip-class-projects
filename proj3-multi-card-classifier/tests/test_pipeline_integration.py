"""Integration tests on the real sample images (skipped when data is absent)."""

import pytest

from cardclassifier import dataset, io_utils, pipeline


def _require(path):
    if not path.exists():
        pytest.skip(f"sample image not present: {path}")


def test_single_card_detected(single_card_path):
    _require(single_card_path)
    res = pipeline.classify_image(io_utils.load_gray(single_card_path))
    assert res.n_cards == 1


def test_multi_card_detects_expected(multi_card_path):
    _require(multi_card_path)
    res = pipeline.classify_image(io_utils.load_gray(multi_card_path))
    expected = len(dataset.parse_truth(multi_card_path))
    if expected:
        assert res.n_cards == expected
    else:
        assert res.n_cards >= 1
    for card in res.cards:
        assert card.card_image is not None
        h, w = card.card_image.shape[:2]
        assert 1.3 < h / w < 1.5


def test_pipeline_without_models(multi_card_path):
    _require(multi_card_path)
    res = pipeline.classify_image(io_utils.load_gray(multi_card_path))
    for card in res.cards:
        assert card.rank is None
        assert card.suit is None
