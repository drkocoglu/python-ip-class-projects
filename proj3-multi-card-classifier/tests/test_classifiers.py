import numpy as np
import pytest

from cardclassifier.rank_classifier import RankClassifier, ink_normalize
from cardclassifier.suit_classifier import ThresholdSuitClassifier


def _glyph(kind: str, rng) -> np.ndarray:
    img = np.full((40, 30), 230, np.uint8)
    if kind == "bar":
        img[5:35, 13:17] = 20
    else:
        img[18:22, 5:25] = 20
    noise = rng.integers(0, 8, img.shape).astype(np.uint8)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def test_ink_normalize_canvas_and_polarity():
    rng = np.random.default_rng(0)
    out = ink_normalize(_glyph("bar", rng).astype(float), (40, 28))
    assert out.shape == (40, 28)
    assert out.max() > 200 and out.min() < 100  # glyph on white canvas


def test_rank_classifier_fit_predict_pickle(tmp_path):
    rng = np.random.default_rng(1)
    images = [_glyph("bar", rng) for _ in range(8)] + [_glyph("dash", rng) for _ in range(8)]
    labels = ["bar"] * 8 + ["dash"] * 8
    clf = RankClassifier(kernel="linear").fit(images, labels)
    probe = _glyph("bar", rng)
    assert clf.predict(probe) in {"bar", "dash"}
    label, margin = clf.predict_margin(probe)
    assert label in {"bar", "dash"} and np.isfinite(margin)
    path = tmp_path / "rank.pkl"
    clf.save(path)
    assert RankClassifier.load(path).predict(probe) == clf.predict(probe)


def test_threshold_suit_rules():
    rules = {"tree": {"stat": "top_bot", "thr": 0.11,
                      "gt": {"suit": "Hearts"},
                      "le": {"stat": "solidity", "thr": 0.8,
                             "le": {"suit": "Clubs"},
                             "gt": {"stat": "aspect", "thr": 1.15,
                                    "gt": {"suit": "Diamonds"},
                                    "le": {"suit": "Spades"}}}}}
    clf = ThresholdSuitClassifier(rules)
    img = np.full((33, 26), 235, np.uint8)  # tall solid diamond-like pip
    cy, cx = 16, 13
    for y in range(33):
        for x in range(26):
            if abs(x - cx) / 11 + abs(y - cy) / 15 < 1:
                img[y, x] = 20
    assert clf.predict(img) in {"Clubs", "Diamonds", "Hearts", "Spades"}


def test_suit_rules_missing_key_raises():
    clf = ThresholdSuitClassifier({"tree": {"stat": "nope", "thr": 0,
                                            "le": {"suit": "Clubs"},
                                            "gt": {"suit": "Spades"}}})
    with pytest.raises(KeyError):
        clf.predict(np.full((30, 30), 200, np.uint8))
