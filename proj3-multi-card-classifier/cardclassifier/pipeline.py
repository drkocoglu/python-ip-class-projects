"""End-to-end orchestration of the card-classification pipeline.

Ties every from-scratch stage together, following the report flowchart:

    input image
      -> binarize -> fill holes -> connected components (= "find & count cards")
      -> for each card:
           mask it out -> rotate upright -> crop & normalise
           -> extract ROI -> segment rank & suit
           -> rank: RankClassifier (BoVW + linear SVM)   [needs a trained model]
           -> suit: SuitClassifier (HOG thresholding)
      -> return per-card results

The single-card path is just the multi-card path with exactly one component, so
both share :func:`classify_image`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import (
    binarize,
    io_utils,
    morphology,
    regionprops,
    rotation,
    segmentation,
)
from .rank_classifier import RankClassifier


@dataclass
class CardResult:
    index: int
    rank: str | None
    suit: str | None
    usable: bool
    card_image: np.ndarray | None = None
    rank_crop: np.ndarray | None = None
    suit_crop: np.ndarray | None = None
    orientation: float | None = None
    mask: np.ndarray | None = None


@dataclass
class PipelineResult:
    n_cards: int
    cards: list[CardResult]


def _detect_cards(image: np.ndarray):
    """Binarize, fill holes, label components -> list of per-card masks."""
    bw = binarize.binarize(image)
    filled = morphology.fill_holes(bw)
    labels, n = morphology.connected_components(filled)
    # Drop tiny specks (noise) -- keep components at least 1% of the frame.
    min_area = 0.01 * image.size
    masks = []
    for m in morphology.component_masks(labels, n):
        if m.sum() >= min_area:
            masks.append(m)
    return masks


def classify_image(image: np.ndarray,
                   rank_clf: RankClassifier | None = None,
                   suit_clf=None,
                   keep_images: bool = True) -> PipelineResult:
    """Run the full pipeline on a grayscale frame containing 1..N cards."""

    masks = _detect_cards(image)
    results: list[CardResult] = []

    for idx, mask in enumerate(masks):
        single = image.copy()
        single[~mask] = 0

        lab, _ = morphology.connected_components(mask)
        props = regionprops.largest_region(lab)
        angle = 90.0 - props.orientation
        rotated = rotation.rotate(single, angle)

        card, usable = segmentation.crop_to_card(rotated)
        if card is None or not usable:
            results.append(CardResult(idx, None, None, False, orientation=props.orientation, mask=mask))
            continue

        segs = segmentation.choose_segments(card)

        rank = suit = None
        if segs.suit is not None and segs.suit.size:
            try:
                suit = suit_clf.predict(segs.suit)
            except Exception:
                suit = None
        if segs.rank is not None and segs.rank.size and rank_clf is not None:
            try:
                crop = segs.rank
                h_c, w_c = crop.shape[:2]
                # A landscape rank crop means the glyph came out sideways
                # (orientation quirk); try upright candidates and keep the
                # highest-confidence read when the classifier exposes margins.
                if w_c > 1.15 * h_c and hasattr(rank_clf, "predict_margin"):
                    cands = [crop, np.rot90(crop, 1), np.rot90(crop, 3)]
                    scored = [rank_clf.predict_margin(c) for c in cands]
                    rank = max(scored, key=lambda t: t[1])[0]
                else:
                    rank = rank_clf.predict(crop)
            except Exception:
                rank = None

        results.append(CardResult(
            index=idx, rank=rank, suit=suit, usable=segs.usable,
            card_image=card if keep_images else None,
            rank_crop=segs.rank if keep_images else None,
            suit_crop=segs.suit if keep_images else None,
            orientation=props.orientation, mask=mask,
        ))

    return PipelineResult(n_cards=len(masks), cards=results)


def classify_path(path: str | Path, **kwargs) -> PipelineResult:
    """Convenience wrapper: load an image file then :func:`classify_image`."""
    return classify_image(io_utils.load_gray(path), **kwargs)
