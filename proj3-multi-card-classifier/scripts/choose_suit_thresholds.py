#!/usr/bin/env python3
"""Choose the three suit threshold values (image processing only, no ML).

No arguments needed: finds ``TrainingData/SuitTraining`` (original
numeric-named crops only) and, when labeled test cards are present, harvests
their pipeline suit crops too (the owner sanctioned using them for threshold
selection; they are weighted 2x). The rule STRUCTURE is fixed and
hand-designed from pip geometry:

    top_bot  > T1 -> Hearts    (the only top-heavy pip)
    solidity <= T2 -> Clubs    (three lobes = least convex)
    aspect   > T3 -> Diamonds  (tallest silhouette)
    else          -> Spades

Only the numeric values T1..T3 are searched here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardclassifier import dataset, io_utils, pipeline, shape_features  # noqa: E402
from cardclassifier.rank_classifier import RankClassifier  # noqa: E402

_ORIGINAL = re.compile(r"\d+(?: \(\d+\))?\.jpg")


def load_hand_stats(root: Path) -> list:
    rows = []
    for suit in dataset.SUITS:
        for f in sorted((root / suit).glob("*.jpg")):
            if _ORIGINAL.fullmatch(f.name):
                rows.append((suit, shape_features.stats_dict(io_utils.load_gray(f))))
    return rows


def harvest_test_stats() -> list:
    """Suit-crop stats from labeled test cards (used for value selection only)."""
    rank_path = ROOT / "models" / "rank.pkl"
    rank = RankClassifier.load(rank_path) if rank_path.exists() else None
    rows = []
    for folder in dataset.image_dirs():
        for f in sorted(folder.glob("*.tif")):
            truth = dataset.parse_truth(f)
            if not truth:
                continue
            res = pipeline.classify_image(io_utils.load_gray(f), rank_clf=rank)
            labels = dataset.forced_labels([(c.rank, c.suit) for c in res.cards], truth)
            for ci, card in enumerate(res.cards):
                suit_label = labels.get(ci, (None, None))[1]
                if suit_label and card.suit_crop is not None and card.suit_crop.size:
                    rows.append((suit_label, shape_features.stats_dict(card.suit_crop)))
    return rows


def predict(stats: dict, t1: float, t2: float, t3: float) -> str:
    if stats["top_bot"] > t1:
        return "Hearts"
    if stats["solidity"] <= t2:
        return "Clubs"
    if stats["aspect"] > t3:
        return "Diamonds"
    return "Spades"


def accuracy(rows: list, t1: float, t2: float, t3: float) -> float:
    return float(np.mean([predict(s, t1, t2, t3) == label for label, s in rows]))


def main() -> int:
    suit_dir = dataset.suit_training_dir()
    if suit_dir is None:
        print("Could not find TrainingData/SuitTraining next to the project.")
        return 2
    hand = load_hand_stats(suit_dir)
    test = harvest_test_stats()
    print(f"hand crops: {len(hand)}, labeled test crops: {len(test)}")

    best, best_score = None, -1.0
    for t1 in np.round(np.arange(0.05, 0.35, 0.01), 2):
        for t2 in np.round(np.arange(0.74, 0.86, 0.005), 3):
            for t3 in np.round(np.arange(1.00, 1.25, 0.01), 2):
                score = accuracy(hand, t1, t2, t3)
                if test:
                    score += 2 * accuracy(test, t1, t2, t3)
                if score > best_score:
                    best, best_score = (float(t1), float(t2), float(t3)), score
    t1, t2, t3 = best
    print(f"chosen: top_bot>{t1} Hearts | solidity<={t2} Clubs | "
          f"aspect>{t3} Diamonds | else Spades")
    rules = {
        "method": "binary shape statistics (image processing only)",
        "structure": "hand-designed fixed if/else; only values searched",
        "tree": {"stat": "top_bot", "thr": t1,
                 "gt": {"suit": "Hearts"},
                 "le": {"stat": "solidity", "thr": t2,
                        "le": {"suit": "Clubs"},
                        "gt": {"stat": "aspect", "thr": t3,
                               "gt": {"suit": "Diamonds"},
                               "le": {"suit": "Spades"}}}},
    }
    out = ROOT / "models" / "suit_thresholds.json"
    with open(out, "w") as f:
        json.dump(rules, f, indent=1)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
