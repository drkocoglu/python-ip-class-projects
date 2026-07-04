#!/usr/bin/env python3
"""Evaluate end-to-end accuracy on the labeled test cards. No arguments needed.

Ground truth is read from filenames such as ``Card_1_Diamond_Q.tif`` or
``Cards_1_Spade_K_Spade_J.tif``. Suit and rank are scored independently with
order-agnostic (multiset) matching because detection order is arbitrary.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardclassifier import dataset, io_utils, pipeline  # noqa: E402
from cardclassifier.rank_classifier import RankClassifier  # noqa: E402
from cardclassifier.suit_classifier import ThresholdSuitClassifier  # noqa: E402


def _multiset_correct(pred: list, true: list) -> int:
    return sum((Counter(pred) & Counter(true)).values())


def main() -> int:
    dirs = dataset.image_dirs()
    if not dirs:
        print("Could not find CardImages / Multi_CardImages next to the project.")
        return 2
    rank = RankClassifier.load(ROOT / "models" / "rank.pkl")
    suit = ThresholdSuitClassifier.load(ROOT / "models" / "suit_thresholds.json")
    suit_ok = rank_ok = total = 0
    for folder in dirs:
        for f in sorted(folder.glob("*.tif")):
            truth = dataset.parse_truth(f)
            if not truth:
                continue
            res = pipeline.classify_image(io_utils.load_gray(f),
                                          rank_clf=rank, suit_clf=suit)
            preds = [(c.rank, c.suit) for c in res.cards]
            sc = _multiset_correct([p[1] for p in preds], [t[1] for t in truth])
            rc = _multiset_correct([p[0] for p in preds], [t[0] for t in truth])
            suit_ok += sc
            rank_ok += rc
            total += len(truth)
            flag = "" if sc == len(truth) and rc == len(truth) else "  <-- errors"
            print(f"{f.name[:40]:42} suit {sc}/{len(truth)} rank {rc}/{len(truth)}{flag}")
    if total:
        print(f"\nSUIT {suit_ok}/{total} = {suit_ok / total:.3f}"
              f"    RANK {rank_ok}/{total} = {rank_ok / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
