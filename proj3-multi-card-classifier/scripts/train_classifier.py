#!/usr/bin/env python3
"""Train the rank model (HOG features -> linear SVM). No arguments needed.

Finds ``TrainingData/RankTraining`` automatically (data folder outside the
project). The labeled test cards are never used. Linear and RBF kernels were
compared during development; linear won end-to-end (26/28 vs 25/28), so
linear is what this trains and ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardclassifier import dataset, training  # noqa: E402
from cardclassifier.rank_classifier import RankClassifier  # noqa: E402


def main() -> int:
    rank_dir = dataset.rank_training_dir()
    if rank_dir is None:
        print("Could not find TrainingData/RankTraining next to the project "
              "(expected e.g. ../data/proj3_data/TrainingData/RankTraining).")
        return 2
    print(f"training data: {rank_dir}")
    images, labels = training.load_labeled(rank_dir, dataset.RANKS)
    print(f"loaded {len(images)} unique images across {len(set(labels))} classes")
    aug_images, aug_labels = training.build_training_set(images, list(labels))
    print(f"training set (augmented + defocus copies): {len(aug_images)}")
    model = RankClassifier(kernel="linear").fit(aug_images, aug_labels)
    out = ROOT / "models" / "rank.pkl"
    model.save(out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
