#!/usr/bin/env python3
"""Evaluate end-to-end accuracy on the labeled test cards. No arguments needed.

Ground truth is read from filenames such as ``Card_1_Diamond_Q.tif`` or
``Cards_1_Spade_K_Spade_J.tif``. Suit and rank are scored independently with
order-agnostic (multiset) matching. Per-card confusion matrices are built by
pairing each detection with the ground truth through the optimal assignment
and are saved to ``results/confusion_matrices.png`` (printed as text when
matplotlib is unavailable).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardclassifier import dataset, io_utils, pipeline  # noqa: E402
from cardclassifier.rank_classifier import RankClassifier  # noqa: E402
from cardclassifier.suit_classifier import ThresholdSuitClassifier  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _multiset_correct(pred: list, true: list) -> int:
    return sum((Counter(pred) & Counter(true)).values())


def _update_confusions(preds, truth, rank_cm, suit_cm) -> None:
    """Pair detections with ground truth (optimal assignment), count per card."""
    assignments = dataset.optimal_assignments(preds, truth)
    if not assignments:
        return
    for card_idx, truth_idx in enumerate(assignments[0]):
        pred_rank, pred_suit = preds[card_idx]
        true_rank, true_suit = truth[truth_idx]
        if pred_rank in dataset.RANKS and true_rank in dataset.RANKS:
            rank_cm[dataset.RANKS.index(true_rank),
                    dataset.RANKS.index(pred_rank)] += 1
        if pred_suit in dataset.SUITS and true_suit in dataset.SUITS:
            suit_cm[dataset.SUITS.index(true_suit),
                    dataset.SUITS.index(pred_suit)] += 1


def _save_matrix_figure(cm, labels, kind: str, out_path: Path) -> None:
    """One professional confusion-matrix heatmap: colorbar legend, gridlines,
    every cell annotated with its card count, accuracy in the title."""
    from matplotlib.ticker import MaxNLocator
    n = len(labels)
    correct, total = int(np.trace(cm)), int(cm.sum())
    fig, ax = plt.subplots(figsize=(9.5, 8) if n > 6 else (6.5, 5.5))
    image = ax.imshow(cm, cmap="Blues", vmin=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("number of cards")
    colorbar.locator = MaxNLocator(integer=True)
    colorbar.update_ticks()
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel(f"Predicted {kind}")
    ax.set_ylabel(f"True {kind}")
    ax.set_title(f"{kind.capitalize()} confusion matrix -- "
                 f"{correct}/{total} correct ({100 * correct / total:.1f}%)")
    threshold = cm.max() / 2 if cm.max() else 1
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if count:
                ax.text(j, i, str(count), ha="center", va="center",
                        fontweight="bold",
                        color="white" if count > threshold else "black")
            else:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=7, color="lightgray")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _print_matrix(cm, labels, title) -> None:
    print(f"\n{title} confusion matrix (rows = true, cols = predicted):")
    print("     " + " ".join(f"{lb:>3}" for lb in labels))
    for i, lb in enumerate(labels):
        print(f"{lb:>4} " + " ".join(f"{cm[i, j]:>3}" for j in range(len(labels))))


def main() -> int:
    dirs = dataset.image_dirs()
    if not dirs:
        print("Could not find CardImages / Multi_CardImages next to the project.")
        return 2
    rank = RankClassifier.load(ROOT / "models" / "rank.pkl")
    suit = ThresholdSuitClassifier.load(ROOT / "models" / "suit_thresholds.json")
    rank_cm = np.zeros((len(dataset.RANKS), len(dataset.RANKS)), dtype=int)
    suit_cm = np.zeros((len(dataset.SUITS), len(dataset.SUITS)), dtype=int)
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
            _update_confusions(preds, truth, rank_cm, suit_cm)
            suit_ok += sc
            rank_ok += rc
            total += len(truth)
            flag = "" if sc == len(truth) and rc == len(truth) else "  <-- errors"
            print(f"{f.name[:40]:42} suit {sc}/{len(truth)} rank {rc}/{len(truth)}{flag}")
    if not total:
        print("No labeled test files found. Ground truth comes from filenames, "
              "e.g. Card_1_Diamond_Q.tif or Cards_1_Spade_K_Spade_J.tif -- "
              "rename your images like that to measure accuracy.")
        return 1
    print(f"\nSUIT {suit_ok}/{total} = {suit_ok / total:.3f}"
          f"    RANK {rank_ok}/{total} = {rank_ok / total:.3f}")
    if plt is not None:
        results = ROOT / "results"
        _save_matrix_figure(rank_cm, dataset.RANKS, "rank",
                            results / "confusion_matrix_ranks.png")
        _save_matrix_figure(suit_cm, dataset.SUITS, "suit",
                            results / "confusion_matrix_suits.png")
        print(f"confusion matrices saved to {results}/confusion_matrix_ranks.png"
              f" and confusion_matrix_suits.png")
    else:
        _print_matrix(rank_cm, dataset.RANKS, "RANK")
        _print_matrix(suit_cm, dataset.SUITS, "SUIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
