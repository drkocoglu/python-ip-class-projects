"""Dataset locations and filename ground truth for the class assignment.

The data lives outside this project, e.g. ``<monorepo>/data/proj3_data`` with
``CardImages``, ``Multi_CardImages`` and ``TrainingData`` inside. Discovery
walks up from the project root, so nothing is hard-coded and every script runs
with no arguments.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
_SUIT_WORDS = {"Diamond": "Diamonds", "Diamand": "Diamonds", "Spade": "Spades",
               "Club": "Clubs", "Heart": "Hearts"}


def find_data_root() -> Path | None:
    """Locate the proj3 data folder relative to this project."""
    for base in (ROOT.parent, ROOT.parent.parent, ROOT):
        for rel in ("data/proj3_data", "proj3_data", "data"):
            p = base / rel
            if (p / "CardImages").is_dir() or (p / "TrainingData").is_dir():
                return p
    for depth in range(1, 4):
        for p in ROOT.parent.glob("/".join(["*"] * depth) + "/proj3_data"):
            if p.is_dir():
                return p
    return None


def image_dirs() -> list[Path]:
    """Existing test-image folders (CardImages, Multi_CardImages)."""
    root = find_data_root()
    if root is None:
        return []
    return [root / n for n in ("CardImages", "Multi_CardImages")
            if (root / n).is_dir()]


def _training_dir(name: str) -> Path | None:
    root = find_data_root()
    if root is None:
        return None
    for rel in (Path("TrainingData") / name,
                Path("TrainingData") / "Training" / name):
        if (root / rel).is_dir():
            return root / rel
    return None


def rank_training_dir() -> Path | None:
    return _training_dir("RankTraining")


def suit_training_dir() -> Path | None:
    return _training_dir("SuitTraining")


def parse_truth(filename: str | Path) -> list[tuple[str, str]]:
    """Ground-truth (rank, suit) pairs encoded in a labeled test filename."""
    tokens = Path(filename).stem.split("_")[2:]
    pairs, i = [], 0
    while i < len(tokens) - 1:
        if tokens[i] in _SUIT_WORDS:
            pairs.append((tokens[i + 1], _SUIT_WORDS[tokens[i]]))
            i += 2
        else:
            i += 1
    return pairs


def optimal_assignments(preds: list[tuple], truth: list[tuple]) -> list[tuple]:
    """All maximum-score assignments of detections to ground-truth pairs.

    ``preds`` holds per-card (rank, suit) predictions (either may be ``None``);
    suit agreement is weighted higher because it anchors the assignment.
    """
    m = min(len(preds), len(truth))
    if m == 0:
        return []
    best, argmax = -1, []
    for perm in permutations(range(len(truth)), m):
        score = 0
        for c in range(m):
            rank_p, suit_p = preds[c]
            rank_t, suit_t = truth[perm[c]]
            score += 10 * (suit_p is not None and suit_p == suit_t)
            score += rank_p is not None and rank_p == rank_t
        if score > best:
            best, argmax = score, [perm]
        elif score == best:
            argmax.append(perm)
    return argmax


def forced_labels(preds: list[tuple], truth: list[tuple]) -> dict:
    """Per-detection labels that every optimal assignment agrees on."""
    argmax = optimal_assignments(preds, truth)
    if not argmax:
        return {}
    m = len(argmax[0])
    out = {}
    for c in range(m):
        ranks = {truth[p[c]][0] for p in argmax}
        suits = {truth[p[c]][1] for p in argmax}
        out[c] = (ranks.pop() if len(ranks) == 1 else None,
                  suits.pop() if len(suits) == 1 else None)
    return out
