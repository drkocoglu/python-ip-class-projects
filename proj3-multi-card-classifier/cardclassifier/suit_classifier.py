"""Suit classification: image-processing statistics + explicit thresholds.

No ML model of any kind -- the suit decision is a fixed, hand-structured
if/else tree over image statistics; only the numeric threshold values were
chosen using sample images (rules in ``models/suit_thresholds.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ThresholdSuitClassifier:
    """Evaluate the explicit threshold rules loaded from JSON."""

    def __init__(self, rules: dict):
        self.rules = rules

    @staticmethod
    def load(path: str | Path) -> "ThresholdSuitClassifier":
        with open(path) as f:
            return ThresholdSuitClassifier(json.load(f))

    def predict(self, suit_img: np.ndarray) -> str:
        from . import shape_features
        stats = shape_features.stats_dict(np.asarray(suit_img))
        node = self.rules["tree"]
        while "suit" not in node:
            node = node["le"] if stats[node["stat"]] <= node["thr"] else node["gt"]
        return node["suit"]
