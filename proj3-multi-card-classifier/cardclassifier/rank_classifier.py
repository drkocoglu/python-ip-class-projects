"""Rank classification: HOG features -> SVM (linear or RBF).

The glyph crop is sharpened (anti-blur), ink-normalized onto a fixed canvas,
described with from-scratch HOG, standardized, and classified by a
scikit-learn SVM. Trained only on the hand-labeled TrainingData; the labeled
test cards are never used for training.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import binarize, hog
from .filters import unsharp


def resize_bilinear(image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize to (rows, cols), implemented with NumPy only."""
    img = np.asarray(image, dtype=np.float64)
    in_h, in_w = img.shape[:2]
    out_h, out_w = out_shape
    if (in_h, in_w) == (out_h, out_w):
        return img
    ry = np.clip((np.arange(out_h) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
    rx = np.clip((np.arange(out_w) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
    y0 = np.floor(ry).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(rx).astype(int)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ry - y0)[:, None]
    wx = (rx - x0)[None, :]
    top = img[np.ix_(y0, x0)] * (1 - wx) + img[np.ix_(y0, x1)] * wx
    bot = img[np.ix_(y1, x0)] * (1 - wx) + img[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def ink_normalize(image: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Isolate the dark glyph, tight-crop it, and center it on a white canvas.

    Applied identically at train and inference time so hand-cropped training
    glyphs and pipeline-segmented glyphs share one geometry.
    """
    g = np.asarray(image, dtype=np.float64)
    ink = ~binarize.binarize(g)
    if ink.mean() > 0.5:
        ink = ~ink
    ys, xs = np.where(ink)
    if ys.size == 0:
        return resize_bilinear(g, canvas)
    glyph = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    oh, ow = canvas
    h, w = glyph.shape
    scale = min((oh - 4) / h, (ow - 4) / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = resize_bilinear(glyph, (nh, nw))
    out = np.full((oh, ow), 255.0)
    y0 = (oh - nh) // 2
    x0 = (ow - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


@dataclass
class RankClassifier:
    """HOG + SVM rank classifier ('10' is a single class)."""

    canvas: tuple = (40, 28)
    cell_size: tuple = (6, 6)
    n_bins: int = 9
    kernel: str = "rbf"
    scaler_: StandardScaler | None = None
    svm_: SVC | LinearSVC | None = None
    labels_: list | None = None

    def _feat(self, image: np.ndarray) -> np.ndarray:
        g = unsharp(np.asarray(image, dtype=np.float64), sigma=1.2, amount=1.0)
        g = ink_normalize(g, self.canvas)
        return hog.hog_features(g, self.cell_size, (2, 2), self.n_bins)

    def fit(self, images: list, labels: list) -> "RankClassifier":
        X = np.array([self._feat(im) for im in images])
        self.scaler_ = StandardScaler().fit(X)
        if self.kernel == "linear":
            self.svm_ = LinearSVC(C=1.0, dual=False, max_iter=5000)
        else:
            self.svm_ = SVC(kernel="rbf", C=10.0, gamma="scale")
        self.svm_.fit(self.scaler_.transform(X), np.array(labels))
        self.labels_ = sorted(set(labels))
        return self

    def _scores(self, image: np.ndarray) -> np.ndarray:
        x = self.scaler_.transform(self._feat(image)[None, :])
        return np.atleast_2d(self.svm_.decision_function(x))[0]

    def predict(self, image: np.ndarray) -> str:
        s = self._scores(image)
        if s.size == 1:
            return str(self.svm_.predict(
                self.scaler_.transform(self._feat(image)[None, :]))[0])
        return str(self.svm_.classes_[int(np.argmax(s))])

    def predict_margin(self, image: np.ndarray) -> tuple[str, float]:
        """Best label and its decision score (used to resolve sideways crops)."""
        s = self._scores(image)
        i = int(np.argmax(s))
        return str(self.svm_.classes_[i]), float(s[i])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RankClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
