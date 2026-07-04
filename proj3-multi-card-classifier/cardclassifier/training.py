"""Training-set utilities: loading labeled folders and augmentation."""

from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import numpy as np

from . import augment, filters, io_utils

_EXTS = ("*.jpg", "*.png", "*.tif", "*.tiff")


def load_labeled(root: str | Path, classes: list[str]):
    """Load one subfolder per class, de-duplicating identical files."""
    X, y, seen = [], [], set()
    for c in classes:
        for ext in _EXTS:
            for f in sorted(glob.glob(str(Path(root) / c / ext))):
                digest = hashlib.md5(open(f, "rb").read()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                X.append(io_utils.load_gray(f))
                y.append(c)
    return X, np.array(y)


def defocus(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic camera-defocus copy (Gaussian blur) of a glyph crop."""
    g = filters.gaussian_blur(np.asarray(image, float), rng.uniform(1.2, 2.2))
    return np.clip(g, 0, 255).astype(np.uint8)


def build_training_set(images: list, labels: list, per_class_target: int = 150):
    """Balanced augmentation plus one defocus copy per original image."""
    Xa, ya = augment.augment_set(list(images), list(labels),
                                 np.random.default_rng(1),
                                 per_class_target=per_class_target)
    Xa = Xa + [defocus(images[i], np.random.default_rng(100 + i))
               for i in range(len(images))]
    return Xa, list(ya) + list(labels)
