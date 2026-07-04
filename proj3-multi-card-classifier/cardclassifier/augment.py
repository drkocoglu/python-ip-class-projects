"""Pipeline-realistic image augmentation (from scratch, NumPy only).

Only transforms that mimic real variation in the pipeline's segmented glyph
crops are used, so augmentation *helps* rather than corrupts:

* small rotations (glyphs are never perfectly upright after card rotation),
* slight zoom / scale jitter,
* brightness & contrast shifts (lighting),
* mild Gaussian blur and noise (camera / segmentation softness).

Deliberately excluded: flips and large rotations (they change glyph identity --
a flipped/So-rotated 'K' or spade is no longer that class).
"""

from __future__ import annotations

import numpy as np

from .rotation import rotate
from .filters import gaussian_blur


def _zoom(img: np.ndarray, factor: float) -> np.ndarray:
    """Zoom in/out about the centre, keeping the same shape (reflect pad)."""
    h, w = img.shape
    if factor == 1.0:
        return img.copy()
    # Resample by nearest index into the original (cheap, adequate for aug).
    yy = np.clip(np.round(np.linspace(0, h - 1, h) / factor + (h - h / factor) / 2), 0, h - 1).astype(int)
    xx = np.clip(np.round(np.linspace(0, w - 1, w) / factor + (w - w / factor) / 2), 0, w - 1).astype(int)
    return img[np.ix_(yy, xx)]


def augment_once(img: np.ndarray, rng: np.random.Generator,
                 max_angle: float = 8.0) -> np.ndarray:
    """Return one random pipeline-realistic variant of a grayscale image."""
    out = np.asarray(img, dtype=np.float64)
    fill = float(np.median(out))               # rotate/pad with background tone

    angle = rng.uniform(-max_angle, max_angle)
    out = rotate(out, angle, fill=fill).astype(np.float64)

    factor = rng.uniform(0.9, 1.1)
    out = _zoom(out, factor)

    gain = rng.uniform(0.85, 1.15)             # brightness/contrast
    bias = rng.uniform(-12, 12)
    out = out * gain + bias

    if rng.random() < 0.5:
        out = gaussian_blur(out, rng.uniform(0.4, 1.0))
    if rng.random() < 0.5:
        out = out + rng.normal(0, rng.uniform(2, 8), out.shape)

    return np.clip(out, 0, 255).astype(np.uint8)


def augment_set(images, labels, rng, per_class_target=None, base_copies=3):
    """Augment a labelled set.

    Keeps every original, then adds variants. If ``per_class_target`` is given,
    minority classes are oversampled up to that count (handles imbalance);
    otherwise each image gets ``base_copies`` variants.

    Returns ``(aug_images, aug_labels)`` including the originals.
    """
    images = list(images)
    labels = list(labels)
    out_i, out_l = list(images), list(labels)

    if per_class_target is not None:
        by = {}
        for im, lb in zip(images, labels):
            by.setdefault(lb, []).append(im)
        for lb, ims in by.items():
            need = per_class_target - len(ims)
            for _ in range(max(0, need)):
                src = ims[rng.integers(len(ims))]
                out_i.append(augment_once(src, rng))
                out_l.append(lb)
    else:
        for im, lb in zip(images, labels):
            for _ in range(base_copies):
                out_i.append(augment_once(im, rng))
                out_l.append(lb)
    return out_i, out_l
