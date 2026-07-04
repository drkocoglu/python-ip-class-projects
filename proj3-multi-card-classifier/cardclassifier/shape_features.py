"""From-scratch binary shape descriptors for suit classification.

Playing-card images here are grayscale, so colour (red vs black) is unavailable
and gradient/appearance features (BoVW, HOG) transfer poorly across the domain
gap between clean training crops and pipeline-segmented crops. Shape is far more
stable, so the suit classifier is built on interpretable binary-shape features:

* aspect ratio, extent, solidity (area / convex-hull area),
* a normalised row-width profile (how wide the pip is at each height), and
  summary statistics of it -- which cleanly separate the four suits:
    - Diamond: symmetric, widest in the middle, very high solidity,
    - Heart:   widest near the top (two lobes), pointed bottom,
    - Spade:   pointed top, wide bottom + stem,
    - Club:    three lobes, widest mid-upper, lowest solidity.

All computed with NumPy only (convex hull via Andrew's monotone chain).
"""

from __future__ import annotations

import numpy as np

from . import binarize, morphology


def _largest_blob(gray: np.ndarray) -> np.ndarray | None:
    """Binarize a suit crop and return the tight-cropped largest pip blob.

    Polarity is decided from the border ring: the background touches the crop
    border, so the pip is whichever class the border is NOT. (A majority vote
    over all pixels fails on tight crops where the pip covers >50%.)
    """
    g = np.asarray(gray)
    bw = binarize.binarize(g)
    border = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
    fg = ~bw if border.mean() > 0.5 else bw
    blob = morphology.largest_component(fg)
    ys, xs = np.where(blob)
    if ys.size == 0:
        return None
    return blob[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _convex_hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of 2-D points (monotone chain + shoelace)."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return float(len(pts))
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def suit_shape_features(gray: np.ndarray, n_rows: int = 12) -> np.ndarray:
    """Return a fixed-length shape-feature vector for a suit crop."""
    blob = _largest_blob(gray)
    if blob is None or blob.sum() < 5:
        return np.zeros(6 + n_rows, dtype=np.float64)

    h, w = blob.shape
    area = float(blob.sum())
    aspect = h / w
    extent = area / (h * w)

    ys, xs = np.where(blob)
    # Hull over pixel corners (not centers) so hull_area >= area, solidity <= 1.
    corners = np.concatenate([np.column_stack([xs + dx, ys + dy])
                              for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
    hull_area = _convex_hull_area(corners)
    solidity = min(area / hull_area, 1.0) if hull_area > 0 else 1.0

    # Row-width profile (fraction of columns filled per row), resampled to n_rows.
    row_w = blob.sum(axis=1).astype(np.float64)
    row_w = row_w / (row_w.max() + 1e-9)
    idx = np.linspace(0, h - 1, n_rows).astype(int)
    profile = row_w[idx]

    top = profile[: n_rows // 3].mean()
    mid = profile[n_rows // 3: 2 * n_rows // 3].mean()
    bot = profile[2 * n_rows // 3:].mean()
    argmax_pos = float(np.argmax(row_w)) / max(h - 1, 1)   # 0=top .. 1=bottom
    top_bot = top - bot                                    # + = top-heavy (heart)

    summary = np.array([aspect, extent, solidity, argmax_pos, top_bot, mid])
    return np.concatenate([summary, profile])


STAT_NAMES = (["aspect", "extent", "solidity", "argmax_pos", "top_bot", "mid"]
              + [f"prof{i}" for i in range(12)])


def stats_dict(gray: "np.ndarray") -> dict:
    """Named shape statistics of a suit pip (see :func:`suit_shape_features`)."""
    v = suit_shape_features(gray, n_rows=12)
    return {k: float(x) for k, x in zip(STAT_NAMES, v)}
