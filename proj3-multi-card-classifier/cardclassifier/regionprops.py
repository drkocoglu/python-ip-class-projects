"""A minimal, from-scratch ``regionprops``.

Only the properties the pipeline actually needs are implemented:

* ``area``        -- pixel count
* ``centroid``    -- (cx, cy)
* ``bounding_box``-- (x, y, width, height), MATLAB-style (top-left corner at
  ``x - 0.5``; here we return integer pixel extents which is what the crop code
  consumes)
* ``orientation`` -- angle in degrees of the region's major axis, using the same
  convention as MATLAB ``regionprops`` (measured counter-clockwise from the
  positive x-axis, in ``(-90, 90]``).

The orientation is derived from the second-order central moments of the region,
which is exactly how MATLAB computes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegionProps:
    label: int
    area: int
    centroid: tuple[float, float]          # (cx, cy)
    bounding_box: tuple[int, int, int, int]  # (x, y, w, h)
    orientation: float                      # degrees


def _region_orientation(ys: np.ndarray, xs: np.ndarray, cy: float, cx: float) -> float:
    """Orientation (degrees) from second central moments, MATLAB convention."""
    x = xs - cx
    y = ys - cy
    n = x.size

    # Normalised second central moments. MATLAB adds 1/12 (the variance of a
    # unit pixel) to the diagonal moments; we mirror that for parity.
    uxx = (x * x).sum() / n + 1.0 / 12.0
    uyy = (y * y).sum() / n + 1.0 / 12.0
    uxy = (x * y).sum() / n

    # MATLAB measures orientation clockwise-positive in image coords, then
    # reports it as counter-clockwise from the x-axis. Reproduce its formula.
    if uyy > uxx:
        num = uyy - uxx + np.sqrt((uyy - uxx) ** 2 + 4 * uxy ** 2)
        den = 2 * uxy
    else:
        num = 2 * uxy
        den = uxx - uyy + np.sqrt((uxx - uyy) ** 2 + 4 * uxy ** 2)

    if num == 0 and den == 0:
        return 0.0
    return float(np.degrees(np.arctan2(num, den)))


def regionprops(labels: np.ndarray, n: int | None = None) -> list[RegionProps]:
    """Compute properties for each labelled region.

    Parameters
    ----------
    labels:
        ``int`` label image (0 = background) as returned by
        :func:`cardclassifier.morphology.connected_components`.
    n:
        Number of labels. If ``None`` it is inferred from ``labels.max()``.
    """
    labels = np.asarray(labels)
    if n is None:
        n = int(labels.max())

    props: list[RegionProps] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        if xs.size == 0:
            continue
        area = int(xs.size)
        cx = float(xs.mean())
        cy = float(ys.mean())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        orient = _region_orientation(ys.astype(np.float64), xs.astype(np.float64), cy, cx)
        props.append(RegionProps(lab, area, (cx, cy), bbox, orient))
    return props


def largest_region(labels: np.ndarray, n: int | None = None) -> RegionProps | None:
    """Return the :class:`RegionProps` of the largest region, or ``None``."""
    props = regionprops(labels, n)
    if not props:
        return None
    return max(props, key=lambda p: p.area)
