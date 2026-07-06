"""Draw the annotated tracking overlay for a single frame.

All drawing is done with OpenCV (no matplotlib) to keep the pipeline fast and
free of GUI/back-end dependencies. Colors follow the original report:
blue worm, pink skeleton, green worm box, purple head/tail boxes, red normal
vectors, and yellow equidistant points.
"""

import cv2
import numpy as np
from scipy import ndimage as ndi

from proj5_ip import tracking_config as cfg
from proj5_ip.worm_segmentation import bounding_box


def _blend_fill(
    frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> None:
    tint = np.array(color, dtype=np.float32)
    region = frame[mask].astype(np.float32)
    frame[mask] = (
        (1.0 - cfg.WORM_ALPHA) * region + cfg.WORM_ALPHA * tint
    ).astype(np.uint8)


def draw_overlay(
    frame: np.ndarray,
    worm: np.ndarray,
    skeleton: np.ndarray,
    points_xy: np.ndarray,
    normals_xy: np.ndarray,
    endpoints_xy: np.ndarray,
) -> np.ndarray:
    """Return a copy of ``frame`` with the full tracking annotation drawn on."""
    canvas = frame.copy()

    _blend_fill(canvas, worm, cfg.COLOR_WORM)
    canvas[ndi.binary_dilation(skeleton, iterations=2)] = cfg.COLOR_SKELETON

    x_min, y_min, x_max, y_max = bounding_box(worm)
    pad = cfg.BBOX_PADDING
    cv2.rectangle(
        canvas,
        (x_min - pad, y_min - pad),
        (x_max + pad, y_max + pad),
        cfg.COLOR_BBOX,
        2,
    )

    for (x, y), normal in zip(points_xy, normals_xy, strict=True):
        base = (int(round(x)), int(round(y)))
        offset = (normal * cfg.NORMAL_LENGTH).round().astype(int)
        cv2.arrowedLine(
            canvas, base, (base[0] + offset[0], base[1] + offset[1]),
            cfg.COLOR_NORMAL, 2, tipLength=0.3,
        )
        cv2.arrowedLine(
            canvas, base, (base[0] - offset[0], base[1] - offset[1]),
            cfg.COLOR_NORMAL, 2, tipLength=0.3,
        )
        cv2.circle(canvas, base, 4, cfg.COLOR_POINT, -1)

    half = cfg.HEADTAIL_BOX_HALF
    for x, y in endpoints_xy:
        top_left = (int(x) - half, int(y) - half)
        bottom_right = (int(x) + half, int(y) + half)
        cv2.rectangle(canvas, top_left, bottom_right, cfg.COLOR_HEADTAIL, 2)

    return canvas
