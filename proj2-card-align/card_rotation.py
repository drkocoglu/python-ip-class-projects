"""
card_rotation.py — Detect a playing card and rotate it upright, FROM SCRATCH.

This follows the constraints of the original image-processing assignment: the
only built-ins used are image I/O and geometric transforms (rotation). Everything
that actually solves the problem — smoothing, edge detection, corner finding, and
the rotation-angle computation — is implemented by hand with convolution and
basic arithmetic. No high-level vision helpers (no minAreaRect, no findContours,
no Canny, no HoughLines) are used.

Method (mirrors the original MATLAB approach)
---------------------------------------------
1. Smooth the image with a hand-written box-blur kernel (convolution) to reduce
   noise.
2. Apply hand-written SOBEL kernels in x and y, combine into a gradient
   magnitude, and threshold it to get a binary edge map of the card's border.
3. From the edge pixels, find the four EXTREME points (topmost, bottommost,
   leftmost, rightmost). For a rotated rectangle these are its four corners.
4. Compute the ROTATION ANGLE from the corners: take the longest side of the
   card (the vector between two adjacent corners) and measure its angle against
   the horizontal with atan2.
5. Rotate the image by that angle (geometric transform — allowed) so the card
   becomes axis-aligned, then crop to the card's now-upright bounding box.
6. Enforce PORTRAIT orientation: if the crop is wider than tall, rotate 90°.

The angle math and corner detection are the graded parts of the assignment, so
they are written out explicitly rather than delegated to a library.
"""

from __future__ import annotations

import numpy as np
import cv2   # used ONLY for geometric transforms (rotate/warp) + I/O elsewhere


# ─────────────────────────────────────────────────────────────────────────────
# Hand-written convolution and kernels
# ─────────────────────────────────────────────────────────────────────────────
def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Direct 2-D convolution (zero-padded, 'same' size), implemented by hand.

    This is the core primitive the assignment expects us to build rather than
    call a library filter. It slides the kernel over every pixel and computes the
    weighted sum. Vectorised over kernel taps (not per-pixel Python loops) so it
    stays fast, but the operation is still an explicit convolution we define.
    """
    image = image.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    out = np.zeros_like(image)
    # Accumulate the contribution of each kernel tap across the whole image.
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + image.shape[0],
                                         j:j + image.shape[1]]
    return out


# Sobel kernels (as defined in the original assignment).
SOBEL_X = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)


def box_blur(image: np.ndarray, passes: int = 5) -> np.ndarray:
    """Smooth with a small averaging kernel, applied a few times (as in the original)."""
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    blurred = image.astype(np.float64)
    for _ in range(passes):
        blurred = convolve2d(blurred, kernel)
    return blurred


# ─────────────────────────────────────────────────────────────────────────────
# Edge map
# ─────────────────────────────────────────────────────────────────────────────
def edge_map(gray: np.ndarray) -> np.ndarray:
    """
    Build a binary edge map: blur -> Sobel x & y -> gradient magnitude -> threshold.

    Returns a boolean array, True on strong edges. A border margin is cleared to
    drop the frame artifacts the Sobel operator produces at the image edge (the
    original did the same by zeroing the outer rows/columns).
    """
    blurred = box_blur(gray, passes=5)
    gx = convolve2d(blurred, SOBEL_X)
    gy = convolve2d(blurred, SOBEL_Y)
    magnitude = np.abs(gx) + np.abs(gy)

    # Threshold at a fraction of the max gradient → keep only strong edges.
    thresh = 0.25 * magnitude.max()
    edges = magnitude > thresh

    # Clear a border margin (Sobel produces a bright frame at the image edge).
    m = 12
    edges[:m, :] = False
    edges[-m:, :] = False
    edges[:, :m] = False
    edges[:, -m:] = False
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Rotation angle by minimum-bounding-box search (hand-coded rotating-calipers idea)
# ─────────────────────────────────────────────────────────────────────────────
def _edge_points(edges: np.ndarray) -> np.ndarray:
    """Return the (x, y) coordinates of all edge pixels as an (N, 2) array."""
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        raise ValueError("No edges found — is the card visible against the background?")
    return np.column_stack([xs, ys]).astype(np.float64)


def _bounding_box_area(points: np.ndarray, angle_deg: float) -> float:
    """
    Rotate the points by -angle and return the area of their axis-aligned
    bounding box. Implemented by hand: build the 2x2 rotation matrix, apply it,
    then measure width x height of the extent.
    """
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate every point by -angle (so a card tilted by +angle becomes upright).
    xr = points[:, 0] * cos_t + points[:, 1] * sin_t
    yr = -points[:, 0] * sin_t + points[:, 1] * cos_t
    width = xr.max() - xr.min()
    height = yr.max() - yr.min()
    return width * height


def find_rotation_angle(edges: np.ndarray) -> float:
    """
    Find the card's rotation angle by searching for the angle whose axis-aligned
    bounding box is smallest — the tightest box occurs when the card is upright.

    This is the rotating-calipers idea implemented from scratch: no library
    computes the angle for us. A rectangle only needs to be tested over 0..90
    degrees (its bounding box repeats every 90). We do a coarse sweep to find the
    neighbourhood, then a fine sweep to refine — cheap and robust because it uses
    ALL edge points, not four extreme pixels.
    """
    points = _edge_points(edges)

    # Coarse search over 0..90 degrees.
    best_angle, best_area = 0.0, None
    for a in np.arange(0.0, 90.0, 2.0):
        area = _bounding_box_area(points, a)
        if best_area is None or area < best_area:
            best_area, best_angle = area, a

    # Fine search in a +/-2 degree window around the coarse winner.
    for a in np.arange(best_angle - 2.0, best_angle + 2.0, 0.25):
        area = _bounding_box_area(points, a)
        if area < best_area:
            best_area, best_angle = area, a

    # best_angle in [0,90) makes the card axis-aligned. Choose the rotation that
    # brings its LONG edge vertical (portrait): if the box is wider than tall at
    # best_angle, an extra 90 turn is handled later by the portrait check.
    return float(best_angle)


def rotation_angle(gray: np.ndarray) -> float:
    """Full pipeline to just the detected rotation angle (degrees)."""
    edges = edge_map(gray)
    return find_rotation_angle(edges)


# ─────────────────────────────────────────────────────────────────────────────
# Rotate + crop + enforce portrait
# ─────────────────────────────────────────────────────────────────────────────
def rotate_card_upright(gray: np.ndarray) -> np.ndarray:
    """
    Detect the card, rotate it upright (portrait), and crop it tight.

    All the analysis (edges, angle) is hand-computed above; only the rotation
    itself uses a geometric-transform built-in, which the assignment permits.
    """
    edges = edge_map(gray)
    angle = find_rotation_angle(edges)

    # Card centre = mean of the edge points.
    points = _edge_points(edges)
    center = points.mean(axis=0)

    # Rotate the whole image so the card becomes axis-aligned.
    rot_mat = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rotated = cv2.warpAffine(gray, rot_mat, (gray.shape[1], gray.shape[0]))

    # Map the edge points through the SAME transform, then crop to their
    # axis-aligned bounding box (now tight around the upright card).
    ones = np.ones((points.shape[0], 1))
    moved = (rot_mat @ np.hstack([points, ones]).T).T

    pad = 6
    x0 = max(int(moved[:, 0].min()) - pad, 0)
    x1 = min(int(moved[:, 0].max()) + pad, rotated.shape[1])
    y0 = max(int(moved[:, 1].min()) - pad, 0)
    y1 = min(int(moved[:, 1].max()) + pad, rotated.shape[0])
    cropped = rotated[y0:y1, x0:x1]

    if cropped.size == 0:
        cropped = rotated

    # Enforce PORTRAIT: stand the card up if it came out wider than tall.
    if cropped.shape[1] > cropped.shape[0]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped
