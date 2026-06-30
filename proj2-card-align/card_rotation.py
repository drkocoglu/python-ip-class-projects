"""
card_rotation.py — Core algorithm: detect a playing card and rotate it upright.

Given a photo of a single playing card lying on a contrasting background, this
module locates the card, undoes its rotation, and crops it tightly so the output
is an axis-aligned, PORTRAIT (taller-than-wide) image of just the card.

Method (robust min-area-rectangle approach)
--------------------------------------------
1. Blur + Otsu threshold to separate the bright card from the darker background.
2. Take the largest external contour — that's the card outline.
3. Fit a MINIMUM-AREA ROTATED RECTANGLE to that contour. Its angle is the card's
   rotation directly, and its width/height are the card's true dimensions. This
   is far more robust than hunting for individual corner points, because the
   rectangle fit uses the entire outline rather than four extreme pixels.
4. Rotate the whole image by that angle so the card becomes axis-aligned, then
   crop to the rectangle.
5. Enforce PORTRAIT orientation: if the cropped card is wider than tall, rotate
   it 90° so it stands upright.

Why portrait, not "perfectly right-side-up": telling upside-down from right-side-up
would require reading the rank/suit, which is out of scope. Portrait orientation
(within a few degrees) is the well-defined, reliably achievable target.
"""

from __future__ import annotations

import cv2
import numpy as np


def _threshold_card(gray: np.ndarray) -> np.ndarray:
    """
    Blur to suppress noise, then Otsu-threshold so the bright card becomes white
    (255) and the darker background becomes black (0).
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu picks the threshold automatically from the image histogram.
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def _largest_contour(binary: np.ndarray) -> np.ndarray:
    """Return the largest external contour — assumed to be the card."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found — is the card visible against the background?")
    return max(contours, key=cv2.contourArea)


def find_card_rectangle(gray: np.ndarray):
    """
    Locate the card and return its minimum-area rotated rectangle.

    Returns
    -------
    rect : ((center_x, center_y), (width, height), angle)
        OpenCV's rotated-rectangle tuple describing the card.
    """
    binary = _threshold_card(gray)
    card = _largest_contour(binary)
    return cv2.minAreaRect(card)


def rotate_card_upright(gray: np.ndarray) -> np.ndarray:
    """
    Detect the card in a grayscale image and return it rotated upright (portrait)
    and cropped tight.

    Parameters
    ----------
    gray : (H, W) uint8 grayscale image containing one card.

    Returns
    -------
    card_upright : (h, w) uint8 image of just the card, axis-aligned and portrait.
    """
    (cx, cy), (w, h), angle = find_card_rectangle(gray)

    # Rotate the whole image about the card's centre so the card becomes
    # axis-aligned. OpenCV's angle is in degrees; warpAffine fills new corners
    # with black by default.
    rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(gray, rot_matrix, (gray.shape[1], gray.shape[0]))

    # Crop to the (now axis-aligned) card bounding box.
    box_w, box_h = int(round(w)), int(round(h))
    x0 = max(int(round(cx - box_w / 2)), 0)
    y0 = max(int(round(cy - box_h / 2)), 0)
    cropped = rotated[y0:y0 + box_h, x0:x0 + box_w]

    # Enforce PORTRAIT: if it came out wider than tall, stand it up with a 90° turn.
    if cropped.shape[1] > cropped.shape[0]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped


def rotation_angle(gray: np.ndarray) -> float:
    """Return just the detected rotation angle (degrees) for a card image."""
    (_, _), (_, _), angle = find_card_rectangle(gray)
    return float(angle)
