"""Image I/O helpers.

Pillow is used only as a file-format codec (reading ``.tif`` and writing
``.png``); every pixel operation in this package is implemented from scratch in
NumPy. Using a codec to decode a TIFF is not the same as using a library to
*do the image processing*, which is what the assignment forbids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image from disk as a 2-D ``uint8`` grayscale array.

    Parameters
    ----------
    path:
        Path to an image file (``.tif``, ``.png``, ``.jpg``, ...).

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(H, W)``.
    """
    img = Image.open(Path(path)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    """Save a 2-D array as a grayscale PNG.

    Float images are assumed to be in ``[0, 1]`` and are scaled to ``[0, 255]``.
    Boolean images are mapped to ``{0, 255}``.
    """
    arr = np.asarray(image)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(Path(path))


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    """Save a 3-channel ``uint8`` ``(H, W, 3)`` array as a PNG."""
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(Path(path))
