"""Image discovery and loading helpers for the day/night classifier."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# File extensions treated as images, matching the MATLAB ``*.JPG`` pattern.
DEFAULT_EXTENSIONS = (".jpg", ".jpeg")


def find_images(directory, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Return image paths in *directory*, sorted by file name.

    Matching is case-insensitive, so ``.JPG`` and ``.jpg`` are both included.

    Raises
    ------
    NotADirectoryError
        If *directory* does not exist or is not a folder.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory not found: {directory}")
    wanted = {ext.lower() for ext in extensions}
    paths = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in wanted
    ]
    return sorted(paths)


def load_image(path) -> NDArray[np.uint8]:
    """Load *path* as an 8-bit RGB array of shape ``(H, W, 3)``.

    Grayscale or palette images are converted to RGB so downstream code can
    always assume three colour channels.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
