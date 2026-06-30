"""
card_data_loader.py — Locate and load the playing-card test images.

The images live OUTSIDE the proj2 script folder, in a sibling data folder:

    <repo_root>/
        data/
            proj2_data/
                Testimage1.tif ... Testimage6.tif
        proj2/
            card_data_loader.py   <- this file
            ...

The path is found by searching UPWARD from this file for data/proj2_data, so the
scripts run no matter where the repo is cloned or launched from — no hard-coded
absolute paths.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

DATA_SUBPATH = os.path.join("data", "proj2_data")
IMAGE_GLOB = "*.tif"


def find_data_dir(start_dir: str | None = None) -> str:
    """Walk upward from this file until proj2_data/data is found."""
    current = start_dir or os.path.dirname(os.path.abspath(__file__))
    checked = []
    while True:
        candidate = os.path.join(current, DATA_SUBPATH)
        checked.append(candidate)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    searched = "\n  ".join(checked)
    raise FileNotFoundError(
        f"Could not find '{DATA_SUBPATH}'. Searched upward in:\n  {searched}"
    )


def list_image_paths() -> list[str]:
    """Return the sorted paths of all test images in the data folder."""
    data_dir = find_data_dir()
    paths = sorted(glob.glob(os.path.join(data_dir, IMAGE_GLOB)))
    if not paths:
        raise FileNotFoundError(f"No {IMAGE_GLOB} images found in {data_dir}")
    return paths


def load_grayscale(path: str) -> np.ndarray:
    """Load a single image as a 2-D uint8 grayscale array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_all_grayscale() -> list[tuple[str, np.ndarray]]:
    """Load every test image; return (path, grayscale_array) pairs."""
    return [(p, load_grayscale(p)) for p in list_image_paths()]
