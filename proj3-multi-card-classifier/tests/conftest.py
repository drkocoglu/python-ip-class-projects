"""Pytest fixtures: project on sys.path and auto-discovered sample images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardclassifier import dataset  # noqa: E402


def _first_tif(folder_name: str) -> Path:
    root = dataset.find_data_root()
    folder = (root / folder_name) if root else None
    if folder is not None and folder.is_dir():
        files = sorted(folder.glob("*.tif"))
        if files:
            return files[0]
    return Path("missing") / folder_name / "none.tif"


@pytest.fixture
def single_card_path() -> Path:
    return _first_tif("CardImages")


@pytest.fixture
def multi_card_path() -> Path:
    return _first_tif("Multi_CardImages")


@pytest.fixture
def solid_square() -> np.ndarray:
    img = np.zeros((100, 100), np.uint8)
    img[30:70, 30:70] = 200
    return img


@pytest.fixture
def square_with_hole() -> np.ndarray:
    mask = np.zeros((50, 50), bool)
    mask[10:40, 10:40] = True
    mask[20:30, 20:30] = False
    return mask
