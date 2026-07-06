"""Pytest fixtures and import-path setup for Project 4 tests.

Adds ``src`` to ``sys.path`` so the ``proj4_ip`` package imports without an
install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def input_image() -> np.ndarray:
    """The real Proj4 input if present, else a synthetic stand-in.

    The synthetic image has a strong periodic pattern on a smooth illumination
    ramp, so tests run even when the shared data folder is absent.
    """
    from proj4_ip import dataio as io

    try:
        return io.load_image(io.find_input_image())
    except FileNotFoundError:
        return _synthetic_image()


def _synthetic_image() -> np.ndarray:
    rows, cols = 96, 128
    yy, xx = np.mgrid[0:rows, 0:cols]
    ramp = 60.0 + 0.4 * xx + 0.2 * yy  # smooth (low-freq) illumination
    pattern = 25.0 * np.cos(2 * np.pi * (6 * xx + 5 * yy) / cols)  # periodic
    return (ramp + pattern).astype(np.float64)
