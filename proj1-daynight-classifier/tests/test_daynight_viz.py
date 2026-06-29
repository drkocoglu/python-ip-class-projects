"""Smoke tests for the figure-building helper."""

from __future__ import annotations

import numpy as np
import pytest

# matplotlib is optional; skip this module entirely if it cannot be imported
# (for example when a security policy blocks its native DLLs).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend so tests need no display

from daynight_viz import build_figure, render  # noqa: E402  (after backend selection)


def _solid(color, size=8):
    return np.tile(np.array(color, np.uint8), (size, size, 1))


def test_build_figure_has_three_panels_with_label():
    import matplotlib.pyplot as plt

    figure = build_figure(_solid((255, 0, 0)), "DAY", score=255.0, title="sample.jpg")
    try:
        # RGB image, middle label panel, HSV image.
        assert len(figure.axes) == 3
        # The DAY/NIGHT label is drawn as text in the middle panel.
        label_texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert "DAY" in label_texts
    finally:
        plt.close(figure)


def test_render_saves_png(tmp_path):
    figure = build_figure(_solid((0, 0, 0)), "NIGHT")
    out = tmp_path / "frame.png"
    render(figure, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
