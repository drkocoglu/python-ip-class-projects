"""Visualisation helpers: an RGB frame next to its HSV view with a DAY/NIGHT tag.

The RGB image is shown on the left and the HSV image on the right. The DAY/NIGHT
label sits in a dedicated gap column between them (bold red text on a yellow,
black-bordered box) so it never overlaps either image.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from daynight_classifier import rgb_to_hsv_uint8


def build_figure(
    image: NDArray,
    label: str,
    score: float | None = None,
    title: str | None = None,
) -> Figure:
    """Build a three-panel figure laid out as ``RGB | label | HSV``.

    Parameters
    ----------
    image:
        8-bit RGB image.
    label:
        ``DAY`` or ``NIGHT`` tag shown in the central gap column.
    score:
        Optional decision score; shown in the figure title, not the label.
    title:
        Optional figure title (typically the file name).
    """
    figure, (rgb_ax, mid_ax, hsv_ax) = plt.subplots(
        1,
        3,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.0, 0.4, 1.0], "wspace": 0.05},
    )

    rgb_ax.imshow(image)
    rgb_ax.set_title("RGB image")
    rgb_ax.axis("off")

    hsv_ax.imshow(rgb_to_hsv_uint8(image))
    hsv_ax.set_title("HSV image")
    hsv_ax.axis("off")

    # The centre column holds only the label, so it never covers the images.
    mid_ax.axis("off")
    mid_ax.text(
        0.5,
        0.5,
        label,
        transform=mid_ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="red",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "yellow",
            "edgecolor": "black",
            "linewidth": 2.5,
        },
    )

    heading = title
    if score is not None:
        score_text = f"score = {score:.2f}"
        heading = f"{title}  \u00b7  {score_text}" if title else score_text
    if heading:
        figure.suptitle(heading)
    return figure


def render(figure: Figure, save_path=None, show: bool = False) -> None:
    """Save and/or display *figure*, then close it to free memory."""
    if save_path is not None:
        figure.savefig(Path(save_path), dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
