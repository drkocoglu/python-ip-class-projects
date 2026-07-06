"""Figure building and saving.

The figures show how the band-pass ring is built from a low-pass and a
high-pass, how it acts on the spectrum, and the resulting images. They are all
rendered and written to disk at once (no interactive display).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import fourier as fz
from .pipeline import IlluminationResult, PatternResult

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

_GRAY = "gray"
_HEAT = "inferno"


def _show_gray(ax: Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap=_GRAY)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _show_heat(ax: Axes, image: np.ndarray, title: str) -> None:
    im = ax.imshow(image, cmap=_HEAT)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def figure_input_and_spectrum(image: np.ndarray) -> Figure:
    """Input image beside its centred log-magnitude spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _show_gray(axes[0], image, "Input image (Proj4.tif)")
    _show_heat(axes[1], fz.log_magnitude(fz.fft2(image)), "log(1 + |F|) spectrum")
    fig.suptitle("Step 0 - Input and its Fourier spectrum", fontweight="bold")
    fig.tight_layout()
    return fig


def figure_bandpass_construction(result: PatternResult) -> Figure:
    """Show band-pass = low-pass x high-pass (the ring that keeps the pattern)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _show_heat(
        axes[0], np.fft.fftshift(result.low_component), "Low-pass\n(keeps below outer)"
    )
    _show_heat(
        axes[1],
        np.fft.fftshift(result.high_component),
        "High-pass\n(removes below inner)",
    )
    _show_heat(
        axes[2], np.fft.fftshift(result.band_filter), "Band-pass = low x high\n(ring)"
    )
    fig.suptitle(
        "Task 1 filter - the band-pass keeps the pattern ring, drops "
        "illumination and noise",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def figure_pattern_extraction(image: np.ndarray, result: PatternResult) -> Figure:
    """Spectrum, band-filtered + thresholded spectrum, and extracted pattern."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _show_heat(axes[0], fz.log_magnitude(result.spectrum), "Original spectrum")
    _show_heat(
        axes[1],
        fz.log_magnitude(result.filtered_spectrum),
        f"Band-pass + threshold\n({result.kept_count} spikes kept)",
    )
    _show_gray(axes[2], result.pattern, "Extracted periodic pattern")
    fig.suptitle(
        "Task 1 - pattern extraction (band-pass + spike threshold)", fontweight="bold"
    )
    fig.tight_layout()
    return fig


def figure_lowpass_surface(result: PatternResult) -> Figure:
    """3-D low-pass surface - the classic 'bright light in the middle' view."""
    low = np.fft.fftshift(result.low_component)
    fig = plt.figure(figsize=(8, 6))
    ax = cast("Axes3D", fig.add_subplot(1, 1, 1, projection="3d"))
    rows, cols = low.shape
    step = max(1, min(rows, cols) // 120)
    ys = np.arange(0, rows, step)
    xs = np.arange(0, cols, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax.plot_surface(
        grid_x, grid_y, low[::step, ::step], cmap=_HEAT, linewidth=0, antialiased=True
    )
    ax.set_title("Low-pass transfer surface (bright peak at DC)", fontsize=11)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    fig.tight_layout()
    return fig


def figure_illumination_correction(
    image: np.ndarray, result: IlluminationResult
) -> Figure:
    """Before/after spectra and images for the high-pass illumination fix."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    _show_heat(axes[0], fz.log_magnitude(result.spectrum), "Original spectrum")
    _show_heat(
        axes[1], fz.log_magnitude(result.filtered_spectrum), "High-pass spectrum"
    )
    _show_gray(axes[2], image, "Input (non-uniform light)")
    _show_gray(axes[3], result.corrected, "Corrected (uniform light)")
    fig.suptitle(
        "Task 2 - illumination correction (Butterworth high-pass)", fontweight="bold"
    )
    fig.tight_layout()
    return fig


def figure_results_summary(
    image: np.ndarray, pattern: np.ndarray, corrected: np.ndarray
) -> Figure:
    """The three deliverable images together."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _show_gray(axes[0], image, "Input")
    _show_gray(axes[1], pattern, "Task 1 - periodic pattern (band-pass)")
    _show_gray(axes[2], corrected, "Task 2 - uniform illumination (high-pass)")
    fig.suptitle("Project 4 - results summary", fontweight="bold")
    fig.tight_layout()
    return fig


def build_figures(
    image: np.ndarray,
    pattern_result: PatternResult,
    illum_result: IlluminationResult,
) -> list[tuple[str, Figure]]:
    """Assemble all figures as ``(name, figure)`` pairs in display order."""
    return [
        ("01_input_and_spectrum", figure_input_and_spectrum(image)),
        ("02_bandpass_construction", figure_bandpass_construction(pattern_result)),
        ("03_pattern_extraction", figure_pattern_extraction(image, pattern_result)),
        ("04_lowpass_surface", figure_lowpass_surface(pattern_result)),
        (
            "05_illumination_correction",
            figure_illumination_correction(image, illum_result),
        ),
        (
            "06_results_summary",
            figure_results_summary(
                image, pattern_result.pattern, illum_result.corrected
            ),
        ),
    ]


def save_figures(figures: list[tuple[str, Figure]], out_dir: Path) -> list[Path]:
    """Save every figure as a PNG, then close it, and return the written paths.

    All figures are written at once (no interactive display), so each is closed
    right after saving to free memory.
    """
    paths: list[Path] = []
    for name, fig in figures:
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
