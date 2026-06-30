"""
card_align_main.py — Align every test card, save the results, and view them.

For each input photo this:
  1. Detects the card, rotates it upright (portrait), and crops it.
  2. Saves two images into results/:
       <name>_edges_and_corners.png  - edges + fitted min-area rectangle whose
                                       four corners (red squares) the rotation is
                                       computed from. (The method fits a rectangle
                                       to the whole outline; it does not detect the
                                       card's own corners directly.)
       <name>_aligned.png            - original next to the upright, cropped card.
  3. Shows the original and aligned card side by side, stepping through them
     interactively from the terminal.

A combined results/showcase_all.png grid of all cards is also saved.

Interactive controls (during viewing):
  * Press ENTER to advance to the next card.
  * Type 'quit' then ENTER to stop.
  * Closing the figure window does NOT stop the program — the next card opens in
    a fresh window.

Run:  python card_align_main.py
The results/ folder is created next to this script if missing; existing result
images are overwritten each run.
"""

from __future__ import annotations

import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from card_data_loader import load_all_grayscale
from card_rotation import (
    rotate_card_upright, rotation_angle,
    _threshold_card, _largest_contour,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders (shared by both the saved files and the interactive view)
# ─────────────────────────────────────────────────────────────────────────────
def _edges_and_corners_figure(base, gray):
    """Edge map + fitted min-area rectangle + its four corners as red squares."""
    binary = _threshold_card(gray)
    card = _largest_contour(binary)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    rect = cv2.minAreaRect(card)
    box = np.intp(cv2.boxPoints(rect))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(edges, cmap="gray")
    poly = np.vstack([box, box[0]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color="lime", linewidth=1.5,
            label="min-area rectangle")
    ax.scatter(box[:, 0], box[:, 1], s=120, marker="s",
               facecolors="none", edgecolors="red", linewidths=2,
               label="rectangle corners")
    ax.set_title(f"{base}: edges + fitted rectangle corners")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _aligned_figure(base, gray):
    """Original vs upright+cropped result, side by side."""
    angle = rotation_angle(gray)
    aligned = rotate_card_upright(gray)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(aligned, cmap="gray")
    axes[1].set_title("Aligned + Cropped")
    axes[1].axis("off")
    fig.suptitle(f"{base}  (angle: {angle:.1f} deg)")
    fig.tight_layout()
    return fig


def _showcase_figure(images):
    """A grid of every card: original on top, aligned below."""
    fig, axes = plt.subplots(2, len(images), figsize=(3 * len(images), 6))
    for col, (path, gray) in enumerate(images):
        angle = rotation_angle(gray)
        aligned = rotate_card_upright(gray)
        axes[0, col].imshow(gray, cmap="gray")
        axes[0, col].set_title(f"In ({angle:.0f} deg)", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(aligned, cmap="gray")
        axes[1, col].set_title("Out", fontsize=9)
        axes[1, col].axis("off")
    fig.suptitle("Card Alignment: Original (top) -> Upright + Cropped (bottom)",
                 fontsize=14)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Saving and viewing
# ─────────────────────────────────────────────────────────────────────────────
def save_results(images):
    """Write all result images (per-card + showcase) into results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for path, gray in images:
        base = os.path.splitext(os.path.basename(path))[0]

        fig = _edges_and_corners_figure(base, gray)
        fig.savefig(os.path.join(RESULTS_DIR, f"{base}_edges_and_corners.png"), dpi=90)
        plt.close(fig)

        fig = _aligned_figure(base, gray)
        fig.savefig(os.path.join(RESULTS_DIR, f"{base}_aligned.png"), dpi=90)
        plt.close(fig)
        print(f"Saved results for {base}")

    fig = _showcase_figure(images)
    fig.savefig(os.path.join(RESULTS_DIR, "showcase_all.png"), dpi=80)
    plt.close(fig)
    print(f"Saved showcase_all.png to {RESULTS_DIR}")


def view_results(images):
    """Step through the original/aligned pairs interactively from the terminal."""
    print("\n" + "-" * 60)
    print("Card alignment viewer.")
    print("  - Press ENTER to see the next card.")
    print("  - Type 'quit' then ENTER to stop.")
    print("  - Closing the image window is fine — the program keeps running.")
    print("-" * 60)

    for path, gray in images:
        base = os.path.splitext(os.path.basename(path))[0]
        angle = rotation_angle(gray)
        print(f"\n{base}: detected rotation angle = {angle:.2f} degrees")

        fig = _aligned_figure(base, gray)
        # Only render under a GUI backend; under Agg (tests/headless) skip silently.
        if matplotlib.get_backend().lower() != "agg":
            plt.show(block=False)
            plt.pause(0.1)

        user = input("Press ENTER for next image, or type 'quit' to stop: ").strip().lower()
        plt.close(fig)
        if user == "quit":
            print("Stopping.")
            break


def main(view=True):
    images = load_all_grayscale()
    save_results(images)          # always write the result images
    if view:                      # then optionally step through them
        view_results(images)


if __name__ == "__main__":
    main()
