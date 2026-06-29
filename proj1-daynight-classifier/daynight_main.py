"""Interactive viewer: step through the camera-trap images one at a time.

Run it directly::

    python daynight_main.py

Each image is shown with its DAY/NIGHT label. Press Enter in the terminal to
move to the next image, or type ``quit`` to stop. Closing a figure window does
not stop the program -- advancing is driven entirely from the terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from daynight_classifier import DayNightClassifier
from daynight_io import find_images, load_image

# The dataset ships one level up from the project: <repo>/data/proj1_data.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "proj1_data"

PROMPT = "Press Enter for the next image (or type 'quit' to stop): "
QUIT_WORDS = {"quit", "q"}


class Classification(NamedTuple):
    """Result of classifying a single image."""

    path: Path
    score: float
    label: str


def classify_directory(data_dir=DEFAULT_DATA_DIR) -> list[Classification]:
    """Classify every image in *data_dir*, sorted by file name."""
    classifier = DayNightClassifier()
    results = []
    for path in find_images(data_dir):
        score = classifier.score(load_image(path))
        results.append(Classification(path, score, classifier.label_for_score(score)))
    return results


def run(data_dir=DEFAULT_DATA_DIR) -> None:
    """Step through the images interactively, one figure at a time.

    The figures are non-blocking: advancing and quitting are controlled from the
    terminal, so closing a window never stops the program.
    """
    import matplotlib.pyplot as plt

    import daynight_viz as viz

    paths = find_images(data_dir)
    if not paths:
        print(f"No images found in {data_dir}")
        return

    classifier = DayNightClassifier()
    plt.ion()  # interactive mode: figures display without blocking
    for path in paths:
        image = load_image(path)
        score = classifier.score(image)
        label = classifier.label_for_score(score)
        print(f"{path.name}  ->  {label}  (score {score:.2f})")

        figure = viz.build_figure(image, label, score=score, title=path.name)
        plt.show(block=False)
        plt.pause(0.1)  # let the window draw before we block on terminal input

        response = input(PROMPT).strip().lower()
        plt.close(figure)
        if response in QUIT_WORDS:
            print("Stopped.")
            plt.close("all")
            return

    plt.close("all")
    print("End of images.")


def main() -> None:
    """Entry point."""
    run()


if __name__ == "__main__":
    main()
