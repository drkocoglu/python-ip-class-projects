"""Static-background model built from the mean of all (binarized) frames.

The worm is dark on a light, stationary background, so each frame is
Otsu-thresholded with the polarity inverted (dark pixels become ``True``).
Averaging those masks over the whole clip yields, per pixel, the fraction of
frames in which it looked like foreground; pixels above ``MEAN_BINARIZE_LEVEL``
are the static background (grid dots, dish rim) that we subtract away.

The resulting boolean background mask is cached to a pickle keyed on the
video's identity, so repeated runs skip the full first pass.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np

from proj5_ip import tracking_config as cfg


def binarize_inverse(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold a grayscale frame so dark pixels become ``True``."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask.astype(bool)


def _cache_key(video_path: Path) -> dict[str, float | int | str]:
    stat = video_path.stat()
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "name": video_path.name,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "frames": frame_count,
        "level": cfg.MEAN_BINARIZE_LEVEL,
    }


def compute_background_mask(video_path: Path, cache_path: Path) -> np.ndarray:
    """Return the static-background mask, using the pickle cache when valid."""
    key = _cache_key(video_path)
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("key") == key:
            print(f"Reusing cached background model ({cache_path.name}).")
            return cached["mask"]

    print("Computing background model from all frames...")
    cap = cv2.VideoCapture(str(video_path))
    accumulator: np.ndarray | None = None
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        foreground = binarize_inverse(gray).astype(np.float64)
        accumulator = foreground if accumulator is None else accumulator + foreground
        count += 1
    cap.release()

    if accumulator is None or count == 0:
        raise ValueError(f"No frames could be read from {video_path}.")

    mask = (accumulator / count) > cfg.MEAN_BINARIZE_LEVEL
    with cache_path.open("wb") as handle:
        pickle.dump({"key": key, "mask": mask}, handle)
    return mask
