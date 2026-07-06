"""Tests for the mean-background model and its pickle cache."""

from pathlib import Path

import cv2
import numpy as np

from proj5_ip import background_model


def _write_clip(path: Path, frames: list[np.ndarray], fps: int = 5) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_binarize_inverse_marks_dark_pixels() -> None:
    gray = np.full((20, 20), 240, dtype=np.uint8)
    gray[5:15, 5:15] = 10  # dark square
    mask = background_model.binarize_inverse(gray)
    assert mask[10, 10]  # dark -> True
    assert not mask[0, 0]  # light -> False


def test_static_pixel_is_background(tmp_path: Path) -> None:
    # A dark bar stays put (background); a moving dot does not.
    frames = []
    for shift in range(10):
        frame = np.full((40, 40, 3), 240, dtype=np.uint8)
        frame[0:40, 5:9] = 10  # static dark bar in every frame
        frame[shift * 3 : shift * 3 + 3, 30:33] = 10  # moving dark dot
        frames.append(frame)
    clip = tmp_path / "clip.avi"
    _write_clip(clip, frames)

    mask = background_model.compute_background_mask(clip, tmp_path / "cache.pkl")
    assert mask[20, 6]  # static bar -> background
    assert not mask[20, 31]  # moving dot -> not background


def test_cache_is_written_and_reused(tmp_path: Path) -> None:
    frames = [np.full((30, 30, 3), 240, dtype=np.uint8) for _ in range(4)]
    for frame in frames:
        frame[10:20, 10:20] = 10
    clip = tmp_path / "clip.avi"
    _write_clip(clip, frames)
    cache = tmp_path / "cache.pkl"

    first = background_model.compute_background_mask(clip, cache)
    assert cache.exists()
    second = background_model.compute_background_mask(clip, cache)
    assert np.array_equal(first, second)
