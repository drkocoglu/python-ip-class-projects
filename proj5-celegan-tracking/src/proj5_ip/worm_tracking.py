"""End-to-end worm-tracking pipeline.

Pass 1 builds (or loads from cache) the static-background mask. Pass 2 streams
the video again, segments and annotates the worm frame by frame, and writes an
annotated ``.avi`` into the project's ``results/`` folder. Frames are processed
one at a time so memory use stays flat regardless of clip length.
"""

from pathlib import Path

import cv2
import numpy as np

from proj5_ip import project_paths
from proj5_ip import tracking_config as cfg
from proj5_ip.background_model import compute_background_mask
from proj5_ip.result_overlay import draw_overlay
from proj5_ip.worm_segmentation import segment_worm
from proj5_ip.worm_skeleton import sample_geometry, skeletonize_worm


def _annotate_frame(
    frame: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Return ``(annotated_frame, worm_detected)`` for a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    worm, _ = segment_worm(gray, background)
    if worm is None:
        return frame, False

    skeleton = skeletonize_worm(worm)
    centerline, points_xy, normals_xy = sample_geometry(skeleton)
    if len(centerline) == 0:
        return frame, False

    endpoints_xy = np.array([centerline[0][::-1], centerline[-1][::-1]])
    annotated = draw_overlay(
        frame, worm, skeleton, points_xy, normals_xy, endpoints_xy
    )
    return annotated, True


def track_worm(video_path: Path, output_path: Path, cache_path: Path) -> Path:
    """Process ``video_path`` and write the annotated result to ``output_path``."""
    background = compute_background_mask(video_path, cache_path)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, cfg.OUTPUT_FPS, (width, height))

    print(f"Processing {total} frames...")
    step = max(1, total // 10)
    target = cfg.PREVIEW_FRAME_INDEX
    if target is None:
        target = total // 2
    best_preview: np.ndarray | None = None
    best_distance: int | None = None

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, detected = _annotate_frame(frame, background)
        writer.write(annotated)

        if detected:
            distance = abs(frame_index - target)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_preview = annotated.copy()

        frame_index += 1
        if frame_index % step == 0 or frame_index == total:
            print(f"  {frame_index / total * 100:5.1f}% ({frame_index}/{total} frames)")

    cap.release()
    writer.release()
    print(f"Saved annotated video to {output_path}")

    if cfg.SAVE_PREVIEW_IMAGE and best_preview is not None:
        preview_path = output_path.parent / cfg.PREVIEW_IMAGE_NAME
        cv2.imwrite(str(preview_path), best_preview)
        print(f"Saved preview image to {preview_path}")

    return output_path


def run() -> Path:
    """Resolve paths, run the pipeline, and return the output video path."""
    video_path = project_paths.find_input_video()
    results_dir = project_paths.get_results_dir()
    output_path = results_dir / cfg.OUTPUT_VIDEO_NAME
    cache_path = results_dir / cfg.MEAN_FRAME_CACHE_NAME
    print(f"Input video: {video_path}")
    return track_worm(video_path, output_path, cache_path)
