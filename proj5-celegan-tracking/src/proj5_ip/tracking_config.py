"""Central configuration for the C. elegans worm-tracking pipeline.

This is the one place to tune the pipeline. Every value below is safe to
edit; nothing else in the project needs to change. Values are grouped by
the stage of the pipeline they affect.

The area thresholds are expressed in pixels and were tuned for the bundled
828x1108 synthetic video (worm area ~= 3,600-7,000 px). The original MATLAB
project used a 2160x2560 recording, so its raw thresholds (23,000 / 30,000)
are ~4x larger; if you swap in a higher-resolution video, scale the
``*_AREA`` values up by roughly ``(width * height) / (828 * 1108)``.
"""

# --- Input / output ------------------------------------------------------
# Folder (under the repo-level ``data/`` directory) that holds the video.
VIDEO_SUBDIR = "proj5_data"
# Which video in ``data/proj5_data`` to process. Set this to the exact file
# name when the folder holds several videos. Leave as "" to auto-detect when
# exactly one video file is present.
# NOTE: the area thresholds below are tuned for the 828x1108 synthetic clip;
# scale them up if you point this at a higher-resolution recording.
INPUT_VIDEO_NAME = "kling_20260706_VIDEO_shot_1_2s__5446_0.avi"
# Name of the annotated video written into the project's ``results/`` folder.
OUTPUT_VIDEO_NAME = "worm_tracking_output.avi"
# Playback speed of the written result video (frames per second).
OUTPUT_FPS = 5
# File the computed mean-background mask is cached to (inside ``results/``).
MEAN_FRAME_CACHE_NAME = "mean_frame_cache.pkl"
# Save one representative annotated frame as a PNG (handy for the README).
SAVE_PREVIEW_IMAGE = True
PREVIEW_IMAGE_NAME = "worm_tracking_preview.png"
# Frame index to use for that preview, or None for the middle of the clip.
# The nearest frame with a detected worm is used if this one has none.
PREVIEW_FRAME_INDEX = None

# --- Background model ----------------------------------------------------
# A pixel is treated as static background when it is part of the (inverted,
# binarized) foreground in at least this fraction of all frames.
MEAN_BINARIZE_LEVEL = 0.85

# --- Worm detection ------------------------------------------------------
# Smallest largest-object area (px) that still counts as a visible worm.
# Frames below this are passed through un-annotated.
MIN_WORM_AREA = 3000
# Above this area the object is assumed to have merged with a bubble/edge,
# so an opening is applied first to break the thin connection.
LARGE_WORM_AREA = 12000
# Structuring-element radii (px) for the morphological clean-up.
OPEN_DISK_RADIUS = 5
CLOSE_DISK_RADIUS = 12

# --- Skeleton & geometry -------------------------------------------------
# Number of equidistant sample points placed along the worm's centerline.
NUM_EQUIDISTANT_POINTS = 10
# Half-window (in centerline samples) used for the finite-difference tangent.
TANGENT_HALF_WINDOW = 3
# Drawn length (px) of each normal vector.
NORMAL_LENGTH = 35

# --- Drawing -------------------------------------------------------------
# Padding (px) added around the worm bounding box.
BBOX_PADDING = 15
# Half-size (px) of the square boxes drawn around the head and tail.
HEADTAIL_BOX_HALF = 30
# Opacity of the blue worm fill overlay (0 = invisible, 1 = opaque).
WORM_ALPHA = 0.5
# Colors are given in OpenCV's BGR order.
COLOR_WORM = (200, 0, 0)
COLOR_SKELETON = (180, 50, 255)
COLOR_BBOX = (0, 255, 0)
COLOR_HEADTAIL = (142, 47, 126)
COLOR_NORMAL = (0, 0, 255)
COLOR_POINT = (0, 255, 255)
