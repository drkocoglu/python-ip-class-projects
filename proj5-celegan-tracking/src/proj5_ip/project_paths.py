"""Locate the repo, data, and results folders by walking up the tree.

The project deliberately has no per-project ``pyproject.toml`` or virtual
environment: the shared ``data/`` folder lives above the project folder, and
paths are resolved at run time so the scripts work no matter where the repo
is checked out.
"""

from pathlib import Path

from proj5_ip import tracking_config as cfg


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor that contains ``data/<VIDEO_SUBDIR>``."""
    origin = (start or Path(__file__)).resolve()
    marker = Path("data") / cfg.VIDEO_SUBDIR
    for parent in (origin, *origin.parents):
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not find 'data/{cfg.VIDEO_SUBDIR}' above {origin}. "
        "Place the video under '<repo>/data/proj5_data/'."
    )


def get_project_root() -> Path:
    """Return the ``proj5-worm-tracking`` folder (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the ``data/proj5_data`` directory holding the input video."""
    return find_repo_root() / "data" / cfg.VIDEO_SUBDIR


def get_results_dir() -> Path:
    """Return the project's ``results/`` folder, creating it if needed."""
    results = get_project_root() / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results


def find_input_video() -> Path:
    """Return the video to process from ``data/proj5_data``.

    Uses ``INPUT_VIDEO_NAME`` when set; otherwise auto-detects the single
    video file present.
    """
    data_dir = get_data_dir()

    if cfg.INPUT_VIDEO_NAME:
        chosen = data_dir / cfg.INPUT_VIDEO_NAME
        if not chosen.is_file():
            raise FileNotFoundError(
                f"INPUT_VIDEO_NAME='{cfg.INPUT_VIDEO_NAME}' was not found in "
                f"{data_dir}. Fix the name in tracking_config.py."
            )
        return chosen

    videos = sorted(
        p
        for p in data_dir.iterdir()
        if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    )
    if not videos:
        raise FileNotFoundError(f"No video file found in {data_dir}.")
    if len(videos) > 1:
        names = ", ".join(p.name for p in videos)
        raise ValueError(
            f"Found {len(videos)} videos in {data_dir}: {names}. Set "
            "INPUT_VIDEO_NAME in tracking_config.py to pick one."
        )
    return videos[0]
