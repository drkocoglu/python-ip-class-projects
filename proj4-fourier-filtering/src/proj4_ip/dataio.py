"""Input/output helpers: locate the data, load it, and save results.

* The input image is found automatically by walking up the directory tree
  (data lives OUTSIDE this project, shared across the monorepo).
* The ``results`` folder is created directly inside the project folder (the one
  that contains ``src``, ``scripts`` and ``tests``).

Monorepo layout::

    <monorepo>/                        <- shared pyproject/uv.lock/.gitignore/.venv
        data/proj4_data/Proj4.tif      <- shared input data (outside proj4)
        proj4/                         <- this project
            scripts/proj4_main.py      <- the runnable entry point
            src/proj4_ip/              <- reusable modules (this package)
            tests/
            results/                   <- created on first run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff

INPUT_FILENAME = "Proj4.tif"
DATA_SUBPATH = Path("data") / "proj4_data"


def project_root() -> Path:
    """Return the project root: the nearest ancestor containing a ``src`` dir."""
    here = Path(__file__).resolve()
    for base in here.parents:
        if (base / "src").is_dir():
            return base
    return here.parents[2]  # fallback: <root>/src/proj4_ip/dataio.py


def find_input_image(
    filename: str = INPUT_FILENAME,
    data_subpath: Path = DATA_SUBPATH,
) -> Path:
    """Locate the input image by searching upward from the project root."""
    start = project_root()
    for base in (start, *start.parents):
        candidate = base / data_subpath / filename
        if candidate.is_file():
            return candidate
    searched = data_subpath / filename
    raise FileNotFoundError(
        f"Could not locate '{searched}' in '{start}' or any parent directory."
    )


def results_dir(create: bool = True) -> Path:
    """Return ``<project_root>/results`` (a direct child of the project)."""
    out = project_root() / "results"
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def load_image(path: Path) -> np.ndarray:
    """Load a grayscale image as a ``float64`` array."""
    image = tiff.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {image.shape}.")
    return image.astype(np.float64)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale a real array to the full 0-255 range (like ``imshow(x, [])``)."""
    finite = np.asarray(image, dtype=np.float64)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(finite.shape, dtype=np.uint8)
    scaled = (finite - lo) / (hi - lo)
    return np.round(scaled * 255.0).astype(np.uint8)


def save_image(image: np.ndarray, path: Path) -> Path:
    """Save ``image`` as an 8-bit TIFF, scaling to 0-255 first."""
    tiff.imwrite(path, to_uint8(image))
    return path
