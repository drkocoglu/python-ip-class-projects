"""Tests for the data I/O helpers, project structure, and config wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proj4_ip import dataio as io


def test_project_has_standard_folders() -> None:
    root = io.project_root()
    assert (root / "src").is_dir()
    assert (root / "scripts").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "scripts" / "proj4_main.py").is_file()


def test_results_dir_is_direct_child_of_root() -> None:
    out = io.results_dir(create=True)
    assert out.parent == io.project_root()
    assert out.name == "results" and out.is_dir()


def test_find_input_image_or_skip() -> None:
    try:
        path = io.find_input_image()
    except FileNotFoundError:
        pytest.skip("shared data folder not present in this checkout")
    assert path.name == "Proj4.tif" and path.parent.name == "proj4_data"


def test_find_input_image_raises_for_missing() -> None:
    with pytest.raises(FileNotFoundError):
        io.find_input_image(filename="does_not_exist.tif")


def test_to_uint8_stretches_and_handles_constant() -> None:
    out = io.to_uint8(np.array([[-3.0, 0.0], [3.0, 6.0]]))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert np.all(io.to_uint8(np.full((4, 4), 7.0)) == 0)


def test_save_image_writes_uint8_tiff(tmp_path: Path) -> None:
    import tifffile as tiff

    path = io.save_image(np.linspace(0, 50, 64).reshape(8, 8), tmp_path / "o.tif")
    loaded = tiff.imread(path)
    assert loaded.dtype == np.uint8 and loaded.shape == (8, 8)


def test_config_values_are_pipeline_defaults() -> None:
    import inspect

    from proj4_ip import config
    from proj4_ip.pipeline import extract_pattern

    defaults = inspect.signature(extract_pattern).parameters
    assert defaults["inner_cutoff"].default == config.PATTERN_INNER_CUTOFF
    assert defaults["log_threshold"].default == config.PATTERN_LOG_THRESHOLD
