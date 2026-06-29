"""Tests for directory-level classification and the interactive viewer."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from daynight_classifier import DAY, NIGHT
from daynight_main import DEFAULT_DATA_DIR, classify_directory, run


def _write_image(path, color, size=8):
    Image.fromarray(np.tile(np.array(color, np.uint8), (size, size, 1))).save(path)


class TestClassifyDirectory:
    def test_labels_synthetic_images(self, tmp_path):
        _write_image(tmp_path / "a_day.jpg", (200, 120, 40))
        _write_image(tmp_path / "b_night.jpg", (0, 0, 0))
        results = classify_directory(tmp_path)
        labels = {r.path.name: r.label for r in results}
        assert labels == {"a_day.jpg": DAY, "b_night.jpg": NIGHT}

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            classify_directory(tmp_path / "missing")


@pytest.fixture
def two_images(tmp_path):
    _write_image(tmp_path / "a.jpg", (200, 120, 40))
    _write_image(tmp_path / "b.jpg", (0, 0, 0))
    return tmp_path


def _patch_display(monkeypatch):
    """Use a headless backend and stub the blocking display calls."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "pause", lambda *a, **k: None)


class TestInteractiveRun:
    def test_advances_through_all_images(self, two_images, monkeypatch, capsys):
        _patch_display(monkeypatch)
        responses = iter(["", ""])  # press Enter for each image
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(responses))

        run(two_images)

        out = capsys.readouterr().out
        assert "End of images." in out
        assert out.count("->") == 2  # one classification line per image

    def test_quit_stops_early(self, two_images, monkeypatch, capsys):
        _patch_display(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *a, **k: "quit")

        run(two_images)

        out = capsys.readouterr().out
        assert "Stopped." in out
        assert "End of images." not in out
        assert out.count("->") == 1  # quit after the first image

    def test_empty_directory_reports_and_returns(self, tmp_path, monkeypatch, capsys):
        _patch_display(monkeypatch)
        # input should never be called for an empty folder
        monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("input called"))

        run(tmp_path)

        assert "No images found" in capsys.readouterr().out


@pytest.mark.skipif(not DEFAULT_DATA_DIR.is_dir(), reason="bundled dataset not present")
def test_bundled_dataset_classifies_cleanly():
    results = classify_directory(DEFAULT_DATA_DIR)
    assert results
    assert all(r.label in {DAY, NIGHT} for r in results)
