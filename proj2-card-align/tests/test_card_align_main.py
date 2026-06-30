"""
test_card_align_main.py — Tests for the merged align/save/view script.

Covers the figure builders, the results-saving step (into a temp folder), and
the interactive viewer loop (with simulated terminal input). Uses the real
images if present and skips cleanly otherwise.
"""

import builtins
import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless: no windows during tests

import card_align_main as main_mod


@pytest.fixture
def sample_images():
    """A couple of real test images, or skip if the data isn't present."""
    from card_data_loader import load_all_grayscale
    try:
        imgs = load_all_grayscale()
    except FileNotFoundError:
        pytest.skip("Test images not found in data/proj2_data")
    return imgs[:2]  # two is enough to exercise the loops


# ── Figure builders ───────────────────────────────────────────────────────────
def test_edges_and_corners_figure(sample_images):
    path, gray = sample_images[0]
    fig = main_mod._edges_and_corners_figure("sample", gray)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_aligned_figure(sample_images):
    path, gray = sample_images[0]
    fig = main_mod._aligned_figure("sample", gray)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_showcase_figure(sample_images):
    fig = main_mod._showcase_figure(sample_images)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── Saving results ────────────────────────────────────────────────────────────
def test_save_results_writes_files(sample_images, tmp_path, monkeypatch):
    # Redirect RESULTS_DIR into the temp folder.
    monkeypatch.setattr(main_mod, "RESULTS_DIR", str(tmp_path / "results"))
    main_mod.save_results(sample_images)

    out = tmp_path / "results"
    assert out.is_dir()
    # Each image yields an _aligned and an _edges_and_corners file, plus showcase.
    files = list(out.glob("*.png"))
    assert any(f.name.endswith("_aligned.png") for f in files)
    assert any(f.name.endswith("_edges_and_corners.png") for f in files)
    assert (out / "showcase_all.png").is_file()


def test_save_results_creates_dir_if_missing(sample_images, tmp_path, monkeypatch):
    target = tmp_path / "nested" / "results"
    monkeypatch.setattr(main_mod, "RESULTS_DIR", str(target))
    assert not target.exists()
    main_mod.save_results(sample_images)
    assert target.is_dir()


# ── Interactive viewer ────────────────────────────────────────────────────────
def test_view_quits_immediately(sample_images, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "quit")
    main_mod.view_results(sample_images)  # should return cleanly


def test_view_advances_then_quits(sample_images, monkeypatch):
    responses = iter(["", "quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    main_mod.view_results(sample_images)


# ── main() orchestration ──────────────────────────────────────────────────────
def test_main_saves_without_viewing(sample_images, tmp_path, monkeypatch):
    monkeypatch.setattr(main_mod, "RESULTS_DIR", str(tmp_path / "results"))
    # view=False skips the interactive loop entirely.
    main_mod.main(view=False)
    assert (tmp_path / "results" / "showcase_all.png").is_file()
