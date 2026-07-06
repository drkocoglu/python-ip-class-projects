"""Tests for repo/data/results path discovery."""

from pathlib import Path

import pytest

from proj5_ip import project_paths


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "data" / "proj5_data").mkdir(parents=True)
    return tmp_path


def test_find_repo_root_walks_up(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    deep = repo / "proj5-worm-tracking" / "src" / "proj5_ip"
    deep.mkdir(parents=True)
    assert project_paths.find_repo_root(deep / "x.py") == repo


def test_find_repo_root_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        project_paths.find_repo_root(tmp_path / "nowhere" / "x.py")


def _patch_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = ""
) -> Path:
    data_dir = tmp_path / "data" / "proj5_data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(project_paths, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(project_paths.cfg, "INPUT_VIDEO_NAME", name)
    return data_dir


def test_find_input_video_explicit_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_data_dir(tmp_path, monkeypatch, name="wanted.avi")
    (data_dir / "wanted.avi").write_bytes(b"stub")
    (data_dir / "other.avi").write_bytes(b"stub")
    assert project_paths.find_input_video().name == "wanted.avi"


def test_find_input_video_explicit_name_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_data_dir(tmp_path, monkeypatch, name="missing.avi")
    with pytest.raises(FileNotFoundError, match="INPUT_VIDEO_NAME"):
        project_paths.find_input_video()


def test_find_input_video_single(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_data_dir(tmp_path, monkeypatch)
    (data_dir / "clip.avi").write_bytes(b"stub")
    assert project_paths.find_input_video().name == "clip.avi"


def test_find_input_video_multiple_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _patch_data_dir(tmp_path, monkeypatch)
    (data_dir / "a.avi").write_bytes(b"stub")
    (data_dir / "b.mp4").write_bytes(b"stub")
    with pytest.raises(ValueError, match="INPUT_VIDEO_NAME"):
        project_paths.find_input_video()


def test_find_input_video_none_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_data_dir(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        project_paths.find_input_video()
