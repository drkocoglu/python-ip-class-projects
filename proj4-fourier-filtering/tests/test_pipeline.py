"""Tests for the pattern (band-pass) and illumination (high-pass) tasks."""

from __future__ import annotations

import numpy as np

from proj4_ip import fourier as fz
from proj4_ip.pipeline import correct_illumination, extract_pattern


def _low_frequency_energy(image: np.ndarray, radius: float = 3.0) -> float:
    spectrum = fz.fft2(image)
    dist = fz.distance_grid(image.shape)
    near_dc = (dist > 0.0) & (dist <= radius)
    return float(np.sum(np.abs(spectrum[near_dc]) ** 2))


def test_extract_pattern_shape_and_real(input_image: np.ndarray) -> None:
    result = extract_pattern(input_image)
    assert result.pattern.shape == input_image.shape
    assert np.isrealobj(result.pattern)


def test_extract_pattern_is_spectrally_sparse(input_image: np.ndarray) -> None:
    result = extract_pattern(input_image)
    assert 0 < result.kept_count / result.spectrum.size < 0.05


def test_bandpass_removes_illumination_pedestal(input_image: np.ndarray) -> None:
    result = extract_pattern(input_image)
    assert result.band_filter[0, 0] == 0.0
    assert _low_frequency_energy(result.pattern) < 0.05 * _low_frequency_energy(
        input_image
    )


def test_extract_pattern_deterministic(input_image: np.ndarray) -> None:
    assert np.array_equal(
        extract_pattern(input_image).pattern, extract_pattern(input_image).pattern
    )


def test_correct_illumination_reduces_low_frequency_energy(
    input_image: np.ndarray,
) -> None:
    result = correct_illumination(input_image)
    before = _low_frequency_energy(input_image)
    assert _low_frequency_energy(result.corrected) < 0.05 * before


def test_correct_illumination_shapes(input_image: np.ndarray) -> None:
    result = correct_illumination(input_image)
    assert result.corrected.shape == input_image.shape
    assert result.highpass.shape == input_image.shape
