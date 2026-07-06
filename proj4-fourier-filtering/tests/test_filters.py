"""Tests for the filters and the Fourier helpers."""

from __future__ import annotations

import numpy as np
import pytest

from proj4_ip import fourier as fz
from proj4_ip.filters import band_pass, high_pass, low_pass

SHAPE = (32, 40)


def test_ifft2_of_fft2_recovers_image() -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(size=SHAPE)
    assert np.allclose(fz.ifft2_real(fz.fft2(image)), image, atol=1e-9)


def test_distance_grid_dc_is_unique_minimum() -> None:
    dist = fz.distance_grid((16, 20))
    assert dist[0, 0] == 0.0
    assert np.count_nonzero(dist == 0.0) == 1


def test_apply_threshold_keeps_strong_attenuates_weak() -> None:
    spectrum = np.array([[1000.0, 1.0], [1.0, 2.0]], dtype=complex)
    out = fz.apply_threshold(spectrum, log_threshold=5.0, attenuation=0.0)
    assert out[0, 0] == spectrum[0, 0]
    assert out[0, 1] == 0.0


def test_lowpass_unity_at_dc_and_monotonic() -> None:
    low = low_pass(SHAPE, cutoff=5.0, order=2)
    assert low[0, 0] == pytest.approx(1.0)
    dist = fz.distance_grid(SHAPE)
    gains = low.ravel()[np.argsort(dist.ravel())]
    assert np.all(np.diff(gains) <= 1e-12)


def test_highpass_blocks_dc_and_complements_lowpass() -> None:
    low = low_pass(SHAPE, cutoff=5.0, order=2)
    high = high_pass(SHAPE, cutoff=5.0, order=2)
    assert high[0, 0] == 0.0
    expected = 1.0 - low
    expected[0, 0] = 0.0
    assert np.allclose(high, expected, atol=1e-9)


def test_bandpass_is_product_of_low_and_high() -> None:
    band = band_pass(SHAPE, inner_cutoff=6.0, outer_cutoff=40.0, order=2)
    expected = high_pass(SHAPE, 6.0, 2) * low_pass(SHAPE, 40.0, 2)
    assert np.allclose(band, expected, atol=1e-12)


def test_bandpass_blocks_dc_and_passes_the_ring() -> None:
    band = band_pass((64, 64), inner_cutoff=6.0, outer_cutoff=15.0, order=2)
    dist = fz.distance_grid((64, 64))
    assert band[0, 0] == 0.0  # DC removed (illumination gone)
    assert band[(dist > 8) & (dist < 13)].mean() > 0.5  # pattern band passed
    assert band[dist > 35].mean() < 0.1  # noise floor removed
