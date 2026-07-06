"""The two Project 4 tasks, in the frequency domain.

Task 1 - Extract the periodic pattern   ->  BAND-PASS + spike threshold
Task 2 - Correct non-uniform illumination  ->  HIGH-PASS

Tunable numbers come from ``config.py`` in this package (used as the default
arguments below), so retuning happens there. Every function still accepts
explicit arguments, so callers can override a value programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from . import fourier as fz
from .filters import band_pass, high_pass, low_pass


@dataclass
class PatternResult:
    """Artifacts from the periodic-pattern extraction task (band-pass)."""

    pattern: np.ndarray
    spectrum: np.ndarray
    band_filter: np.ndarray
    low_component: np.ndarray
    high_component: np.ndarray
    filtered_spectrum: np.ndarray
    kept_count: int


@dataclass
class IlluminationResult:
    """Artifacts from the illumination-correction task (high-pass)."""

    corrected: np.ndarray
    spectrum: np.ndarray
    highpass: np.ndarray
    lowpass: np.ndarray
    filtered_spectrum: np.ndarray


def extract_pattern(
    image: np.ndarray,
    inner_cutoff: float = config.PATTERN_INNER_CUTOFF,
    outer_cutoff: float = config.PATTERN_OUTER_CUTOFF,
    order: int = config.PATTERN_ORDER,
    log_threshold: float = config.PATTERN_LOG_THRESHOLD,
    attenuation: float = config.PATTERN_ATTENUATION,
) -> PatternResult:
    """Isolate the periodic pattern with a band-pass, then a spike threshold."""
    shape = image.shape
    spectrum = fz.fft2(image)

    band_filter = band_pass(shape, inner_cutoff, outer_cutoff, order)
    banded = band_filter * spectrum

    filtered = fz.apply_threshold(banded, log_threshold, attenuation)
    pattern = fz.ifft2_real(filtered)

    kept = fz.spectral_threshold_mask(banded, log_threshold)
    return PatternResult(
        pattern=pattern,
        spectrum=spectrum,
        band_filter=band_filter,
        low_component=low_pass(shape, outer_cutoff, order),
        high_component=high_pass(shape, inner_cutoff, order),
        filtered_spectrum=filtered,
        kept_count=int(kept.sum()),
    )


def correct_illumination(
    image: np.ndarray,
    cutoff: float = config.ILLUMINATION_CUTOFF,
    order: int = config.ILLUMINATION_ORDER,
) -> IlluminationResult:
    """Flatten non-uniform illumination with a Butterworth high-pass filter."""
    shape = image.shape
    spectrum = fz.fft2(image)
    highpass = high_pass(shape, cutoff, order)
    filtered = highpass * spectrum
    corrected = fz.ifft2_real(filtered)
    return IlluminationResult(
        corrected=corrected,
        spectrum=spectrum,
        highpass=highpass,
        lowpass=low_pass(shape, cutoff, order),
        filtered_spectrum=filtered,
    )
