"""
Test `sinethesizer.utils.waves` module.

Author: Nikolay Lysenko
"""


from math import pi, sqrt
from typing import Optional

import pytest
import numpy as np

from sinethesizer.utils.waves import generate_mono_wave


@pytest.mark.parametrize(
    "waveform, frequency, amplitude_envelope, frame_rate, phase, "
    "amplitude_modulator, phase_modulator, expected",
    [
        (
            # `waveform`
            'sine',
            # `frequency`
            1.0,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 2, 2, 2]),
            # `frame_rate`
            8,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([0, 1 / sqrt(2), 1, 1 / sqrt(2), 0, -2 * 1 / sqrt(2), -2])
        ),
        (
            # `waveform`
            'sine',
            # `frequency`
            1.0,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 2, 2, 2]),
            # `frame_rate`
            8,
            # `phase`
            pi,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([0, -1 / sqrt(2), -1, -1 / sqrt(2), 0, 2 * 1 / sqrt(2), 2])
        ),
        (
            # `waveform`
            'sine',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            12,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([0, 0.5, sqrt(3) / 2, 1, sqrt(3) / 2, 0.5, 0, -0.5])
        ),
        (
            # `waveform`
            'square',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 2, 1, 2, 1, 2, 1, 2]),
            # `frame_rate`
            8,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([1.0, 2, 1, 2, -1, -2, -1, -2])
        ),
        (
            # `waveform`
            'triangle',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            8,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5])
        ),
        (
            # `waveform`
            'sawtooth',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            8,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([-1, -0.75, -0.5, -0.25,  0,  0.25,  0.5,  0.75])
        ),
        (
            # `waveform`
            'sawtooth',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            8,
            # `phase`
            0,
            # `amplitude_modulator`
            np.array([0.5, 4, 2, 1, 4, 2, 1, 2]),
            # `phase_modulator`
            None,
            # `expected`
            np.array([-0.5, -3, -1, -0.25, 0, 0.5, 0.5, 1.5])
        ),
        (
            # `waveform`
            'sawtooth',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            8,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            np.array([0, 0, 0, 0, 0, 0, 1, 0]),
            # `expected`
            np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.8183099, 0.75])
        ),
    ]
)
def test_generate_mono_wave(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float,
        amplitude_modulator: Optional[np.ndarray],
        phase_modulator: Optional[np.ndarray], expected: np.ndarray
) -> None:
    """Test `generate_mono_wave` function."""
    result = generate_mono_wave(
        waveform, frequency, amplitude_envelope, frame_rate, phase,
        amplitude_modulator, phase_modulator
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "amplitude_envelope, expected_len",
    [
        (np.array([1, 1, 1, 1]), 4)
    ]
)
def test_generate_mono_wave_with_white_noise(
        amplitude_envelope: np.ndarray, expected_len: int
) -> None:
    """Test `generate_mono_wave` function with white noise."""
    result = generate_mono_wave('white_noise', 440, amplitude_envelope, 8)
    assert len(result) == expected_len
