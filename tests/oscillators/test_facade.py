"""
Test `sinethesizer.oscillators.facade` module.

Author: Nikolay Lysenko
"""


from math import pi, sqrt
from typing import Optional

import pytest
import numpy as np

from sinethesizer.oscillators.facade import generate_mono_wave


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
            np.array([0.0, 2, 1, 2, 0, -2, -1, -2])
        ),
        (
            # `waveform`
            'square',
            # `frequency`
            10,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            128,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([
                0, 1, 1, 1, 1, 1, 0.64, -0.84, -1, -1, -1, -1, -0.96, 0.36, 1
            ])
        ),
        (
            # `waveform`
            'triangle',
            # `frequency`
            1,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            12,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([
                -0.8444444, -0.6611111, -0.3333333, 0, 0.3333333, 0.6611111,
                0.8444444, 0.6611111, 0.3333333, 0, -0.3333333, -0.6611111
            ])
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
            np.array([0, -0.75, -0.5, -0.25,  0,  0.25,  0.5,  0.75])
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
            np.array([0, -3, -1, -0.25, 0, 0.5, 0.5, 1.5])
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
            np.array([0, -0.75, -0.5, -0.25, 0, 0.25, 0.74365, 0.75])
        ),
        (
            # `waveform`
            'sawtooth',
            # `frequency`
            10,
            # `amplitude_envelope`
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            # `frame_rate`
            128,
            # `phase`
            0,
            # `amplitude_modulator`
            None,
            # `phase_modulator`
            None,
            # `expected`
            np.array([
                0, -0.84375, -0.6875, -0.53125, -0.375, -0.21875,
                -0.0625, 0.09375, 0.25, 0.40625, 0.5625, 0.71875,
                0.835, -0.32875, -0.8125, -0.65625, -0.5, -0.34375
            ])
        ),
    ]
)
def test_generate_mono_wave_with_analog_waveforms(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float,
        amplitude_modulator: Optional[np.ndarray],
        phase_modulator: Optional[np.ndarray], expected: np.ndarray
) -> None:
    """Test analog waveforms produced by `generate_mono_wave` function."""
    result = generate_mono_wave(
        waveform, frequency, amplitude_envelope, frame_rate, phase,
        amplitude_modulator, phase_modulator
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "waveform, amplitude_envelope, expected_len",
    [
        ('karplus_strong', np.ones(100), 100),
    ]
)
def test_generate_mono_wave_with_model_based_waves(
        waveform: str, amplitude_envelope: np.ndarray, expected_len: int
) -> None:
    """Test model-based waves produced by `generate_mono_wave` function."""
    result = generate_mono_wave(waveform, 440, amplitude_envelope, 1024)
    assert len(result) == expected_len


@pytest.mark.parametrize(
    "waveform, amplitude_envelope, expected_len",
    [
        ('white_noise', np.array([1, 1, 1, 1]), 4),
        ('pink_noise', np.array([1, 1, 1, 1]), 4),
    ]
)
def test_generate_mono_wave_with_noises(
        waveform: str, amplitude_envelope: np.ndarray, expected_len: int
) -> None:
    """Test noises produced by `generate_mono_wave` function."""
    result = generate_mono_wave(waveform, 440, amplitude_envelope, 1024)
    assert len(result) == expected_len
