"""
Test `sinethesizer.utils.waves` module.

Author: Nikolay Lysenko
"""


from math import pi, sqrt, tan

import pytest
import numpy as np

from sinethesizer.utils.waves import generate_mono_wave, generate_stereo_wave


@pytest.mark.parametrize(
    "form, frequency, amplitudes, frame_rate, expected",
    [
        (
            'sine', 1, np.array([1, 1, 1, 1, 2, 2, 2]), 8,
            np.array([0, 1 / sqrt(2), 1, 1 / sqrt(2), 0, -2 * 1 / sqrt(2), -2])
        ),
        (
            'sine', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 12,
            np.array([0, 0.5, sqrt(3) / 2, 1, sqrt(3) / 2, 0.5, 0, -0.5])
        ),
        (
            'square', 1, np.array([1, 2, 1, 2, 1, 2, 1, 2]), 8,
            np.array([1.0, 2, 1, 2, -1, -2, -1, -2])
        ),
        (
            'triangle', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 8,
            np.array([-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5])
        ),
        (
            'sawtooth', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 8,
            np.array([-1, -0.75, -0.5, -0.25,  0,  0.25,  0.5,  0.75])
        ),
    ]
)
def test_generate_mono_wave(
        form: str, frequency: float, amplitudes: np.ndarray, frame_rate: int,
        expected: np.ndarray
) -> None:
    """Test `generate_mono_wave` function."""
    result = generate_mono_wave(form, frequency, amplitudes, frame_rate)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "form, frequency, amplitudes, frame_rate, location, max_channel_delay, "
    "expected",
    [
        (
            'sine', 1, np.array([1, 1, 1, 1, 2, 2, 2]), 8, -1, 0.125,
            np.array([
                [0, 1 / sqrt(2), 1, 1 / sqrt(2), 0, -2 * 1 / sqrt(2), -2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ])
        ),
        (
            'sine', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 12, 1, 0.25,
            np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.5, sqrt(3) / 2, 1, sqrt(3) / 2, 0.5, 0, -0.5, 0, 0, 0],
            ])
        ),
        (
            'sine', 1, np.array([1, 1, 1, 1]), 12, 0.5, 0.5,
            np.array([
                [
                    sqrt(1 / (1 + 1.1 ** (2 * tan(pi / 2 * 0.5)))) * x
                    for x in [0, 0, 0, 0, 0.5, sqrt(3) / 2, 1]
                ],
                [
                    sqrt(1.1 ** (2 * tan(pi / 2 * 0.5)) / (1 + 1.1 ** (2 * tan(pi / 2 * 0.5)))) * x
                    for x in [0, 0.5, sqrt(3) / 2, 1, 0, 0, 0]
                ],
            ])
        ),
    ]
)
def test_generate_stereo_wave(
        form: str, frequency: float, amplitudes: np.ndarray, frame_rate: int,
        location: float, max_channel_delay: float, expected: np.ndarray
) -> None:
    """Test `generate_stereo_wave` function."""
    result = generate_stereo_wave(
        form, frequency, amplitudes, frame_rate, location, max_channel_delay
    )
    np.testing.assert_almost_equal(result, expected)
