"""
Test `sinethesizer.synth.waves` module.

Author: Nikolay Lysenko
"""


from math import sqrt

import pytest
import numpy as np

from sinethesizer.synth.waves import generate_wave


@pytest.mark.parametrize(
    "form, frequency, amplitudes, frame_rate, location, max_channel_delay, "
    "expected",
    [
        (
            'sine', 1, np.array([1, 1, 1, 1, 2, 2, 2]), 8, 0, 10,
            np.array([
                [0, 1 / sqrt(2), 1, 1 / sqrt(2), 0, -2 * 1 / sqrt(2), -2],
                [0, 1 / sqrt(2), 1, 1 / sqrt(2), 0, -2 * 1 / sqrt(2), -2],
            ])
        ),
        (
            'sine', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 12, 0, 10,
            np.array([
                [0, 0.5, sqrt(3) / 2, 1, sqrt(3) / 2, 0.5, 0, -0.5],
                [0, 0.5, sqrt(3) / 2, 1, sqrt(3) / 2, 0.5, 0, -0.5],
            ])
        ),
        (
            'sine', 1, np.array([1, 1, 1, 1]), 12, 0.5, 0.5,
            np.array([
                [0, 0, 0, 0, 0.25, sqrt(3) / 4, 0.5],
                [0, 0.75, 1.5 * sqrt(3) / 2, 1.5, 0, 0, 0],
            ])
        ),
        (
            'square', 1, np.array([1, 2, 1, 2, 1, 2, 1, 2]), 8, 0, 10,
            np.array([
                [1.0, 2, 1, 2, -1, -2, -1, -2],
                [1.0, 2, 1, 2, -1, -2, -1, -2],
            ])
        ),
        (
            'triangle', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 8, -0.5, 0.25,
            np.array([
                [-1.5, -0.75, 0, 0.75, 1.5, 0.75, 0, -0.75, 0],
                [0, -0.5, -0.25, 0, 0.25, 0.5, 0.25, 0, -0.25],
            ])
        ),
        (
            'sawtooth', 1, np.array([1, 1, 1, 1, 1, 1, 1, 1]), 8, 0, 10,
            np.array([
                [-1, -0.75, -0.5, -0.25,  0,  0.25,  0.5,  0.75],
                [-1, -0.75, -0.5, -0.25,  0,  0.25,  0.5,  0.75],
            ])
        ),
    ]
)
def test_generate_wave(
        form: str, frequency: float, amplitudes: np.ndarray, frame_rate: int,
        location: float, max_channel_delay: float, expected: np.ndarray
) -> None:
    """Test `generate_wave` function."""
    result = generate_wave(
        form, frequency, amplitudes, frame_rate, location, max_channel_delay
    )
    np.testing.assert_almost_equal(result, expected)
