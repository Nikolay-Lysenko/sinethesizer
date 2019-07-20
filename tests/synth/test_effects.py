"""
Test `sinethesizer.synth.effects` module.

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
import pytest

from sinethesizer.synth.effects import (
    frequency_filter, overdrive, tremolo, vibrato
)
from sinethesizer.synth.waves import generate_wave


@pytest.mark.parametrize(
    "frequency, frame_rate, min_frequency, max_frequency, invert, order",
    [
        (440, 44100, 330, 550, True, 10),
        (440, 44100, 880, None, False, 10),
        (440, 44100, None, 220, False, 10),
    ]
)
def test_frequency_filter_with_sine_waves(
        frequency: float, frame_rate: int, min_frequency: float,
        max_frequency: float, invert: bool, order: int
) -> None:
    """Test that `frequency_filter` function removes requested frequencies."""
    sound = generate_wave('sine', frequency, np.ones(frame_rate), frame_rate)
    result = frequency_filter(
        sound, frame_rate, min_frequency, max_frequency, invert, order
    )
    l1_norm = np.sum(np.abs(result))
    assert l1_norm / result.shape[1] < 0.01


@pytest.mark.parametrize(
    "frequencies, frame_rate, min_frequency, max_frequency, invert, order",
    [
        ([220, 440, 660], 44100, 330, 550, False, 10),
        ([220, 440, 660], 44100, 330, 550, True, 10),
    ]
)
def test_frequency_filter_with_arbitrary_input(
        frequencies: List[float], frame_rate: int, min_frequency: float,
        max_frequency: float, invert: bool, order: int
) -> None:
    """Test that `frequency_filter` function returns something finite."""
    waves = [
        generate_wave('sine', frequency, np.ones(frame_rate), frame_rate)
        for frequency in frequencies
    ]
    sound = sum(waves)
    result = frequency_filter(
        sound, frame_rate, min_frequency, max_frequency, invert, order
    )
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "sound, frame_rate, fraction_to_clip, strength, expected",
    [
        (
            np.array([
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            ]),
            10, 0.25, 0,
            np.array([
                [1, 2, 3, 3, 3, 2, 1, 2, 3, 3, 3, 2, 1, 2, 3, 3],
                [1, 2, 3, 3, 3, 2, 1, 2, 3, 3, 3, 2, 1, 2, 3, 3]
            ])
        ),
        (
            np.array([
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            ]),
            10, 3 / 16, 0,
            np.array([
                [1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875],
                [1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875]
            ])
        ),
        (
            np.array([
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            ]),
            10, 0.25, 0.3,
            np.array([
                [
                    1.38095238, 2.47619048, 3, 3,
                    3, 2.47619048, 1.38095238, 2.47619048,
                    3, 3, 3, 2.47619048,
                    1.38095238, 2.47619048, 3, 3
                ],
                [
                    1.38095238, 2.47619048, 3, 3,
                    3, 2.47619048, 1.38095238, 2.47619048,
                    3, 3, 3, 2.47619048,
                    1.38095238, 2.47619048, 3, 3
                ]
            ])
        ),
    ]
)
def test_overdrive(
        sound: np.ndarray, frame_rate: int,
        fraction_to_clip: float, strength: float, expected: np.ndarray
) -> None:
    """Test `overdrive` function."""
    result = overdrive(
        sound, frame_rate, fraction_to_clip, strength
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "sound, frame_rate, frequency, amplitude, expected",
    [
        (
            np.vstack((
                np.arange(12, dtype=float),
                np.arange(12, dtype=float)
            )),
            12, 3, 0.25,
            np.array([
                [0, 1.25, 2, 2.25, 4, 6.25, 6, 5.25, 8, 11.25, 10, 8.25],
                [0, 1.25, 2, 2.25, 4, 6.25, 6, 5.25, 8, 11.25, 10, 8.25]
            ])
        ),
    ]
)
def test_tremolo(
        sound: np.ndarray, frame_rate: int,
        frequency: float, amplitude: float, expected: np.ndarray
) -> None:
    """Test `tremolo` function."""
    result = tremolo(sound, frame_rate, frequency, amplitude)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "sound, frame_rate, frequency, width, expected",
    [
        (
            np.vstack((np.arange(12), np.arange(12))), 12, 4, 2,
            np.array([
                [
                    0, 1.02385798, 1.97614202, 3, 4.02385798, 4.97614202,
                    6, 7.02385798, 7.97614202, 9, 10.02385798, 10.97614202
                ],
                [
                    0, 1.02385798, 1.97614202, 3, 4.02385798, 4.97614202,
                    6, 7.02385798, 7.97614202, 9, 10.02385798, 10.97614202
                ],
            ])
        ),
        (
            np.vstack((np.arange(12), np.arange(12))), 12, 1, 10,
            np.array([
                [
                    0, 1.2683738, 2.46483706, 3.53674761, 4.46483706, 5.2683738,
                    6, 6.7316262, 7.53516294, 8.46325239, 9.53516294, 10.7316262

                ],
                [
                    0, 1.2683738, 2.46483706, 3.53674761, 4.46483706, 5.2683738,
                    6, 6.7316262, 7.53516294, 8.46325239, 9.53516294, 10.7316262
                ],
            ])
        ),
    ]
)
def test_vibrato(
        sound: np.ndarray, frame_rate: int,
        frequency: float, width: float, expected: np.ndarray
) -> None:
    """Test `vibrato` function."""
    result = vibrato(sound, frame_rate, frequency, width)
    np.testing.assert_almost_equal(result, expected)
