"""
Test `sinethesizer.effects.stereo` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.effects.stereo import (
    apply_panning, apply_stereo_delay, apply_stereo_to_mono_conversion
)
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, event, left_amplitude_ratio, right_amplitude_ratio, expected",
    [
        (
            # `sound`
            np.array([
                [1.0, 2, 3],
                [2, 3, 4],
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0.0,
                duration=1.0,
                frequency=440.0,
                velocity=0.0,
                effects='',
                frame_rate=20
            ),
            # `left_amplitude_ratio`
            0.5,
            # `right_amplitude_ratio`
            0.1,
            # `expected`
            np.array([
                [0.5, 1.0, 1.5],
                [0.2, 0.3, 0.4],
            ]),
        ),
    ]
)
def test_apply_panning(
        sound: np.ndarray, event: Event,
        left_amplitude_ratio: float, right_amplitude_ratio: float,
        expected: np.ndarray
) -> None:
    """Test `apply_panning` function."""
    result = apply_panning(sound, event, left_amplitude_ratio, right_amplitude_ratio)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "sound, event, delay, expected",
    [
        (
            # `sound`
            np.array([
                [1, 2, 3],
                [2, 3, 4],
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0.0,
                duration=1.0,
                frequency=440.0,
                velocity=0.0,
                effects='',
                frame_rate=20
            ),
            # `delay`
            -0.05,
            # `expected`
            np.array([
                [1, 2, 3, 0],
                [0, 2, 3, 4],
            ]),
        ),
        (
            # `sound`
            np.array([
                [1, 2, 3],
                [2, 3, 4],
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0.0,
                duration=1.0,
                frequency=440.0,
                velocity=0.0,
                effects='',
                frame_rate=20
            ),
            # `delay`
            0.05,
            # `expected`
            np.array([
                [0, 1, 2, 3],
                [2, 3, 4, 0],
            ]),
        ),
    ]
)
def test_apply_stereo_delay(
        sound: np.ndarray, event: Event, delay: float, expected: np.ndarray
) -> None:
    """Test `apply_stereo_delay` function."""
    result = apply_stereo_delay(sound, event, delay)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "sound, expected",
    [
        (
            # `sound`
            np.array([[1, 2, 3, 4], [-1, 2, -1, 0]]),
            # `expected`
            np.array([[0, 2, 1, 2], [0, 2, 1, 2]])
        ),
    ]
)
def test_apply_stereo_to_mono_conversion(
        sound: np.ndarray, expected: np.ndarray
) -> None:
    """Test `apply_stereo_to_mono_conversion` function."""
    result = apply_stereo_to_mono_conversion(sound, None)
    np.testing.assert_equal(result, expected)
