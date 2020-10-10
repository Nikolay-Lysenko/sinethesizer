"""
Test `sinethesizer.effects.stereo` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.effects.stereo import apply_haas_effect, apply_panning
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, event, location, max_channel_delay, expected",
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
            # `location`
            -0.5,
            # `max_channel_delay`
            0.1,
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
            # `location`
            0.5,
            # `max_channel_delay`
            0.1,
            # `expected`
            np.array([
                [0, 1, 2, 3],
                [2, 3, 4, 0],
            ]),
        ),
    ]
)
def test_apply_haas_effect(
        sound: np.ndarray, event: Event,
        location: float, max_channel_delay: float,
        expected: np.ndarray
) -> None:
    """Test `apply_haas_effect` function."""
    result = apply_haas_effect(sound, event, location, max_channel_delay)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "sound, event, left_volume_ratio, right_volume_ratio, expected",
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
            # `left_volume_ratio`
            0.5,
            # `right_volume_ratio`
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
        left_volume_ratio: float, right_volume_ratio: float,
        expected: np.ndarray
) -> None:
    """Test `apply_panning` function."""
    result = apply_panning(sound, event, left_volume_ratio, right_volume_ratio)
    np.testing.assert_almost_equal(result, expected)
