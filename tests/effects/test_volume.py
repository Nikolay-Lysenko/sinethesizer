"""
Test `sinethesizer.effects.volume` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.effects.volume import apply_amplitude_normalization, apply_compressor
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, velocity, value_at_max_velocity, quantile, "
    "value_on_velocity_order, value_at_zero_velocity, expected",
    [
        (
            # `sound`
            np.array([
                [1.0, 2, 3, 4, 3, 2, 1, 2],
                [-1, -1, -1, -1, -1, -1, -1, -1]
            ]),
            # `velocity`
            1,
            # `value_at_max_velocity`
            6,
            # `quantile`
            7 / 8,
            # `value_on_velocity_order`
            0.5,
            # `value_at_zero_velocity`
            1,
            # `expected`
            np.array([
                [2, 4, 6, 8, 6, 4, 2, 4],
                [-2, -2, -2, -2, -2, -2, -2, -2]
            ])
        ),
    ]
)
def test_apply_amplitude_normalization(
        sound: np.ndarray, velocity: float, value_at_max_velocity: float,
        quantile: float, value_on_velocity_order: float,
        value_at_zero_velocity: float, expected: np.ndarray
) -> None:
    """Test `apply_amplitude_normalization` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=8
    )
    result = apply_amplitude_normalization(
        sound, event, value_at_max_velocity, quantile, value_on_velocity_order,
        value_at_zero_velocity
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "sound, frame_rate, frequency, threshold, frame_size_in_cycles, expected",
    [
        (
            # `sound`
            np.array([
                [0.1, 0.3, 0.5, 0.7, 1.0, 0.8, 0.8, 0.6, 0.5, 0.1],
                [0.01, 0.03, 0.05, 0.07, 0.1, 0.08, 0.08, 0.06, 0.05, 0.01],
            ]),
            # `frame_rate`
            9,
            # `frequency`
            3,
            # `threshold`
            0.7,
            # `frame_size_in_cycles`
            1,
            # `expected`
            np.array([
                [0.1, 0.3, 0.5, 0.7, 1.0, 0.72, 0.64, 0.42, 0.4, 0.09],
                [0.01, 0.03, 0.05, 0.07, 0.1, 0.072, 0.064, 0.042, 0.04, 0.009],
            ])
        ),
    ]
)
def test_apply_compressor(
        sound: np.ndarray, frame_rate: int, frequency: float, threshold: float,
        frame_size_in_cycles: float, expected: np.ndarray
) -> None:
    """Test `apply_compressor` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=sound.shape[1] / frame_rate,
        frequency=frequency,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_compressor(sound, event, threshold, frame_size_in_cycles)
    np.testing.assert_almost_equal(result, expected)
