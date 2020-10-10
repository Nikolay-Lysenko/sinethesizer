"""
Test `sinethesizer.effects.volume` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.effects.volume import apply_amplitude_normalization
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
