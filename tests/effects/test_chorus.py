"""
Test `sinethesizer.effects.chorus` module.

Author: Nikolay Lysenko
"""


from typing import Any

import numpy as np
import pytest

from sinethesizer.effects.chorus import apply_chorus
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, frame_rate, original_sound_gain, copies_params, expected",
    [
        (
            # `sound`
            np.array([
                [1.0, 2, 3, 4],
                [5, 6, 7, 8]
            ]),
            # `frame_rate`
            4,
            # `original_sound_gain`
            0.5,
            # `copies_params`
            [
                {'delay': 0.5, 'gain': 1.0, 'width': 0},
                {'delay': 0.25, 'gain': 0.25, 'width': 0},
            ],
            # `expected`
            np.array([
                [0.5, 1.25, 3, 4.75, 4, 4],
                [2.5, 4.25, 10, 11.75, 9, 8],
            ])
        )
    ]
)
def test_apply_chorus(
        sound: np.ndarray, frame_rate: int, original_sound_gain: float,
        copies_params: list[dict[str, Any]], expected: np.ndarray
) -> None:
    """Test `apply_chorus` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=1,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_chorus(sound, event, original_sound_gain, copies_params)
    np.testing.assert_equal(result, expected)
