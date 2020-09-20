"""
Test `sinethesizer.effects.overdrive` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.effects.overdrive import apply_overdrive
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, frame_rate, fraction_to_clip, strength, expected",
    [
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            ]),
            # `frame_rate`
            10,
            # `fraction_to_clip`
            0.25,
            # `strength`
            0,
            # `expected`
            np.array([
                [1, 2, 3, 3, 3, 2, 1, 2, 3, 3, 3, 2, 1, 2, 3, 3],
                [1, 2, 3, 3, 3, 2, 1, 2, 3, 3, 3, 2, 1, 2, 3, 3]
            ])
        ),
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            ]),
            # `frame_rate`
            10,
            # `fraction_to_clip`
            3 / 16,
            # `strength`
            0,
            # `expected`
            np.array([
                [1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875],
                [1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875, 3, 2, 1, 2, 3, 3.1875]
            ])
        ),
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            ]),
            # `frame_rate`
            10,
            # `fraction_to_clip`
            0.25,
            # `strength`
            0.3,
            # `expected`
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
def test_apply_overdrive(
        sound: np.ndarray, frame_rate: int,
        fraction_to_clip: float, strength: float, expected: np.ndarray
) -> None:
    """Test `apply_overdrive` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=440,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_overdrive(sound, event, fraction_to_clip, strength)
    np.testing.assert_almost_equal(result, expected)
