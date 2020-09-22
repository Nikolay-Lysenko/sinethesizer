"""
Test `sinethesizer.effects.vibrato` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict

import numpy as np
import pytest

from sinethesizer.effects.vibrato import apply_vibrato
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, frame_rate, sound_frequency, kind, kwargs, expected",
    [
        (
            # `sound`
            np.vstack((np.arange(12), np.arange(12))),
            # `frame_rate`
            12,
            # `sound_frequency`
            None,
            # `kind`
            'absolute',
            # `kwargs`
            {'frequency': 4, 'width': 2},
            # `expected`
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
            # `sound`
            np.vstack((np.arange(12), np.arange(12))),
            # `frame_rate`
            12,
            # `sound_frequency`
            1,
            # `kind`
            'relative',
            # `kwargs`
            {'frequency_ratio': 1, 'width': 10},
            # `expected`
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
def test_apply_vibrato(
        sound: np.ndarray, frame_rate: int, sound_frequency: float, kind: str,
        kwargs: Dict[str, Any], expected: np.ndarray
) -> None:
    """Test `apply_vibrato` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=sound_frequency,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_vibrato(sound, event, kind, **kwargs)
    np.testing.assert_almost_equal(result, expected)
