"""
Test `sinethesizer.envelopes.misc` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.envelopes.misc import constant
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "duration, frame_rate, value, expected",
    [
        (0.5, 5, 3, np.array([3, 3, 3])),
    ]
)
def test_constant(
        duration: float, frame_rate: int, value: float, expected: np.ndarray
) -> None:
    """Test `constant` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0.0,
        duration=duration,
        frequency=440,
        velocity=1.0,
        effects='',
        frame_rate=frame_rate
    )
    result = constant(event, value)
    np.testing.assert_equal(result, expected)
