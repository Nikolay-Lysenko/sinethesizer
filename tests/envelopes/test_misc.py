"""
Test `sinethesizer.envelopes.misc` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.envelopes.misc import create_constant_envelope
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "duration, frame_rate, value, expected",
    [
        (0.5, 5, 3, np.array([3, 3, 3])),
    ]
)
def test_create_constant_envelope(
        duration: float, frame_rate: int, value: float, expected: np.ndarray
) -> None:
    """Test `create_constant_envelope` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0.0,
        duration=duration,
        frequency=440,
        velocity=1.0,
        effects='',
        frame_rate=frame_rate
    )
    result = create_constant_envelope(event, value)
    np.testing.assert_equal(result, expected)
