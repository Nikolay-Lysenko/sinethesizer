"""
Test `sinethesizer.synth.timeline` module.

Author: Nikolay Lysenko
"""


from math import sqrt
from typing import Any, Dict

import numpy as np
import pytest

from sinethesizer.synth.envelopes import trapezoid
from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.synth.timeline import add_event_to_timeline


@pytest.mark.parametrize(
    "timeline, event, timbres_registry, max_channel_delay, frame_rate, "
    "expected",
    [
        (
            # `timeline`
            np.array([
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, -4, -5, -6, -7, -8, -9]
            ]),
            # `event`
            {
                'timbre': 'sine',
                'start_time': 2,
                'duration': 1,
                'frequency': 1,
                'volume': 1,
                'location': 0,
                'effects': {},
            },
            # `timbres_registry`
            {'sine': TimbreSpec('sine', trapezoid, [], [])},
            # `max_channel_delay`
            0.01,
            # `frame_rate`
            4,
            # `expected`
            np.array([
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 1 / sqrt(2), 0, -1 / sqrt(2)],
                [1, 2, 3, -4, -5, -6, -7, -8, -9, 1 / sqrt(2), 0, -1 / sqrt(2)]
            ])
        ),
    ]
)
def test_add_event_to_timeline(
        timeline: np.ndarray, event: Dict[str, Any],
        timbres_registry: Dict[str, TimbreSpec], max_channel_delay: float,
        frame_rate: int, expected: np.ndarray
) -> None:
    """Test `add_event_to_timeline` function."""
    result = add_event_to_timeline(
        timeline, event, timbres_registry, max_channel_delay, frame_rate
    )
    np.testing.assert_almost_equal(result, expected)
