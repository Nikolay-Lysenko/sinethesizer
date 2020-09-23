"""
Test `sinethesizer.envelopes.user_defined` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import numpy as np
import pytest

from sinethesizer.envelopes.user_defined import create_user_defined_envelope
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "parts, ratio_at_zero_velocity, envelope_values_on_velocity_order, "
    "expected",
    [
        (
            # `duration`
            2,
            # `velocity`
            0.375,
            # `frame_rate`
            10,
            # `parts`
            [
                {
                    'values': [0.0, 1.0, 0.9, 0.1, 0.0],
                    'max_duration': None
                },
            ],
            # `ratio_at_zero_velocity`
            0.2,
            # `envelope_values_on_velocity_order`
            1.0,
            # `expected`
            np.array([
                0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.49, 0.48, 0.47, 0.46,
                0.45, 0.35, 0.25, 0.15, 0.05,
                0.04, 0.03, 0.02, 0.01, 0.0
            ])
        ),
        (
            # `duration`
            2,
            # `velocity`
            1,
            # `frame_rate`
            10,
            # `parts`
            [
                {
                    'values': [0.0, 0.5, 1.0, 0.0],
                    'max_duration': None
                },
            ],
            # `ratio_at_zero_velocity`
            0.2,
            # `envelope_values_on_velocity_order`
            1.0,
            # `expected`
            np.array([
                0, 1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12,
                0.5, 8 / 14, 9 / 14, 10 / 14, 11 / 14, 12 / 14, 13 / 14,
                1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 0
            ])
        ),
        (
            # `duration`
            2,
            # `velocity`
            1,
            # `frame_rate`
            10,
            # `parts`
            [
                {
                    'values': [0.0, 0.5, 1.0],
                    'max_duration': 0.2
                },
                {
                    'values': [1.0, 0.7, 0.1],
                    'max_duration': None
                },
            ],
            # `ratio_at_zero_velocity`
            0.2,
            # `envelope_values_on_velocity_order`
            1.0,
            # `expected`
            np.array([
                0.5, 1.0,
                1.0, 0.9625, 0.925, 0.8875, 0.85, 0.8125, 0.775, 0.7375, 0.7,
                19 / 30, 17 / 30, 15 / 30, 13 / 30, 11 / 30, 9 / 30, 7 / 30,
                5 / 30, 3 / 30
            ])
        ),
        (
            # `duration`
            2,
            # `velocity`
            1,
            # `frame_rate`
            10,
            # `parts`
            [
                {
                    'values': [0.0, 0.5, 1.0],
                    'max_duration': None
                },
                {
                    'values': [1.0, 0.7, 0.1],
                    'max_duration': None
                },
            ],
            # `ratio_at_zero_velocity`
            0.2,
            # `envelope_values_on_velocity_order`
            1.0,
            # `expected`
            np.array([
                0, 0.125, 0.25, 0.375, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                1, 0.925, 0.85, 0.775, 0.7, 0.58, 0.46, 0.34, 0.22, 0.1
            ])
        ),
    ]
)
def test_create_user_defined_envelope(
        duration: float, velocity: float, frame_rate: int,
        parts: List[Dict[str, Any]], ratio_at_zero_velocity: float,
        envelope_values_on_velocity_order: float, expected: np.ndarray
) -> None:
    """Test `create_user_defined_envelope` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = create_user_defined_envelope(
        event, parts, ratio_at_zero_velocity, envelope_values_on_velocity_order
    )
    np.testing.assert_almost_equal(result, expected)
