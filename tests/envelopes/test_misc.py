"""
Test `sinethesizer.envelopes.misc` module.

Author: Nikolay Lysenko
"""


from typing import Optional

import numpy as np
import pytest

from sinethesizer.envelopes.misc import (
    create_constant_envelope, create_exponentially_decaying_envelope
)
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


@pytest.mark.parametrize(
    "duration, frame_rate, "
    "attack_to_ad_max_ratio, max_attack_duration, attack_degree, "
    "decay_half_life, decay_half_life_ratio, "
    "max_release_duration, release_duration_on_velocity_order, release_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order, "
    "expected",
    [
        (
            # `duration`
            1,
            # `frame_rate`
            8,
            # `attack_to_ad_max_ratio`
            3 / 8,
            # max_attack_duration`
            1,
            # `attack_degree`
            1,
            # `decay_half_life`
            None,
            # `decay_half_life_ratio`
            0.25,
            # `max_release_duration`
            0,
            # `release_duration_on_velocity_order`
            1,
            # `release_degree`
            1,
            # `peak_value`
            1,
            # `ratio_at_zero_velocity`
            0,
            # `envelope_values_on_velocity_order`
            1,
            # `expected`
            np.array([
                # Attack
                0, 0.5, 1.0,
                # Decay
                1.0, 0.5, 0.25, 0.125, 0.0625
            ])
        ),
        (
            # `duration`
            1,
            # `frame_rate`
            9,
            # `attack_to_ad_max_ratio`
            0,
            # max_attack_duration`
            0,
            # `attack_degree`
            1,
            # `decay_half_life`
            None,
            # `decay_half_life_ratio`
            0.25,
            # `max_release_duration`
            0,
            # `release_duration_on_velocity_order`
            1,
            # `release_degree`
            1,
            # `peak_value`
            1,
            # `ratio_at_zero_velocity`
            0,
            # `envelope_values_on_velocity_order`
            1,
            # `expected`
            np.array([
                # Decay
                1.0, 2 ** (-1/2), 0.5, 0.5 * 2 ** (-1/2), 0.25,
                0.25 * 2 ** (-1/2), 0.125, 0.125 * 2 ** (-1/2), 0.0625
            ])
        ),
        (
            # `duration`
            1,
            # `frame_rate`
            8,
            # `attack_to_ad_max_ratio`
            3 / 8,
            # max_attack_duration`
            1,
            # `attack_degree`
            1,
            # `decay_half_life`
            5 / 8,
            # `decay_half_life_ratio`
            None,
            # `max_release_duration`
            0.7,
            # `release_duration_on_velocity_order`
            1,
            # `release_degree`
            1,
            # `peak_value`
            1,
            # `ratio_at_zero_velocity`
            0,
            # `envelope_values_on_velocity_order`
            1,
            # `expected`
            np.array([
                # Attack
                0, 0.5, 1.0,
                # Decay
                1, 0.84089642, 0.70710678, 0.59460356, 0.5,
                # Release
                0.5, 0.375, 0.25, 0.125, 0
            ])
        ),
    ]
)
def test_create_exponentially_decaying_envelope(
        duration: float, frame_rate: int, attack_to_ad_max_ratio: float,
        max_attack_duration: float, attack_degree: float,
        decay_half_life: Optional[float],
        decay_half_life_ratio: Optional[float], max_release_duration: float,
        release_duration_on_velocity_order: float, release_degree: float,
        peak_value: float, ratio_at_zero_velocity: float,
        envelope_values_on_velocity_order: float, expected: np.ndarray
) -> None:
    """Test `create_exponentially_decaying_envelope` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0.0,
        duration=duration,
        frequency=440,
        velocity=1.0,
        effects='',
        frame_rate=frame_rate
    )
    result = create_exponentially_decaying_envelope(
        event, attack_to_ad_max_ratio, max_attack_duration, attack_degree,
        decay_half_life, decay_half_life_ratio, max_release_duration,
        release_duration_on_velocity_order, release_degree, peak_value,
        ratio_at_zero_velocity, envelope_values_on_velocity_order
    )
    np.testing.assert_almost_equal(result, expected)
