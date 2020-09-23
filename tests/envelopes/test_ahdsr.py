"""
Test `sinethesizer.envelopes.ahdsr` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.envelopes.ahdsr import (
    create_generic_ahdsr_envelope,
    create_relative_ahdsr_envelope,
    create_trapezoid_envelope
)
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "attack_to_ahds_max_ratio, max_attack_duration, attack_degree, "
    "hold_to_hds_max_ratio, max_hold_duration, "
    "decay_to_ds_max_ratio, max_decay_duration, decay_degree, "
    "sustain_level, max_sustain_duration, "
    "max_release_duration, release_duration_on_velocity_order, "
    "release_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order, "
    "expected",
    [
        (
            1.0,  # `duration`
            1.0,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.1,  # `hold_to_hds_max_ratio`
            0.05,  # `max_hold_duration`
            0.3,  # `decay_to_ds_max_ratio`
            0.25,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.4,  # `max_release_duration`
            0.5,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 1 / 3, 2 / 3, 1.0,
                # Hold
                1.0,
                # Decay
                1.0, 1 - 0.4 / 3, 1 - 0.8 / 3, 0.6,
                # Sustain
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                # Release
                0.6, 6 / 7 * 0.6, 5 / 7 * 0.6, 4 / 7 * 0.6, 3 / 7 * 0.6,
                2 / 7 * 0.6, 1 / 7 * 0.6, 0.0
            ])
        ),
        (
            1.0,  # `duration`
            2 / 7,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.1,  # `hold_to_hds_max_ratio`
            0.05,  # `max_hold_duration`
            0.3,  # `decay_to_ds_max_ratio`
            0.25,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.4,  # `max_release_duration`
            0.5,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 1 / 6, 1 / 3, 0.5,
                # Hold
                0.5,
                # Decay
                0.5, 0.5 - 0.2 / 3, 0.5 - 0.4 / 3, 0.3,
                # Sustain
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                # Release
                0.3, 0.2, 0.1, 0.0
            ])
        ),
        (
            1.0,  # `duration`
            2 / 7,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.1,  # `hold_to_hds_max_ratio`
            0.05,  # `max_hold_duration`
            0.3,  # `decay_to_ds_max_ratio`
            0.25,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.4,  # `max_release_duration`
            0.0,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 1 / 6, 1 / 3, 0.5,
                # Hold
                0.5,
                # Decay
                0.5, 0.5 - 0.2 / 3, 0.5 - 0.4 / 3, 0.3,
                # Sustain
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                # Release
                0.3, 6 / 7 * 0.3, 5 / 7 * 0.3, 4 / 7 * 0.3, 3 / 7 * 0.3,
                2 / 7 * 0.3, 1 / 7 * 0.3, 0.0
            ])
        ),
        (
            1.0,  # `duration`
            1.0,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.1,  # `hold_to_hds_max_ratio`
            0.05,  # `max_hold_duration`
            0.3,  # `decay_to_ds_max_ratio`
            0.25,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.4,  # `max_release_duration`
            0.5,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            2.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 2 / 3, 4 / 3, 2.0,
                # Hold
                2.0,
                # Decay
                2.0, 2 - 0.8 / 3, 2 - 1.6 / 3, 1.2,
                # Sustain
                1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                # Release
                1.2, 6 / 7 * 1.2, 5 / 7 * 1.2, 4 / 7 * 1.2, 3 / 7 * 1.2,
                2 / 7 * 1.2, 1 / 7 * 1.2, 0.0
            ])
        ),
        (
            3.0,  # `duration`
            1.0,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.0,  # `hold_to_hds_max_ratio`
            0.0,  # `max_hold_duration`
            0.25,  # `decay_to_ds_max_ratio`
            1.0,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.5,  # `max_sustain_duration`
            1.05,  # `max_release_duration`
            0.6,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 0.25, 0.5, 0.75, 1.0,
                # No hold
                # Decay
                1.0, 1.0 - 0.1 / 3, 1.0 - 2 * 0.1 / 3, 1.0 - 3 * 0.1 / 3,
                1.0 - 4 * 0.1 / 3, 1.0 - 5 * 0.1 / 3, 1.0 - 6 * 0.1 / 3,
                1.0 - 7 * 0.1 / 3, 1.0 - 8 * 0.1 / 3, 1.0 - 9 * 0.1 / 3,
                1.0 - 10 * 0.1 / 3, 1.0 - 11 * 0.1 / 3, 1.0 - 12 * 0.1 / 3,
                # Sustain
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                # Release
                0.6, 0.57, 0.54, 0.51, 0.48, 0.45, 0.42, 0.39, 0.36, 0.33,
                0.3, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03, 0.0
            ])
        ),
        (
            1.0,  # `duration`
            0.5,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.0,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.0,  # `hold_to_hds_max_ratio`
            0.0,  # `max_hold_duration`
            0.25,  # `decay_to_ds_max_ratio`
            0.0,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.3,  # `max_release_duration`
            1.0,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Sustain
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                # Release
                0.3, 0.15, 0.0
            ])
        ),
        (
            1.0,  # `duration`
            0.1,  # `velocity`
            20,  # `frame_rate`
            0.2,  # `attack_to_ahds_max_ratio`
            0.0,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.0,  # `hold_to_hds_max_ratio`
            0.0,  # `max_hold_duration`
            0.25,  # `decay_to_ds_max_ratio`
            0.0,  # `max_decay_duration`
            1.0,  # `decay_degree`
            1.0,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.3,  # `max_release_duration`
            1.0,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Sustain
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ])
        ),
    ]
)
def test_create_generic_ahdsr_envelope(
        duration: float, velocity: float, frame_rate: int,
        attack_to_ahds_max_ratio: float,
        max_attack_duration: float,
        attack_degree: float,
        hold_to_hds_max_ratio: float,
        max_hold_duration: float,
        decay_to_ds_max_ratio: float,
        max_decay_duration: float,
        decay_degree: float,
        sustain_level: float,
        max_sustain_duration: float,
        max_release_duration: float,
        release_duration_on_velocity_order: float,
        release_degree: float,
        peak_value: float,
        ratio_at_zero_velocity: float,
        envelope_values_on_velocity_order: float,
        expected: np.ndarray
) -> None:
    """Test `create_generic_ahdsr_envelope` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = create_generic_ahdsr_envelope(
        event,
        attack_to_ahds_max_ratio, max_attack_duration, attack_degree,
        hold_to_hds_max_ratio, max_hold_duration,
        decay_to_ds_max_ratio, max_decay_duration, decay_degree,
        sustain_level, max_sustain_duration,
        max_release_duration, release_duration_on_velocity_order,
        release_degree,
        peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "attack_to_ahds_ratio, attack_degree, hold_to_ahds_ratio, "
    "decay_to_ahds_ratio, decay_degree, sustain_level, "
    "max_release_duration, release_duration_on_velocity_order, "
    "release_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order, "
    "expected",
    [
        (
            1.0,  # `duration`
            1.0,  # `velocity`
            10,  # `frame_rate`
            0.2,  # `attack_to_ahds_ratio`
            1.0,  # `attack_degree`,
            0.2,  # `hold_to_ahds_ratio`
            0.2,  # `decay_to_ahds_ratio`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            0.4,  # `max_release_duration`
            1.0,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 1.0,
                # Hold
                1.0, 1.0,
                # Decay
                1.0, 0.6,
                # Sustain
                0.6, 0.6, 0.6, 0.6,
                # Release
                0.6, 0.4, 0.2, 0.0
            ])
        ),
        (
            1.0,  # `duration`
            0.5,  # `velocity`
            10,  # `frame_rate`
            0.0,  # `attack_to_ahds_ratio`
            1.0,  # `attack_degree`,
            0.0,  # `hold_to_ahds_ratio`
            0.0,  # `decay_to_ahds_ratio`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_release_duration`
            1.0,  # `release_duration_on_velocity_order`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            0.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Sustain
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                # Release
                0.6, 0.45, 0.3, 0.15, 0.0
            ])
        ),
    ]
)
def test_create_relative_ahdsr_envelope(
        duration: float, velocity: float, frame_rate: int,
        attack_to_ahds_ratio: float, attack_degree: float,
        hold_to_ahds_ratio: float,
        decay_to_ahds_ratio: float, decay_degree: float,
        sustain_level: float,
        max_release_duration: float, release_duration_on_velocity_order: float,
        release_degree: float,
        peak_value: float, ratio_at_zero_velocity: float,
        envelope_values_on_velocity_order: float,
        expected: np.ndarray
) -> None:
    """Test `create_relative_ahdsr_envelope` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = create_relative_ahdsr_envelope(
        event,
        attack_to_ahds_ratio, attack_degree, hold_to_ahds_ratio,
        decay_to_ahds_ratio, decay_degree, sustain_level,
        max_release_duration, release_duration_on_velocity_order,
        release_degree,
        peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "attack_share, attack_degree, decay_share, decay_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order, "
    "expected",
    [
        (
            1.0,  # `duration`
            1.0,  # `velocity`
            10,  # `frame_rate`
            0.2,  # `attack_share`
            1.0,  # `attack_degree`
            0.5,  # `decay_share`
            1.0,  # `decay_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            0.0,  # `envelope_values_on_velocity_order`
            np.array([
                # Attack
                0, 1.0,
                # Hold
                1.0, 1.0, 1.0,
                # Decay
                1.0, 0.75, 0.5, 0.25, 0.0
            ])
        ),
    ]
)
def test_create_trapezoid_envelope(
        duration: float, velocity: float, frame_rate: int,
        attack_share: float, attack_degree: float,
        decay_share: float, decay_degree: float,
        peak_value: float, ratio_at_zero_velocity: float,
        envelope_values_on_velocity_order: float,
        expected: np.ndarray
) -> None:
    """Test `create_trapezoid_envelope` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = create_trapezoid_envelope(
        event,
        attack_share, attack_degree, decay_share, decay_degree,
        peak_value, ratio_at_zero_velocity, envelope_values_on_velocity_order
    )
    np.testing.assert_almost_equal(result, expected)
