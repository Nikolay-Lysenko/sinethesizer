"""
Test `sinethesizer.envelopes.ahdsr` module.

Author: Nikolay Lysenko
"""


from typing import Optional

import numpy as np
import pytest

from sinethesizer.envelopes.ahdsr import (
    generic_ahdsr, relative_ahdsr, trapezoid
)
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "attack_to_ahds_max_ratio, max_attack_duration, attack_degree, "
    "hold_to_hds_max_ratio, max_hold_duration, "
    "decay_to_ds_max_ratio, max_decay_duration, decay_degree, "
    "sustain_level, max_sustain_duration, "
    "max_release_duration, release_sensitivity_to_velocity, release_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_sensitivity_to_velocity, "
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
            0.5,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0, 0.25, 0.5, 0.75,
                1.0,
                1.0, 0.9, 0.8, 0.7, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.525, 0.45, 0.375, 0.3, 0.225, 0.15, 0.075
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
            0.5,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0, 0.125, 0.25, 0.375,
                0.5,
                0.5, 0.45, 0.4, 0.35, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.225, 0.15, 0.075
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
            0.0,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0, 0.125, 0.25, 0.375,
                0.5,
                0.5, 0.45, 0.4, 0.35, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.2625, 0.225, 0.1875, 0.15, 0.1125, 0.075, 0.0375
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
            0.5,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            2.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0, 0.5, 1.0, 1.5,
                2.0,
                2.0, 1.8, 1.6, 1.4, 1.2,
                1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                1.2, 1.05, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15
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
            1.0,  # `max_release_duration`
            0.6,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.3,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0, 0.2, 0.4, 0.6, 0.8,
                1.0, 1.0 - 0.4 / 13, 1.0 - 2 * 0.4 / 13, 1.0 - 3 * 0.4 / 13,
                1.0 - 4 * 0.4 / 13, 1.0 - 5 * 0.4 / 13, 1.0 - 6 * 0.4 / 13,
                1.0 - 7 * 0.4 / 13, 1.0 - 8 * 0.4 / 13, 1.0 - 9 * 0.4 / 13,
                1.0 - 10 * 0.4 / 13, 1.0 - 11 * 0.4 / 13, 1.0 - 12 * 0.4 / 13,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.57, 0.54, 0.51, 0.48, 0.45, 0.42, 0.39, 0.36, 0.33,
                0.3, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03
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
            1.0,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.2, 0.1
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
            1.0,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
            ])
        ),
    ]
)
def test_generic_ahdsr(
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
        release_sensitivity_to_velocity: float,
        release_degree: float,
        peak_value: float,
        ratio_at_zero_velocity: float,
        envelope_sensitivity_to_velocity: float,
        expected: np.ndarray
) -> None:
    """Test `generic_ahdsr` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = generic_ahdsr(
        event,
        attack_to_ahds_max_ratio, max_attack_duration, attack_degree,
        hold_to_hds_max_ratio, max_hold_duration,
        decay_to_ds_max_ratio, max_decay_duration, decay_degree,
        sustain_level, max_sustain_duration,
        max_release_duration, release_sensitivity_to_velocity, release_degree,
        peak_value, ratio_at_zero_velocity, envelope_sensitivity_to_velocity
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "attack_to_ahds_ratio, attack_degree, hold_to_ahds_ratio, "
    "decay_to_ahds_ratio, decay_degree, sustain_level, "
    "max_release_duration, release_sensitivity_to_velocity, release_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_sensitivity_to_velocity, "
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
            1.0,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            1.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0, 0.5,
                1.0, 1.0,
                1.0, 0.8,
                0.6, 0.6, 0.6, 0.6,
                0.6, 0.45, 0.3, 0.15
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
            1.0,  # `release_sensitivity_to_velocity`
            1.0,  # `release_degree`
            1.0,  # `peak_value`
            0.0,  # `ratio_at_zero_velocity`
            0.0,  # `envelope_sensitivity_to_velocity`
            np.array([
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.48, 0.36, 0.24, 0.12
            ])
        ),
    ]
)
def test_relative_ahdsr(
        duration: float, velocity: float, frame_rate: int,
        attack_to_ahds_ratio: float, attack_degree: float,
        hold_to_ahds_ratio: float,
        decay_to_ahds_ratio: float, decay_degree: float,
        sustain_level: float,
        max_release_duration: float, release_sensitivity_to_velocity: float,
        release_degree: float,
        peak_value: float, ratio_at_zero_velocity: float,
        envelope_sensitivity_to_velocity: float,
        expected: np.ndarray
) -> None:
    """Test `relative_ahdsr` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = relative_ahdsr(
        event,
        attack_to_ahds_ratio, attack_degree, hold_to_ahds_ratio,
        decay_to_ahds_ratio, decay_degree, sustain_level,
        max_release_duration, release_sensitivity_to_velocity, release_degree,
        peak_value, ratio_at_zero_velocity, envelope_sensitivity_to_velocity
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, velocity, frame_rate, "
    "attack_share, attack_degree, decay_share, decay_degree, "
    "peak_value, ratio_at_zero_velocity, envelope_sensitivity_to_velocity, "
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
            0.0,  # `envelope_sensitivity_to_velocity`
            np.array([0, 0.5, 1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2])
        ),
    ]
)
def test_trapezoid(
        duration: float, velocity: float, frame_rate: int,
        attack_share: float, attack_degree: float,
        decay_share: float, decay_degree: float,
        peak_value: float, ratio_at_zero_velocity: float,
        envelope_sensitivity_to_velocity: float,
        expected: np.ndarray
) -> None:
    """Test `trapezoid` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=duration,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    result = trapezoid(
        event,
        attack_share, attack_degree, decay_share, decay_degree,
        peak_value, ratio_at_zero_velocity, envelope_sensitivity_to_velocity
    )
    np.testing.assert_almost_equal(result, expected)
