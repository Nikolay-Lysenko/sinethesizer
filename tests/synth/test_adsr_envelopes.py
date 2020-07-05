"""
Test `sinethesizer.synth.adsr_envelopes` module.

Author: Nikolay Lysenko
"""


from typing import Optional

import numpy as np
import pytest

from sinethesizer.synth.adsr_envelopes import (
    generic_adsr, relative_adsr, spike, trapezoid
)


@pytest.mark.parametrize(
    "duration, frame_rate, "
    "attack_to_adsr_max_ratio, max_attack_duration, attack_degree, "
    "decay_to_dsr_max_ratio, max_decay_duration, decay_degree, "
    "sustain_level, max_sustain_duration, "
    "release_to_sr_approx_ratio, max_release_duration, release_degree, "
    "expected",
    [
        (
            1.0,  # `duration`
            20,  # `frame_rate`
            0.2,  # `attack_to_adsr_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.25,  # `decay_to_dsr_max_ratio`
            0.25,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.6666,  # `release_to_sr_approx_ratio`
            1.0,  # `max_release_duration`
            1.0,  # `release_degree`,
            np.array([
                0, 0.25, 0.5, 0.75,
                1.0, 0.9, 0.8, 0.7,
                0.6, 0.6, 0.6, 0.6,
                0.6, 0.525, 0.45, 0.375, 0.3, 0.225, 0.15, 0.075
            ])
        ),
        (
            3.0,  # `duration`
            20,  # `frame_rate`
            0.2,  # `attack_to_adsr_max_ratio`
            0.25,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.25,  # `decay_to_dsr_max_ratio`
            1.0,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.6666,  # `release_to_sr_approx_ratio`
            1.0,  # `max_release_duration`
            1.0,  # `release_degree`,
            np.array([
                0, 0.2, 0.4, 0.6, 0.8,
                1.0, 1.0 - 0.4 / 13, 1.0 - 2 * 0.4 / 13, 1.0 - 3 * 0.4 / 13,
                1.0 - 4 * 0.4 / 13, 1.0 - 5 * 0.4 / 13, 1.0 - 6 * 0.4 / 13,
                1.0 - 7 * 0.4 / 13, 1.0 - 8 * 0.4 / 13, 1.0 - 9 * 0.4 / 13,
                1.0 - 10 * 0.4 / 13, 1.0 - 11 * 0.4 / 13, 1.0 - 12 * 0.4 / 13,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.57, 0.54, 0.51, 0.48, 0.45, 0.42, 0.39, 0.36, 0.33,
                0.3, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.03,
                0, 0
            ])
        ),
        (
            1.0,  # `duration`
            20,  # `frame_rate`
            0.2,  # `attack_to_adsr_max_ratio`
            0.0,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.25,  # `decay_to_dsr_max_ratio`
            0.0,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            1.0,  # `max_sustain_duration`
            0.6666,  # `release_to_sr_approx_ratio`
            0.0,  # `max_release_duration`
            1.0,  # `release_degree`,
            np.array([
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6
            ])
        ),
        (
            1.0,  # `duration`
            20,  # `frame_rate`
            0.2,  # `attack_to_adsr_max_ratio`
            0.0,  # `max_attack_duration`
            1.0,  # `attack_degree`
            0.25,  # `decay_to_dsr_max_ratio`
            0.0,  # `max_decay_duration`
            1.0,  # `decay_degree`
            0.6,  # `sustain_level`
            None,  # `max_sustain_duration`
            0.6666,  # `release_to_sr_approx_ratio`
            None,  # `max_release_duration`
            1.0,  # `release_degree`,
            np.array([
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 13 / 14 * 0.6, 12 / 14 * 0.6, 11 / 14 * 0.6,
                10 / 14 * 0.6, 9 / 14 * 0.6, 8 / 14 * 0.6, 7 / 14 * 0.6,
                6 / 14 * 0.6, 5 / 14 * 0.6, 4 / 14 * 0.6, 3 / 14 * 0.6,
                2 / 14 * 0.6, 1 / 14 * 0.6
            ])
        ),
    ]
)
def test_generic_adsr(
        duration: float, frame_rate: int,
        attack_to_adsr_max_ratio: float,
        max_attack_duration: float,
        attack_degree: float,
        decay_to_dsr_max_ratio: float,
        max_decay_duration: float,
        decay_degree: float,
        sustain_level: float,
        max_sustain_duration: Optional[float],
        release_to_sr_approx_ratio: float,
        max_release_duration: Optional[float],
        release_degree: float,
        expected: np.ndarray
) -> None:
    """Test `generic_adsr` function."""
    result = generic_adsr(
        duration, frame_rate,
        attack_to_adsr_max_ratio, max_attack_duration, attack_degree,
        decay_to_dsr_max_ratio, max_decay_duration, decay_degree,
        sustain_level, max_sustain_duration,
        release_to_sr_approx_ratio, max_release_duration, release_degree
    )
    print(result)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, frame_rate, "
    "attack_share, decay_share, sustain_level, release_share, expected",
    [
        (
            1, 10, 0.2, 0.2, 0.6, 0.2,
            np.array([0, 0.5, 1, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3])
        ),
        (
            1, 10, 0, 0, 0.6, 0,
            np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        ),
    ]
)
def test_relative_adsr(
        duration: float, frame_rate: int,
        attack_share: float, decay_share: float,
        sustain_level: float, release_share: float,
        expected: np.ndarray
) -> None:
    """Test `relative_adsr` function."""
    result = relative_adsr(
        duration, frame_rate,
        attack_share, decay_share, sustain_level, release_share
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, frame_rate, breakpoint_location, expected",
    [
        (
            1, 10, 0.4,
            np.array([0, 0.25, 0.5, 0.75, 1, 5 / 6, 2 / 3, 0.5, 1 / 3, 1 / 6])
        ),
    ]
)
def test_spike(
        duration: float, frame_rate: int,
        breakpoint_location: float, expected: np.ndarray
) -> None:
    """Test `spike` function."""
    result = spike(duration, frame_rate, breakpoint_location)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "duration, frame_rate, begin_share, end_share, expected",
    [
        (
            1, 10, 0.2, 0.5,
            np.array([0, 0.5, 1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2])
        ),
    ]
)
def test_trapezoid(
        duration: float, frame_rate: int,
        begin_share: float, end_share: float, expected: np.ndarray
) -> None:
    """Test `trapezoid` function."""
    result = trapezoid(
        duration, frame_rate, begin_share, end_share
    )
    np.testing.assert_almost_equal(result, expected)
