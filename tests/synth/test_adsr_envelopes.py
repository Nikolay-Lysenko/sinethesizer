"""
Test `sinethesizer.synth.adsr_envelopes` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.synth.adsr_envelopes import (
    relative_adsr, absolute_adsr, spike, constant_with_linear_ends
)


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
    "duration, frame_rate, "
    "attack_time, decay_time, sustain_level, release_time, expected",
    [
        (
            1, 10, 0.2, 0.2, 0.6, 0.2,
            np.array([0, 0.5, 1, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3])
        ),
        (
            1, 10, 1, 0.5, 0.6, 0.5,
            np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.6, 0.3])
        ),
        (
            1, 10, 0, 0, 0.6, 0,
            np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        ),
    ]
)
def test_absolute_adsr(
        duration: float, frame_rate: int,
        attack_time: float, decay_time: float,
        sustain_level: float, release_time: float,
        expected: np.ndarray
) -> None:
    """Test `absolute_adsr` function."""
    result = absolute_adsr(
        duration, frame_rate,
        attack_time, decay_time, sustain_level, release_time
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
def test_constant_with_linear_ends(
        duration: float, frame_rate: int,
        begin_share: float, end_share: float, expected: np.ndarray
) -> None:
    """Test `constant_with_linear_ends` function."""
    result = constant_with_linear_ends(
        duration, frame_rate, begin_share, end_share
    )
    np.testing.assert_almost_equal(result, expected)
