"""
Test `sinethesizer.effects.amplitude` module.

Author: Nikolay Lysenko
"""


from typing import Any

import numpy as np
import pytest

from sinethesizer.effects.amplitude import (
    apply_amplitude_normalization, apply_compressor, apply_envelope_shaper
)
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, velocity, value_at_max_velocity, quantile, "
    "value_on_velocity_order, value_at_zero_velocity, expected",
    [
        (
            # `sound`
            np.array([
                [1.0, 2, 3, 4, 3, 2, 1, 2],
                [-1, -1, -1, -1, -1, -1, -1, -1]
            ]),
            # `velocity`
            1,
            # `value_at_max_velocity`
            6,
            # `quantile`
            7 / 8,
            # `value_on_velocity_order`
            0.5,
            # `value_at_zero_velocity`
            1,
            # `expected`
            np.array([
                [2, 4, 6, 8, 6, 4, 2, 4],
                [-2, -2, -2, -2, -2, -2, -2, -2]
            ])
        ),
    ]
)
def test_apply_amplitude_normalization(
        sound: np.ndarray, velocity: float, value_at_max_velocity: float,
        quantile: float, value_on_velocity_order: float,
        value_at_zero_velocity: float, expected: np.ndarray
) -> None:
    """Test `apply_amplitude_normalization` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=440,
        velocity=velocity,
        effects='',
        frame_rate=8
    )
    result = apply_amplitude_normalization(
        sound, event, value_at_max_velocity, quantile, value_on_velocity_order,
        value_at_zero_velocity
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "sound, frame_rate, frequency, threshold, quantile, chunk_size_in_cycles, expected",
    [
        (
            # `sound`
            np.array([
                [0.1, 0.3, 0.5, 0.7, 1.0, 0.8, 0.8, 0.6, 0.5, 0.1],
                [0.01, 0.03, 0.05, 0.07, 0.1, 0.08, 0.08, 0.06, 0.05, 0.01],
            ]),
            # `frame_rate`
            9,
            # `frequency`
            3,
            # `threshold`
            0.7,
            # `quantile`
            1,
            # `chunk_size_in_cycles`
            1,
            # `expected`
            np.array([
                [0.1, 0.3, 0.5, 0.7, 1.0, 0.72, 0.64, 0.42, 0.4, 0.09],
                [0.01, 0.03, 0.05, 0.07, 0.1, 0.072, 0.064, 0.042, 0.04, 0.009],
            ])
        ),
    ]
)
def test_apply_compressor(
        sound: np.ndarray, frame_rate: int, frequency: float, threshold: float, quantile: float,
        chunk_size_in_cycles: float, expected: np.ndarray
) -> None:
    """Test `apply_compressor` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=sound.shape[1] / frame_rate,
        frequency=frequency,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_compressor(sound, event, threshold, quantile, chunk_size_in_cycles)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "sound, frame_rate, frequency, envelope_params, quantile, chunk_size_in_cycles, "
    "initial_rescaling_ratio, forced_fading_ratio, expected",
    [
        (
            # `sound`
            np.array([
                [1.0, 2, 3, 4, 5, 6, 7, 8],
                [-1, 2, -3, 4, 5, 6, 7, -8],
            ]),
            # `frame_rate`
            4,
            # `frequency`
            2,
            # `envelope_params`
            {
                'name': 'trapezoid',
                'attack_share': 0.5,
                'attack_degree': 1.0,
                'decay_share': 0.5,
                'decay_degree': 1.0,
                'peak_value': 1.0,
            },
            # `quantile`
            1,
            # `chunk_size_in_cycles`
            2,
            # `initial_rescaling_ratio`
            1,
            # `forced_fading_ratio`
            0,
            # `expected`
            np.array([
                [1, 25 / 16, 27 / 16, 11 / 8, 5 / 8, 21 / 32, 21 / 32, 5 / 8],
                [-1, 25 / 16, -27 / 16, 11 / 8, 5 / 8, 21 / 32, 21 / 32, -5 / 8],
            ])
        ),
        (
            # `sound`
            np.array([
                [1.0, 2, 3, 4, 5, 6, 7, 8],
                [-1, 2, -3, 4, 5, 6, 7, -8],
            ]),
            # `frame_rate`
            4,
            # `frequency`
            2,
            # `envelope_params`
            {
                'name': 'trapezoid',
                'attack_share': 0.5,
                'attack_degree': 1.0,
                'decay_share': 0.5,
                'decay_degree': 1.0,
                'peak_value': 1.0,
            },
            # `quantile`
            1,
            # `chunk_size_in_cycles`
            2,
            # `initial_rescaling_ratio`
            1,
            # `forced_fading_ratio`
            0.4,
            # `expected`
            np.array([
                [1, 25 / 16, 27 / 16, 11 / 8, 5 / 8, 21 / 32, 21 / 48, 5 / 24],
                [-1, 25 / 16, -27 / 16, 11 / 8, 5 / 8, 21 / 32, 21 / 48, -5 / 24],
            ])
        ),
    ]
)
def test_apply_envelope_shaper(
        sound: np.ndarray, frame_rate: int, frequency: float, envelope_params: dict[str, Any],
        quantile: float, chunk_size_in_cycles: float, initial_rescaling_ratio: float,
        forced_fading_ratio: float, expected: np.ndarray
) -> None:
    """Test `apply_envelope_shaper` function."""
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=sound.shape[1] / frame_rate,
        frequency=frequency,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_envelope_shaper(
        sound, event, envelope_params, quantile, chunk_size_in_cycles,
        initial_rescaling_ratio, forced_fading_ratio
    )
    np.testing.assert_almost_equal(result, expected)
