"""
Test `sinethesizer.oscillators.analog` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.oscillators.analog import (
    generate_pulse_wave, generate_sawtooth_wave, generate_triangle_wave
)


TWO_PI = 2 * np.pi


@pytest.mark.parametrize(
    "xs, xs_step, duty_cycle, expected",
    [
        (
            # `xs`
            np.array([
                0.9 * np.pi, 0.95 * np.pi, np.pi, 1.05 * np.pi,
                0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI
            ]),
            # `xs_step`
            0.03 * TWO_PI,
            # `duty_cycle`
            0.5,
            # `expected`
            np.array([
                1, 0.9722222, 0, -0.9722222, -1, -0.9722222, 0, 0.9722222
            ])
        ),
        (
            # `xs`
            np.array([
                0.9 * np.pi, 0.95 * np.pi, np.pi, 1.05 * np.pi,
                0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI
            ]),
            # `xs_step`
            0.02 * TWO_PI,
            # `duty_cycle`
            0.5,
            # `expected`
            np.array([1.0, 1, 0, -1, -1, -1, 0, 1])
        ),
        (
            # `xs`
            np.array([
                0, 0.1 * np.pi, 0.2 * np.pi, 0.3 * np.pi, 0.4 * np.pi,
                0.5 * np.pi, 0.6 * np.pi, 0.7 * np.pi, 0.8 * np.pi, 0.9 * np.pi
            ]),
            # `xs_step`
            0.05 * TWO_PI,
            # `duty_cycle`
            0.3,
            # `expected`
            np.array([0, 1, 1, 1, 1, 1, 0, -1, -1, -1])
        ),
        (
            # `xs`
            np.array([
                0.01 * np.pi, 0.11 * np.pi, 0.21 * np.pi, 0.31 * np.pi, 0.41 * np.pi,
                0.51 * np.pi, 0.61 * np.pi, 0.71 * np.pi, 0.81 * np.pi, 0.91 * np.pi
            ]),
            # `xs_step`
            0.05 * TWO_PI,
            # `duty_cycle`
            0.3,
            # `expected`
            np.array([0.19, 1, 1, 1, 1, 0.99, -0.19, -1, -1, -1])
        ),
    ]
)
def test_generate_pulse_wave(
        xs: np.ndarray, xs_step: float, duty_cycle: float, expected: np.ndarray
) -> None:
    """Test `generate_pulse_wave` function."""
    result = generate_pulse_wave(xs, xs_step, duty_cycle)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "xs, xs_step, expected",
    [
        (
            # `xs`
            np.array([0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI]),
            # `xs_step`
            0.03 * TWO_PI,
            # `expected`
            np.array([0.9, 0.9222222, 0, -0.9222222])
        ),
        (
            # `xs`
            np.array([0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI]),
            # `xs_step`
            0.02 * TWO_PI,
            # `expected`
            np.array([0.9, 0.95, 0, -0.95])
        ),
    ]
)
def test_generate_sawtooth_wave(
        xs: np.ndarray, xs_step: float, expected: np.ndarray
) -> None:
    """Test `generate_sawtooth_wave` function."""
    result = generate_sawtooth_wave(xs, xs_step)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "xs, xs_step, expected",
    [
        (
            # `xs`
            np.array([
                0.9 * np.pi, 0.95 * np.pi, np.pi, 1.05 * np.pi,
                0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI
            ]),
            # `xs_step`
            0.03 * TWO_PI,
            # `expected`
            np.array([
                0.7999918, 0.8956782, 0.944, 0.8956782,
                -0.7999918, -0.8956782, -0.944, -0.8956782
            ])
        ),
        (
            # `xs`
            np.array([
                0.9 * np.pi, 0.95 * np.pi, np.pi, 1.05 * np.pi,
                0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI
            ]),
            # `xs_step`
            0.02 * TWO_PI,
            # `expected`
            np.array([
                0.8, 0.8996836, 0.9626667, 0.8996836,
                -0.8, -0.8996836, -0.9626667, -0.8996836
            ])
        ),
        (
            # `xs`
            np.array([
                0.9 * np.pi, 0.95 * np.pi, np.pi, 1.05 * np.pi,
                0.95 * TWO_PI, 0.975 * TWO_PI, TWO_PI, 1.025 * TWO_PI
            ]),
            # `xs_step`
            0.01 * TWO_PI,
            # `expected`
            np.array([
                0.8, 0.9, 0.9813333, 0.9,
                -0.8, -0.9, -0.9813333, -0.9
            ])
        ),
    ]
)
def test_generate_triangle_wave(
        xs: np.ndarray, xs_step: float, expected: np.ndarray
) -> None:
    """Test `generate_triangle_wave` function."""
    result = generate_triangle_wave(xs, xs_step)
    np.testing.assert_almost_equal(result, expected)
