"""
Test `sinethesizer.oscillators.analog` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.oscillators.analog import (
    generate_sawtooth_wave, generate_square_wave, generate_triangle_wave
)


TWO_PI = 2 * np.pi


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
            # `expected`
            np.array([1.0, 1, 0, -1, -1, -1, 0, 1])
        ),
    ]
)
def test_generate_square_wave(
        xs: np.ndarray, xs_step: float, expected: np.ndarray
) -> None:
    """Test `generate_square_wave` function."""
    result = generate_square_wave(xs, xs_step)
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
