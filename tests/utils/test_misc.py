"""
Test `sinethesizer.utils.misc` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.utils.misc import mix_with_original_sound, sum_two_sounds


@pytest.mark.parametrize(
    "sound, factor, original_sound_weight, expected",
    [
        (np.array([1.0, 2, 3]), 2, 0.25, np.array([1.75, 3.5, 5.25])),
    ]
)
def test_mix_with_original_sound(
        sound: np.ndarray, factor: float, original_sound_weight: float, expected: np.ndarray
) -> None:
    """Test `mix_with_original_sound` decorator."""
    @mix_with_original_sound
    def multiply(sound: np.ndarray, factor: float, **kwargs) -> np.ndarray:
        return factor * sound

    result = multiply(sound, factor, original_sound_weight=original_sound_weight)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "first_sound, second_sound, expected",
    [
        (
            np.array([[1, 2, 3], [2, 3, 4]]),
            np.array([[7, 9], [8, 9]]),
            np.array([[8, 11, 3], [10, 12, 4]])
        ),
        (
            np.array([[7, 9], [8, 9]]),
            np.array([[1, 2, 3], [2, 3, 4]]),
            np.array([[8, 11, 3], [10, 12, 4]])
        ),
    ]
)
def test_sum_two_sounds(
        first_sound: np.ndarray, second_sound: np.ndarray, expected: np.ndarray
) -> None:
    """Test `sum_two_sounds` function."""
    result = sum_two_sounds(first_sound, second_sound)
    np.testing.assert_almost_equal(result, expected)
