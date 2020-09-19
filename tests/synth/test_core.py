"""
Test `sinethesizer.synth.synth` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.synth.synth import sum_two_sounds


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
