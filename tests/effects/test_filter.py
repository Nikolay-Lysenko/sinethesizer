"""
Test `sinethesizer.effects.filter` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.effects.filter import (
    apply_frequency_filter,
    filter_absolute_frequencies,
    filter_relative_frequencies,
)
from sinethesizer.utils.waves import generate_wave


@pytest.mark.parametrize(
    "frequencies, frame_rate, kind, kwargs, spectrogram_params, expected",
    [
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `kind`
            'absolute',
            # `kwargs`
            {
                'min_frequency': 300,
                'max_frequency': 500,
                'invert': False,
                'order': 10,
            },
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0024158, 0.0056049, 0.0075644, 0.0097482, 0.5070437,
                    0.0096938, 0.0075109, 0.0048818, 0.0025853, 0.0010571
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `kind`
            'relative',
            # `kwargs`
            {
                'min_frequency_ratio': 1.5,
                'max_frequency_ratio': 2.5,
                'invert': True,
                'order': 10,
            },
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0080094, 0.0173524, 0.5157133, 0.0158572, 0.0154468,
                    0.014754, 0.5136481, 0.0109237, 0.007874, 0.0049527
                ]
            )
        ),
    ]
)
def test_apply_frequency_filter(
        frequencies: List[float], frame_rate: int, kind: str,
        kwargs: Dict[str, Any], spectrogram_params: Dict[str, Any],
        expected: np.ndarray
) -> None:
    """Test `apply_frequency_filter` function."""
    waves = [
        generate_wave('sine', frequency, np.ones(frame_rate), frame_rate)
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound_info = {
        'frame_rate': frame_rate,
        'fundamental_frequency': min(frequencies)
    }
    sound = apply_frequency_filter(sound, sound_info, kind, **kwargs)
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequencies, frame_rate, min_frequency, max_frequency, invert, order, "
    "spectrogram_params, expected",
    [
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `min_frequency`
            300,
            # `max_frequency`
            500,
            # `invert`
            False,
            # `order`
            10,
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0024158, 0.0056049, 0.0075644, 0.0097482, 0.5070437,
                    0.0096938, 0.0075109, 0.0048818, 0.0025853, 0.0010571
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `min_frequency`
            300,
            # `max_frequency`
            500,
            # `invert`
            True,
            # `order`
            10,
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0080094, 0.0173524, 0.5157133, 0.0158572, 0.0154468,
                    0.014754, 0.5136481, 0.0109237, 0.007874, 0.0049527
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `min_frequency`
            300,
            # `max_frequency`
            None,
            # `invert`
            False,
            # `order`
            10,
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0027162, 0.0067605, 0.0103526, 0.0146349, 0.5163752,
                    0.0197767, 0.5190597, 0.0146632, 0.0101723, 0.0059893
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `min_frequency`
            None,
            # `max_frequency`
            300,
            # `invert`
            False,
            # `order`
            10,
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.007438026, 0.01572727, 0.5129805, 0.0106836, 0.0093715,
                    0.00494968, 0.002616551, 0.00108557, 0.00031414, 0.00004582
                ]
            )
        ),
    ]
)
def test_filter_absolute_frequencies(
        frequencies: List[float], frame_rate: int,
        min_frequency: float, max_frequency: float, invert: bool, order: int,
        spectrogram_params: Dict[str, Any], expected: np.ndarray
) -> None:
    """Test `filter_absolute_frequencies` function."""
    waves = [
        generate_wave('sine', frequency, np.ones(frame_rate), frame_rate)
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound_info = {
        'frame_rate': frame_rate,
        'fundamental_frequency': min(frequencies)
    }
    sound = filter_absolute_frequencies(
        sound, sound_info, min_frequency, max_frequency, invert, order
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequencies, frame_rate, min_frequency_ratio, max_frequency_ratio,"
    "invert, order, spectrogram_params, expected",
    [
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `min_frequency_ratio`
            1.5,
            # `max_frequency_ratio`
            None,
            # `invert`
            False,
            # `order`
            10,
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0027162, 0.0067605, 0.0103526, 0.0146349, 0.5163752,
                    0.0197767, 0.5190597, 0.0146632, 0.0101723, 0.0059893
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `min_frequency_ratio`
            None,
            # `max_frequency_ratio`
            1.5,
            # `invert`
            False,
            # `order`
            10,
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed power for
            # frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.007438026, 0.01572727, 0.5129805, 0.0106836, 0.0093715,
                    0.00494968, 0.002616551, 0.00108557, 0.00031414, 0.00004582
                ]
            )
        ),
    ]
)
def test_filter_relative_frequencies(
        frequencies: List[float], frame_rate: int,
        min_frequency_ratio: float, max_frequency_ratio: float,
        invert: bool, order: int, spectrogram_params: Dict[str, Any],
        expected: np.ndarray
) -> None:
    """Test `filter_relative_frequencies` function."""
    waves = [
        generate_wave('sine', frequency, np.ones(frame_rate), frame_rate)
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound_info = {
        'frame_rate': frame_rate,
        'fundamental_frequency': min(frequencies)
    }
    sound = filter_relative_frequencies(
        sound, sound_info,
        min_frequency_ratio, max_frequency_ratio, invert, order
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)
