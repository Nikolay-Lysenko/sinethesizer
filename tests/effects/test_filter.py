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
from sinethesizer.utils.waves import generate_stereo_wave


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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0012079, 0.0028025, 0.0037822, 0.0048741, 0.2535218,
                    0.0048469, 0.0037555, 0.0024409, 0.0012927, 0.0005285
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0040047, 0.0086762, 0.2578567, 0.0079286, 0.0077234,
                    0.007377, 0.256824, 0.0054619, 0.003937, 0.0024764
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
        generate_stereo_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0012079, 0.0028025, 0.0037822, 0.0048741, 0.2535218,
                    0.0048469, 0.0037555, 0.0024409, 0.0012927, 0.0005285
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0040047, 0.0086762, 0.2578567, 0.0079286, 0.0077234,
                    0.007377, 0.256824, 0.0054619, 0.003937, 0.0024764
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0013581, 0.0033803, 0.0051763, 0.0073174, 0.2581876,
                    0.0098884, 0.2595298, 0.0073316, 0.0050862, 0.0029946
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.003719, 0.007863635, 0.25649025, 0.0053418, 0.0046857499,
                    0.002474844, 0.00130828, 0.0005427843, 0.0001571, 0.0000229
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
        generate_stereo_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0013581, 0.0033803, 0.0051763, 0.0073174, 0.2581876,
                    0.0098884, 0.2595298, 0.0073316, 0.0050862, 0.0029946
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.003719, 0.007863635, 0.25649025, 0.0053418, 0.0046857499,
                    0.002474844, 0.00130828, 0.0005427843, 0.0001571, 0.0000229
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
        generate_stereo_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
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
