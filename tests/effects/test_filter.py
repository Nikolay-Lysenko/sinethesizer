"""
Test `sinethesizer.effects.filter` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.effects.filter import (
    apply_frequency_filter,
    filter_absolute_frequencies,
    filter_absolute_frequencies_wrt_velocity,
    filter_relative_frequencies,
    filter_relative_frequencies_wrt_velocity
)
from sinethesizer.synth.core import Event
from sinethesizer.oscillators import generate_mono_wave


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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
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
            # `kind`
            'absolute_wrt_velocity',
            # `kwargs`
            {
                'min_frequency_at_zero_velocity': 300,
                'min_frequency_at_max_velocity': 300,
                'min_frequency_on_velocity_order': 1.0,
                'max_frequency_at_zero_velocity': 500,
                'max_frequency_at_max_velocity': 500,
                'max_frequency_on_velocity_order': 1.0,
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
            'relative_wrt_velocity',
            # `kwargs`
            {
                'min_frequency_ratio_at_zero_velocity': 1.5,
                'min_frequency_ratio_at_max_velocity': 1.5,
                'min_frequency_ratio_on_velocity_order': 1.0,
                'max_frequency_ratio_at_zero_velocity': 2.5,
                'max_frequency_ratio_at_max_velocity': 2.5,
                'max_frequency_ratio_on_velocity_order': 1.0,
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
        generate_mono_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound = np.vstack((sound, sound))
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=min(frequencies),
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    sound = apply_frequency_filter(sound, event, kind, **kwargs)
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.007438023, 0.01572642, 0.5129808, 0.01068358, 0.00937196,
                    0.00494969, 0.002616557, 0.00108557, 0.00031414, 0.00004582
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
        generate_mono_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound = np.vstack((sound, sound))
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=min(frequencies),
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    sound = filter_absolute_frequencies(
        sound, event, min_frequency, max_frequency, invert, order
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequencies, velocity, frame_rate, "
    "min_frequency_at_zero_velocity, min_frequency_at_max_velocity, "
    "min_frequency_on_velocity_order, "
    "max_frequency_at_zero_velocity, max_frequency_at_max_velocity, "
    "max_frequency_on_velocity_order, "
    "invert, order, "
    "spectrogram_params, expected",
    [
        (
                # `frequencies`
                [200, 400, 600],
                # `velocity`
                0.5,
                # `frame_rate`
                10000,
                # `min_frequency_at_zero_velocity`
                200,
                # `min_frequency_at_max_velocity`
                400,
                # `min_frequency_on_velocity_order`
                1.0,
                # `max_frequency_at_zero_velocity`
                None,
                # `max_frequency_at_max_velocity`
                None,
                # `max_frequency_on_velocity_order`
                1.0,
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
                        0.0027162, 0.0067605, 0.0103526, 0.0146349, 0.5163752,
                        0.0197767, 0.5190597, 0.0146632, 0.0101723, 0.0059893
                    ]
                )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `velocity`
            0.5,
            # `frame_rate`
            10000,
            # `min_frequency_at_zero_velocity`
            None,
            # `min_frequency_at_zero_velocity`
            None,
            # `min_frequency_on_velocity_order`
            1.0,
            # `max_frequency_at_zero_velocity`
            200,
            # `max_frequency_at_max_velocity`
            400,
            # `max_frequency_on_velocity_order`
            1.0,
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
                    0.007438023, 0.01572642, 0.5129808, 0.01068358, 0.00937196,
                    0.00494969, 0.002616557, 0.00108557, 0.00031414, 0.00004582
                ]
            )
        ),
    ]
)
def test_filter_absolute_frequencies_wrt_velocity(
        frequencies: List[float], velocity: float, frame_rate: int,
        min_frequency_at_zero_velocity: Optional[float],
        min_frequency_at_max_velocity: Optional[float],
        min_frequency_on_velocity_order: Optional[float],
        max_frequency_at_zero_velocity: Optional[float],
        max_frequency_at_max_velocity: Optional[float],
        max_frequency_on_velocity_order: Optional[float],
        invert: bool, order: int, spectrogram_params: Dict[str, Any],
        expected: np.ndarray
) -> None:
    """Test `filter_absolute_frequencies_wrt_velocity` function."""
    waves = [
        generate_mono_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound = np.vstack((sound, sound))
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=min(frequencies),
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    sound = filter_absolute_frequencies_wrt_velocity(
        sound, event,
        min_frequency_at_zero_velocity, min_frequency_at_max_velocity,
        min_frequency_on_velocity_order,
        max_frequency_at_zero_velocity, max_frequency_at_max_velocity,
        max_frequency_on_velocity_order,
        invert, order
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
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.007438023, 0.01572642, 0.5129808, 0.01068358, 0.00937196,
                    0.00494969, 0.002616557, 0.00108557, 0.00031414, 0.00004582
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
        generate_mono_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound = np.vstack((sound, sound))
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=min(frequencies),
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    sound = filter_relative_frequencies(
        sound, event, min_frequency_ratio, max_frequency_ratio, invert, order
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequencies, velocity, frame_rate, "
    "min_frequency_ratio_at_zero_velocity, min_frequency_ratio_at_max_velocity, "
    "min_frequency_ratio_on_velocity_order, "
    "max_frequency_ratio_at_zero_velocity, max_frequency_ratio_at_max_velocity, "
    "max_frequency_ratio_on_velocity_order, "
    "invert, order, "
    "spectrogram_params, expected",
    [
        (
                # `frequencies`
                [200, 400, 600],
                # `velocity`
                0.5,
                # `frame_rate`
                10000,
                # `min_frequency_ratio_at_zero_velocity`
                1.0,
                # `min_frequency_ratio_at_max_velocity`
                2.0,
                # `min_frequency_ratio_on_velocity_order`
                1.0,
                # `max_frequency_ratio_at_zero_velocity`
                None,
                # `max_frequency_ratio_at_max_velocity`
                None,
                # `max_frequency_ratio_on_velocity_order`
                1.0,
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
                        0.0027162, 0.0067605, 0.0103526, 0.0146349, 0.5163752,
                        0.0197767, 0.5190597, 0.0146632, 0.0101723, 0.0059893
                    ]
                )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `velocity`
            0.5,
            # `frame_rate`
            10000,
            # `min_frequency_ratio_at_zero_velocity`
            None,
            # `min_frequency_ratio_at_zero_velocity`
            None,
            # `min_frequency_ratio_on_velocity_order`
            1.0,
            # `max_frequency_ratio_at_zero_velocity`
            1.0,
            # `max_frequency_ratio_at_max_velocity`
            2.0,
            # `max_frequency_ratio_on_velocity_order`
            1.0,
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
                    0.007438023, 0.01572642, 0.5129808, 0.01068358, 0.00937196,
                    0.00494969, 0.002616557, 0.00108557, 0.00031414, 0.00004582
                ]
            )
        ),
    ]
)
def test_filter_relative_frequencies_wrt_velocity(
        frequencies: List[float], velocity: float, frame_rate: int,
        min_frequency_ratio_at_zero_velocity: Optional[float],
        min_frequency_ratio_at_max_velocity: Optional[float],
        min_frequency_ratio_on_velocity_order: Optional[float],
        max_frequency_ratio_at_zero_velocity: Optional[float],
        max_frequency_ratio_at_max_velocity: Optional[float],
        max_frequency_ratio_on_velocity_order: Optional[float],
        invert: bool, order: int, spectrogram_params: Dict[str, Any],
        expected: np.ndarray
) -> None:
    """Test `filter_relative_frequencies_wrt_velocity` function."""
    waves = [
        generate_mono_wave(
            'sine', frequency, np.ones(frame_rate), frame_rate
        )
        for frequency in frequencies
    ]
    sound = sum(waves)
    sound = np.vstack((sound, sound))
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=min(frequencies),
        velocity=velocity,
        effects='',
        frame_rate=frame_rate
    )
    sound = filter_relative_frequencies_wrt_velocity(
        sound, event,
        min_frequency_ratio_at_zero_velocity,
        min_frequency_ratio_at_max_velocity,
        min_frequency_ratio_on_velocity_order,
        max_frequency_ratio_at_zero_velocity,
        max_frequency_ratio_at_max_velocity,
        max_frequency_ratio_on_velocity_order,
        invert, order
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)
