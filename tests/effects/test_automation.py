"""
Test `sinethesizer.effects.automation` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.effects.automation import apply_automated_effect
from sinethesizer.synth.core import Event
from sinethesizer.oscillators import generate_mono_wave


@pytest.mark.parametrize(
    "sound, event, automated_effect_name, break_points, expected",
    [
        (
            # `sound`
            np.array([
                [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=8
            ),
            # `automated_effect_name`
            'panning',
            # `break_points`
            [
                {
                    'relative_position': 0,
                    'left_volume_ratio': 1.0,
                    'right_volume_ratio': 1.0,
                },
                {
                    'relative_position': 0.25,
                    'left_volume_ratio': 0.25,
                    'right_volume_ratio': 1.0,
                },
                {
                    'relative_position': 0.5,
                    'left_volume_ratio': 0.5,
                    'right_volume_ratio': 1.0,
                },
                {
                    'relative_position': 1,
                    'left_volume_ratio': 1.0,
                    'right_volume_ratio': 1.0,
                },
            ],
            # `expected`
            np.array([
                [
                    1, 0.8125, 0.625, 0.4375, 0.25, 0.3125, 0.375, 0.4375,
                    0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375
                ],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ])
        ),
    ]
)
def test_apply_automated_effect(
        sound: np.ndarray, event: Event, automated_effect_name: str,
        break_points: List[Dict[str, Any]], expected: np.ndarray
) -> None:
    """Test `apply_automated_effect` function."""
    result = apply_automated_effect(
        sound, event, automated_effect_name, break_points
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequencies, frame_rate, automated_effect_name, break_points, "
    "spectrogram_params, expected",
    [
        (
            # `frequencies`
            [100 * x for x in range(1, 20)],
            # `frame_rate`
            10000,
            # `automated_effect_name`
            'equalizer',
            # `break_points`
            [
                {
                    'relative_position': 0,
                    'breakpoint_frequencies': [400, 400],
                    'gains': [0, 1]
                },
                {
                    'relative_position': 0.25,
                    'breakpoint_frequencies': [0, 5000],
                    'gains': [1, 1]
                },
                {
                    'relative_position': 0.5,
                    'breakpoint_frequencies': [0, 1000, 1500],
                    'gains': [1, 0.8, 0]
                },
                {
                    'relative_position': 1,
                    'breakpoint_frequencies': [400, 400],
                    'gains': [1, 0]
                },
            ],
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, two integer numbers define start and end
            # segments of spectrogram and the array contains summed over time
            # power for frequencies 0, 100, 200, ..., 1900 respectively for
            # spectrogram segments between the start and the end.
            [
                (
                    0,
                    5,
                    np.array([
                        0.0002183, 0.0004449, 0.0007268, 0.0013604, 0.0075382,
                        0.0220401, 0.0224595, 0.0227577, 0.0230772, 0.0232628,
                        0.0234817, 0.0234934, 0.0235646, 0.0234514, 0.0235033,
                        0.0230645, 0.0231514, 0.0225386, 0.0227911, 0.0221979
                    ])
                ),
                (
                    26,
                    31,
                    np.array([
                        0.0004725, 0.0246534, 0.0265909, 0.0265413, 0.0278156,
                        0.0287562, .028847, 0.029115, 0.0292235, 0.029249,
                        0.029214, 0.0290024, 0.028756, 0.0284386, 0.028084,
                        0.0279104, 0.0273389, 0.0269038, 0.027062, 0.0250446
                    ])
                ),
                (
                    52,
                    57,
                    np.array([
                        0.0001076, 0.025058, 0.028553, 0.0278598, 0.0256236,
                        0.023996, 0.0231824, 0.0225178, 0.0217181, 0.0207106,
                        0.0187084, 0.0133343, 0.0081357, 0.0041491, 0.0015651,
                        0.0004121, 0.0002923, 0.0002912, 0.000276, 0.0002442
                    ])
                ),
                (
                    107,
                    112,
                    np.array([
                        0.0016849, 0.0218725, 0.0187567, 0.0202654, 0.0068659,
                        0.00134, 0.0008968, 0.0004696, 0.0002146, 0.000099,
                        0.0000694, 0.0000482, 0.0000291, 0.0000162, 0.0000067,
                        0.00000213, 0.0000009, 0.0000004, 0.0000001, 0.00000006
                    ])
                ),
            ]
        ),
    ]
)
def test_apply_automated_effect_with_spectrogram_checks(
        frequencies: List[float], frame_rate: int, automated_effect_name: str,
        break_points: List[Dict[str, Union[float, List[float]]]],
        spectrogram_params: Dict[str, Any],
        expected: List[Tuple[int, int, np.ndarray]]
) -> None:
    """Test `apply_automated_effect` function with spectrogram checks."""
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
    sound = apply_automated_effect(
        sound, event, automated_effect_name, break_points
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    for start_segment, end_segment, expected_distribution in expected:
        spc_slice = spc[:len(expected_distribution), start_segment:end_segment]
        result = spc_slice.sum(axis=1)
        np.testing.assert_almost_equal(result, expected_distribution)
