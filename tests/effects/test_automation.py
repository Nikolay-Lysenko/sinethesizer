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
from sinethesizer.utils.waves import generate_mono_wave


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
                        0.0002183, 0.0004449, 0.0007267, 0.0013604, 0.0075377,
                        0.0220382, 0.0224577, 0.0227558, 0.0230753, 0.0232609,
                        0.0234798, 0.0234915, 0.0235627, 0.0234495, 0.0235014,
                        0.0230627, 0.0231495, 0.0225367, 0.0227892, 0.0221962
                    ])
                ),
                (
                    26,
                    31,
                    np.array([
                        0.0004726, 0.0246527, 0.0265902, 0.0265404, 0.0278111,
                        0.0287469, 0.0288369, 0.0291045, 0.0292126, 0.029238,
                        0.0292028, 0.0289911, 0.0287447, 0.0284273, 0.0280728,
                        0.0278993, 0.0273279, 0.0268932, 0.0270518, 0.0250346
                    ])
                ),
                (
                    52,
                    57,
                    np.array([
                        0.0001075, 0.0250414, 0.0285338, 0.0278403, 0.0256055,
                        0.0239786, 0.0231653, 0.0225009, 0.0217015, 0.0206943,
                        0.0186929, 0.0133212, 0.0081255, 0.0041418, 0.0015606,
                        0.0004099, 0.0002907, 0.0002895, 0.0002744, 0.0002428
                    ])
                ),
                (
                    107,
                    112,
                    np.array([
                        0.0016842, 0.0218644, 0.0187497, 0.0202582, 0.0068620,
                        0.0013392, 0.0008960, 0.0004691, 0.0002142, 0.0000986,
                        0.0000691, 0.0000479, 0.0000290, 0.0000161, 0.0000067,
                        0.0000021, 0.0000009, 0.0000004, 0.0000001, 0.00000006
                    ])
                ),
            ]
        ),
        (
            # `frequencies`
            [100],
            # `frame_rate`
            10000,
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
                        0.0003073, 0.0208728, 0.0004687, 0.0003312, 0.0001999
                    ])
                ),
                (
                    26,
                    31,
                    np.array([
                        0.00003104, 0.0016946, 0.0000454, 0.00003239, 0.0000193
                    ])
                ),
                (
                    52,
                    57,
                    np.array([
                        0.0001143, 0.00510456, 0.0001717, 0.00011839, 0.0000688
                    ])
                ),
                (
                    90,
                    95,
                    np.array([
                        0.0003272, 0.0147043, 0.0004909, 0.0003388, 0.0001971
                    ])
                ),
            ]
        ),
    ]
)
def test_apply_automated_effect(
        frequencies: List[float], frame_rate: int, automated_effect_name: str,
        break_points: List[Dict[str, Union[float, List[float]]]],
        spectrogram_params: Dict[str, Any],
        expected: List[Tuple[int, int, np.ndarray]]
) -> None:
    """Test `apply_automated_effect` function."""
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
