"""
Test `sinethesizer.effects.equalizer` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.effects.equalizer import (
    apply_equalizer, apply_equalizer_envelope
)
from sinethesizer.synth.core import Event
from sinethesizer.utils.waves import generate_mono_wave


@pytest.mark.parametrize(
    "frequencies, frame_rate, breakpoint_frequencies, gains, "
    "spectrogram_params, expected",
    [
        (
            # `frequencies`
            [100 * x for x in range(1, 20)],
            # `frame_rate`
            10000,
            # `breakpoint_frequencies`
            [300, 700],
            # `gains`
            [0.2, 1.0],
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 1900 respectively.
            np.array(
                [
                    0.0021011, 0.0249528, 0.0277226, 0.0387388, 0.0996291,
                    0.2081294, 0.3571571, 0.5181565, 0.55258, 0.557289,
                    0.5601418, 0.5615491, 0.5621033, 0.5622196, 0.5619461,
                    0.5608991, 0.5583538, 0.5535695, 0.5462548, 0.536942
                ]
            )
        ),
        (
            # `frequencies`
            [100 * x for x in range(1, 20)],
            # `frame_rate`
            10000,
            # `breakpoint_frequencies`
            [0, 500, 1200, 1900],
            # `gains`
            [0, 1.0, 0.1, 1.0],
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 1900 respectively.
            np.array(
                [
                    0.0062764, 0.0342341, 0.0986968, 0.2045612, 0.3501325,
                    0.4880824, 0.4132437, 0.306272, 0.2138001, 0.1371348,
                    0.0776751, 0.03646, 0.0184661, 0.0364665, 0.0775099,
                    0.136432, 0.2119483, 0.3025262, 0.4070148, 0.5069672
                ]
            )
        ),
        (
            # `frequencies`
            [100 * x for x in range(1, 20)],
            # `frame_rate`
            10000,
            # `breakpoint_frequencies`
            [0, 500, 1200, 1900, 5000],
            # `gains`
            [0, 1.0, 0.1, 1.0, 1.0],
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 1900 respectively.
            np.array(
                [
                    0.0062764, 0.0342341, 0.0986968, 0.2045612, 0.3501325,
                    0.4880824, 0.4132437, 0.306272, 0.2138001, 0.1371348,
                    0.0776751, 0.03646, 0.0184661, 0.0364665, 0.0775099,
                    0.136432, 0.2119483, 0.3025262, 0.4070148, 0.5069672
                ]
            )
        ),
    ]
)
def test_apply_equalizer(
        frequencies: List[float], frame_rate: int,
        breakpoint_frequencies: List[float], gains: List[float],
        spectrogram_params: Dict[str, Any], expected: np.ndarray
) -> None:
    """Test `apply_equalizer` function."""
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
    sound = apply_equalizer(sound, event, breakpoint_frequencies, gains)
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequencies, frame_rate, envelope_points, spectrogram_params, expected",
    [
        (
            # `frequencies`
            [100 * x for x in range(1, 20)],
            # `frame_rate`
            10000,
            # `envelope_points`
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
    ]
)
def test_apply_equalizer_envelope(
        frequencies: List[float], frame_rate: int,
        envelope_points: List[Dict[str, Union[float, List[float]]]],
        spectrogram_params: Dict[str, Any],
        expected: List[Tuple[int, int, np.ndarray]]
) -> None:
    """Test `apply_equalizer_envelope` function."""
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
    sound = apply_equalizer_envelope(sound, event, envelope_points)
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    for start_segment, end_segment, expected_distribution in expected:
        spc_slice = spc[:len(expected_distribution), start_segment:end_segment]
        result = spc_slice.sum(axis=1)
        np.testing.assert_almost_equal(result, expected_distribution)
