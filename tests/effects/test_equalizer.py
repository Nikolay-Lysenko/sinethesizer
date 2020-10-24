"""
Test `sinethesizer.effects.equalizer` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.effects.equalizer import apply_equalizer
from sinethesizer.synth.core import Event
from sinethesizer.oscillators import generate_mono_wave


@pytest.mark.parametrize(
    "frequencies, frame_rate, kind, kwargs, spectrogram_params, expected",
    [
        (
            # `frequencies`
            [100 * x for x in range(1, 20)],
            # `frame_rate`
            10000,
            # `kind`
            'absolute',
            # `kwargs`
            {
                'breakpoint_frequencies': [300, 700],
                'gains': [0.2, 1.0],
            },
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
            # `kind`
            'absolute',
            # `kwargs`
            {
                'breakpoint_frequencies': [0, 500, 1200, 1900],
                'gains': [0, 1.0, 0.1, 1.0],
            },
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
            # `kind`
            'absolute',
            # `kwargs`
            {
                'breakpoint_frequencies': [0, 500, 1200, 1900, 5000],
                'gains': [0, 1.0, 0.1, 1.0, 1.0],
            },
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
            # `kind`
            'relative',
            # `kwargs`
            {
                'breakpoint_frequencies_ratios': [0, 5, 12, 19, 50],
                'gains': [0, 1.0, 0.1, 1.0, 1.0],
            },
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
        frequencies: List[float], frame_rate: int, kind: str,
        kwargs: Dict[str, Any], spectrogram_params: Dict[str, Any],
        expected: np.ndarray
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
    sound = apply_equalizer(sound, event, kind, **kwargs)
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)
