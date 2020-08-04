"""
Test `sinethesizer.effects.filter_sweep` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.effects.filter_sweep import (
    apply_filter_sweep,
    apply_phaser,
    oscillate_between_sounds,
)
from sinethesizer.utils.waves import generate_stereo_wave


@pytest.mark.parametrize(
    "frequencies, frame_rate, kind, bands, invert, order, frequency, waveform,"
    "spectrogram_params, expected",
    [
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `kind`
            'absolute',
            # `bands`
            [(180, 220), (230, 270), (280, 320), (330, 370), (380, 420)],
            # `invert`
            True,
            # `order`
            10,
            # `frequency`
            5,
            # `waveform`
            'sine',
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0031053, 0.0079471, 0.1784642, 0.0107419, 0.1794305,
                    0.0106748, 0.2592777, 0.0069855, 0.0048323, 0.002871
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `kind`
            'absolute',
            # `bands`
            [(180, 220), (230, 270)],
            # `invert`
            True,
            # `order`
            10,
            # `frequency`
            5,
            # `waveform`
            'sine',
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0029946, 0.0068664, 0.1012633, 0.0097042, 0.2597036,
                    0.0106444, 0.2601111, 0.007498, 0.0051409, 0.0030093
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `kind`
            'absolute',
            # `bands`
            [(380, 420), (430, 470)],
            # `invert`
            False,
            # `order`
            10,
            # `frequency`
            5,
            # `waveform`
            'sine',
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0006736, 0.0015094, 0.0017339, 0.0020243, 0.0954683,
                    0.0018791, 0.0013814, 0.0009007, 0.0004846, 0.0002025
                ]
            )
        ),
        (
            # `frequencies`
            [200, 400, 600],
            # `frame_rate`
            10000,
            # `kind`
            'absolute',
            # `bands`
            [(380, 420)],
            # `invert`
            False,
            # `order`
            10,
            # `frequency`
            5,
            # `waveform`
            'sine',
            # `spectrogram_params`
            {'nperseg': 100},
            # `expected`
            # In this test case, `expected` contains summed over time power
            # for frequencies 0, 100, 200, ..., 900 respectively.
            np.array(
                [
                    0.0012617, 0.0028942, 0.0037543, 0.0046438, 0.2426652,
                    0.0045922, 0.0035679, 0.002328, 0.0012373, 0.000508
                ]
            )
        ),
    ]
)
def test_apply_filter_sweep(
        frequencies: List[float], frame_rate: int, kind: str,
        bands: List[Tuple[Optional[float], Optional[float]]],
        invert: bool, order: int, frequency: float, waveform: str,
        spectrogram_params: Dict[str, Any], expected: np.ndarray
) -> None:
    """Test `apply_filter_sweep` function."""
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
    sound = apply_filter_sweep(
        sound, sound_info, kind, bands, invert, order, frequency, waveform
    )
    spc = spectrogram(sound[0], frame_rate, **spectrogram_params)[2]
    result = spc.sum(axis=1)[:len(expected)]
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "frequency, frame_rate, kind",
    [
        (440, 44100, 'absolute'),
        (440, 44100, 'relative'),
    ]
)
def test_apply_phaser(frequency: float, frame_rate: int, kind: str) -> None:
    """Test that `apply_phaser` function runs without failures."""
    sound = generate_stereo_wave(
        'sine', frequency, np.ones(frame_rate), frame_rate
    )
    sound_info = {'frame_rate': frame_rate, 'fundamental_frequency': frequency}
    result = apply_phaser(sound, sound_info, kind)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "sounds, frame_rate, frequency, waveform, expected",
    [
        (
            # `sounds`
            np.array(
                [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[-1, -2, -3, -4, -5, -6, -7, -8]]
                ]
            ),
            # `frame_rate`
            4,
            # `frequency`
            1,
            # `waveform`
            'sine',
            # `expected`
            np.array([[0, -2, 0, 4, 0, -6, 0, 8]])
        ),
        (
            # `sounds`
            np.array(
                [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[-1, -1, -1, -1, -1, -1, -1, -1]],
                    [[-2, -2, -2, -2, -2, -2, -2, -2]],
                    [[11, 12, 13, 14, 15, 16, 17, 18]],
                ]
            ),
            # `frame_rate`
            4,
            # `frequency`
            0.5,
            # `waveform`
            'triangle',
            # `expected`
            np.array([[1, -0.25, -1.5, 2, 15, 2.5, -1.5, 1.25]])
        ),
        (
            # `sounds`
            np.array(
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [1, 2, 3, 4, 5, 6, 7, 8]
                    ],
                    [
                        [-1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1]
                    ],
                    [
                        [-2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -2, -2]
                    ],
                    [
                        [11, 12, 13, 14, 15, 16, 17, 18],
                        [11, 12, 13, 14, 15, 16, 17, 18]
                    ],
                ]
            ),
            # `frame_rate`
            4,
            # `frequency`
            0.5,
            # `waveform`
            'triangle',
            # `expected`
            np.array([
                [1, -0.25, -1.5, 2, 15, 2.5, -1.5, 1.25],
                [1, -0.25, -1.5, 2, 15, 2.5, -1.5, 1.25]
            ])
        ),
    ]
)
def test_oscillate_between_sounds(
        sounds: np.ndarray, frame_rate: int, frequency: float,
        waveform: str, expected: np.ndarray
) -> None:
    """Test `oscillate_between_sounds` function."""
    result = oscillate_between_sounds(sounds, frame_rate, frequency, waveform)
    np.testing.assert_almost_equal(result, expected)
