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
from sinethesizer.synth.core import Event
from sinethesizer.oscillators import generate_mono_wave


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
                    0.0062106, 0.0158941, 0.3569285, 0.0214838, 0.3588611,
                    0.0213496, 0.5185554, 0.013971, 0.0096646, 0.005742
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
                    0.0059891, 0.0137329, 0.2025266, 0.0194084, 0.5194071,
                    0.0212887, 0.5202222, 0.014996, 0.0102818, 0.0060186
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
                    0.0013472, 0.0030189, 0.0034677, 0.0040487, 0.1909366,
                    0.0037583, 0.0027628, 0.0018015, 0.0009691, 0.0004051
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
                    0.0025234, 0.0057885, 0.0075087, 0.0092877, 0.4853304,
                    0.0091845, 0.0071357, 0.004656, 0.0024747, 0.0010161
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
    sound = apply_filter_sweep(
        sound, event, kind, bands, invert, order, frequency, waveform
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
    sound = generate_mono_wave(
        'sine', frequency, np.ones(frame_rate), frame_rate
    )
    sound = np.vstack((sound, sound))
    event = Event(
        instrument='any_instrument',
        start_time=0,
        duration=1,
        frequency=frequency,
        velocity=1,
        effects='',
        frame_rate=frame_rate
    )
    result = apply_phaser(sound, event, kind)
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
            'raw_triangle',
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
            'raw_triangle',
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
