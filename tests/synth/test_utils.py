"""
Test `sinethesizer.synth.utils` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest

from sinethesizer.synth.adsr_envelopes import trapezoid
from sinethesizer.synth.timbre import OvertoneSpec, TimbreSpec
from sinethesizer.synth.utils import (
    oscillate_between_sounds, validate_timbre_spec
)


@pytest.mark.parametrize(
    "sounds, frame_rate, frequency, waveform, expected",
    [
        (
            np.array(
                [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[-1, -2, -3, -4, -5, -6, -7, -8]]
                ]
            ),
            4, 1, 'sine',
            np.array([[0, -2, 0, 4, 0, -6, 0, 8]])
        ),
        (
            np.array(
                [
                    [[1, 2, 3, 4, 5, 6, 7, 8]],
                    [[-1, -1, -1, -1, -1, -1, -1, -1]],
                    [[-2, -2, -2, -2, -2, -2, -2, -2]],
                    [[11, 12, 13, 14, 15, 16, 17, 18]],
                ]
            ),
            4, 0.5, 'triangle',
            np.array([[1, -0.25, -1.5, 2, 15, 2.5, -1.5, 1.25]])
        ),
        (
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
            4, 0.5, 'triangle',
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


@pytest.mark.parametrize(
    "timbre_spec, is_correct",
    [
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=trapezoid,
                fundamental_effects=[],
                overtones_specs=[]
            ),
            True
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=trapezoid,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='sine',
                        frequency_ratio=1.5,
                        volume_share=0.4,
                        volume_envelope_fn=trapezoid,
                        effects=[]
                    )
                ]
            ),
            True
        ),
        (
            TimbreSpec(
                fundamental_waveform='unknown',
                fundamental_volume_envelope_fn=trapezoid,
                fundamental_effects=[],
                overtones_specs=[]
            ),
            False
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=trapezoid,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='unknown',
                        frequency_ratio=1.5,
                        volume_share=0.4,
                        volume_envelope_fn=trapezoid,
                        effects=[]
                    )
                ]
            ),
            False
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=trapezoid,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='sine',
                        frequency_ratio=0.5,
                        volume_share=0.4,
                        volume_envelope_fn=trapezoid,
                        effects=[]
                    )
                ]
            ),
            False
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=trapezoid,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='sine',
                        frequency_ratio=1.5,
                        volume_share=2,
                        volume_envelope_fn=trapezoid,
                        effects=[]
                    )
                ]
            ),
            False
        ),
    ]
)
def test_validate_timbre_spec(
        timbre_spec: TimbreSpec, is_correct: bool
) -> None:
    """Test `validate_timbre_spec` function."""
    if is_correct:
        validate_timbre_spec(timbre_spec)
    else:
        with pytest.raises(ValueError):
            validate_timbre_spec(timbre_spec)
