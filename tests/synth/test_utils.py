"""
Test `sinethesizer.synth.utils` module.

Author: Nikolay Lysenko
"""


import pytest

from sinethesizer.synth.adsr_envelopes import constant_with_linear_ends
from sinethesizer.synth.timbre import OvertoneSpec, TimbreSpec
from sinethesizer.synth.utils import validate_timbre_spec


@pytest.mark.parametrize(
    "timbre_spec, is_correct",
    [
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=constant_with_linear_ends,
                fundamental_effects=[],
                overtones_specs=[]
            ),
            True
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=constant_with_linear_ends,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='sine',
                        frequency_ratio=1.5,
                        volume_share=0.4,
                        volume_envelope_fn=constant_with_linear_ends,
                        effects=[]
                    )
                ]
            ),
            True
        ),
        (
            TimbreSpec(
                fundamental_waveform='unknown',
                fundamental_volume_envelope_fn=constant_with_linear_ends,
                fundamental_effects=[],
                overtones_specs=[]
            ),
            False
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=constant_with_linear_ends,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='unknown',
                        frequency_ratio=1.5,
                        volume_share=0.4,
                        volume_envelope_fn=constant_with_linear_ends,
                        effects=[]
                    )
                ]
            ),
            False
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=constant_with_linear_ends,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='sine',
                        frequency_ratio=0.5,
                        volume_share=0.4,
                        volume_envelope_fn=constant_with_linear_ends,
                        effects=[]
                    )
                ]
            ),
            False
        ),
        (
            TimbreSpec(
                fundamental_waveform='sine',
                fundamental_volume_envelope_fn=constant_with_linear_ends,
                fundamental_effects=[],
                overtones_specs=[
                    OvertoneSpec(
                        waveform='sine',
                        frequency_ratio=1.5,
                        volume_share=2,
                        volume_envelope_fn=constant_with_linear_ends,
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
