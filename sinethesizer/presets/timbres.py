"""
Define basic timbres.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Tuple

from sinethesizer.synth.timbre import TimbreSpec, OvertoneSpec
from sinethesizer.presets.adsr_envelopes import (
    relative_adsr, constant_with_linear_ends
)


sine = TimbreSpec(
    fundamental_waveform='sine',
    fundamental_volume_envelope_fn=constant_with_linear_ends,
    overtones_specs=[]
)

poor_organ = TimbreSpec(
    fundamental_waveform='sine',
    fundamental_volume_envelope_fn=constant_with_linear_ends,
    overtones_specs=[
        OvertoneSpec(
            waveform='sine',
            frequency_ratio=1.5,
            volume_share=0.4,
            volume_envelope_fn=constant_with_linear_ends
        )
    ]
)


def define_sine_with_n_harmonics(n: int) -> Tuple[str, TimbreSpec]:
    """
    Define a parametric family of timbres called 'sine_with_n_harmonics'.

    :param n:
        number of harmonics
    :return:
        name of timbre and its specification
    """
    timbre_name = f'sine_with_{n}_harmonics'
    timbre_spec = TimbreSpec(
        fundamental_waveform='sine',
        fundamental_volume_envelope_fn=partial(
            relative_adsr,
            attack_share=0.15,
            decay_share=0.15,
            sustain_level=0.6,
            release_share=0.2
        ),
        overtones_specs=[
            OvertoneSpec(
                waveform='sine',
                frequency_ratio=i,
                volume_share=0.2 / 2 ** (i-2),
                volume_envelope_fn=partial(
                    relative_adsr,
                    attack_share=0.15,
                    decay_share=0.15,
                    sustain_level=0.6,
                    release_share=0.1 * (i + 1)
                )
            )
            for i in range(2, n+2)
        ]
    )
    return timbre_name, timbre_spec
