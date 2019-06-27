"""
Define basic timbres.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Tuple

from sinethesizer.synth.timbres import TimbreSpec, OvertoneSpec
from sinethesizer.presets.basic_envelopes import (
    constant_with_linear_ends, wide_spike
)


sine = TimbreSpec(
    fundamental_waveform='sine',
    fundamental_volume_envelope_fn=wide_spike,
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
            constant_with_linear_ends, end_share=0.1
        ),
        overtones_specs=[
            OvertoneSpec(
                waveform='sine',
                frequency_ratio=i,
                volume_share=0.2 / 2 ** (i-2),
                volume_envelope_fn=partial(
                    constant_with_linear_ends, end_share=0.1 * i
                )
            )
            for i in range(2, n+2)
        ]
    )
    return timbre_name, timbre_spec
